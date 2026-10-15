/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qwen3_asr_backend.hpp"

#include <sndfile.h>
#include <curl/curl.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "audio_utils.hpp"

namespace asr {

// ============================================================================
// Base64 encoder (RFC 4648)
// ============================================================================

static const char kBase64Table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string Qwen3ASRBackend::base64Encode(const void* data, size_t len) {
    const auto* src = static_cast<const uint8_t*>(data);
    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = static_cast<uint32_t>(src[i]) << 16;
        if (i + 1 < len) n |= static_cast<uint32_t>(src[i + 1]) << 8;
        if (i + 2 < len) n |= static_cast<uint32_t>(src[i + 2]);

        out += kBase64Table[(n >> 18) & 0x3F];
        out += kBase64Table[(n >> 12) & 0x3F];
        out += (i + 1 < len) ? kBase64Table[(n >> 6) & 0x3F] : '=';
        out += (i + 2 < len) ? kBase64Table[n & 0x3F] : '=';
    }
    return out;
}

// ============================================================================
// WAV in-memory encoder (PCM16, mono)
// ============================================================================

std::string Qwen3ASRBackend::wavEncode(
    const float* samples, size_t count, int sample_rate) {
    // Convert float [-1,1] to int16
    std::vector<int16_t> pcm16(count);
    for (size_t i = 0; i < count; ++i) {
        float v = samples[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm16[i] = static_cast<int16_t>(v * 32767.0f);
    }

    // Build WAV using libsndfile virtual IO
    SF_INFO info{};
    info.samplerate = sample_rate;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SF_VIRTUAL_IO vio;
    struct MemBuf {
        std::string data;
        sf_count_t pos = 0;
    } buf;

    vio.get_filelen = [](void* ud) -> sf_count_t {
        return static_cast<MemBuf*>(ud)->data.size();
    };
    vio.seek = [](sf_count_t offset, int whence, void* ud) -> sf_count_t {
        auto* b = static_cast<MemBuf*>(ud);
        sf_count_t new_pos;
        switch (whence) {
            case SEEK_SET: new_pos = offset; break;
            case SEEK_CUR: new_pos = b->pos + offset; break;
            case SEEK_END: new_pos = static_cast<sf_count_t>(b->data.size()) + offset; break;
            default: return -1;
        }
        if (new_pos < 0) return -1;
        b->pos = new_pos;
        return new_pos;
    };
    vio.read = [](void* ptr, sf_count_t cnt, void* ud) -> sf_count_t {
        auto* b = static_cast<MemBuf*>(ud);
        sf_count_t avail = static_cast<sf_count_t>(b->data.size()) - b->pos;
        if (avail <= 0) return 0;
        sf_count_t n = (cnt < avail) ? cnt : avail;
        std::memcpy(ptr, b->data.data() + b->pos, static_cast<size_t>(n));
        b->pos += n;
        return n;
    };
    vio.write = [](const void* ptr, sf_count_t cnt, void* ud) -> sf_count_t {
        auto* b = static_cast<MemBuf*>(ud);
        size_t end = static_cast<size_t>(b->pos) + static_cast<size_t>(cnt);
        if (end > b->data.size()) b->data.resize(end);
        std::memcpy(&b->data[static_cast<size_t>(b->pos)], ptr, static_cast<size_t>(cnt));
        b->pos += cnt;
        return cnt;
    };
    vio.tell = [](void* ud) -> sf_count_t {
        return static_cast<MemBuf*>(ud)->pos;
    };

    SNDFILE* sf = sf_open_virtual(&vio, SFM_WRITE, &info, &buf);
    if (!sf) return {};

    sf_write_short(sf, pcm16.data(), static_cast<sf_count_t>(count));
    sf_close(sf);

    return buf.data;
}

// ============================================================================
// Minimal JSON content extractor
// ============================================================================

// Extracts the value of the first "content" key from llama-server JSON response.
// Handles both string content and array-of-objects content.
std::string Qwen3ASRBackend::extractContent(const std::string& json) {
    // Find "content" key
    const std::string key = "\"content\"";
    auto pos = json.find(key);
    if (pos == std::string::npos) return {};

    // Skip past "content" and any whitespace/colon
    pos += key.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n'))
        ++pos;

    if (pos >= json.size()) return {};

    if (json[pos] == '"') {
        // String content – extract until unescaped closing quote
        ++pos;
        std::string result;
        while (pos < json.size()) {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                char next = json[pos + 1];
                switch (next) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    default: result += next; break;
                }
                pos += 2;
            } else if (json[pos] == '"') {
                break;
            } else {
                result += json[pos++];
            }
        }
        return result;
    }

    if (json[pos] == '[') {
        // Array content – find all "text" values and concatenate
        auto end = json.find(']', pos);
        if (end == std::string::npos) end = json.size();
        std::string segment = json.substr(pos, end - pos);

        std::string result;
        const std::string text_key = "\"text\"";
        size_t search = 0;
        while ((search = segment.find(text_key, search)) != std::string::npos) {
            search += text_key.size();
            while (search < segment.size() && segment[search] != '"') ++search;
            if (search >= segment.size()) break;
            ++search;  // skip opening quote
            while (search < segment.size() && segment[search] != '"') {
                if (segment[search] == '\\' && search + 1 < segment.size()) {
                    result += segment[search + 1];
                    search += 2;
                } else {
                    result += segment[search++];
                }
            }
        }
        return result;
    }

    return {};
}

// ============================================================================
// HTTP POST via libcurl
// ============================================================================

static size_t curlWriteCallback(char* ptr, size_t size, size_t nmemb, void* ud) {
    auto* buf = static_cast<std::string*>(ud);
    buf->append(ptr, size * nmemb);
    return size * nmemb;
}

std::string Qwen3ASRBackend::httpPost(
    const std::string& url, const std::string& body,
    int64_t timeout_sec, std::string& err_msg) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        err_msg = "curl_easy_init failed";
        return {};
    }

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<int64_t>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_sec);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        err_msg = curl_easy_strerror(res);
        response.clear();
    } else {
        int64_t http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200) {
            err_msg = "HTTP " + std::to_string(http_code);
            // keep response for debugging
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return response;
}

// ============================================================================
// Language → prompt string
// ============================================================================

std::string Qwen3ASRBackend::languageToPrompt(Language lang) {
    switch (lang) {
        case Language::ZH:  return "Chinese";
        case Language::EN:  return "English";
        case Language::JA:  return "Japanese";
        case Language::KO:  return "Korean";
        case Language::YUE: return "Chinese";
        default:            return "auto";
    }
}

// ============================================================================
// Audio conversion
// ============================================================================

std::vector<float> Qwen3ASRBackend::convertToFloat(const AudioChunk& audio) {
    std::vector<float> result;
    if (!audio.data || audio.size_bytes == 0) return result;

    switch (audio.format) {
        case AudioFormat::PCM_S16LE: {
            const auto* data = static_cast<const int16_t*>(audio.data);
            size_t samples = audio.size_bytes / sizeof(int16_t);
            result.resize(samples);
            for (size_t i = 0; i < samples; ++i)
                result[i] = static_cast<float>(data[i]) / 32768.0f;
            break;
        }
        case AudioFormat::PCM_F32LE: {
            const auto* data = static_cast<const float*>(audio.data);
            size_t samples = audio.size_bytes / sizeof(float);
            result.assign(data, data + samples);
            break;
        }
        default:
            break;
    }

    // Stereo to mono
    if (audio.channels > 1 && !result.empty()) {
        size_t mono_samples = result.size() / audio.channels;
        std::vector<float> mono(mono_samples);
        for (size_t i = 0; i < mono_samples; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < audio.channels; ++ch)
                sum += result[i * audio.channels + ch];
            mono[i] = sum / audio.channels;
        }
        return mono;
    }
    return result;
}

// ============================================================================
// Lifecycle
// ============================================================================

Qwen3ASRBackend::Qwen3ASRBackend() = default;
Qwen3ASRBackend::~Qwen3ASRBackend() { shutdown(); }

ErrorInfo Qwen3ASRBackend::initialize(const ASRConfig& config) {
    if (initialized_.load())
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Already initialized");

    config_ = config;

    // Read extra_params
    auto get = [&](const std::string& key, const std::string& def) -> std::string {
        auto it = config_.extra_params.find(key);
        return (it != config_.extra_params.end() && !it->second.empty()) ? it->second : def;
    };

    endpoint_ = get("endpoint", "http://127.0.0.1:8063/v1/chat/completions");
    model_ = get("model", "qwen3-asr");
    timeout_sec_ = std::stol(get("timeout", "60"));

    std::cout << "[Qwen3ASR] endpoint=" << endpoint_
            << " model=" << model_
            << " timeout=" << timeout_sec_ << "s" << std::endl;

    initialized_.store(true);
    return ErrorInfo::ok();
}

void Qwen3ASRBackend::shutdown() {
    initialized_.store(false);
}

// ============================================================================
// Core transcription
// ============================================================================

ErrorInfo Qwen3ASRBackend::transcribe(
    const float* samples, size_t count,
    int sample_rate, std::string& out_text) {
    // 1. Encode to WAV bytes
    std::string wav = wavEncode(samples, count, sample_rate);
    if (wav.empty()) {
        std::cerr << "[Qwen3ASR] WAV encoding failed" << std::endl;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "WAV encoding failed");
    }
    std::cerr << "[Qwen3ASR] WAV encoded: " << wav.size() << " bytes, "
            << count << " samples @ " << sample_rate << " Hz" << std::endl;

    // 2. Base64 encode
    std::string b64 = base64Encode(wav.data(), wav.size());
    std::cerr << "[Qwen3ASR] Base64 length: " << b64.size() << std::endl;

    // 3. Build JSON payload
    std::string lang_prompt = languageToPrompt(config_.language);

    // Escape b64 is safe (only base64 chars). Language prompt is also safe.
    std::ostringstream json;
    json << R"({"model":")" << model_
        << R"(","messages":[{"role":"user","content":[)"
        << R"({"type":"input_audio","input_audio":{"data":")" << b64
        << R"(","format":"wav"}},)"
        << R"({"type":"text","text":"language )" << lang_prompt << R"(<asr_text>"})"
        << R"(]}],"max_tokens":)" << 512
        << R"(,"temperature":0})";

    std::cerr << "[Qwen3ASR] POST " << endpoint_
            << " payload=" << json.str().size() << " bytes" << std::endl;

    // 4. HTTP POST
    std::string err_msg;
    std::string response = httpPost(endpoint_, json.str(), timeout_sec_, err_msg);

    if (!err_msg.empty()) {
        std::cerr << "[Qwen3ASR] HTTP error: " << err_msg << std::endl;
    }
    if (response.empty() && !err_msg.empty())
        return ErrorInfo::error(ErrorCode::NETWORK_ERROR,
                                "llama-server request failed: " + err_msg);

    std::cerr << "[Qwen3ASR] Response (" << response.size() << " bytes): "
                << response.substr(0, 300) << std::endl;

    // 5. Parse response
    out_text = extractContent(response);
    std::cerr << "[Qwen3ASR] Extracted text: [" << out_text << "]" << std::endl;

    if (out_text.empty() && response.find("\"choices\"") == std::string::npos)
        return ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
                                "Invalid response from llama-server",
                                response.substr(0, 200));

    return ErrorInfo::ok();
}

// ============================================================================
// Offline recognition
// ============================================================================

ErrorInfo Qwen3ASRBackend::recognize(const AudioChunk& audio, RecognitionResult& result) {
    if (!initialized_.load())
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Not initialized");

    auto t0 = std::chrono::steady_clock::now();

    auto audio_float = convertToFloat(audio);
    if (audio_float.empty())
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty audio");

    if (audio.sample_rate <= 0)
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            "Invalid sample rate: " + std::to_string(audio.sample_rate) + " Hz");

    int64_t audio_ms = (audio_float.size() * 1000) / audio.sample_rate;

    auto model_audio = audio_utils::normalizeSampleRate(
        std::move(audio_float), audio.sample_rate, config_.sample_rate);
    if (model_audio.empty())
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty audio after resampling");

    std::string text;
    auto err = transcribe(model_audio.data(), model_audio.size(),
                            config_.sample_rate, text);
    if (!err.isOk()) return err;

    auto t1 = std::chrono::steady_clock::now();
    int64_t proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    result = buildResult(text, audio_ms, proc_ms);
    return ErrorInfo::ok();
}

ErrorInfo Qwen3ASRBackend::recognizeFile(const std::string& file_path,
                                        RecognitionResult& result) {
    if (!initialized_.load())
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Not initialized");

    auto t0 = std::chrono::steady_clock::now();

    // Read file via libsndfile
    SF_INFO info{};
    SNDFILE* sf = sf_open(file_path.c_str(), SFM_READ, &info);
    if (!sf)
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
            "Cannot open audio file: " + file_path, sf_strerror(nullptr));

    std::vector<float> audio(info.frames * info.channels);
    sf_read_float(sf, audio.data(), static_cast<sf_count_t>(audio.size()));
    sf_close(sf);

    // Stereo to mono
    if (info.channels > 1) {
        std::vector<float> mono(info.frames);
        for (sf_count_t i = 0; i < info.frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < info.channels; ++ch)
                sum += audio[i * info.channels + ch];
            mono[i] = sum / info.channels;
        }
        audio = std::move(mono);
    }

    if (info.samplerate <= 0)
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            "Invalid sample rate: " + std::to_string(info.samplerate) + " Hz");

    int64_t audio_ms = (info.frames * 1000) / info.samplerate;

    auto model_audio = audio_utils::normalizeSampleRate(
        std::move(audio), info.samplerate, config_.sample_rate);
    if (model_audio.empty())
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty audio after resampling");

    std::string text;
    auto err = transcribe(model_audio.data(), model_audio.size(), config_.sample_rate, text);
    if (!err.isOk()) return err;

    auto t1 = std::chrono::steady_clock::now();
    int64_t proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    result = buildResult(text, audio_ms, proc_ms);
    return ErrorInfo::ok();
}

// ============================================================================
// Result builder
// ============================================================================

RecognitionResult Qwen3ASRBackend::buildResult(const std::string& text,
                                                int64_t audio_duration_ms,
                                                int64_t processing_time_ms) {
    RecognitionResult result;
    SentenceResult sentence;
    sentence.text = text;
    sentence.begin_time_ms = 0;
    sentence.end_time_ms = static_cast<int32_t>(audio_duration_ms);
    sentence.confidence = 1.0f;
    sentence.is_final = true;
    sentence.detected_language = config_.language;

    result.sentences.push_back(sentence);
    result.audio_duration_ms = audio_duration_ms;
    result.processing_time_ms = processing_time_ms;
    result.rtf = (audio_duration_ms > 0)
        ? static_cast<float>(processing_time_ms) / audio_duration_ms
        : 0.0f;
    result.first_result_latency_ms = processing_time_ms;
    result.final_result_latency_ms = processing_time_ms;
    return result;
}

}  // namespace asr
