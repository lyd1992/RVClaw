/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "tts_service.h"

#if defined(__linux__)
#include <malloc.h>
#endif

#include <cstdint>

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "backends/tts_backend.hpp"

namespace {

void releaseUnusedHeapMemory() noexcept {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

}  // namespace

namespace SpacemiT {

// =============================================================================
// TtsEngineResult 实现
// =============================================================================

struct TtsEngineResult::Impl {
    std::vector<float> audio_float;
    int sample_rate = 22050;
    int duration_ms = 0;
    int processing_time_ms = 0;
    bool success = false;
    bool is_sentence_end = false;
    std::string message;
    std::string request_id;
};

TtsEngineResult::TtsEngineResult() : impl_(std::make_unique<Impl>()) {}
TtsEngineResult::~TtsEngineResult() = default;

TtsEngineResult::TtsEngineResult(TtsEngineResult&&) noexcept = default;
TtsEngineResult& TtsEngineResult::operator=(TtsEngineResult&&) noexcept = default;

std::vector<uint8_t> TtsEngineResult::GetAudioData() const {
    auto int16_data = GetAudioInt16();
    std::vector<uint8_t> result(int16_data.size() * 2);
    for (size_t i = 0; i < int16_data.size(); ++i) {
        result[i * 2] = static_cast<uint8_t>(int16_data[i] & 0xFF);
        result[i * 2 + 1] = static_cast<uint8_t>((int16_data[i] >> 8) & 0xFF);
    }
    return result;
}

std::vector<float> TtsEngineResult::GetAudioFloat() const {
    return impl_->audio_float;
}

std::vector<int16_t> TtsEngineResult::GetAudioInt16() const {
    std::vector<int16_t> result(impl_->audio_float.size());
    for (size_t i = 0; i < impl_->audio_float.size(); ++i) {
        float sample = impl_->audio_float[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        result[i] = static_cast<int16_t>(sample * 32767.0f);
    }
    return result;
}

std::string TtsEngineResult::GetTimestamp() const {
    return "{}";  // 未实现时间戳
}

std::string TtsEngineResult::GetResponse() const {
    return "{}";  // JSON 响应
}

std::string TtsEngineResult::GetRequestId() const {
    return impl_->request_id;
}

bool TtsEngineResult::IsSuccess() const {
    return impl_->success;
}

std::string TtsEngineResult::GetCode() const {
    return impl_->success ? "0" : "1";
}

std::string TtsEngineResult::GetMessage() const {
    return impl_->message;
}

bool TtsEngineResult::IsEmpty() const {
    return impl_->audio_float.empty();
}

bool TtsEngineResult::IsSentenceEnd() const {
    return impl_->is_sentence_end;
}

int TtsEngineResult::GetSampleRate() const {
    return impl_->sample_rate;
}

int TtsEngineResult::GetDurationMs() const {
    return impl_->duration_ms;
}

int TtsEngineResult::GetProcessingTimeMs() const {
    return impl_->processing_time_ms;
}

float TtsEngineResult::GetRTF() const {
    if (impl_->duration_ms == 0) return 0.0f;
    return static_cast<float>(impl_->processing_time_ms) / impl_->duration_ms;
}

bool TtsEngineResult::SaveToFile(const std::string& file_path) const {
    if (impl_->audio_float.empty()) {
        return false;
    }

    std::ofstream file(file_path, std::ios::binary);
    if (!file) {
        return false;
    }

    auto int16_data = GetAudioInt16();
    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = impl_->sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int data_size = static_cast<int>(int16_data.size() * 2);
    int file_size = 36 + data_size;

    // Write WAV header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    int fmt_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    int16_t audio_format = 1;  // PCM
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    int16_t channels = static_cast<int16_t>(num_channels);
    file.write(reinterpret_cast<const char*>(&channels), 2);
    file.write(reinterpret_cast<const char*>(&impl_->sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    int16_t block_align_short = static_cast<int16_t>(block_align);
    file.write(reinterpret_cast<const char*>(&block_align_short), 2);
    int16_t bits_short = static_cast<int16_t>(bits_per_sample);
    file.write(reinterpret_cast<const char*>(&bits_short), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);

    // Write audio data
    file.write(reinterpret_cast<const char*>(int16_data.data()), data_size);

    return file.good();
}

// =============================================================================
// TtsEngine 实现
// =============================================================================

// 转换 SpacemiT::BackendType 到 tts::BackendType
static tts::BackendType convertBackendType(BackendType type) {
    switch (type) {
        case BackendType::MATCHA_ZH:
            return tts::BackendType::MATCHA_ZH;
        case BackendType::MATCHA_EN:
            return tts::BackendType::MATCHA_EN;
        case BackendType::MATCHA_ZH_EN:
            return tts::BackendType::MATCHA_ZH_EN;
        case BackendType::VITS:
            return tts::BackendType::VITS;
        case BackendType::PIPER:
            return tts::BackendType::PIPER;
        case BackendType::KOKORO:
            return tts::BackendType::KOKORO;
        default:
            return tts::BackendType::MATCHA_ZH;
    }
}

struct TtsEngine::Impl {
    mutable std::mutex mutex;
    std::unique_ptr<tts::ITtsBackend> backend;
    TtsConfig config;
    bool initialized = false;

    bool init(const TtsConfig& cfg) {
        config = cfg;

        // 创建后端
        auto backend_type = convertBackendType(cfg.backend);
        backend = tts::TtsBackendFactory::create(backend_type);
        if (!backend) {
            std::cerr << "Failed to create TTS backend" << std::endl;
            return false;
        }

        // 转换配置
        tts::TtsConfig internal_config;
        internal_config.backend = backend_type;
        internal_config.model_dir = cfg.model_dir;
        internal_config.voice = cfg.voice;
        internal_config.speaker_id = cfg.speaker_id;
        internal_config.speech_rate = cfg.speech_rate;
        internal_config.sample_rate = cfg.sample_rate;
        internal_config.num_threads = cfg.num_threads;
        internal_config.enable_warmup = cfg.enable_warmup;
        internal_config.provider = cfg.provider;

        // 初始化
        auto error = backend->initialize(internal_config);
        if (!error.isOk()) {
            std::cerr << "Failed to initialize TTS backend: " << error.message << std::endl;
            return false;
        }

        initialized = true;

        // 应用初始 lexicon
        if (!cfg.lexicon.empty()) {
            std::vector<tts::ITtsBackend::LexiconEntry> lex_entries;
            lex_entries.reserve(cfg.lexicon.size());
            for (const auto& e : cfg.lexicon) {
                lex_entries.push_back({e.word, e.phoneme, e.locale});
            }
            backend->updateLexicon(lex_entries);
        }

        return true;
    }

    void shutdownUnlocked() noexcept {
        if (!backend) {
            initialized = false;
            return;
        }
        try {
            backend->shutdown();
        } catch (const std::exception& e) {
            std::cerr << "TTS backend shutdown failed: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "TTS backend shutdown failed: unknown error" << std::endl;
        }
        backend.reset();
        initialized = false;
        releaseUnusedHeapMemory();
    }

    void shutdownThreadSafe() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        shutdownUnlocked();
    }
};

TtsEngine::TtsEngine(BackendType backend, const std::string& model_dir)
    : impl_(std::make_unique<Impl>()) {
    TtsConfig config;
    config.backend = backend;
    config.model_dir = model_dir;

    switch (backend) {
        case BackendType::MATCHA_ZH:
            config.sample_rate = 22050;
            break;
        case BackendType::MATCHA_EN:
            config.sample_rate = 22050;
            break;
        case BackendType::MATCHA_ZH_EN:
            config.sample_rate = 16000;
            break;
        case BackendType::KOKORO:
            config.sample_rate = 24000;
            break;
        default:
            config.sample_rate = 22050;
    }

    impl_->init(config);
}

TtsEngine::TtsEngine(const TtsConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->init(config);
}

TtsEngine::~TtsEngine() {
    impl_->shutdownUnlocked();
}

void TtsEngine::Shutdown() {
    impl_->shutdownThreadSafe();
}

std::shared_ptr<TtsEngineResult> TtsEngine::Call(const std::string& text,
                                                const TtsConfig& config) {
    auto result = std::make_shared<TtsEngineResult>();
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->initialized || !impl_->backend) {
        result->impl_->success = false;
        result->impl_->message = "Engine not initialized";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    tts::SynthesisResult synthesis_result;
    auto error = impl_->backend->synthesize(text, synthesis_result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!error.isOk()) {
        result->impl_->success = false;
        result->impl_->message = error.message;
        return result;
    }

    result->impl_->audio_float = std::move(synthesis_result.audio.samples);
    result->impl_->sample_rate = synthesis_result.audio.sample_rate;
    result->impl_->duration_ms = static_cast<int>(synthesis_result.audio_duration_ms);
    result->impl_->processing_time_ms = static_cast<int>(duration.count());
    result->impl_->success = true;
    result->impl_->is_sentence_end = true;

    return result;
}

bool TtsEngine::CallToFile(const std::string& text, const std::string& file_path) {
    auto result = Call(text);
    if (!result || !result->IsSuccess()) {
        return false;
    }
    return result->SaveToFile(file_path);
}

void TtsEngine::StreamingCall(const std::string& text,
    std::shared_ptr<TtsResultCallback> callback,
    const TtsConfig& config) {
    if (callback) {
        callback->OnOpen();
    }

    auto result = Call(text, config);

    if (callback) {
        if (result && result->IsSuccess()) {
            callback->OnEvent(result);
            callback->OnComplete();
        } else {
            callback->OnError(result ? result->GetMessage() : "Synthesis failed");
        }
        callback->OnClose();
    }
}

std::shared_ptr<TtsEngine::DuplexStream> TtsEngine::StartDuplexStream(
    std::shared_ptr<TtsResultCallback> callback,
    const TtsConfig& config) {
    // 简单实现：不支持真正的双向流
    return nullptr;
}

void TtsEngine::SetSpeed(float speed) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config.speech_rate = speed;
    if (impl_->backend) {
        impl_->backend->setSpeed(speed);
    }
}

void TtsEngine::SetSpeaker(int speaker_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config.speaker_id = speaker_id;
    if (impl_->backend) {
        impl_->backend->setSpeaker(speaker_id);
    }
}

void TtsEngine::SetVolume(int volume) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config.volume = volume;
    if (impl_->backend) {
        impl_->backend->setVolume(volume / 100.0f);
    }
}

void TtsEngine::UpdateLexicon(const std::vector<PronunciationEntry>& entries) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->backend) return;
    std::vector<tts::ITtsBackend::LexiconEntry> lex_entries;
    lex_entries.reserve(entries.size());
    for (const auto& e : entries) {
        lex_entries.push_back({e.word, e.phoneme, e.locale});
    }
    impl_->backend->updateLexicon(lex_entries);
}

TtsConfig TtsEngine::GetConfig() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->config;
}

bool TtsEngine::IsInitialized() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->initialized;
}

std::string TtsEngine::GetEngineName() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->backend) {
        return impl_->backend->getName();
    }
    return "Unknown";
}

BackendType TtsEngine::GetBackendType() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->config.backend;
}

int TtsEngine::GetNumSpeakers() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->backend) {
        return impl_->backend->getNumSpeakers();
    }
    return 1;
}

int TtsEngine::GetSampleRate() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->backend) {
        return impl_->backend->getSampleRate();
    }
    return impl_->config.sample_rate;
}

std::string TtsEngine::GetLastRequestId() const {
    return "";
}

}  // namespace SpacemiT
