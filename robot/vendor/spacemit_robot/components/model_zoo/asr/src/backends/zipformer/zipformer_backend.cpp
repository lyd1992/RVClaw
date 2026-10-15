/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backends/zipformer/zipformer_backend.hpp"

#include <sndfile.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "backends/zipformer/zipformer_ctc_decoder.h"
#include "backends/zipformer/zipformer_ctc_model.h"
#include "backends/zipformer/zipformer_feature_extractor.h"
#include "backends/zipformer/zipformer_symbol_table.h"
#include "model_downloader.hpp"

namespace asr {

static std::string expandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    if (!home) return path;
    return std::string(home) + path.substr(1);
}

ZipformerBackend::ZipformerBackend() = default;

ZipformerBackend::~ZipformerBackend() { shutdown(); }

ErrorInfo ZipformerBackend::initialize(const ASRConfig& config) {
    if (initialized_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED,
                                "Backend already initialized");
    }

    config_ = config;

    // Check and download models if needed
    std::string model_dir = "~/.cache/models/asr/zipformer";
    if (!config_.model_path.empty()) {
        size_t last_slash = config_.model_path.rfind('/');
        if (last_slash != std::string::npos) {
            model_dir = config_.model_path.substr(0, last_slash);
        }
    }

    ModelDownloader downloader({
        .model_dir = model_dir,
        .url = "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/zipformer.tar.gz",
        .archive_name = "zipformer.tar.gz",
        .archive_subdir = "zipformer",
        .required_files = {"ctc-epoch-20-avg-1-chunk-16-left-128.q.onnx", "tokens.txt"},
    });
    if (!downloader.ensure()) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
                                "Zipformer models not available");
    }

    try {
        std::string vocab_path = expandPath(config_.vocab_path);
        sym_ = std::make_unique<::zipformer::SymbolTable>();
        if (!sym_->Load(vocab_path)) {
            return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
                                    "Failed to load tokens.txt",
                                    "Path: " + vocab_path);
        }

        ::zipformer::CtcModel::Config model_config;
        model_config.model_path = expandPath(config_.model_path);
        model_config.num_threads = config_.num_threads;

        auto it_provider = config_.extra_params.find("provider");
        if (it_provider != config_.extra_params.end()) {
            model_config.provider = it_provider->second;
        }
        auto it_qmodel = config_.extra_params.find("quantized_model_path");
        if (it_qmodel != config_.extra_params.end()) {
            model_config.quantized_model_path = expandPath(it_qmodel->second);
        }

        std::cout << "[ZipformerBackend] model_path: " << model_config.model_path
                    << std::endl;
        std::cout << "[ZipformerBackend] quantized_model_path: "
                    << model_config.quantized_model_path << std::endl;
        std::cout << "[ZipformerBackend] provider: " << model_config.provider
                    << std::endl;

        model_ = std::make_unique<::zipformer::CtcModel>(model_config);

        decoder_ = std::make_unique<::zipformer::CtcDecoder>(*sym_, 0);

        ::zipformer::FeatureExtractor::Config feat_config;
        feat_config.sample_rate = config_.sample_rate;
        feat_ = std::make_unique<::zipformer::FeatureExtractor>(feat_config);
    } catch (const std::exception& e) {
        std::cerr << "[ZipformerBackend] Init failed: " << e.what() << std::endl;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
                                "Failed to initialize Zipformer model", e.what());
    }

    initialized_.store(true);
    std::cout << "[ZipformerBackend] Initialized successfully" << std::endl;
    return ErrorInfo::ok();
}

void ZipformerBackend::shutdown() {
    if (stream_active_.load()) {
        stopStream();
    }
    decoder_.reset();
    model_.reset();
    feat_.reset();
    sym_.reset();
    initialized_.store(false);
}

// ---------------------------------------------------------------------------
// Core recognition logic
// ---------------------------------------------------------------------------

std::string ZipformerBackend::recognizeAudio(const float* data, size_t samples,
                                                int sample_rate) {
    ::zipformer::FeatureExtractor::Config feat_config;
    feat_config.sample_rate = config_.sample_rate;
    ::zipformer::FeatureExtractor feat(feat_config);

    feat.AcceptWaveform(sample_rate, data, static_cast<int32_t>(samples));
    feat.InputFinished();

    int32_t num_frames = feat.NumFramesReady();
    int32_t dim = feat.FeatureDim();
    if (num_frames <= 0) return "";

    auto features = feat.GetFrames(0, num_frames);
    auto log_probs = model_->Run(features, num_frames, dim);
    return decoder_->Decode(log_probs.data(), model_->OutputFrames(),
                            model_->VocabSize());
}

// ---------------------------------------------------------------------------
// Offline recognition
// ---------------------------------------------------------------------------

ErrorInfo ZipformerBackend::recognize(const AudioChunk& audio,
                                        RecognitionResult& result) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED,
                                "Backend not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    auto audio_float = convertToFloat(audio);
    if (audio_float.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
                                "Empty or invalid audio data");
    }

    int64_t audio_duration_ms =
            static_cast<int64_t>(audio_float.size()) * 1000 / config_.sample_rate;

    std::string text;
    try {
        text = recognizeAudio(audio_float.data(), audio_float.size(),
                                audio.sample_rate);
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
                                "Zipformer inference failed", e.what());
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count();

    result = buildResult(text, audio_duration_ms, processing_time, true);
    return ErrorInfo::ok();
}

ErrorInfo ZipformerBackend::recognizeFile(const std::string& file_path,
                                            RecognitionResult& result) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED,
                                "Backend not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    SF_INFO sf_info;
    memset(&sf_info, 0, sizeof(sf_info));

    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sf_info);
    if (!file) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
                                "Failed to open audio file: " + file_path,
                                sf_strerror(nullptr));
    }

    std::vector<float> audio_data(sf_info.frames * sf_info.channels);
    sf_count_t frames_read =
            sf_read_float(file, audio_data.data(), audio_data.size());
    sf_close(file);

    if (frames_read <= 0) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
                                "Failed to read audio data from file");
    }

    std::vector<float>* audio_ptr = &audio_data;
    std::vector<float> mono_audio;

    if (sf_info.channels > 1) {
        mono_audio.resize(sf_info.frames);
        for (sf_count_t i = 0; i < sf_info.frames; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < sf_info.channels; ++ch) {
                sum += audio_data[i * sf_info.channels + ch];
            }
            mono_audio[i] = sum / sf_info.channels;
        }
        audio_ptr = &mono_audio;
    }

    int64_t audio_duration_ms =
            static_cast<int64_t>(sf_info.frames) * 1000 / sf_info.samplerate;

    std::string text;
    try {
        text = recognizeAudio(audio_ptr->data(), audio_ptr->size(),
                                sf_info.samplerate);
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
                                "Zipformer inference failed", e.what());
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count();

    result = buildResult(text, audio_duration_ms, processing_time, true);
    return ErrorInfo::ok();
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

ErrorInfo ZipformerBackend::startStream() {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED,
                                "Backend not initialized");
    }
    if (stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED,
                                "Stream already active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);
    audio_buffer_.clear();
    stream_active_.store(true);
    notifyStart();

    std::cout << "[ZipformerBackend] Stream started" << std::endl;
    return ErrorInfo::ok();
}

ErrorInfo ZipformerBackend::feedAudio(const AudioChunk& audio) {
    if (!stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);
    auto audio_float = convertToFloat(audio);
    audio_buffer_.insert(audio_buffer_.end(), audio_float.begin(),
                            audio_float.end());

    return ErrorInfo::ok();
}

ErrorInfo ZipformerBackend::stopStream() {
    if (!stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);

    if (!audio_buffer_.empty()) {
        processBufferedAudio(true);
    }

    stream_active_.store(false);
    audio_buffer_.clear();

    notifyComplete();
    notifyClose();

    std::cout << "[ZipformerBackend] Stream stopped" << std::endl;
    return ErrorInfo::ok();
}

ErrorInfo ZipformerBackend::flushStream() {
    if (!stream_active_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not active");
    }

    std::lock_guard<std::mutex> lock(stream_mutex_);

    if (!audio_buffer_.empty()) {
        processBufferedAudio(true);
    }

    std::cout << "[ZipformerBackend] Stream flushed" << std::endl;
    return ErrorInfo::ok();
}

void ZipformerBackend::processBufferedAudio(bool force_final) {
    if (audio_buffer_.empty() || !model_) return;

    auto start_time = std::chrono::steady_clock::now();

    int64_t audio_duration_ms =
            static_cast<int64_t>(audio_buffer_.size()) * 1000 / config_.sample_rate;

    std::string text;
    try {
        text = recognizeAudio(audio_buffer_.data(), audio_buffer_.size(),
                                config_.sample_rate);
    } catch (const std::exception& e) {
        notifyError(ErrorInfo::error(ErrorCode::INFERENCE_FAILED,
                                        "Zipformer inference failed", e.what()));
        return;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count();

    auto result = buildResult(text, audio_duration_ms, processing_time,
                                force_final);
    notifyResult(result);

    audio_buffer_.clear();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<float> ZipformerBackend::convertToFloat(const AudioChunk& audio) {
    std::vector<float> result;
    if (audio.data == nullptr || audio.size_bytes == 0) return result;

    switch (audio.format) {
        case AudioFormat::PCM_S16LE: {
            const int16_t* data = static_cast<const int16_t*>(audio.data);
            size_t samples = audio.size_bytes / sizeof(int16_t);
            result.resize(samples);
            for (size_t i = 0; i < samples; ++i) {
                result[i] = static_cast<float>(data[i]) / 32768.0f;
            }
            break;
        }
        case AudioFormat::PCM_F32LE: {
            const float* data = static_cast<const float*>(audio.data);
            size_t samples = audio.size_bytes / sizeof(float);
            result.assign(data, data + samples);
            break;
        }
        default:
            std::cerr << "[ZipformerBackend] Unsupported audio format" << std::endl;
            break;
    }

    if (audio.channels > 1 && !result.empty()) {
        size_t mono_samples = result.size() / audio.channels;
        std::vector<float> mono(mono_samples);
        for (size_t i = 0; i < mono_samples; ++i) {
            float sum = 0.0f;
            for (int ch = 0; ch < audio.channels; ++ch) {
                sum += result[i * audio.channels + ch];
            }
            mono[i] = sum / audio.channels;
        }
        return mono;
    }

    return result;
}

RecognitionResult ZipformerBackend::buildResult(const std::string& text,
                                                int64_t audio_duration_ms,
                                                int64_t processing_time_ms,
                                                bool is_final) {
    RecognitionResult result;

    SentenceResult sentence;
    sentence.text = text;
    sentence.begin_time_ms = 0;
    sentence.end_time_ms = static_cast<int32_t>(audio_duration_ms);
    sentence.confidence = 1.0f;
    sentence.is_final = is_final;
    sentence.detected_language = config_.language;

    result.sentences.push_back(sentence);
    result.audio_duration_ms = audio_duration_ms;
    result.processing_time_ms = processing_time_ms;

    if (audio_duration_ms > 0) {
        result.rtf =
                static_cast<float>(processing_time_ms) / audio_duration_ms;
    }

    result.first_result_latency_ms = processing_time_ms;
    result.final_result_latency_ms = processing_time_ms;

    return result;
}

}  // namespace asr
