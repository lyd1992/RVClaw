/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QWEN3_ASR_BACKEND_HPP
#define QWEN3_ASR_BACKEND_HPP

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "../asr_backend.hpp"

namespace asr {

/// Qwen3-ASR backend: delegates recognition to an external llama-server
/// via its OpenAI-compatible /v1/chat/completions endpoint.
///
/// Config keys used via extra_params:
///   "endpoint"  – full URL  (default http://127.0.0.1:8063/v1/chat/completions)
///   "model"     – model tag (default "qwen3-asr")
///   "timeout"   – seconds   (default "60")
class Qwen3ASRBackend : public IASRBackend {
public:
    Qwen3ASRBackend();
    ~Qwen3ASRBackend() override;

    // Lifecycle
    ErrorInfo initialize(const ASRConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }

    // Metadata
    BackendType getType() const override { return BackendType::QWEN3_ASR; }
    std::string getName() const override { return "Qwen3-ASR"; }
    std::string getVersion() const override { return "1.0.2"; }
    bool supportsStreaming() const override { return false; }

    std::vector<AudioFormat> getSupportedFormats() const override {
        return {AudioFormat::PCM_S16LE, AudioFormat::PCM_F32LE};
    }
    std::vector<int> getSupportedSampleRates() const override {
        return {16000};
    }

    // Offline recognition
    ErrorInfo recognize(const AudioChunk& audio, RecognitionResult& result) override;
    ErrorInfo recognizeFile(const std::string& file_path, RecognitionResult& result) override;

private:
    ASRConfig config_;
    std::atomic<bool> initialized_{false};

    std::string endpoint_;
    std::string model_;
    int64_t timeout_sec_ = 60;

    // Core: send audio samples to llama-server, return transcribed text.
    ErrorInfo transcribe(const float* samples, size_t count,
                            int sample_rate, std::string& out_text);

    // Audio helpers
    static std::vector<float> convertToFloat(const AudioChunk& audio);
    static std::string languageToPrompt(Language lang);

    // HTTP + encoding helpers
    static std::string base64Encode(const void* data, size_t len);
    static std::string wavEncode(const float* samples, size_t count, int sample_rate);
    static std::string extractContent(const std::string& json);
    static std::string httpPost(const std::string& url, const std::string& body,
                                    int64_t timeout_sec, std::string& err_msg);

    // Result builder
    RecognitionResult buildResult(const std::string& text,
                                    int64_t audio_duration_ms,
                                    int64_t processing_time_ms);
};

}  // namespace asr

#endif  // QWEN3_ASR_BACKEND_HPP
