/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZIPFORMER_BACKEND_HPP
#define ZIPFORMER_BACKEND_HPP

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../asr_backend.hpp"

namespace zipformer {
class CtcModel;
class FeatureExtractor;
class SymbolTable;
class CtcDecoder;
}  // namespace zipformer

namespace asr {

class ZipformerBackend : public IASRBackend {
public:
    ZipformerBackend();
    ~ZipformerBackend() override;

    ErrorInfo initialize(const ASRConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }

    BackendType getType() const override { return BackendType::ZIPFORMER; }
    std::string getName() const override { return "Zipformer"; }
    std::string getVersion() const override { return "1.0.2"; }

    bool supportsStreaming() const override { return true; }

    std::vector<AudioFormat> getSupportedFormats() const override {
        return {AudioFormat::PCM_S16LE, AudioFormat::PCM_F32LE};
    }

    std::vector<int> getSupportedSampleRates() const override {
        return {16000};
    }

    ErrorInfo recognize(const AudioChunk& audio,
                        RecognitionResult& result) override;
    ErrorInfo recognizeFile(const std::string& file_path,
                            RecognitionResult& result) override;

    ErrorInfo startStream() override;
    ErrorInfo feedAudio(const AudioChunk& audio) override;
    ErrorInfo stopStream() override;
    ErrorInfo flushStream() override;
    bool isStreamActive() const override { return stream_active_.load(); }

private:
    ASRConfig config_;
    std::atomic<bool> initialized_{false};

    std::unique_ptr<zipformer::CtcModel> model_;
    std::unique_ptr<zipformer::FeatureExtractor> feat_;
    std::unique_ptr<zipformer::SymbolTable> sym_;
    std::unique_ptr<zipformer::CtcDecoder> decoder_;

    std::atomic<bool> stream_active_{false};
    std::mutex stream_mutex_;
    std::vector<float> audio_buffer_;

    std::string recognizeAudio(const float* data, size_t samples,
                                int sample_rate);
    std::vector<float> convertToFloat(const AudioChunk& audio);
    RecognitionResult buildResult(const std::string& text,
                                    int64_t audio_duration_ms,
                                    int64_t processing_time_ms, bool is_final);
    void processBufferedAudio(bool force_final);
};

}  // namespace asr

#endif  // ZIPFORMER_BACKEND_HPP
