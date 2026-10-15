/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SENSEVOICE_BACKEND_HPP
#define SENSEVOICE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>

#include "../asr_backend.hpp"

namespace asr {

// Forward declaration
namespace sensevoice {
    class SenseVoiceModel;
}

// =============================================================================
// SenseVoice Backend (SenseVoice后端实现)
// =============================================================================
//
// 基于SenseVoice ONNX模型的本地ASR后端。
// 支持离线识别和流式识别。
//
// 模型要求:
// - model_quant_optimized.onnx (量化模型)
// - config.json (配置文件)
// - vocab.txt (词表)
//

class SenseVoiceBackend : public IASRBackend {
public:
    SenseVoiceBackend();
    ~SenseVoiceBackend() override;

    // -------------------------------------------------------------------------
    // IASRBackend 接口实现
    // -------------------------------------------------------------------------

    ErrorInfo initialize(const ASRConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_.load(); }

    BackendType getType() const override { return BackendType::SENSEVOICE; }
    std::string getName() const override { return "SenseVoice"; }
    std::string getVersion() const override { return "1.0.2"; }

    bool supportsStreaming() const override { return true; }

    std::vector<AudioFormat> getSupportedFormats() const override {
        return {AudioFormat::PCM_S16LE, AudioFormat::PCM_F32LE};
    }

    std::vector<int> getSupportedSampleRates() const override {
        return {16000};
    }

    // 离线识别
    ErrorInfo recognize(const AudioChunk& audio, RecognitionResult& result) override;
    ErrorInfo recognizeFile(const std::string& file_path, RecognitionResult& result) override;

    // 流式识别
    ErrorInfo startStream() override;
    ErrorInfo feedAudio(const AudioChunk& audio) override;
    ErrorInfo stopStream() override;
    ErrorInfo flushStream() override;
    bool isStreamActive() const override { return stream_active_.load(); }

    // 热更新
    ErrorInfo updateHotwords(const std::vector<std::string>& hotwords) override;
    ErrorInfo setLanguage(Language language) override;

private:
    ASRConfig config_;
    std::atomic<bool> initialized_{false};

    // 核心组件 (使用SenseVoice模型)
    std::unique_ptr<sensevoice::SenseVoiceModel> model_;

    // 流式识别状态
    std::atomic<bool> stream_active_{false};
    std::mutex stream_mutex_;

    // 音频缓冲 (流式模式)
    std::vector<float> audio_buffer_;
    int64_t buffer_timestamp_ms_ = 0;

    // VAD状态 (流式模式, 暂不实现)
    bool vad_speech_started_ = false;
    int64_t speech_start_time_ms_ = 0;

    // 内部方法
    ErrorInfo initializeASRModel();

    // 音频转换
    std::vector<float> convertToFloat(const AudioChunk& audio);
    std::vector<float> trimEndpointSilence(const std::vector<float>& audio) const;

    // 流式处理
    void processBufferedAudio(bool force_final);

    // 结果构建
    RecognitionResult buildResult(
        const std::string& text, const std::string& emotion,
        int64_t audio_duration_ms,
        int64_t processing_time_ms, bool is_final);
};

}  // namespace asr

#endif  // SENSEVOICE_BACKEND_HPP
