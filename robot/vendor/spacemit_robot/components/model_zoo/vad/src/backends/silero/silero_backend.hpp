/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Silero VAD Backend
 *
 * 基于 Silero VAD ONNX 模型的后端实现。
 */

#ifndef SILERO_BACKEND_HPP
#define SILERO_BACKEND_HPP

#include <onnxruntime_cxx_api.h>

#include <deque>
#include <memory>
#include <string>
#include <vector>

// Include from vad/include/
#include "../vad_backend.hpp"

namespace vad {

/**
 * @brief Silero VAD 后端
 *
 * 使用 Silero VAD ONNX 模型进行语音活动检测。
 * 支持 16kHz 采样率，512 样本窗口。
 */
class SileroBackend : public IVadBackend {
public:
    SileroBackend();
    ~SileroBackend() override;

    // IVadBackend interface
    ErrorInfo initialize(const VadConfig& config) override;
    void shutdown() override;
    ErrorInfo detect(const AudioChunk& audio, DetectionResult& result) override;
    void reset() override;
    BackendType getType() const override { return BackendType::SILERO; }
    std::string getName() const override { return "Silero VAD"; }
    bool isInitialized() const override { return session_ != nullptr; }

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    // Model configuration
    VadConfig config_;
    std::string model_path_;

    // LSTM state (Silero uses 2-layer LSTM with 128 hidden units)
    std::vector<float> state_h_;  // (2, 1, 128)
    std::vector<float> state_c_;  // (2, 1, 128)
    std::vector<float> context_;  // context window

    // Probability smoothing
    std::deque<float> prob_history_;

    // State tracking for detect()
    VadState current_state_ = VadState::SILENCE;
    int64_t speech_start_time_ = -1;
    int64_t frame_count_ = 0;

    // Input/output tensor info
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Helper methods
    bool initializeSession();
    float runInference(const float* audio, size_t length);
    float applySmoothing(float prob);
    VadState updateState(float prob, DetectionResult& result);
    std::string findModelPath(const std::string& model_dir);
    std::string expandPath(const std::string& path);
    bool downloadModel(const std::string& dest_dir);
};

}  // namespace vad

#endif  // SILERO_BACKEND_HPP
