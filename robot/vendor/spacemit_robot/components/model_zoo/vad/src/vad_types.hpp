/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * VAD Internal Types
 *
 * 内部类型定义，包括错误码、检测结果、音频块等。
 */

#ifndef VAD_TYPES_HPP
#define VAD_TYPES_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <map>

namespace vad {

// =============================================================================
// VAD State
// =============================================================================

enum class VadState {
    SILENCE,        ///< 静音状态
    SPEECH_START,   ///< 语音开始（从静音转为语音）
    SPEECH,         ///< 语音进行中
    SPEECH_END,     ///< 语音结束（从语音转为静音）
};

inline const char* vadStateToString(VadState state) {
    switch (state) {
        case VadState::SILENCE:      return "silence";
        case VadState::SPEECH_START: return "speech_start";
        case VadState::SPEECH:       return "speech";
        case VadState::SPEECH_END:   return "speech_end";
        default:                     return "unknown";
    }
}

// =============================================================================
// Backend Type
// =============================================================================

enum class BackendType {
    SILERO,         ///< Silero VAD (ONNX, 深度学习)
    ENERGY,         ///< 基于能量的 VAD (轻量级)
    WEBRTC,         ///< WebRTC VAD (预留)
    FSMN,           ///< 阿里 FSMN VAD (预留)
    CUSTOM,         ///< 自定义后端
};

inline const char* backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::SILERO: return "silero";
        case BackendType::ENERGY: return "energy";
        case BackendType::WEBRTC: return "webrtc";
        case BackendType::FSMN:   return "fsmn";
        case BackendType::CUSTOM: return "custom";
        default:                  return "unknown";
    }
}

// =============================================================================
// Error Code
// =============================================================================

enum class ErrorCode {
    OK = 0,

    // 配置错误 (1xx)
    INVALID_CONFIG = 100,
    MODEL_NOT_FOUND = 101,
    UNSUPPORTED_SAMPLE_RATE = 102,
    INVALID_PARAMETER = 103,

    // 运行时错误 (2xx)
    NOT_INITIALIZED = 200,
    ALREADY_INITIALIZED = 201,
    ALREADY_STARTED = 202,
    NOT_STARTED = 203,
    INFERENCE_FAILED = 204,

    // 内部错误 (4xx)
    INTERNAL_ERROR = 400,
    OUT_OF_MEMORY = 401,
    BACKEND_ERROR = 402,
};

// =============================================================================
// Error Info
// =============================================================================

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::string detail;

    bool isOk() const { return code == ErrorCode::OK; }

    static ErrorInfo ok() {
        return {ErrorCode::OK, "", ""};
    }

    static ErrorInfo error(ErrorCode code, const std::string& msg,
                            const std::string& detail = "") {
        return {code, msg, detail};
    }
};

// =============================================================================
// VAD Config (Internal)
// =============================================================================

struct VadConfig {
    // 后端选择
    BackendType backend = BackendType::SILERO;

    // 模型配置
    std::string model_dir;              ///< 模型目录路径

    // 音频参数
    int sample_rate = 16000;            ///< 采样率 (Hz)
    int window_size = 512;              ///< 窗口大小 (样本数)
    int context_size = 64;              ///< 上下文大小 (样本数)

    // 检测参数
    float trigger_threshold = 0.5f;     ///< 语音开始阈值
    float stop_threshold = 0.35f;       ///< 语音结束阈值
    int min_speech_duration_ms = 250;   ///< 最小语音持续时间
    int min_silence_duration_ms = 100;  ///< 最小静音持续时间

    // 平滑参数
    int smoothing_window = 10;          ///< 平滑窗口大小
    bool use_smoothing = true;          ///< 是否使用概率平滑

    // 性能配置
    int num_threads = 1;                ///< 推理线程数

    // 扩展参数
    std::map<std::string, std::string> extra_params;
};

// =============================================================================
// Detection Result
// =============================================================================

struct DetectionResult {
    float probability = 0.0f;           ///< 原始语音概率
    float smoothed_probability = 0.0f;  ///< 平滑后概率
    VadState state = VadState::SILENCE;  ///< 当前状态
    bool is_speech = false;             ///< 是否为语音

    int64_t timestamp_ms = 0;           ///< 帧时间戳
    int processing_time_ms = 0;         ///< 处理耗时

    // 语音段信息
    int64_t speech_start_ms = -1;       ///< 语音开始时间 (-1 表示无)
    int64_t speech_end_ms = -1;         ///< 语音结束时间 (-1 表示进行中)
    int speech_duration_ms = 0;         ///< 语音持续时间
};

// =============================================================================
// Audio Chunk
// =============================================================================

struct AudioChunk {
    const float* data = nullptr;
    size_t num_samples = 0;
    int sample_rate = 16000;
    int64_t timestamp_ms = -1;

    static AudioChunk fromFloat(const float* data, size_t samples,
                                int sample_rate = 16000, int64_t timestamp = -1) {
        return {data, samples, sample_rate, timestamp};
    }

    static AudioChunk fromVector(const std::vector<float>& vec,
                                int sample_rate = 16000, int64_t timestamp = -1) {
        return {vec.data(), vec.size(), sample_rate, timestamp};
    }

    bool isEmpty() const { return data == nullptr || num_samples == 0; }
};

}  // namespace vad

#endif  // VAD_TYPES_HPP
