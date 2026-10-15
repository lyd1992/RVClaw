/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_TYPES_HPP
#define ASR_TYPES_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>

namespace asr {

// =============================================================================
// Audio Format
// =============================================================================

enum class AudioFormat {
    PCM_S16LE,      // 16-bit signed little-endian PCM (raw)
    PCM_F32LE,      // 32-bit float little-endian PCM
    WAV,            // WAV container (PCM inside)
    MP3,
    OPUS,
    AAC,
    // Add more as needed
};

// =============================================================================
// Recognition Mode
// =============================================================================

enum class RecognitionMode {
    OFFLINE,        // 离线批处理模式 - 一次性输入完整音频
    STREAMING,      // 流式实时模式 - 分块输入音频
};

// =============================================================================
// Language Codes
// =============================================================================

enum class Language {
    AUTO,           // 自动检测
    ZH,             // 中文
    EN,             // 英文
    JA,             // 日语
    KO,             // 韩语
    YUE,            // 粤语
};

inline const char* languageToString(Language lang) {
    switch (lang) {
        case Language::AUTO: return "auto";
        case Language::ZH:   return "zh";
        case Language::EN:   return "en";
        case Language::JA:   return "ja";
        case Language::KO:   return "ko";
        case Language::YUE:  return "yue";
        default:             return "auto";
    }
}

// =============================================================================
// Word-level Result (词级别结果)
// =============================================================================

struct WordResult {
    std::string text;           // 文字内容
    int32_t begin_time_ms;      // 开始时间(毫秒)
    int32_t end_time_ms;        // 结束时间(毫秒)
    float confidence;           // 置信度 [0.0, 1.0]
    std::string punctuation;    // 标点符号(如有)
};

// =============================================================================
// Sentence Result (句子级别结果)
// =============================================================================

struct SentenceResult {
    std::string text;                   // 完整句子文本
    int32_t begin_time_ms;              // 句子开始时间
    int32_t end_time_ms;                // 句子结束时间
    float confidence;                   // 整句置信度
    bool is_final;                      // 是否为最终结果(流式模式)
    std::vector<WordResult> words;      // 词级别详情(可选)
    Language detected_language;         // 检测到的语言
    std::string emotion;                // 情绪标签（SenseVoice SER，未知时为空）
};

// =============================================================================
// Recognition Result (识别结果)
// =============================================================================

struct RecognitionResult {
    std::string request_id;             // 请求ID(用于追踪)
    std::vector<SentenceResult> sentences;  // 句子列表

    // 性能指标
    int64_t audio_duration_ms;          // 音频时长
    int64_t processing_time_ms;         // 处理耗时
    float rtf;                          // Real-Time Factor (处理时间/音频时长)

    // 延迟指标(流式模式)
    int64_t first_result_latency_ms;    // 首包延迟
    int64_t final_result_latency_ms;    // 最终结果延迟

    // 便捷方法: 获取完整文本
    std::string getText() const {
        std::string result;
        for (const auto& s : sentences) {
            result += s.text;
        }
        return result;
    }

    // 便捷方法: 是否为空结果
    bool isEmpty() const {
        return sentences.empty() || getText().empty();
    }
};

// =============================================================================
// Error Info (错误信息)
// =============================================================================

enum class ErrorCode {
    OK = 0,

    // 配置错误 (1xx)
    INVALID_CONFIG = 100,
    MODEL_NOT_FOUND = 101,
    UNSUPPORTED_FORMAT = 102,
    UNSUPPORTED_SAMPLE_RATE = 103,

    // 运行时错误 (2xx)
    NOT_INITIALIZED = 200,
    ALREADY_STARTED = 201,
    NOT_STARTED = 202,
    INFERENCE_FAILED = 203,
    TIMEOUT = 204,

    // 网络错误 (3xx) - 用于云端模式
    NETWORK_ERROR = 300,
    CONNECTION_FAILED = 301,
    AUTH_FAILED = 302,

    // 内部错误 (4xx)
    INTERNAL_ERROR = 400,
    OUT_OF_MEMORY = 401,
};

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::string detail;         // 详细信息(调试用)

    bool isOk() const { return code == ErrorCode::OK; }

    static ErrorInfo ok() {
        return {ErrorCode::OK, "", ""};
    }

    static ErrorInfo error(ErrorCode code, const std::string& msg, const std::string& detail = "") {
        return {code, msg, detail};
    }
};

// =============================================================================
// Backend Type (后端类型)
// =============================================================================

enum class BackendType {
    SENSEVOICE,     // SenseVoice (ONNX, 本地)
    FUNASR,         // FunASR (云端或本地)
    WHISPER,        // Whisper (未来扩展)
    PARAFORMER,     // Paraformer (未来扩展)
    QWEN3_ASR,      // Qwen3-ASR (llama-server HTTP)
    ZIPFORMER,      // Zipformer CTC (ONNX, 本地流式)
    CUSTOM,         // 自定义后端
};

inline const char* backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::SENSEVOICE: return "sensevoice";
        case BackendType::FUNASR:     return "funasr";
        case BackendType::WHISPER:    return "whisper";
        case BackendType::PARAFORMER: return "paraformer";
        case BackendType::QWEN3_ASR:  return "qwen3-asr";
        case BackendType::ZIPFORMER:  return "zipformer";
        case BackendType::CUSTOM:     return "custom";
        default:                      return "unknown";
    }
}

// =============================================================================
// Audio Chunk (音频块 - 用于流式输入)
// =============================================================================

struct AudioChunk {
    const void* data;           // 音频数据指针
    size_t size_bytes;          // 数据大小(字节)
    AudioFormat format;         // 音频格式
    int sample_rate;            // 采样率
    int channels;               // 声道数
    int64_t timestamp_ms;       // 时间戳(可选, -1表示未知)

    // 便捷构造: PCM S16
    static AudioChunk fromPCM16(const int16_t* data, size_t samples, int sample_rate, int channels = 1) {
        return {data, samples * sizeof(int16_t) * channels, AudioFormat::PCM_S16LE, sample_rate, channels, -1};
    }

    // 便捷构造: PCM Float
    static AudioChunk fromPCMFloat(const float* data, size_t samples, int sample_rate, int channels = 1) {
        return {data, samples * sizeof(float) * channels, AudioFormat::PCM_F32LE, sample_rate, channels, -1};
    }
};

}  // namespace asr

#endif  // ASR_TYPES_HPP
