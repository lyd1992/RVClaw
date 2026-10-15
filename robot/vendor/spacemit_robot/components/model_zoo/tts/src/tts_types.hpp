/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TTS_TYPES_HPP
#define TTS_TYPES_HPP

#include <cstdint>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts {

// =============================================================================
// Audio Format (音频格式)
// =============================================================================

enum class AudioFormat {
    PCM_S16LE,      // 16-bit signed little-endian PCM (raw)
    PCM_F32LE,      // 32-bit float little-endian PCM
    WAV,            // WAV container
    MP3,            // MP3 (预留)
    OGG,            // OGG Vorbis (预留)
};

inline const char* audioFormatToString(AudioFormat format) {
    switch (format) {
        case AudioFormat::PCM_S16LE: return "pcm_s16le";
        case AudioFormat::PCM_F32LE: return "pcm_f32le";
        case AudioFormat::WAV:       return "wav";
        case AudioFormat::MP3:       return "mp3";
        case AudioFormat::OGG:       return "ogg";
        default:                     return "unknown";
    }
}

// =============================================================================
// Synthesis Mode (合成模式)
// =============================================================================

enum class SynthesisMode {
    OFFLINE,        // 离线批处理模式 - 一次性合成完整音频
    STREAMING,      // 流式实时模式 - 按句子输出音频
};

// =============================================================================
// Backend Type (后端类型)
// =============================================================================

enum class BackendType {
    // Matcha-TTS 系列
    MATCHA_ZH,          // 中文 (matcha-icefall-zh-baker, 22050Hz)
    MATCHA_EN,          // 英文 (matcha-icefall-en_US-ljspeech, 22050Hz)
    MATCHA_ZH_EN,       // 中英混合 (matcha-icefall-zh-en, 16000Hz)

    // 预留扩展
    COSYVOICE,          // CosyVoice
    VITS,               // VITS
    PIPER,              // Piper TTS
    KOKORO,             // Kokoro TTS
    CUSTOM,             // 自定义后端
};

inline const char* backendTypeToString(BackendType type) {
    switch (type) {
        case BackendType::MATCHA_ZH:    return "matcha-zh";
        case BackendType::MATCHA_EN:    return "matcha-en";
        case BackendType::MATCHA_ZH_EN: return "matcha-zh-en";
        case BackendType::COSYVOICE:    return "cosyvoice";
        case BackendType::VITS:         return "vits";
        case BackendType::PIPER:        return "piper";
        case BackendType::KOKORO:       return "kokoro";
        case BackendType::CUSTOM:       return "custom";
        default:                        return "unknown";
    }
}

inline int getDefaultSampleRate(BackendType type) {
    switch (type) {
        case BackendType::MATCHA_ZH:    return 22050;
        case BackendType::MATCHA_EN:    return 22050;
        case BackendType::MATCHA_ZH_EN: return 16000;
        case BackendType::COSYVOICE:    return 22050;
        case BackendType::VITS:         return 22050;
        case BackendType::PIPER:        return 22050;
        case BackendType::KOKORO:       return 24000;
        default:                        return 22050;
    }
}

// =============================================================================
// Error Code (错误码)
// =============================================================================

enum class ErrorCode {
    OK = 0,

    // 配置错误 (1xx)
    INVALID_CONFIG = 100,
    MODEL_NOT_FOUND = 101,
    UNSUPPORTED_FORMAT = 102,
    UNSUPPORTED_LANGUAGE = 103,
    INVALID_TEXT = 104,

    // 运行时错误 (2xx)
    NOT_INITIALIZED = 200,
    ALREADY_STARTED = 201,
    NOT_STARTED = 202,
    SYNTHESIS_FAILED = 203,
    TIMEOUT = 204,
    TEXT_TOO_LONG = 205,

    // 网络错误 (3xx) - 用于云端模式
    NETWORK_ERROR = 300,
    CONNECTION_FAILED = 301,
    AUTH_FAILED = 302,

    // 内部错误 (4xx)
    INTERNAL_ERROR = 400,
    OUT_OF_MEMORY = 401,
    FILE_WRITE_ERROR = 402,
};

inline const char* errorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::OK:                 return "OK";
        case ErrorCode::INVALID_CONFIG:     return "INVALID_CONFIG";
        case ErrorCode::MODEL_NOT_FOUND:    return "MODEL_NOT_FOUND";
        case ErrorCode::UNSUPPORTED_FORMAT: return "UNSUPPORTED_FORMAT";
        case ErrorCode::UNSUPPORTED_LANGUAGE: return "UNSUPPORTED_LANGUAGE";
        case ErrorCode::INVALID_TEXT:       return "INVALID_TEXT";
        case ErrorCode::NOT_INITIALIZED:    return "NOT_INITIALIZED";
        case ErrorCode::ALREADY_STARTED:    return "ALREADY_STARTED";
        case ErrorCode::NOT_STARTED:        return "NOT_STARTED";
        case ErrorCode::SYNTHESIS_FAILED:   return "SYNTHESIS_FAILED";
        case ErrorCode::TIMEOUT:            return "TIMEOUT";
        case ErrorCode::TEXT_TOO_LONG:      return "TEXT_TOO_LONG";
        case ErrorCode::NETWORK_ERROR:      return "NETWORK_ERROR";
        case ErrorCode::CONNECTION_FAILED:  return "CONNECTION_FAILED";
        case ErrorCode::AUTH_FAILED:        return "AUTH_FAILED";
        case ErrorCode::INTERNAL_ERROR:     return "INTERNAL_ERROR";
        case ErrorCode::OUT_OF_MEMORY:      return "OUT_OF_MEMORY";
        case ErrorCode::FILE_WRITE_ERROR:   return "FILE_WRITE_ERROR";
        default:                            return "UNKNOWN";
    }
}

// =============================================================================
// Error Info (错误信息)
// =============================================================================

struct ErrorInfo {
    ErrorCode code;
    std::string message;
    std::string detail;  // 详细信息(调试用)

    bool isOk() const { return code == ErrorCode::OK; }

    static ErrorInfo ok() {
        return {ErrorCode::OK, "", ""};
    }

    static ErrorInfo error(ErrorCode code, const std::string& msg, const std::string& detail = "") {
        return {code, msg, detail};
    }
};

// =============================================================================
// Audio Chunk (音频块 - 用于流式输出)
// =============================================================================

struct AudioChunk {
    std::vector<float> samples;     // 音频样本 (float32, [-1.0, 1.0])
    int sample_rate;                // 采样率 (Hz)
    int channels = 1;               // 声道数 (默认单声道)
    bool is_final;                  // 是否为最终块 (句子结束)
    int sentence_index;             // 句子索引 (流式模式)
    int64_t timestamp_ms;           // 时间戳 (毫秒, -1表示未知)

    // 便捷方法: 获取时长 (毫秒)
    int getDurationMs() const {
        if (samples.empty() || sample_rate <= 0) return 0;
        return static_cast<int>(samples.size() * 1000 / sample_rate);
    }

    // 便捷方法: 获取样本数
    size_t getNumSamples() const {
        return samples.size();
    }

    // 便捷方法: 是否为空
    bool isEmpty() const {
        return samples.empty();
    }

    // 便捷方法: 转换为 int16 格式
    std::vector<int16_t> toInt16() const {
        std::vector<int16_t> result(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            float s = samples[i];
            if (s > 1.0f) s = 1.0f;
            if (s < -1.0f) s = -1.0f;
            result[i] = static_cast<int16_t>(s * 32767.0f);
        }
        return result;
    }

    // 便捷方法: 转换为字节数组 (PCM S16LE)
    std::vector<uint8_t> toBytes() const {
        auto int16_data = toInt16();
        std::vector<uint8_t> result(int16_data.size() * 2);
        for (size_t i = 0; i < int16_data.size(); ++i) {
            result[i * 2] = static_cast<uint8_t>(int16_data[i] & 0xFF);
            result[i * 2 + 1] = static_cast<uint8_t>((int16_data[i] >> 8) & 0xFF);
        }
        return result;
    }

    // 静态构造: 从 float 数据
    static AudioChunk fromFloat(const std::vector<float>& data, int sample_rate, bool is_final = true) {
        AudioChunk chunk;
        chunk.samples = data;
        chunk.sample_rate = sample_rate;
        chunk.channels = 1;
        chunk.is_final = is_final;
        chunk.sentence_index = 0;
        chunk.timestamp_ms = -1;
        return chunk;
    }

    // 静态构造: 从 int16 数据
    static AudioChunk fromInt16(const std::vector<int16_t>& data, int sample_rate, bool is_final = true) {
        AudioChunk chunk;
        chunk.samples.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            chunk.samples[i] = static_cast<float>(data[i]) / 32768.0f;
        }
        chunk.sample_rate = sample_rate;
        chunk.channels = 1;
        chunk.is_final = is_final;
        chunk.sentence_index = 0;
        chunk.timestamp_ms = -1;
        return chunk;
    }
};

// =============================================================================
// Phoneme Info (音素信息 - 用于时间戳)
// =============================================================================

struct PhonemeInfo {
    std::string phoneme;        // 音素文本
    int32_t begin_time_ms;      // 开始时间 (毫秒)
    int32_t end_time_ms;        // 结束时间 (毫秒)
};

// =============================================================================
// Word Info (词信息 - 用于时间戳)
// =============================================================================

struct WordInfo {
    std::string text;                       // 词文本
    int32_t begin_time_ms;                  // 开始时间 (毫秒)
    int32_t end_time_ms;                    // 结束时间 (毫秒)
    std::vector<PhonemeInfo> phonemes;      // 音素详情 (可选)
};

// =============================================================================
// Sentence Info (句子信息)
// =============================================================================

struct SentenceInfo {
    std::string text;                   // 句子文本
    int32_t begin_time_ms;              // 开始时间 (毫秒)
    int32_t end_time_ms;                // 结束时间 (毫秒)
    std::vector<WordInfo> words;        // 词详情 (可选)
    bool is_final;                      // 是否为最终结果 (流式模式)
};

// =============================================================================
// Synthesis Result (合成结果 - 内部使用)
// =============================================================================

struct SynthesisResult {
    std::string request_id;                 // 请求ID (用于追踪)
    AudioChunk audio;                       // 音频数据
    std::vector<SentenceInfo> sentences;    // 句子列表 (包含时间戳)

    // 性能指标
    int64_t audio_duration_ms = 0;          // 音频时长 (毫秒)
    int64_t processing_time_ms = 0;         // 处理耗时 (毫秒)
    float rtf = 0.0f;                       // Real-Time Factor (处理时间/音频时长)

    // 状态
    bool success = false;                   // 是否成功
    ErrorInfo error;                        // 错误信息

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
        return audio.isEmpty();
    }

    // 便捷方法: 计算 RTF
    void calculateRTF() {
        if (audio_duration_ms > 0) {
            rtf = static_cast<float>(processing_time_ms) / static_cast<float>(audio_duration_ms);
        }
    }
};

// =============================================================================
// Callback Interface (回调接口 - 内部使用)
// =============================================================================

class ITtsCallback {
public:
    virtual ~ITtsCallback() = default;

    /// @brief 合成开始
    virtual void onOpen() {}

    /// @brief 收到音频块
    /// @param chunk 音频数据块
    virtual void onAudioChunk(const AudioChunk& chunk) {}

    /// @brief 合成完成
    virtual void onComplete() {}

    /// @brief 发生错误
    /// @param error 错误信息
    virtual void onError(const ErrorInfo& error) {}

    /// @brief 会话关闭
    virtual void onClose() {}
};

}  // namespace tts

#endif  // TTS_TYPES_HPP
