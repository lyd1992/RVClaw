/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_CONFIG_HPP
#define ASR_CONFIG_HPP

#include <string>
#include <vector>
#include <map>

#include "asr_types.hpp"

namespace asr {

// =============================================================================
// ASR Configuration (ASR配置)
// =============================================================================

struct ASRConfig {
    // -------------------------------------------------------------------------
    // 后端选择
    // -------------------------------------------------------------------------

    BackendType backend = BackendType::SENSEVOICE;

    // -------------------------------------------------------------------------
    // 模型配置 (本地模型后端使用)
    // -------------------------------------------------------------------------

    std::string model_path;         // 模型文件路径 (ONNX等)
    std::string config_path;        // 模型配置文件路径
    std::string vocab_path;         // 词表文件路径
    std::string decoder_path;       // 解码器路径 (可选)

    // 模型选项
    bool use_quantized = true;      // 使用量化模型
    int num_threads = 2;            // 推理线程数

    // -------------------------------------------------------------------------
    // 云端配置 (云端后端使用)
    // -------------------------------------------------------------------------

    std::string api_key;            // API密钥
    std::string api_endpoint;       // API端点 URL
    std::string model_id;           // 云端模型标识 (如 "fun-asr-realtime")

    // -------------------------------------------------------------------------
    // 音频格式配置
    // -------------------------------------------------------------------------

    AudioFormat audio_format = AudioFormat::PCM_S16LE;
    int sample_rate = 16000;        // 采样率 (Hz)
    int channels = 1;               // 声道数

    // -------------------------------------------------------------------------
    // 识别模式
    // -------------------------------------------------------------------------

    RecognitionMode mode = RecognitionMode::OFFLINE;

    // -------------------------------------------------------------------------
    // 语言配置
    // -------------------------------------------------------------------------

    Language language = Language::AUTO;
    std::vector<Language> language_hints = {Language::ZH, Language::EN};  // 多语言提示

    // -------------------------------------------------------------------------
    // VAD (语音活动检测) 配置
    // -------------------------------------------------------------------------

    bool vad_enabled = true;                    // 启用VAD
    int vad_silence_threshold_ms = 300;         // 静音判定阈值 (毫秒)
    int vad_max_sentence_silence_ms = 1300;     // 句子最大静音间隔
    float vad_threshold = 0.5f;                 // VAD灵敏度 [0.0, 1.0]

    // -------------------------------------------------------------------------
    // 标点与格式化
    // -------------------------------------------------------------------------

    bool punctuation_enabled = true;            // 自动添加标点
    bool itn_enabled = true;                    // ITN (Inverse Text Normalization)
                                                // 将 "一二三" 转为 "123"

    // -------------------------------------------------------------------------
    // 热词配置
    // -------------------------------------------------------------------------

    std::string hotword_file;                   // 热词文件路径
    std::vector<std::string> hotwords;          // 热词列表
    float hotword_boost = 1.0f;                 // 热词增强系数

    // -------------------------------------------------------------------------
    // 流式配置
    // -------------------------------------------------------------------------

    int chunk_size_ms = 100;                    // 每块音频时长 (毫秒)
    bool return_partial_results = true;         // 返回中间结果
    bool return_word_timestamps = false;        // 返回词级别时间戳
    bool enable_emotion = false;                // 启用 SenseVoice 情绪识别（默认关闭）
    int max_audio_duration_s = 60;              // 最大音频时长 (秒)

    // -------------------------------------------------------------------------
    // 性能配置
    // -------------------------------------------------------------------------

    int timeout_ms = 30000;                     // 超时时间 (毫秒)
    bool enable_profiling = false;              // 启用性能分析

    // -------------------------------------------------------------------------
    // 扩展参数 (用于后端特定配置)
    // -------------------------------------------------------------------------

    std::map<std::string, std::string> extra_params;

    // -------------------------------------------------------------------------
    // 便捷构建方法
    // -------------------------------------------------------------------------

    /// @brief 创建SenseVoice本地识别配置
    static ASRConfig sensevoice(const std::string& model_dir) {
        ASRConfig config;
        config.backend = BackendType::SENSEVOICE;
        config.model_path = model_dir + "/model_quant_optimized.onnx";
        config.config_path = model_dir + "/am.mvn";  // CMVN file
        config.vocab_path = model_dir + "/tokens.txt";
        config.decoder_path = model_dir + "/sensevoice_decoder_model.onnx";
        config.use_quantized = true;
        return config;
    }

    /// @brief 创建Zipformer CTC本地识别配置
    static ASRConfig zipformer(const std::string& model_dir) {
        ASRConfig config;
        config.backend = BackendType::ZIPFORMER;
        config.model_path = model_dir + "/ctc-epoch-20-avg-1-chunk-16-left-128.q.onnx";
        config.vocab_path = model_dir + "/tokens.txt";
        config.use_quantized = true;
        return config;
    }

    /// @brief 创建FunASR云端识别配置
    static ASRConfig funasrCloud(const std::string& api_key, const std::string& model_id = "fun-asr-realtime") {
        ASRConfig config;
        config.backend = BackendType::FUNASR;
        config.api_key = api_key;
        config.model_id = model_id;
        config.api_endpoint = "wss://dashscope.aliyuncs.com/api-ws/v1/inference";
        config.mode = RecognitionMode::STREAMING;
        return config;
    }

    /// @brief 创建流式识别配置 (基于已有配置)
    ASRConfig withStreaming(int chunk_ms = 100) const {
        ASRConfig config = *this;
        config.mode = RecognitionMode::STREAMING;
        config.chunk_size_ms = chunk_ms;
        config.return_partial_results = true;
        return config;
    }

    /// @brief 设置语言
    ASRConfig withLanguage(Language lang) const {
        ASRConfig config = *this;
        config.language = lang;
        return config;
    }

    /// @brief 禁用VAD
    ASRConfig withoutVAD() const {
        ASRConfig config = *this;
        config.vad_enabled = false;
        return config;
    }

    /// @brief 启用词级别时间戳
    ASRConfig withWordTimestamps() const {
        ASRConfig config = *this;
        config.return_word_timestamps = true;
        return config;
    }

    /// @brief 启用/关闭情绪标签输出
    ASRConfig withEmotion(bool enabled = true) const {
        ASRConfig config = *this;
        config.enable_emotion = enabled;
        return config;
    }
};

// =============================================================================
// Config Validator (配置验证器)
// =============================================================================

class ConfigValidator {
public:
    static ErrorInfo validate(const ASRConfig& config) {
        // 检查后端类型
        if (config.backend == BackendType::SENSEVOICE ||
            config.backend == BackendType::WHISPER ||
            config.backend == BackendType::PARAFORMER ||
            config.backend == BackendType::ZIPFORMER) {
            // 本地后端需要模型路径
            if (config.model_path.empty()) {
                return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
                    "Model path is required for local backend");
            }
        }

        // Qwen3-ASR is HTTP-based, no local model path needed

        if (config.backend == BackendType::FUNASR) {
            // 云端后端需要API配置
            if (config.api_key.empty()) {
                return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
                    "API key is required for cloud backend");
            }
        }

        // 检查采样率
        if (config.sample_rate != 16000 && config.sample_rate != 8000) {
            return ErrorInfo::error(ErrorCode::UNSUPPORTED_SAMPLE_RATE,
                "Sample rate must be 16000 or 8000 Hz");
        }

        // 检查声道数
        if (config.channels < 1 || config.channels > 2) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
                "Channels must be 1 or 2");
        }

        return ErrorInfo::ok();
    }
};

}  // namespace asr

#endif  // ASR_CONFIG_HPP
