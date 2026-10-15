/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TTS_CONFIG_HPP
#define TTS_CONFIG_HPP

#include <string>

#include "tts_types.hpp"

namespace tts {

// =============================================================================
// TTS Config (TTS配置 - 内部使用)
// =============================================================================

struct TtsConfig {
    // -------------------------------------------------------------------------
    // 后端选择
    // -------------------------------------------------------------------------

    BackendType backend = BackendType::MATCHA_ZH_EN;  ///< 后端类型

    // -------------------------------------------------------------------------
    // 模型配置
    // -------------------------------------------------------------------------

    std::string model;                  ///< 模型名称
    std::string model_dir;              ///< 模型目录路径，空则使用默认路径
    std::string acoustic_model_path;    ///< 声学模型路径 (可选，自动推断)
    std::string vocoder_path;           ///< 声码器路径 (可选，自动推断)
    std::string voice = "default";      ///< 音色名称

    // -------------------------------------------------------------------------
    // 说话人配置
    // -------------------------------------------------------------------------

    int speaker_id = 0;                 ///< 说话人ID (多说话人模型)

    // -------------------------------------------------------------------------
    // 音频参数
    // -------------------------------------------------------------------------

    AudioFormat format = AudioFormat::WAV;  ///< 输出格式
    int sample_rate = 16000;            ///< 输出采样率 (Hz)
    int output_sample_rate = 0;         ///< 重采样后采样率 (0=不重采样)
    int volume = 50;                    ///< 音量 [0, 100]

    // -------------------------------------------------------------------------
    // 合成参数
    // -------------------------------------------------------------------------

    float speech_rate = 1.0f;           ///< 语速 (>1.0快, <1.0慢)
    float pitch = 1.0f;                 ///< 音调
    float noise_scale = 1.0f;           ///< 变化控制 (Matcha 特有)
    float noise_scale_w = 1.0f;         ///< 持续时间变化 (Matcha 特有)

    // -------------------------------------------------------------------------
    // 音频后处理
    // -------------------------------------------------------------------------

    float target_rms = 0.15f;           ///< 目标RMS电平
    float compression_ratio = 2.0f;     ///< 压缩比
    bool use_rms_norm = true;           ///< 使用RMS归一化
    bool remove_clicks = true;          ///< 移除爆音

    // -------------------------------------------------------------------------
    // 性能配置
    // -------------------------------------------------------------------------

    int num_threads = 2;                ///< 推理线程数
    bool enable_warmup = true;          ///< 启动时预热
    std::string provider = "auto";      ///< 推理后端: auto/cpu/spacemit

    // -------------------------------------------------------------------------
    // 链式配置
    // -------------------------------------------------------------------------

    TtsConfig withSpeed(float speed) const {
        auto c = *this;
        c.speech_rate = speed;
        return c;
    }

    TtsConfig withSpeaker(int id) const {
        auto c = *this;
        c.speaker_id = id;
        return c;
    }

    TtsConfig withVolume(int vol) const {
        auto c = *this;
        c.volume = vol;
        return c;
    }

    TtsConfig withPitch(float p) const {
        auto c = *this;
        c.pitch = p;
        return c;
    }

    TtsConfig withModelDir(const std::string& dir) const {
        auto c = *this;
        c.model_dir = dir;
        return c;
    }

    TtsConfig withSampleRate(int rate) const {
        auto c = *this;
        c.sample_rate = rate;
        return c;
    }

    TtsConfig withFormat(AudioFormat fmt) const {
        auto c = *this;
        c.format = fmt;
        return c;
    }

    TtsConfig withProvider(const std::string& value) const {
        auto c = *this;
        c.provider = value;
        return c;
    }

    // -------------------------------------------------------------------------
    // 工具方法
    // -------------------------------------------------------------------------

    /// @brief 获取模型目录的完整路径
    /// @return 展开后的路径
    std::string getExpandedModelDir() const {
        if (model_dir.empty()) {
            return "~/.cache/models/tts/matcha-tts";
        }
        // 展开 ~ 到 HOME 目录
        if (model_dir[0] == '~') {
            const char* home = getenv("HOME");
            if (home) {
                return std::string(home) + model_dir.substr(1);
            }
        }
        return model_dir;
    }

    /// @brief 验证配置是否有效
    /// @return 错误信息
    ErrorInfo validate() const {
        if (sample_rate <= 0) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid sample rate");
        }
        if (speech_rate <= 0) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Invalid speech rate");
        }
        if (volume < 0 || volume > 100) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Volume must be 0-100");
        }
        return ErrorInfo::ok();
    }
};

}  // namespace tts

#endif  // TTS_CONFIG_HPP
