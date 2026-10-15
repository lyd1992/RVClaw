/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef AUDIO_PROCESSOR_HPP
#define AUDIO_PROCESSOR_HPP

/**
 * AudioProcessor - 音频后处理模块
 *
 * 提供音频归一化、动态压缩、去爆音等功能。
 */

#include <cstdint>

#include <vector>

namespace tts {
namespace audio {

// =============================================================================
// 音频处理配置
// =============================================================================

struct AudioProcessConfig {
    // RMS 归一化
    float target_rms = 0.15f;           ///< 目标 RMS 电平
    bool use_rms_norm = true;           ///< 是否使用 RMS 归一化
    float max_gain = 3.0f;              ///< 最大增益倍数，非正值按默认 3.0f 处理

    // 动态压缩
    float compression_ratio = 2.0f;     ///< 压缩比
    float compression_threshold = 0.5f;  ///< 压缩阈值

    // 去爆音
    bool remove_clicks = true;          ///< 是否移除爆音

    // 默认配置
    static AudioProcessConfig Default() {
        return AudioProcessConfig();
    }
};

// =============================================================================
// 音频处理函数
// =============================================================================

/**
 * @brief 计算音频的 RMS (Root Mean Square)
 * @param audio 音频样本
 * @return RMS 值
 */
float calculateRMS(const std::vector<float>& audio);

/**
 * @brief 应用动态压缩
 * @param audio 输入音频
 * @param threshold 压缩阈值
 * @param ratio 压缩比
 * @return 压缩后的音频
 */
std::vector<float> applyCompression(const std::vector<float>& audio,
                                    float threshold,
                                    float ratio);

/**
 * @brief 应用 RMS 响度归一化，不做动态压缩
 * @param audio 输入音频
 * @param config 处理配置
 * @return 归一化后的音频
 */
std::vector<float> normalizeRmsLevel(
        const std::vector<float>& audio,
        const AudioProcessConfig& config);

/**
 * @brief 音频归一化 (RMS 或峰值归一化)
 * @param audio 输入音频
 * @param config 处理配置
 * @return 归一化后的音频
 */
std::vector<float> normalizeAudio(
        const std::vector<float>& audio,
        const AudioProcessConfig& config);

/**
 * @brief 移除爆音和直流偏移
 *
 * 包括：
 * - 移除直流偏移
 * - 淡入淡出
 * - 高通滤波
 *
 * @param audio 输入音频
 * @return 处理后的音频
 */
std::vector<float> removeClicksAndPops(const std::vector<float>& audio);

/**
 * @brief 重采样音频 (线性插值)
 * @param audio 输入音频
 * @param src_rate 源采样率
 * @param dst_rate 目标采样率
 * @return 重采样后的音频
 */
std::vector<float> resampleAudio(
        const std::vector<float>& audio,
        int src_rate,
        int dst_rate);

/**
 * @brief 完整的音频后处理流程
 *
 * 依次执行：归一化 -> 去爆音 (可选)
 *
 * @param audio 输入音频
 * @param config 处理配置
 * @return 处理后的音频
 */
std::vector<float> processAudio(const std::vector<float>& audio,
                                const AudioProcessConfig& config);

// =============================================================================
// 格式转换
// =============================================================================

/**
 * @brief float 转 int16
 * @param audio float 音频 [-1.0, 1.0]
 * @return int16 音频
 */
std::vector<int16_t> floatToInt16(const std::vector<float>& audio);

/**
 * @brief int16 转 float
 * @param audio int16 音频
 * @return float 音频 [-1.0, 1.0]
 */
std::vector<float> int16ToFloat(const std::vector<int16_t>& audio);

/**
 * @brief float 转字节数组 (PCM S16LE)
 * @param audio float 音频
 * @return 字节数组
 */
std::vector<uint8_t> floatToBytes(const std::vector<float>& audio);

}  // namespace audio
}  // namespace tts

#endif  // AUDIO_PROCESSOR_HPP
