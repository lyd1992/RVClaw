/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef VOCODER_HPP
#define VOCODER_HPP

/**
 * Vocoder - 声码器模块
 *
 * 提供 ISTFT (逆短时傅里叶变换) 等声码器相关功能。
 */

#include <cstdint>

#include <vector>

namespace tts {
namespace vocoder {

// =============================================================================
// ISTFT 参数配置
// =============================================================================

struct ISTFTConfig {
    int32_t n_fft = 1024;           ///< FFT 窗口大小
    int32_t hop_length = 256;       ///< 帧移
    int32_t win_length = 1024;      ///< 窗口长度
};

// =============================================================================
// ISTFT 函数
// =============================================================================

/**
 * @brief 创建 Hann 窗口
 * @param length 窗口长度
 * @return Hann 窗口系数
 */
std::vector<float> createHannWindow(int32_t length);

/**
 * @brief 执行逆短时傅里叶变换 (ISTFT)
 * @param stft_real STFT 实部数据 (frames x n_fft_bins)
 * @param stft_imag STFT 虚部数据 (frames x n_fft_bins)
 * @param num_frames 帧数
 * @param n_fft_bins FFT bin 数量 (通常为 n_fft/2 + 1)
 * @param config ISTFT 配置参数
 * @return 重建的时域音频信号
 *
 * 使用 FFTW3 库执行 IFFT，采用 overlap-add 方法重建音频。
 * 支持 RISC-V 架构的内存对齐优化。
 */
std::vector<float> istft(
        const std::vector<float>& stft_real,
        const std::vector<float>& stft_imag,
        int32_t num_frames,
        int32_t n_fft_bins,
        const ISTFTConfig& config = ISTFTConfig());

}  // namespace vocoder
}  // namespace tts

#endif  // VOCODER_HPP
