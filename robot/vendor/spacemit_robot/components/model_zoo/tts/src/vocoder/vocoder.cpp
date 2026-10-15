/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "vocoder/vocoder.hpp"

#include <fftw3.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tts {
namespace vocoder {

// =============================================================================
// Hann 窗口
// =============================================================================

std::vector<float> createHannWindow(int32_t length) {
    std::vector<float> window(length);
    for (int32_t i = 0; i < length; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (length - 1)));
    }
    return window;
}

// =============================================================================
// ISTFT 实现
// =============================================================================

std::vector<float> istft(const std::vector<float>& stft_real,
    const std::vector<float>& stft_imag,
    int32_t num_frames,
    int32_t n_fft_bins,
    const ISTFTConfig& config) {
    int32_t n_fft = config.n_fft;
    int32_t hop_length = config.hop_length;
    int32_t win_length = config.win_length;

    // Calculate audio length
    int32_t audio_length = n_fft + (num_frames - 1) * hop_length;
    std::vector<float> audio(audio_length, 0.0f);
    std::vector<float> denominator(audio_length, 0.0f);

    // Create Hann window
    std::vector<float> window = createHannWindow(win_length);

    // Process each frame using overlap-add ISTFT
    for (int32_t frame = 0; frame < num_frames; ++frame) {
        // Extract real and imag for this frame
        const float* p_real = stft_real.data() + frame * n_fft_bins;
        const float* p_imag = stft_imag.data() + frame * n_fft_bins;

        // Allocate aligned memory for FFTW (RISC-V compatibility)
        fftwf_complex* in = nullptr;
        float* out = nullptr;

        const size_t alignment = 16;
        size_t in_size = sizeof(fftwf_complex) * (n_fft / 2 + 1);
        size_t out_size = sizeof(float) * n_fft;

        if (posix_memalign(reinterpret_cast<void**>(&in), alignment, in_size) != 0) {
            throw std::runtime_error("Failed to allocate aligned memory for FFT input");
        }
        if (posix_memalign(reinterpret_cast<void**>(&out), alignment, out_size) != 0) {
            free(in);
            throw std::runtime_error("Failed to allocate aligned memory for FFT output");
        }

        // Create FFTW plan with FFTW_UNALIGNED for RISC-V compatibility
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_fft, in, out, FFTW_ESTIMATE | FFTW_UNALIGNED);

        // Copy to FFTW input format
        for (int32_t i = 0; i < n_fft_bins && i < (n_fft / 2 + 1); ++i) {
            in[i][0] = p_real[i];
            in[i][1] = p_imag[i];
        }

        // Execute IFFT
        fftwf_execute(plan);

        // Apply IFFT normalization
        float scale = 1.0f / n_fft;
        for (int32_t i = 0; i < n_fft; ++i) {
            out[i] *= scale;
        }

        // Apply window
        for (int32_t i = 0; i < win_length && i < n_fft; ++i) {
            out[i] *= window[i];
        }

        // Overlap-add
        int32_t start_pos = frame * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            if (start_pos + i < audio_length) {
                audio[start_pos + i] += out[i];
                denominator[start_pos + i] += window[i] * window[i];
            }
        }

        // Cleanup
        fftwf_destroy_plan(plan);
        free(in);
        free(out);
    }

    // Normalize by window overlap
    for (int32_t i = 0; i < audio_length; ++i) {
        if (denominator[i] > 1e-8f) {
            audio[i] /= denominator[i];
        }
    }

    return audio;
}

}  // namespace vocoder
}  // namespace tts
