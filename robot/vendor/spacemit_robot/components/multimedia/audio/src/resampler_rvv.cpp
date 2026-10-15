/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_resampler.hpp"

#ifdef USE_RVV_OPTIMIZATION

// RVV support detection (same pattern as audio_mixer.hpp)
#if defined(__riscv) && defined(__riscv_v)
#define RESAMPLER_HAS_RVV 1
#include <riscv_vector.h>
#else
#define RESAMPLER_HAS_RVV 0
#endif

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#if RESAMPLER_HAS_RVV

// ============================================================================
// RVV-Optimized Linear Resampling Implementation
// ============================================================================
// Uses RISC-V Vector Extension (RVV) to accelerate linear interpolation
// for audio resampling. Provides significant speedup for high sample rates.
// ============================================================================

namespace {
    // Helper: Precompute source positions and fractional parts for vectorization
    struct ResampleLUT {
        std::vector<size_t> indices;   // Source frame indices
        std::vector<float> fractions;  // Fractional parts for interpolation

        void compute(size_t output_frames, double ratio, size_t max_input_frames) {
            indices.resize(output_frames);
            fractions.resize(output_frames);

            for (size_t i = 0; i < output_frames; ++i) {
                double src_pos = i / ratio;
                size_t src_idx = static_cast<size_t>(src_pos);
                double frac = src_pos - src_idx;

                // Clamp to valid range
                if (src_idx >= max_input_frames - 1) {
                    src_idx = max_input_frames - 2;
                    frac = 1.0;
                }

                indices[i] = src_idx;
                fractions[i] = static_cast<float>(frac);
            }
        }
    };
}  // namespace

std::vector<float> Resampler::linearUpsample_RVV(const float* input, size_t num_samples) {
    size_t num_frames = num_samples / config_.channels;
    size_t output_frames = static_cast<size_t>(std::ceil(num_frames * ratio_));
    std::vector<float> output(output_frames * config_.channels);

    if (output_frames == 0 || num_frames < 2) {
        return output;
    }

    // Process each channel separately
    for (int ch = 0; ch < config_.channels; ++ch) {
        const float* input_ch = input + ch;
        float* output_ch = output.data() + ch;
        const int ch_stride = config_.channels;

        size_t i = 0;
        while (i < output_frames) {
            size_t vl = __riscv_vsetvl_e32m4(output_frames - i);

            // Compute source positions for this batch
            // Pre-allocate on stack for small vl (typically 4-16)
            float frac_buf[64];
            float sample0_buf[64];
            float sample1_buf[64];

            for (size_t j = 0; j < vl; ++j) {
                double src_pos = (i + j) / ratio_;
                size_t src_idx = static_cast<size_t>(src_pos);
                float frac = static_cast<float>(src_pos - src_idx);

                if (src_idx >= num_frames - 1) {
                    src_idx = num_frames - 2;
                    frac = 1.0f;
                }

                frac_buf[j] = frac;
                sample0_buf[j] = input_ch[src_idx * ch_stride];
                sample1_buf[j] = input_ch[(src_idx + 1) * ch_stride];
            }

            // Vectorized interpolation
            vfloat32m4_t v_frac = __riscv_vle32_v_f32m4(frac_buf, vl);
            vfloat32m4_t v_weight0 = __riscv_vfrsub_vf_f32m4(v_frac, 1.0f, vl);
            vfloat32m4_t v_sample0 = __riscv_vle32_v_f32m4(sample0_buf, vl);
            vfloat32m4_t v_sample1 = __riscv_vle32_v_f32m4(sample1_buf, vl);

            vfloat32m4_t v_result = __riscv_vfmul_vv_f32m4(v_sample0, v_weight0, vl);
            v_result = __riscv_vfmacc_vv_f32m4(v_result, v_frac, v_sample1, vl);

            // Store results (strided for multi-channel)
            float result_buf[64];
            __riscv_vse32_v_f32m4(result_buf, v_result, vl);
            for (size_t j = 0; j < vl; ++j) {
                output_ch[(i + j) * ch_stride] = result_buf[j];
            }

            i += vl;
        }
    }

    return output;
}

// ============================================================================
// RVV-Optimized Linear Downsampling
// ============================================================================

std::vector<float> Resampler::linearDownsample_RVV(const float* input, size_t num_samples) {
    size_t num_frames = num_samples / config_.channels;
    size_t output_frames = static_cast<size_t>(std::ceil(num_frames * ratio_));
    std::vector<float> output(output_frames * config_.channels);

    if (output_frames == 0 || num_frames < 2) {
        return output;
    }

    // Process each channel separately
    for (int ch = 0; ch < config_.channels; ++ch) {
        const float* input_ch = input + ch;
        float* output_ch = output.data() + ch;
        const int ch_stride = config_.channels;

        size_t i = 0;
        while (i < output_frames) {
            size_t vl = __riscv_vsetvl_e32m4(output_frames - i);

            // Stack-allocated buffers (avoid heap allocation)
            float frac_buf[64];
            float sample0_buf[64];
            float sample1_buf[64];

            for (size_t j = 0; j < vl; ++j) {
                double src_pos = (i + j) / ratio_;
                size_t src_idx = static_cast<size_t>(src_pos);
                float frac = static_cast<float>(src_pos - src_idx);

                if (src_idx >= num_frames - 1) {
                    sample0_buf[j] = input_ch[(num_frames - 1) * ch_stride];
                    sample1_buf[j] = sample0_buf[j];
                    frac_buf[j] = 0.0f;
                } else {
                    frac_buf[j] = frac;
                    sample0_buf[j] = input_ch[src_idx * ch_stride];
                    sample1_buf[j] = input_ch[(src_idx + 1) * ch_stride];
                }
            }

            // Vectorized interpolation
            vfloat32m4_t v_frac = __riscv_vle32_v_f32m4(frac_buf, vl);
            vfloat32m4_t v_weight0 = __riscv_vfrsub_vf_f32m4(v_frac, 1.0f, vl);
            vfloat32m4_t v_sample0 = __riscv_vle32_v_f32m4(sample0_buf, vl);
            vfloat32m4_t v_sample1 = __riscv_vle32_v_f32m4(sample1_buf, vl);

            vfloat32m4_t v_result = __riscv_vfmul_vv_f32m4(v_sample0, v_weight0, vl);
            v_result = __riscv_vfmacc_vv_f32m4(v_result, v_frac, v_sample1, vl);

            // Store results
            float result_buf[64];
            __riscv_vse32_v_f32m4(result_buf, v_result, vl);
            for (size_t j = 0; j < vl; ++j) {
                output_ch[(i + j) * ch_stride] = result_buf[j];
            }

            i += vl;
        }
    }

    return output;
}

#endif  // RESAMPLER_HAS_RVV

// ============================================================================
// Stub implementations when RVV is not available at compile time
// ============================================================================
#if !RESAMPLER_HAS_RVV

std::vector<float> Resampler::linearUpsample_RVV(const float* input, size_t num_samples) {
    // This should never be called - isRVVAvailable() returns false
    // But we need to provide a definition to satisfy the linker
    (void)input;
    (void)num_samples;
    return {};
}

std::vector<float> Resampler::linearDownsample_RVV(const float* input, size_t num_samples) {
    // This should never be called - isRVVAvailable() returns false
    (void)input;
    (void)num_samples;
    return {};
}

#endif  // !RESAMPLER_HAS_RVV

// ============================================================================
// RVV Capability Detection
// ============================================================================

bool Resampler::isRVVAvailable() {
#if RESAMPLER_HAS_RVV
    // RVV is available at compile time
    static bool first_call = true;
    if (first_call) {
        std::cout << "[Resampler] RVV intrinsics available at compile time (VLEN detected at runtime)" << std::endl;
        first_call = false;
    }
    return true;
#else
    static bool first_call = true;
    if (first_call) {
        std::cout << "[Resampler] RVV not compiled in (missing __riscv_v macro)" << std::endl;
        first_call = false;
    }
    return false;
#endif
}

#endif  // USE_RVV_OPTIMIZATION
