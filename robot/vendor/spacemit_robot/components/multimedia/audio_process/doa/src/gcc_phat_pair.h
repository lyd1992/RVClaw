/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Private GCC-PHAT helper for one microphone pair.
 */

#ifndef GCC_PHAT_PAIR_H
#define GCC_PHAT_PAIR_H

#include <fftw3.h>

#include <cstddef>

namespace SpacemitAudio {
namespace internal {

struct GccPhatPairConfig {
    int sample_rate = 16000;
    int avg_frames = 4;
    int padded_size = 0;
    int spectrum_size = 0;
    int upsample_factor = 1;
    int upsampled_size = 0;
    int up_spectrum_size = 0;
    float sound_speed = 343.0f;
    float pair_distance_m = 0.0f;
    float max_frequency_hz = 0.0f;
    bool use_fftw_measure = false;
};

struct GccPhatPairResult {
    bool ready = false;
    float tdoa_seconds = 0.0f;
    float delay_samples = 0.0f;
    float peak_value = 0.0f;
    const float* gcc = nullptr;
    int gcc_size = 0;
};

class GccPhatPair {
public:
    GccPhatPair() = default;
    ~GccPhatPair();

    GccPhatPair(const GccPhatPair&) = delete;
    GccPhatPair& operator=(const GccPhatPair&) = delete;
    GccPhatPair(GccPhatPair&& other) noexcept;
    GccPhatPair& operator=(GccPhatPair&& other) noexcept;

    bool Initialize(const GccPhatPairConfig& config);
    void Reset();
    void Destroy();

    bool Accumulate(const fftwf_complex* spectrum_i,
                    const fftwf_complex* spectrum_j);
    bool IsReady() const;
    GccPhatPairResult IfftAndPeak();

    int GetMaxDelaySamples() const;
    int GetBandLimitBinMax() const;
    const float* GetGccData() const;
    int GetGccSize() const;

private:
    void MoveFrom(GccPhatPair&& other) noexcept;

    GccPhatPairConfig config_;
    fftwf_complex* phat_accumulator_ = nullptr;
    fftwf_complex* gcc_spectrum_up_ = nullptr;
    float* gcc_result_up_ = nullptr;
    fftwf_plan ifft_plan_ = nullptr;
    int avg_count_ = 0;
    int max_delay_samples_ = 0;
    int band_limit_bin_max_ = 0;
};

}  // namespace internal
}  // namespace SpacemitAudio

#endif  // GCC_PHAT_PAIR_H
