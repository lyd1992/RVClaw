/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Private GCC-PHAT helper for one microphone pair.
 */

#include "gcc_phat_pair.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace SpacemitAudio {
namespace internal {
namespace {

constexpr float kEpsilon = 1e-10f;

}  // namespace

GccPhatPair::~GccPhatPair() {
    Destroy();
}

GccPhatPair::GccPhatPair(GccPhatPair&& other) noexcept {
    MoveFrom(std::move(other));
}

GccPhatPair& GccPhatPair::operator=(GccPhatPair&& other) noexcept {
    if (this != &other) {
        Destroy();
        MoveFrom(std::move(other));
    }
    return *this;
}

void GccPhatPair::MoveFrom(GccPhatPair&& other) noexcept {
    config_ = other.config_;
    phat_accumulator_ = other.phat_accumulator_;
    gcc_spectrum_up_ = other.gcc_spectrum_up_;
    gcc_result_up_ = other.gcc_result_up_;
    ifft_plan_ = other.ifft_plan_;
    avg_count_ = other.avg_count_;
    max_delay_samples_ = other.max_delay_samples_;
    band_limit_bin_max_ = other.band_limit_bin_max_;

    other.phat_accumulator_ = nullptr;
    other.gcc_spectrum_up_ = nullptr;
    other.gcc_result_up_ = nullptr;
    other.ifft_plan_ = nullptr;
    other.avg_count_ = 0;
    other.max_delay_samples_ = 0;
    other.band_limit_bin_max_ = 0;
}

bool GccPhatPair::Initialize(const GccPhatPairConfig& config) {
    Destroy();
    config_ = config;

    if (config_.sample_rate <= 0 || config_.avg_frames <= 0
        || config_.padded_size <= 0 || config_.spectrum_size <= 0
        || config_.upsample_factor <= 0 || config_.upsampled_size <= 0
        || config_.up_spectrum_size <= 0 || config_.sound_speed <= 0.0f) {
        return false;
    }

    phat_accumulator_ = fftwf_alloc_complex(config_.spectrum_size);
    gcc_spectrum_up_ = fftwf_alloc_complex(config_.up_spectrum_size);
    gcc_result_up_ = fftwf_alloc_real(config_.upsampled_size);

    if (!phat_accumulator_ || !gcc_spectrum_up_ || !gcc_result_up_) {
        Destroy();
        return false;
    }

    const unsigned fftw_flags = config_.use_fftw_measure
                                    ? FFTW_MEASURE : FFTW_ESTIMATE;
    ifft_plan_ = fftwf_plan_dft_c2r_1d(config_.upsampled_size,
        gcc_spectrum_up_, gcc_result_up_,
        fftw_flags);
    if (!ifft_plan_) {
        Destroy();
        return false;
    }

    max_delay_samples_ = static_cast<int>(std::ceil(
        config_.pair_distance_m / config_.sound_speed
        * static_cast<float>(config_.sample_rate)));
    max_delay_samples_ = std::max(0, max_delay_samples_);
    max_delay_samples_ = std::min(max_delay_samples_, config_.padded_size / 2);

    const float nyquist = 0.5f * static_cast<float>(config_.sample_rate);
    if (config_.max_frequency_hz > 0.0f
        && config_.max_frequency_hz < nyquist) {
        const float bin = config_.max_frequency_hz
            * static_cast<float>(config_.padded_size)
            / static_cast<float>(config_.sample_rate);
        band_limit_bin_max_ = static_cast<int>(std::floor(bin));
        band_limit_bin_max_ = std::clamp(
            band_limit_bin_max_, 0, config_.spectrum_size - 1);
    } else {
        band_limit_bin_max_ = config_.spectrum_size - 1;
    }

    Reset();
    return true;
}

void GccPhatPair::Destroy() {
    if (ifft_plan_) {
        fftwf_destroy_plan(ifft_plan_);
        ifft_plan_ = nullptr;
    }
    if (phat_accumulator_) {
        fftwf_free(phat_accumulator_);
        phat_accumulator_ = nullptr;
    }
    if (gcc_spectrum_up_) {
        fftwf_free(gcc_spectrum_up_);
        gcc_spectrum_up_ = nullptr;
    }
    if (gcc_result_up_) {
        fftwf_free(gcc_result_up_);
        gcc_result_up_ = nullptr;
    }
    avg_count_ = 0;
    max_delay_samples_ = 0;
    band_limit_bin_max_ = 0;
}

void GccPhatPair::Reset() {
    if (phat_accumulator_) {
        std::memset(phat_accumulator_, 0,
                    config_.spectrum_size * sizeof(fftwf_complex));
    }
    if (gcc_spectrum_up_) {
        std::memset(gcc_spectrum_up_, 0,
                    config_.up_spectrum_size * sizeof(fftwf_complex));
    }
    if (gcc_result_up_) {
        std::memset(gcc_result_up_, 0,
                    config_.upsampled_size * sizeof(float));
    }
    avg_count_ = 0;
}

bool GccPhatPair::Accumulate(const fftwf_complex* spectrum_i,
    const fftwf_complex* spectrum_j) {
    if (!phat_accumulator_ || !spectrum_i || !spectrum_j) {
        return false;
    }

    for (int k = 0; k < config_.spectrum_size; ++k) {
        if (k > band_limit_bin_max_) {
            continue;
        }

        const float ar = spectrum_i[k][0];
        const float ai = spectrum_i[k][1];
        const float br = spectrum_j[k][0];
        const float bi = spectrum_j[k][1];

        // X_i * conj(X_j), matching the shipped 2-channel kernel.
        const float cross_r = ar * br + ai * bi;
        const float cross_i = ai * br - ar * bi;
        const float mag = std::sqrt(cross_r * cross_r + cross_i * cross_i);

        if (mag > kEpsilon) {
            const float inv_mag = 1.0f / mag;
            phat_accumulator_[k][0] += cross_r * inv_mag;
            phat_accumulator_[k][1] += cross_i * inv_mag;
        }
    }

    ++avg_count_;
    return true;
}

bool GccPhatPair::IsReady() const {
    return avg_count_ >= config_.avg_frames;
}

GccPhatPairResult GccPhatPair::IfftAndPeak() {
    GccPhatPairResult result;
    result.gcc = gcc_result_up_;
    result.gcc_size = config_.upsampled_size;

    if (!IsReady() || !gcc_spectrum_up_ || !gcc_result_up_) {
        return result;
    }

    const float inv_avg = 1.0f / static_cast<float>(avg_count_);
    std::memset(gcc_spectrum_up_, 0,
                config_.up_spectrum_size * sizeof(fftwf_complex));
    for (int k = 0; k <= band_limit_bin_max_; ++k) {
        gcc_spectrum_up_[k][0] = phat_accumulator_[k][0] * inv_avg;
        gcc_spectrum_up_[k][1] = phat_accumulator_[k][1] * inv_avg;
    }

    fftwf_execute(ifft_plan_);

    const float inv_n = 1.0f / static_cast<float>(config_.padded_size);
    for (int i = 0; i < config_.upsampled_size; ++i) {
        gcc_result_up_[i] *= inv_n;
    }

    const int M = config_.upsampled_size;
    const int md_up = max_delay_samples_ * config_.upsample_factor;
    float peak_val = -1e30f;
    int peak_idx = 0;

    for (int i = 0; i <= md_up && i < M; ++i) {
        if (gcc_result_up_[i] > peak_val) {
            peak_val = gcc_result_up_[i];
            peak_idx = i;
        }
    }
    for (int i = M - md_up; i < M; ++i) {
        if (i >= 0 && gcc_result_up_[i] > peak_val) {
            peak_val = gcc_result_up_[i];
            peak_idx = i;
        }
    }

    float delta = 0.0f;
    const int left_idx = (peak_idx - 1 + M) % M;
    const int right_idx = (peak_idx + 1) % M;
    const float y_l = gcc_result_up_[left_idx];
    const float y_c = gcc_result_up_[peak_idx];
    const float y_r = gcc_result_up_[right_idx];
    const float denom = y_l - 2.0f * y_c + y_r;
    if (std::fabs(denom) > kEpsilon) {
        delta = 0.5f * (y_l - y_r) / denom;
        delta = std::clamp(delta, -0.5f, 0.5f);
    }

    float up_delay = 0.0f;
    if (peak_idx <= M / 2) {
        up_delay = static_cast<float>(peak_idx) + delta;
    } else {
        up_delay = static_cast<float>(peak_idx) - static_cast<float>(M) + delta;
    }

    result.ready = true;
    result.delay_samples = up_delay / static_cast<float>(config_.upsample_factor);
    result.tdoa_seconds = result.delay_samples
        / static_cast<float>(config_.sample_rate);
    // [P1.1] Normalize peak so confidence is invariant to fft_size /
    // band-limit. Ideal noise-free peak after IFFT * (1/padded_size) ≈
    // (2 * active_bins - 1) / padded_size, where active_bins counts the
    // PHAT-summed bins (including DC: indices 0..band_limit_bin_max_
    // inclusive). Post-normalize, multi-channel peak_value ≈ 1.0 for a
    // clean source and is meaningful to compare against the same
    // confidence_threshold across different mic geometries / band limits.
    // (2-ch SoundLocator is full-band so its raw peak is already ≈ 1.0
    // by construction; it intentionally does NOT apply this normalization
    // to preserve the bit-stable contract.)
    const int active_bins = band_limit_bin_max_ + 1;
    const float ideal_peak =
        (2.0f * static_cast<float>(active_bins) - 1.0f)
        / static_cast<float>(config_.padded_size);
    result.peak_value = (ideal_peak > 1e-10f)
        ? (peak_val / ideal_peak)
        : peak_val;

    std::memset(phat_accumulator_, 0,
                config_.spectrum_size * sizeof(fftwf_complex));
    avg_count_ = 0;
    return result;
}

int GccPhatPair::GetMaxDelaySamples() const {
    return max_delay_samples_;
}

int GccPhatPair::GetBandLimitBinMax() const {
    return band_limit_bin_max_;
}

const float* GccPhatPair::GetGccData() const {
    return gcc_result_up_;
}

int GccPhatPair::GetGccSize() const {
    return config_.upsampled_size;
}

}  // namespace internal
}  // namespace SpacemitAudio
