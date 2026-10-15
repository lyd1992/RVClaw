/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GCC-PHAT based dual-channel sound source localization.
 *
 * Precision strategy:
 *   1. Forward FFT at padded_size = 2 * frame_size
 *   2. PHAT-weighted cross-spectrum (spectrum_size bins)
 *   3. Frequency-domain zero-padding to upsampled_size = padded_size * R
 *   4. Upsampled IFFT → GCC with 1/R-sample resolution
 *   5. Parabolic interpolation on upsampled peak for final refinement
 *   6. Multi-frame averaging for temporal stability
 */

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "doa_service.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SpacemitAudio {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct SoundLocator::Impl {
    SoundLocatorConfig config;

    // Effective upsample factor (computed from config)
    int upsample_factor = 1;

    // Forward FFT dimensions
    int padded_size = 0;       // 2 * frame_size (time-domain zero-padded)
    int spectrum_size = 0;     // padded_size / 2 + 1

    // Upsampled IFFT dimensions
    int upsampled_size = 0;    // padded_size * upsample_factor
    int up_spectrum_size = 0;  // upsampled_size / 2 + 1

    // FFTW resources — forward FFTs
    fftwf_plan fft_plan_a = nullptr;
    fftwf_plan fft_plan_b = nullptr;
    float* fft_in_a = nullptr;
    float* fft_in_b = nullptr;
    fftwf_complex* fft_out_a = nullptr;
    fftwf_complex* fft_out_b = nullptr;

    // FFTW resources — upsampled inverse FFT
    fftwf_plan ifft_plan = nullptr;
    fftwf_complex* gcc_spectrum_up = nullptr;   // up_spectrum_size bins
    float* gcc_result_up = nullptr;             // upsampled_size points

    // Pre-computed Hann window
    std::vector<float> hann_window;

    // Frequency-domain accumulator (spectrum_size bins)
    fftwf_complex* phat_accumulator = nullptr;
    int avg_count = 0;

    // Physical constraint (in original sample units)
    int max_delay_samples = 0;

    // Cached results (latest batch)
    float tdoa = 0.0f;
    float doa = 0.0f;
    float confidence = 0.0f;
    bool valid = false;

    // Global accumulation across batches (for GetAverageDOA).
    // We build a confidence-weighted histogram in upsampled delay space so
    // the summary estimate follows the dominant delay cluster instead of a
    // fragile arithmetic angle mean near 0° / 180°.
    std::vector<float> global_delay_hist_weight;
    std::vector<float> global_delay_hist_sum;
    int global_delay_hist_offset = 0;
    int global_result_count = 0;

    // Conversion buffers (reused across calls)
    std::vector<float> buf_a;
    std::vector<float> buf_b;

    bool initialized = false;

    // helpers
    void DestroyFFT();
    bool ComputeGCC(const float* ch0, const float* ch1, size_t n);
    bool FindPeakAndConvert();
};

// ---------------------------------------------------------------------------
// Impl helpers
// ---------------------------------------------------------------------------

void SoundLocator::Impl::DestroyFFT() {
    if (fft_plan_a) {
        fftwf_destroy_plan(fft_plan_a);
        fft_plan_a = nullptr;
    }
    if (fft_plan_b) {
        fftwf_destroy_plan(fft_plan_b);
        fft_plan_b = nullptr;
    }
    if (ifft_plan) {
        fftwf_destroy_plan(ifft_plan);
        ifft_plan = nullptr;
    }

    if (fft_in_a) {
        fftwf_free(fft_in_a);
        fft_in_a = nullptr;
    }
    if (fft_in_b) {
        fftwf_free(fft_in_b);
        fft_in_b = nullptr;
    }
    if (fft_out_a) {
        fftwf_free(fft_out_a);
        fft_out_a = nullptr;
    }
    if (fft_out_b) {
        fftwf_free(fft_out_b);
        fft_out_b = nullptr;
    }
    if (gcc_spectrum_up) {
        fftwf_free(gcc_spectrum_up);
        gcc_spectrum_up = nullptr;
    }
    if (gcc_result_up) {
        fftwf_free(gcc_result_up);
        gcc_result_up = nullptr;
    }
    if (phat_accumulator) {
        fftwf_free(phat_accumulator);
        phat_accumulator = nullptr;
    }

    initialized = false;
}

/// Run GCC-PHAT on one frame: FFT → cross-spectrum → accumulate in freq domain.
/// When enough frames are accumulated, do a single IFFT.
bool SoundLocator::Impl::ComputeGCC(const float* ch0, const float* ch1, size_t n) {
    const int fs = config.frame_size;
    const int N = padded_size;

    // --- 1. Windowing + zero-padding ---
    const size_t copy_len = std::min(static_cast<size_t>(fs), n);
    for (size_t i = 0; i < copy_len; ++i) {
        fft_in_a[i] = ch0[i] * hann_window[i];
        fft_in_b[i] = ch1[i] * hann_window[i];
    }
    if (copy_len < static_cast<size_t>(N)) {
        std::memset(fft_in_a + copy_len, 0, (N - copy_len) * sizeof(float));
        std::memset(fft_in_b + copy_len, 0, (N - copy_len) * sizeof(float));
    }

    // --- 2. Forward FFT (R2C, size N) ---
    fftwf_execute(fft_plan_a);
    fftwf_execute(fft_plan_b);

    // --- 3. Cross-power spectrum + PHAT weighting → accumulate in freq domain ---
    constexpr float kEpsilon = 1e-10f;

    for (int k = 0; k < spectrum_size; ++k) {
        const float ar = fft_out_a[k][0], ai = fft_out_a[k][1];
        const float br = fft_out_b[k][0], bi = fft_out_b[k][1];

        const float cross_r = ar * br + ai * bi;
        const float cross_i = ai * br - ar * bi;

        const float mag = std::sqrt(cross_r * cross_r + cross_i * cross_i);

        if (mag > kEpsilon) {
            const float inv_mag = 1.0f / mag;
            phat_accumulator[k][0] += cross_r * inv_mag;
            phat_accumulator[k][1] += cross_i * inv_mag;
        }
    }
    ++avg_count;

    if (avg_count < config.avg_frames) {
        return false;
    }

    // --- 4. Prepare upsampled spectrum and do single IFFT ---
    const int M = upsampled_size;
    const float inv_avg = 1.0f / static_cast<float>(avg_count);

    // Zero the entire upsampled spectrum, then fill from accumulator
    std::memset(gcc_spectrum_up, 0, up_spectrum_size * sizeof(fftwf_complex));
    for (int k = 0; k < spectrum_size; ++k) {
        gcc_spectrum_up[k][0] = phat_accumulator[k][0] * inv_avg;
        gcc_spectrum_up[k][1] = phat_accumulator[k][1] * inv_avg;
    }

    // Single IFFT for the entire batch
    fftwf_execute(ifft_plan);

    // Normalise by N (not M): only spectrum_size bins carry energy
    const float inv_n = 1.0f / static_cast<float>(N);
    for (int i = 0; i < M; ++i) {
        gcc_result_up[i] *= inv_n;
    }

    // Reset freq-domain accumulator
    std::memset(phat_accumulator, 0, spectrum_size * sizeof(fftwf_complex));
    avg_count = 0;

    return true;
}

/// Find the peak in the averaged GCC (in gcc_result_up after single IFFT),
/// refine with parabolic interpolation, convert to TDOA / DOA.
bool SoundLocator::Impl::FindPeakAndConvert() {
    const int M = upsampled_size;
    const int R = upsample_factor;
    // Search range in upsampled indices
    const int md_up = max_delay_samples * R;

    // --- Peak search in valid range ---
    float peak_val = -1e30f;
    int peak_idx = 0;

    // Positive delays: indices [0 .. md_up]
    for (int i = 0; i <= md_up && i < M; ++i) {
        if (gcc_result_up[i] > peak_val) {
            peak_val = gcc_result_up[i];
            peak_idx = i;
        }
    }
    // Negative delays: indices [M - md_up .. M - 1]
    for (int i = M - md_up; i < M; ++i) {
        if (i >= 0 && gcc_result_up[i] > peak_val) {
            peak_val = gcc_result_up[i];
            peak_idx = i;
        }
    }

    // --- Parabolic interpolation on GCC ---
    float delta = 0.0f;
    {
        const int left_idx  = (peak_idx - 1 + M) % M;
        const int right_idx = (peak_idx + 1) % M;

        const float y_l = gcc_result_up[left_idx];
        const float y_c = gcc_result_up[peak_idx];
        const float y_r = gcc_result_up[right_idx];

        const float denom = y_l - 2.0f * y_c + y_r;
        if (std::fabs(denom) > 1e-10f) {
            delta = 0.5f * (y_l - y_r) / denom;
            delta = std::max(-0.5f, std::min(0.5f, delta));
        }
    }

    // --- Convert upsampled index to signed delay in original samples ---
    float up_delay;
    if (peak_idx <= M / 2) {
        up_delay = static_cast<float>(peak_idx) + delta;
    } else {
        up_delay = static_cast<float>(peak_idx) - static_cast<float>(M) + delta;
    }
    // Divide by upsample factor to get delay in original sample units
    float delay = up_delay / static_cast<float>(R);

    tdoa = delay / static_cast<float>(config.sample_rate);

    float sin_theta = config.sound_speed * tdoa / config.mic_distance;
    sin_theta = std::max(-1.0f, std::min(1.0f, sin_theta));
    doa = 90.0f - std::asin(sin_theta) * 180.0f / static_cast<float>(M_PI);

    // 2-ch path is full-band: active_bins = spectrum_size = padded/2+1, so
    // ideal_peak = (padded+1)/padded ≈ 1.0 already — scale-invariant w.r.t.
    // fft_size by construction. No P1.1 normalization needed; preserves the
    // doa_service.h:11 "2-channel SoundLocator is preserved bit-stable
    // across releases" contract. P1.1 normalization is only applied on the
    // band-limited multi path (gcc_phat_pair.cpp) where active_bins varies
    // with band-limit / mic geometry.
    confidence = peak_val;
    valid = (confidence >= config.confidence_threshold);

    // Accumulate globally for GetAverageDOA
    if (valid) {
        const int hist_idx = std::clamp(
            static_cast<int>(std::lround(up_delay)) + global_delay_hist_offset,
            0,
            static_cast<int>(global_delay_hist_weight.size()) - 1);
        global_delay_hist_weight[hist_idx] += confidence;
        global_delay_hist_sum[hist_idx] += up_delay * confidence;
        ++global_result_count;
    }

    return valid;
}

// ---------------------------------------------------------------------------
// SoundLocator public API
// ---------------------------------------------------------------------------

SoundLocator::SoundLocator(const SoundLocatorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

SoundLocator::~SoundLocator() {
    if (impl_) {
        impl_->DestroyFFT();
    }
}

SoundLocator::SoundLocator(SoundLocator&& other) noexcept = default;
SoundLocator& SoundLocator::operator=(SoundLocator&& other) noexcept = default;

bool SoundLocator::Initialize() {
    auto& d = *impl_;

    d.DestroyFFT();

    // Forward FFT size: 2× frame_size (time-domain zero-pad)
    d.padded_size = d.config.fft_size > 0
                        ? d.config.fft_size
                        : 2 * d.config.frame_size;
    if (d.padded_size % 2 != 0) {
        ++d.padded_size;
    }
    d.spectrum_size = d.padded_size / 2 + 1;

    // Compute effective upsample factor
    if (d.config.upsample_factor > 0) {
        d.upsample_factor = d.config.upsample_factor;
    } else {
        // Adaptive: target ~4μs resolution → ceil(1 / (4e-6 * sample_rate))
        d.upsample_factor = std::max(1,
            static_cast<int>(std::ceil(
                1.0 / (4e-6 * d.config.sample_rate))));
    }

    // Upsampled IFFT size
    d.upsampled_size = d.padded_size * d.upsample_factor;
    d.up_spectrum_size = d.upsampled_size / 2 + 1;

    // Physical max delay (original samples)
    d.max_delay_samples = static_cast<int>(
        std::ceil(d.config.mic_distance / d.config.sound_speed
                  * static_cast<float>(d.config.sample_rate)));
    if (d.max_delay_samples > d.padded_size / 2) {
        d.max_delay_samples = d.padded_size / 2;
    }

    // --- Allocate forward FFT buffers ---
    d.fft_in_a  = fftwf_alloc_real(d.padded_size);
    d.fft_in_b  = fftwf_alloc_real(d.padded_size);
    d.fft_out_a = fftwf_alloc_complex(d.spectrum_size);
    d.fft_out_b = fftwf_alloc_complex(d.spectrum_size);

    // --- Allocate upsampled IFFT buffers ---
    d.gcc_spectrum_up = fftwf_alloc_complex(d.up_spectrum_size);
    d.gcc_result_up   = fftwf_alloc_real(d.upsampled_size);

    // --- Allocate frequency-domain accumulator ---
    d.phat_accumulator = fftwf_alloc_complex(d.spectrum_size);

    if (!d.fft_in_a || !d.fft_in_b || !d.fft_out_a || !d.fft_out_b
        || !d.gcc_spectrum_up || !d.gcc_result_up || !d.phat_accumulator) {
        d.DestroyFFT();
        return false;
    }

    // --- Create FFTW plans ---
    const unsigned fftw_flags = d.config.use_fftw_measure
                                    ? FFTW_MEASURE : FFTW_ESTIMATE;
    d.fft_plan_a = fftwf_plan_dft_r2c_1d(
        d.padded_size, d.fft_in_a, d.fft_out_a, fftw_flags);
    d.fft_plan_b = fftwf_plan_dft_r2c_1d(
        d.padded_size, d.fft_in_b, d.fft_out_b, fftw_flags);
    d.ifft_plan = fftwf_plan_dft_c2r_1d(
        d.upsampled_size, d.gcc_spectrum_up, d.gcc_result_up, fftw_flags);

    if (!d.fft_plan_a || !d.fft_plan_b || !d.ifft_plan) {
        d.DestroyFFT();
        return false;
    }

    // Pre-compute Hann window
    d.hann_window.resize(d.config.frame_size);
    for (int i = 0; i < d.config.frame_size; ++i) {
        d.hann_window[i] = 0.5f * (1.0f - std::cos(
            2.0f * static_cast<float>(M_PI) * i / (d.config.frame_size - 1)));
    }

    // Initialize frequency-domain accumulator
    std::memset(d.phat_accumulator, 0, d.spectrum_size * sizeof(fftwf_complex));
    d.avg_count = 0;
    d.global_delay_hist_offset = d.max_delay_samples * d.upsample_factor;
    d.global_delay_hist_weight.assign(
        2 * d.global_delay_hist_offset + 1, 0.0f);
    d.global_delay_hist_sum.assign(
        2 * d.global_delay_hist_offset + 1, 0.0f);

    d.tdoa = 0.0f;
    d.doa = 0.0f;
    d.confidence = 0.0f;
    d.valid = false;

    d.initialized = true;
    return true;
}

void SoundLocator::Reset() {
    if (!impl_ || !impl_->initialized) return;
    std::memset(impl_->phat_accumulator, 0,
                impl_->spectrum_size * sizeof(fftwf_complex));
    impl_->avg_count = 0;
    impl_->tdoa = 0.0f;
    impl_->doa = 0.0f;
    impl_->confidence = 0.0f;
    impl_->valid = false;
    std::fill(impl_->global_delay_hist_weight.begin(),
            impl_->global_delay_hist_weight.end(), 0.0f);
    std::fill(impl_->global_delay_hist_sum.begin(),
            impl_->global_delay_hist_sum.end(), 0.0f);
    impl_->global_result_count = 0;
}

// --- Process: separated float channels ---
bool SoundLocator::Process(const float* ch0, const float* ch1, size_t num_frames) {
    if (!impl_ || !impl_->initialized || !ch0 || !ch1 || num_frames == 0) {
        return false;
    }

    const size_t fs = static_cast<size_t>(impl_->config.frame_size);
    size_t offset = 0;
    bool result_ready = false;

    while (offset < num_frames) {
        const size_t chunk = std::min(fs, num_frames - offset);
        if (impl_->ComputeGCC(ch0 + offset, ch1 + offset, chunk)) {
            impl_->FindPeakAndConvert();
            result_ready = true;
        }
        offset += chunk;
    }
    return result_ready;
}

// --- Process: interleaved float ---
bool SoundLocator::Process(const float* interleaved, size_t num_frames) {
    if (!impl_ || !interleaved || num_frames == 0) return false;

    impl_->buf_a.resize(num_frames);
    impl_->buf_b.resize(num_frames);

    for (size_t i = 0; i < num_frames; ++i) {
        impl_->buf_a[i] = interleaved[2 * i];
        impl_->buf_b[i] = interleaved[2 * i + 1];
    }
    return Process(impl_->buf_a.data(), impl_->buf_b.data(), num_frames);
}

// --- Process: interleaved PCM16 ---
bool SoundLocator::Process(const int16_t* interleaved, size_t num_frames) {
    if (!impl_ || !interleaved || num_frames == 0) return false;

    impl_->buf_a.resize(num_frames);
    impl_->buf_b.resize(num_frames);

    constexpr float kScale = 1.0f / 32768.0f;
    for (size_t i = 0; i < num_frames; ++i) {
        impl_->buf_a[i] = static_cast<float>(interleaved[2 * i])     * kScale;
        impl_->buf_b[i] = static_cast<float>(interleaved[2 * i + 1]) * kScale;
    }
    return Process(impl_->buf_a.data(), impl_->buf_b.data(), num_frames);
}

// --- Result accessors ---
float SoundLocator::GetTDOA() const       { return impl_->tdoa; }
float SoundLocator::GetDOA() const        { return impl_->doa; }
float SoundLocator::GetConfidence() const { return impl_->confidence; }
bool  SoundLocator::IsValid() const       { return impl_->valid; }

float SoundLocator::GetAverageDOA() const {
    if (impl_->global_result_count <= 0
        || impl_->global_delay_hist_weight.empty()) {
        return 90.0f;
    }

    const auto best_it = std::max_element(
        impl_->global_delay_hist_weight.begin(),
        impl_->global_delay_hist_weight.end());
    if (best_it == impl_->global_delay_hist_weight.end() || *best_it <= 0.0f) {
        return 90.0f;
    }

    const int best_idx = static_cast<int>(
        std::distance(impl_->global_delay_hist_weight.begin(), best_it));
    const int radius = std::max(1, impl_->upsample_factor / 4);

    float sum_w = 0.0f;
    float sum_up_delay = 0.0f;
    const int start = std::max(0, best_idx - radius);
    const int end = std::min(
        static_cast<int>(impl_->global_delay_hist_weight.size()) - 1,
        best_idx + radius);
    for (int i = start; i <= end; ++i) {
        sum_w += impl_->global_delay_hist_weight[i];
        sum_up_delay += impl_->global_delay_hist_sum[i];
    }
    if (sum_w <= 0.0f) {
        return 90.0f;
    }

    const float up_delay = sum_up_delay / sum_w;
    const float delay = up_delay / static_cast<float>(impl_->upsample_factor);
    const float tdoa = delay / static_cast<float>(impl_->config.sample_rate);

    float sin_theta = impl_->config.sound_speed * tdoa / impl_->config.mic_distance;
    sin_theta = std::max(-1.0f, std::min(1.0f, sin_theta));
    return 90.0f - std::asin(sin_theta) * 180.0f / static_cast<float>(M_PI);
}

int SoundLocator::GetResultCount() const {
    return impl_->global_result_count;
}

const SoundLocatorConfig& SoundLocator::GetConfig() const {
    return impl_->config;
}

int SoundLocator::GetMaxDelaySamples() const {
    return impl_->max_delay_samples;
}

}  // namespace SpacemitAudio
