/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Multi-channel planar sound source localization using pairwise GCC-PHAT
 * and MPCC-LSQ fusion.
 *
 * Pair ordering convention (N=3): pair index 0=(0,1), 1=(0,2), 2=(1,2)
 * (lexicographic on (i, j) with i<j). For N>3 the same lexicographic rule
 * extends to all C(N,2) pairs.
 *
 * GCC sign convention: gcc_phat_pair.Accumulate(spec_i, spec_j) returns a
 * TDOA whose sign matches τ_ij = arrival_i − arrival_j. Closure for N=3
 * is then τ_01 + τ_12 − τ_02 = 0; we compute it as
 *   closure = τ(0,1) + τ(1,2) + τ(2,0) = τ_01 + τ_12 − τ_02
 * by negating the (0,2) pair's stored TDOA (pairs are stored in i<j form).
 */

// Single-header public API: this .cpp implements the MultiSoundLocator class
// declared in `doa_service.h` (the merged single public header for the doa
// component; see v1.1 changelog A12). cpplint's basename-derived include
// ordering rule does not match this layout, so the system / std headers
// below are tagged with NOLINT(build/include_order).
#include "doa_service.h"

#include <fftw3.h>  // NOLINT(build/include_order)

#include <algorithm>  // NOLINT(build/include_order)
#include <cmath>  // NOLINT(build/include_order)
#include <cstring>  // NOLINT(build/include_order)
#include <deque>  // NOLINT(build/include_order)
#include <limits>  // NOLINT(build/include_order)
#include <stdexcept>  // NOLINT(build/include_order)
#include <string>  // NOLINT(build/include_order)
#include <utility>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include "gcc_phat_pair.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace SpacemitAudio {
namespace {

constexpr float kEpsilon = 1e-10f;

float NormalizeAngle360(float degrees) {
    float value = std::fmod(degrees, 360.0f);
    if (value < 0.0f) {
        value += 360.0f;
    }
    if (value >= 360.0f) {
        value -= 360.0f;
    }
    return value;
}

float Distance2D(const MicrophonePosition& a, const MicrophonePosition& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

struct MultiSoundLocator::Impl {
    struct PairInfo {
        int i = 0;
        int j = 0;
        float row_x = 0.0f;  // p_j.x - p_i.x, matching GCC sign t_i - t_j
        float row_y = 0.0f;  // p_j.y - p_i.y
        float distance_m = 0.0f;
    };

    struct PairPeakCandidate {
        float tdoa_seconds = 0.0f;
        float delay_samples = 0.0f;
        float peak_value = 0.0f;
    };

    MultiSoundLocatorConfig config;
    int n_channels = 0;
    int upsample_factor = 1;
    int padded_size = 0;
    int spectrum_size = 0;
    int upsampled_size = 0;
    int up_spectrum_size = 0;

    std::vector<float*> fft_in;
    std::vector<fftwf_complex*> fft_out;
    std::vector<fftwf_plan> fft_plans;
    std::vector<float> hann_window;

    std::vector<PairInfo> pairs;
    std::vector<internal::GccPhatPair> gcc_pairs;
    std::vector<float> latest_tdoa;
    std::vector<float> latest_pair_confidences;  // [A3] cached per-pair GCC peak

    // Cholesky factor of A^T A = L * L^T for the 2D LSQ solve.
    float chol_l00 = 0.0f;
    float chol_l10 = 0.0f;
    float chol_l11 = 0.0f;

    MultiSoundLocatorResult result;

    // Sliding-window accumulator for GetAverageAzimuth. If
    // config.max_avg_seconds <= 0, the deque is unbounded (matches v1
    // behavior). Otherwise it is capped at max_avg_batches and the running
    // sum is decremented on eviction so GetAverageAzimuth stays O(1).
    //
    // [A6] BatchEntry now also carries `weight` (= confidence) so that
    // sum_weights stays in lockstep when an entry is evicted. The
    // resultant length R = ‖(avg_x, avg_y)‖ / sum_weights.
    struct BatchEntry { float weighted_x; float weighted_y; float weight; };
    std::deque<BatchEntry> avg_history;
    int   max_avg_batches = 0;   // 0 = unbounded
    float average_x = 0.0f;
    float average_y = 0.0f;
    double sum_weights = 0.0;   // [A6] Σ confidence over the active window
    int   result_count = 0;

    std::vector<std::vector<float>> planar_buffers;
    std::vector<float> int16_float_buffer;
    std::vector<const float*> channel_view;

    // [P1.1] Effective closure threshold (samples), computed once at
    // Initialize from max(config.closure_threshold_samples,
    // config.closure_threshold_fraction * max_physical_TDOA_samples).
    // Cached so SolveReadyPairs is array-scale invariant without
    // recomputing per call.
    float effective_closure_samples = 0.0f;

    bool initialized = false;

    void DestroyFFT();
    bool BuildGeometry();
    bool SolveNormal(float rhs_x, float rhs_y, float& out_x, float& out_y) const;
    void ValidateChannelCount(int n) const;
    bool ProcessPlanar(const float* const* channel_ptrs, size_t n_frames);
    bool ProcessChunk(const float* const* channel_ptrs, size_t chunk);
    bool SolveReadyPairs();
    std::vector<PairPeakCandidate> CollectPairPeakCandidates(
        size_t pair_idx, const internal::GccPhatPairResult& primary_result) const;
    bool SelectThreeChannelPeaks(
        const std::vector<internal::GccPhatPairResult>& pair_results,
        std::vector<PairPeakCandidate>* selected_peaks) const;
    bool SamplePairGccValue(
        size_t pair_idx, const internal::GccPhatPairResult& pair_result,
        float delay_samples, float* value) const;
    bool SelectSrpPhatPeaks(
        const std::vector<internal::GccPhatPairResult>& pair_results,
        std::vector<PairPeakCandidate>* selected_peaks) const;
    int FindPairIndex(int i, int j) const;
};

void MultiSoundLocator::Impl::DestroyFFT() {
    for (auto plan : fft_plans) {
        if (plan) {
            fftwf_destroy_plan(plan);
        }
    }
    fft_plans.clear();

    for (auto* ptr : fft_in) {
        if (ptr) {
            fftwf_free(ptr);
        }
    }
    fft_in.clear();

    for (auto* ptr : fft_out) {
        if (ptr) {
            fftwf_free(ptr);
        }
    }
    fft_out.clear();

    gcc_pairs.clear();
    pairs.clear();
    latest_tdoa.clear();
    latest_pair_confidences.clear();
    hann_window.clear();
    initialized = false;
}

bool MultiSoundLocator::Impl::BuildGeometry() {
    pairs.clear();

    float max_pair_distance = 0.0f;
    for (int i = 0; i < n_channels; ++i) {
        for (int j = i + 1; j < n_channels; ++j) {
            PairInfo pair;
            pair.i = i;
            pair.j = j;
            pair.row_x = config.microphones[j].x - config.microphones[i].x;
            pair.row_y = config.microphones[j].y - config.microphones[i].y;
            pair.distance_m = Distance2D(config.microphones[i],
                config.microphones[j]);
            max_pair_distance = std::max(max_pair_distance, pair.distance_m);
            pairs.push_back(pair);
        }
    }

    if (pairs.empty() || max_pair_distance <= 0.0f) {
        return false;
    }

    if (config.max_frequency_hz <= 0.0f) {
        config.max_frequency_hz = config.sound_speed / (2.0f * max_pair_distance);
    }

    // [P1.1] Derive array-scale-invariant closure threshold once at init.
    // max_physical_TDOA_samples = max_pair_distance / sound_speed * sample_rate.
    // Effective threshold = max(explicit samples, fraction * physical_max).
    // Both <= 0 disables the gate entirely.
    const float max_physical_tdoa_samples = (config.sound_speed > 0.0f)
        ? (max_pair_distance / config.sound_speed
           * static_cast<float>(config.sample_rate))
        : 0.0f;
    const float from_fraction = std::max(0.0f, config.closure_threshold_fraction)
                              * max_physical_tdoa_samples;
    const float from_samples = std::max(0.0f, config.closure_threshold_samples);
    effective_closure_samples = std::max(from_samples, from_fraction);

    float ata00 = 0.0f;
    float ata01 = 0.0f;
    float ata11 = 0.0f;
    for (const auto& pair : pairs) {
        ata00 += pair.row_x * pair.row_x;
        ata01 += pair.row_x * pair.row_y;
        ata11 += pair.row_y * pair.row_y;
    }

    if (ata00 <= kEpsilon) {
        return false;
    }
    chol_l00 = std::sqrt(ata00);
    chol_l10 = ata01 / chol_l00;
    const float diag = ata11 - chol_l10 * chol_l10;
    if (diag <= kEpsilon) {
        return false;
    }
    chol_l11 = std::sqrt(diag);
    return true;
}

bool MultiSoundLocator::Impl::SolveNormal(
    float rhs_x, float rhs_y, float& out_x, float& out_y) const {
    if (chol_l00 <= 0.0f || chol_l11 <= 0.0f) {
        return false;
    }

    const float y0 = rhs_x / chol_l00;
    const float y1 = (rhs_y - chol_l10 * y0) / chol_l11;
    out_y = y1 / chol_l11;
    out_x = (y0 - chol_l10 * out_y) / chol_l00;
    return true;
}

void MultiSoundLocator::Impl::ValidateChannelCount(int n) const {
    if (n != n_channels) {
        throw std::invalid_argument(
            "MultiSoundLocator::Process expected " + std::to_string(n_channels)
            + " channels, got " + std::to_string(n));
    }
}

bool MultiSoundLocator::Impl::ProcessPlanar(
    const float* const* channel_ptrs, size_t n_frames) {
    if (!initialized || !channel_ptrs || n_frames == 0) {
        return false;
    }

    for (int ch = 0; ch < n_channels; ++ch) {
        if (!channel_ptrs[ch]) {
            return false;
        }
    }

    const size_t frame = static_cast<size_t>(config.frame_size);
    size_t offset = 0;
    bool result_ready = false;

    while (offset < n_frames) {
        const size_t chunk = std::min(frame, n_frames - offset);
        channel_view.resize(n_channels);
        for (int ch = 0; ch < n_channels; ++ch) {
            channel_view[ch] = channel_ptrs[ch] + offset;
        }
        if (ProcessChunk(channel_view.data(), chunk)) {
            result_ready = true;
        }
        offset += chunk;
    }

    return result_ready;
}

bool MultiSoundLocator::Impl::ProcessChunk(
    const float* const* channel_ptrs, size_t chunk) {
    const size_t copy_len = std::min(
        chunk, static_cast<size_t>(config.frame_size));

    if (config.min_signal_rms > 0.0f) {
        double energy = 0.0;
        size_t sample_count = 0;
        for (int ch = 0; ch < n_channels; ++ch) {
            for (size_t i = 0; i < copy_len; ++i) {
                const double sample = static_cast<double>(channel_ptrs[ch][i]);
                energy += sample * sample;
            }
            sample_count += copy_len;
        }
        const float rms = sample_count > 0
            ? static_cast<float>(std::sqrt(energy / sample_count)) : 0.0f;
        if (rms < config.min_signal_rms) {
            for (auto& pair : gcc_pairs) {
                pair.Reset();
            }
            std::fill(latest_tdoa.begin(), latest_tdoa.end(), 0.0f);
            std::fill(latest_pair_confidences.begin(),
                latest_pair_confidences.end(), 0.0f);
            result = MultiSoundLocatorResult{};
            return false;
        }
    }

    for (int ch = 0; ch < n_channels; ++ch) {
        std::memset(fft_in[ch], 0, padded_size * sizeof(float));
        for (size_t i = 0; i < copy_len; ++i) {
            fft_in[ch][i] = channel_ptrs[ch][i] * hann_window[i];
        }
        fftwf_execute(fft_plans[ch]);
    }

    for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
        const auto& pair = pairs[pair_idx];
        if (!gcc_pairs[pair_idx].Accumulate(fft_out[pair.i], fft_out[pair.j])) {
            return false;
        }
    }

    const bool all_ready = std::all_of(
        gcc_pairs.begin(), gcc_pairs.end(),
        [](const internal::GccPhatPair& pair) { return pair.IsReady(); });
    if (!all_ready) {
        return false;
    }

    return SolveReadyPairs();
}

std::vector<MultiSoundLocator::Impl::PairPeakCandidate>
MultiSoundLocator::Impl::CollectPairPeakCandidates(
    size_t pair_idx, const internal::GccPhatPairResult& primary_result) const {
    std::vector<PairPeakCandidate> candidates;
    if (!primary_result.ready) {
        return candidates;
    }

    auto add_candidate = [&candidates](const PairPeakCandidate& candidate) {
        if (!std::isfinite(candidate.tdoa_seconds)
            || !std::isfinite(candidate.delay_samples)
            || !std::isfinite(candidate.peak_value)
            || candidate.peak_value <= 0.0f) {
            return;
        }

        constexpr float kDuplicateDelaySamples = 0.20f;
        for (auto& existing : candidates) {
            if (std::fabs(existing.delay_samples - candidate.delay_samples)
                <= kDuplicateDelaySamples) {
                if (candidate.peak_value > existing.peak_value) {
                    existing = candidate;
                }
                return;
            }
        }
        candidates.push_back(candidate);
    };

    add_candidate(PairPeakCandidate{
        primary_result.tdoa_seconds,
        primary_result.delay_samples,
        primary_result.peak_value});

    if (pair_idx >= gcc_pairs.size() || !primary_result.gcc
        || primary_result.gcc_size <= 0 || upsample_factor <= 0
        || config.sample_rate <= 0 || padded_size <= 0) {
        return candidates;
    }

    const auto& gcc_pair = gcc_pairs[pair_idx];
    const float* gcc = primary_result.gcc;
    const int M = primary_result.gcc_size;
    const int md_up = std::min(
        gcc_pair.GetMaxDelaySamples() * upsample_factor, M / 2);
    if (md_up < 0) {
        return candidates;
    }

    const int active_bins = gcc_pair.GetBandLimitBinMax() + 1;
    const float ideal_peak =
        (2.0f * static_cast<float>(active_bins) - 1.0f)
        / static_cast<float>(padded_size);

    auto make_candidate = [&](int idx) {
        const int left_idx = (idx - 1 + M) % M;
        const int right_idx = (idx + 1) % M;
        const float y_l = gcc[left_idx];
        const float y_c = gcc[idx];
        const float y_r = gcc[right_idx];
        float delta = 0.0f;
        const float denom = y_l - 2.0f * y_c + y_r;
        if (std::fabs(denom) > kEpsilon) {
            delta = 0.5f * (y_l - y_r) / denom;
            delta = std::clamp(delta, -0.5f, 0.5f);
        }

        float up_delay = 0.0f;
        if (idx <= M / 2) {
            up_delay = static_cast<float>(idx) + delta;
        } else {
            up_delay = static_cast<float>(idx) - static_cast<float>(M) + delta;
        }

        PairPeakCandidate candidate;
        candidate.delay_samples =
            up_delay / static_cast<float>(upsample_factor);
        candidate.tdoa_seconds =
            candidate.delay_samples / static_cast<float>(config.sample_rate);
        candidate.peak_value = (ideal_peak > kEpsilon)
            ? (y_c / ideal_peak)
            : y_c;
        add_candidate(candidate);
    };

    auto is_local_max = [&](int idx) {
        const int left_idx = (idx - 1 + M) % M;
        const int right_idx = (idx + 1) % M;
        return gcc[idx] >= gcc[left_idx] && gcc[idx] >= gcc[right_idx];
    };

    for (int idx = 0; idx <= md_up && idx < M; ++idx) {
        if (is_local_max(idx)) {
            make_candidate(idx);
        }
    }
    for (int idx = M - md_up; idx < M; ++idx) {
        if (idx >= 0 && is_local_max(idx)) {
            make_candidate(idx);
        }
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const PairPeakCandidate& a, const PairPeakCandidate& b) {
            return a.peak_value > b.peak_value;
        });
    constexpr size_t kMaxCandidatesPerPair = 6;
    if (candidates.size() > kMaxCandidatesPerPair) {
        candidates.resize(kMaxCandidatesPerPair);
    }
    return candidates;
}

bool MultiSoundLocator::Impl::SelectThreeChannelPeaks(
    const std::vector<internal::GccPhatPairResult>& pair_results,
    std::vector<PairPeakCandidate>* selected_peaks) const {
    if (!selected_peaks || n_channels != 3 || pairs.size() != 3
        || pair_results.size() != 3) {
        return false;
    }

    std::vector<PairPeakCandidate> candidates[3];
    for (size_t pair_idx = 0; pair_idx < 3; ++pair_idx) {
        candidates[pair_idx] =
            CollectPairPeakCandidates(pair_idx, pair_results[pair_idx]);
        if (candidates[pair_idx].empty()) {
            return false;
        }
    }

    const float closure_limit =
        effective_closure_samples > 0.0f
            ? effective_closure_samples
            : std::numeric_limits<float>::infinity();

    struct CombinationScore {
        bool found = false;
        bool closure_ok = false;
        double score = -std::numeric_limits<double>::infinity();
        float closure_samples = std::numeric_limits<float>::infinity();
        float norm_error = std::numeric_limits<float>::infinity();
        PairPeakCandidate peaks[3];
    };

    CombinationScore best;

    auto consider = [&](const PairPeakCandidate& p0,
                        const PairPeakCandidate& p1,
                        const PairPeakCandidate& p2) {
        const PairPeakCandidate peaks[3] = {p0, p1, p2};
        float rhs_x = 0.0f;
        float rhs_y = 0.0f;
        double log_peak_sum = 0.0;
        for (size_t pair_idx = 0; pair_idx < 3; ++pair_idx) {
            const float distance_delta =
                config.sound_speed * peaks[pair_idx].tdoa_seconds;
            rhs_x += pairs[pair_idx].row_x * distance_delta;
            rhs_y += pairs[pair_idx].row_y * distance_delta;
            log_peak_sum += std::log(
                std::max(static_cast<double>(peaks[pair_idx].peak_value),
                        1e-12));
        }

        float wave_x = 0.0f;
        float wave_y = 0.0f;
        if (!SolveNormal(rhs_x, rhs_y, wave_x, wave_y)) {
            return;
        }

        const float norm = std::sqrt(wave_x * wave_x + wave_y * wave_y);
        float quality = 1.0f - std::fabs(1.0f - norm);
        if (quality < 0.0f) quality = 0.0f;
        if (quality > 1.0f) quality = 1.0f;

        const float closure_sec =
            p0.tdoa_seconds + p2.tdoa_seconds - p1.tdoa_seconds;
        const float closure_samples =
            std::fabs(closure_sec) * static_cast<float>(config.sample_rate);
        const bool closure_ok = closure_samples <= closure_limit;
        const float closure_norm = std::isfinite(closure_limit)
            ? closure_samples / std::max(closure_limit, kEpsilon)
            : closure_samples;
        const double score =
            (log_peak_sum / 3.0)
            + std::log(std::max(static_cast<double>(quality), 1e-12))
            - 2.0 * static_cast<double>(closure_norm);
        const float norm_error = std::fabs(1.0f - norm);

        const bool better =
            !best.found
            || (closure_ok && !best.closure_ok)
            || (closure_ok == best.closure_ok
                && (score > best.score + 1e-9
                    || (std::fabs(score - best.score) <= 1e-9
                        && (closure_samples < best.closure_samples
                            || (std::fabs(closure_samples
                                    - best.closure_samples) <= 1e-6f
                                && norm_error < best.norm_error)))));
        if (better) {
            best.found = true;
            best.closure_ok = closure_ok;
            best.score = score;
            best.closure_samples = closure_samples;
            best.norm_error = norm_error;
            best.peaks[0] = p0;
            best.peaks[1] = p1;
            best.peaks[2] = p2;
        }
    };

    for (const auto& p0 : candidates[0]) {
        for (const auto& p1 : candidates[1]) {
            for (const auto& p2 : candidates[2]) {
                consider(p0, p1, p2);
            }
        }
    }

    if (!best.found) {
        return false;
    }

    for (size_t pair_idx = 0; pair_idx < 3; ++pair_idx) {
        (*selected_peaks)[pair_idx] = best.peaks[pair_idx];
    }
    return true;
}

bool MultiSoundLocator::Impl::SamplePairGccValue(
    size_t pair_idx, const internal::GccPhatPairResult& pair_result,
    float delay_samples, float* value) const {
    if (!value || pair_idx >= gcc_pairs.size() || !pair_result.gcc
        || pair_result.gcc_size <= 1 || upsample_factor <= 0
        || padded_size <= 0) {
        return false;
    }

    const auto& gcc_pair = gcc_pairs[pair_idx];
    const float max_delay =
        static_cast<float>(gcc_pair.GetMaxDelaySamples()) + 1e-3f;
    if (std::fabs(delay_samples) > max_delay) {
        return false;
    }

    const int M = pair_result.gcc_size;
    float up_delay = delay_samples * static_cast<float>(upsample_factor);
    float wrapped = up_delay;
    if (wrapped < 0.0f) {
        wrapped += static_cast<float>(M);
    }
    if (wrapped < 0.0f || wrapped >= static_cast<float>(M)) {
        wrapped = std::fmod(wrapped, static_cast<float>(M));
        if (wrapped < 0.0f) {
            wrapped += static_cast<float>(M);
        }
    }

    const int idx0 = static_cast<int>(std::floor(wrapped));
    const int idx1 = (idx0 + 1) % M;
    const float frac = wrapped - static_cast<float>(idx0);
    const float raw =
        pair_result.gcc[idx0] * (1.0f - frac) + pair_result.gcc[idx1] * frac;

    const int active_bins = gcc_pair.GetBandLimitBinMax() + 1;
    const float ideal_peak =
        (2.0f * static_cast<float>(active_bins) - 1.0f)
        / static_cast<float>(padded_size);
    *value = (ideal_peak > kEpsilon) ? (raw / ideal_peak) : raw;
    return std::isfinite(*value);
}

bool MultiSoundLocator::Impl::SelectSrpPhatPeaks(
    const std::vector<internal::GccPhatPairResult>& pair_results,
    std::vector<PairPeakCandidate>* selected_peaks) const {
    if (!selected_peaks || selected_peaks->size() != pairs.size()
        || pair_results.size() != pairs.size() || pairs.empty()
        || config.sample_rate <= 0 || config.sound_speed <= 0.0f) {
        return false;
    }

    for (const auto& pair_result : pair_results) {
        if (!pair_result.ready || !pair_result.gcc || pair_result.gcc_size <= 0) {
            return false;
        }
    }

    float step_deg = config.search_step_deg;
    if (!std::isfinite(step_deg) || step_deg <= 0.0f) {
        step_deg = 1.0f;
    }
    step_deg = std::clamp(step_deg, 0.25f, 15.0f);

    struct SrpScore {
        bool found = false;
        float angle_deg = 0.0f;
        double score = -std::numeric_limits<double>::infinity();
        std::vector<PairPeakCandidate> peaks;
    };

    SrpScore best;
    const int steps = std::max(1,
        static_cast<int>(std::ceil(360.0f / step_deg)));

    for (int step = 0; step < steps; ++step) {
        const float angle_deg = std::min(
            static_cast<float>(step) * step_deg, 360.0f - step_deg);
        const float angle_rad =
            angle_deg * static_cast<float>(M_PI) / 180.0f;
        const float dir_x = std::cos(angle_rad);
        const float dir_y = std::sin(angle_rad);

        std::vector<PairPeakCandidate> peaks(pairs.size());
        double score = 0.0;
        bool ok = true;
        for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
            const auto& pair = pairs[pair_idx];
            const float distance_delta =
                pair.row_x * dir_x + pair.row_y * dir_y;
            const float delay_samples =
                distance_delta / config.sound_speed
                * static_cast<float>(config.sample_rate);
            float value = 0.0f;
            if (!SamplePairGccValue(pair_idx, pair_results[pair_idx],
                    delay_samples, &value)) {
                ok = false;
                break;
            }
            peaks[pair_idx] = PairPeakCandidate{
                delay_samples / static_cast<float>(config.sample_rate),
                delay_samples,
                std::max(value, 0.0f)};
            score += static_cast<double>(value);
        }

        if (!ok) {
            continue;
        }

        if (!best.found || score > best.score) {
            best.found = true;
            best.angle_deg = angle_deg;
            best.score = score;
            best.peaks = std::move(peaks);
        }
    }

    if (!best.found || best.score <= 0.0) {
        return false;
    }

    *selected_peaks = std::move(best.peaks);
    return true;
}

bool MultiSoundLocator::Impl::SolveReadyPairs() {
    float rhs_x = 0.0f;
    float rhs_y = 0.0f;
    float confidence_sum = 0.0f;  // [A4] arithmetic mean → kept as peak_score
    int valid_pairs_count = 0;

    std::vector<internal::GccPhatPairResult> pair_results(pairs.size());
    std::vector<PairPeakCandidate> selected_peaks(pairs.size());
    for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
        const auto pair_result = gcc_pairs[pair_idx].IfftAndPeak();
        if (!pair_result.ready) {
            return false;
        }
        pair_results[pair_idx] = pair_result;
        selected_peaks[pair_idx] = PairPeakCandidate{
            pair_result.tdoa_seconds,
            pair_result.delay_samples,
            pair_result.peak_value};
    }

    if (!SelectSrpPhatPeaks(pair_results, &selected_peaks)
        && n_channels == 3 && pairs.size() == 3) {
        SelectThreeChannelPeaks(pair_results, &selected_peaks);
    }

    for (size_t pair_idx = 0; pair_idx < pairs.size(); ++pair_idx) {
        latest_tdoa[pair_idx] = selected_peaks[pair_idx].tdoa_seconds;
        latest_pair_confidences[pair_idx] =
            selected_peaks[pair_idx].peak_value;  // [A3]
        const float distance_delta =
            config.sound_speed * selected_peaks[pair_idx].tdoa_seconds;
        rhs_x += pairs[pair_idx].row_x * distance_delta;
        rhs_y += pairs[pair_idx].row_y * distance_delta;
        confidence_sum += selected_peaks[pair_idx].peak_value;
        if (selected_peaks[pair_idx].peak_value
            >= config.confidence_threshold) {
            ++valid_pairs_count;
        }
    }

    float wave_x = 0.0f;
    float wave_y = 0.0f;
    if (!SolveNormal(rhs_x, rhs_y, wave_x, wave_y)) {
        return false;
    }

    const float norm = std::sqrt(wave_x * wave_x + wave_y * wave_y);
    const float azimuth_rad = std::atan2(wave_y, wave_x);
    const float azimuth_deg = NormalizeAngle360(
        azimuth_rad * 180.0f / static_cast<float>(M_PI)
        + config.azimuth_offset_deg);

    // [A5] clamped quality: symmetric penalty around ‖k̂‖ = 1.
    float quality = 1.0f - std::fabs(1.0f - norm);
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 1.0f) quality = 1.0f;

    // [A4] peak_score = pre-A4 arithmetic mean (backward-compat); confidence
    // = geometric mean × quality. For N pairs, geo-mean = (Π peak_i)^(1/N).
    // If any pair has peak <= 0 the geo-mean collapses to 0 (correct: a
    // dead pair invalidates the joint estimate).
    const int n_pairs = static_cast<int>(pairs.size());
    const float peak_score = n_pairs > 0
        ? confidence_sum / static_cast<float>(n_pairs)
        : 0.0f;
    float pair_geo_mean = 0.0f;
    if (n_pairs > 0) {
        double log_sum = 0.0;
        bool any_nonpositive = false;
        for (int p = 0; p < n_pairs; ++p) {
            const float v = latest_pair_confidences[p];
            if (v <= 0.0f) {
                any_nonpositive = true;
                break;
            }
            log_sum += std::log(static_cast<double>(v));
        }
        if (!any_nonpositive) {
            pair_geo_mean = static_cast<float>(
                std::exp(log_sum / static_cast<double>(n_pairs)));
        }
    }
    const float confidence = pair_geo_mean * quality;

    // [A1/A2] TDOA closure (only meaningful for N=3).
    float closure_sec = 0.0f;
    float closure_samples = 0.0f;
    bool closure_ok = true;  // default-true when N != 3 (gate not applicable)
    if (n_pairs == 3 && n_channels == 3) {
        // pairs[] indexed lexicographically: 0=(0,1), 1=(0,2), 2=(1,2)
        // τ(0,1) + τ(1,2) − τ(0,2) should ≈ 0 for a consistent batch
        closure_sec = latest_tdoa[0] + latest_tdoa[2] - latest_tdoa[1];
        const float abs_sec = std::fabs(closure_sec);
        closure_samples = abs_sec * static_cast<float>(config.sample_rate);
        // [P1.1] Use effective_closure_samples (computed once at init from
        // max(closure_threshold_samples, fraction * max_physical_TDOA)) so
        // the gate is array-scale invariant.
        if (effective_closure_samples > 0.0f) {
            closure_ok = closure_samples <= effective_closure_samples;
        }
    }

    result.azimuth_deg = azimuth_deg;
    result.confidence = confidence;
    result.peak_score = peak_score;
    result.score_margin = norm;
    result.quality = quality;
    result.closure_residual_sec = std::fabs(closure_sec);
    result.valid_pairs = valid_pairs_count;
    // valid gates:
    //   - confidence_threshold (P1.1 normalized, scale-invariant default 0.1)
    //   - margin_threshold (default 0.6, geometry consistency)
    //   - quality_threshold (default 0 = disabled; opt-in 0.5 if stricter)
    //   - closure gate (N=3 only) via effective_closure_samples derived
    //     from max(closure_threshold_samples, closure_threshold_fraction *
    //     max_physical_TDOA). Pre-P1.1 `closure_threshold_samples=2.0` was
    //     scale-sensitive; new default `samples=0.0, fraction=0.3` is
    //     array-scale invariant.
    result.valid = confidence >= config.confidence_threshold
                && norm >= config.margin_threshold
                && quality >= config.quality_threshold
                && closure_ok;

    if (result.valid && confidence > 0.0f) {
        const float robot_rad = azimuth_deg * static_cast<float>(M_PI) / 180.0f;
        BatchEntry entry;
        entry.weighted_x = confidence * std::cos(robot_rad);
        entry.weighted_y = confidence * std::sin(robot_rad);
        entry.weight = confidence;
        average_x += entry.weighted_x;
        average_y += entry.weighted_y;
        sum_weights += static_cast<double>(entry.weight);
        avg_history.push_back(entry);
        ++result_count;
        // Sliding-window eviction: drop oldest batches until the deque fits
        // the configured max_avg_batches. Running sum decrements in lockstep
        // so GetAverageAzimuth + GetAverageResultantLength stay O(1).
        if (max_avg_batches > 0) {
            while (static_cast<int>(avg_history.size()) > max_avg_batches) {
                const auto& front = avg_history.front();
                average_x -= front.weighted_x;
                average_y -= front.weighted_y;
                sum_weights -= static_cast<double>(front.weight);
                --result_count;
                avg_history.pop_front();
            }
        }
    }

    return true;
}

int MultiSoundLocator::Impl::FindPairIndex(int i, int j) const {
    if (i < 0 || j < 0 || i >= n_channels || j >= n_channels || i == j) {
        throw std::out_of_range("microphone pair index out of range");
    }

    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        if (pairs[idx].i == lo && pairs[idx].j == hi) {
            return static_cast<int>(idx);
        }
    }

    throw std::out_of_range("microphone pair index out of range");
}

MultiSoundLocator::MultiSoundLocator(const MultiSoundLocatorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MultiSoundLocator::~MultiSoundLocator() {
    if (impl_) {
        impl_->DestroyFFT();
    }
}

MultiSoundLocator::MultiSoundLocator(MultiSoundLocator&& other) noexcept = default;
MultiSoundLocator& MultiSoundLocator::operator=(MultiSoundLocator&& other) noexcept = default;

bool MultiSoundLocator::Initialize() {
    auto& d = *impl_;
    d.DestroyFFT();

    d.n_channels = static_cast<int>(d.config.microphones.size());
    if (d.n_channels < 2 || d.config.sample_rate <= 0
        || d.config.frame_size <= 0 || d.config.avg_frames <= 0
        || d.config.sound_speed <= 0.0f) {
        return false;
    }

    d.padded_size = d.config.fft_size > 0
                        ? d.config.fft_size
                        : 2 * d.config.frame_size;
    if (d.padded_size < d.config.frame_size) {
        d.padded_size = d.config.frame_size;
    }
    if (d.padded_size % 2 != 0) {
        ++d.padded_size;
    }
    d.spectrum_size = d.padded_size / 2 + 1;

    if (d.config.upsample_factor > 0) {
        d.upsample_factor = d.config.upsample_factor;
    } else {
        d.upsample_factor = std::max(1,
            static_cast<int>(std::ceil(
                1.0 / (4e-6 * static_cast<double>(d.config.sample_rate)))));
    }
    d.upsampled_size = d.padded_size * d.upsample_factor;
    d.up_spectrum_size = d.upsampled_size / 2 + 1;

    if (!d.BuildGeometry()) {
        d.DestroyFFT();
        return false;
    }

    d.fft_in.assign(d.n_channels, nullptr);
    d.fft_out.assign(d.n_channels, nullptr);
    d.fft_plans.assign(d.n_channels, nullptr);

    const unsigned fftw_flags = d.config.use_fftw_measure
                                    ? FFTW_MEASURE : FFTW_ESTIMATE;
    for (int ch = 0; ch < d.n_channels; ++ch) {
        d.fft_in[ch] = fftwf_alloc_real(d.padded_size);
        d.fft_out[ch] = fftwf_alloc_complex(d.spectrum_size);
        if (!d.fft_in[ch] || !d.fft_out[ch]) {
            d.DestroyFFT();
            return false;
        }
        d.fft_plans[ch] = fftwf_plan_dft_r2c_1d(
            d.padded_size, d.fft_in[ch], d.fft_out[ch], fftw_flags);
        if (!d.fft_plans[ch]) {
            d.DestroyFFT();
            return false;
        }
    }

    d.hann_window.resize(d.config.frame_size);
    if (d.config.frame_size == 1) {
        d.hann_window[0] = 1.0f;
    } else {
        for (int i = 0; i < d.config.frame_size; ++i) {
            d.hann_window[i] = 0.5f * (1.0f - std::cos(
                2.0f * static_cast<float>(M_PI) * i
                / static_cast<float>(d.config.frame_size - 1)));
        }
    }

    d.gcc_pairs.clear();
    d.gcc_pairs.reserve(d.pairs.size());
    for (const auto& pair : d.pairs) {
        internal::GccPhatPairConfig pair_config;
        pair_config.sample_rate = d.config.sample_rate;
        pair_config.avg_frames = d.config.avg_frames;
        pair_config.padded_size = d.padded_size;
        pair_config.spectrum_size = d.spectrum_size;
        pair_config.upsample_factor = d.upsample_factor;
        pair_config.upsampled_size = d.upsampled_size;
        pair_config.up_spectrum_size = d.up_spectrum_size;
        pair_config.sound_speed = d.config.sound_speed;
        pair_config.pair_distance_m = pair.distance_m;
        pair_config.max_frequency_hz = d.config.max_frequency_hz;
        pair_config.use_fftw_measure = d.config.use_fftw_measure;

        internal::GccPhatPair gcc_pair;
        if (!gcc_pair.Initialize(pair_config)) {
            d.DestroyFFT();
            return false;
        }
        d.gcc_pairs.push_back(std::move(gcc_pair));
    }

    d.latest_tdoa.assign(d.pairs.size(), 0.0f);
    d.latest_pair_confidences.assign(d.pairs.size(), 0.0f);
    d.planar_buffers.assign(d.n_channels, {});
    d.channel_view.assign(d.n_channels, nullptr);
    d.result = MultiSoundLocatorResult{};
    d.avg_history.clear();
    d.average_x = 0.0f;
    d.average_y = 0.0f;
    d.sum_weights = 0.0;
    d.result_count = 0;

    // Compute sliding-window cap for GetAverageAzimuth from max_avg_seconds.
    // 0 (default) = unbounded since Reset (v1 backward-compat).
    // >0 = ceil(seconds / batch_duration_seconds), min 1.
    d.max_avg_batches = 0;
    if (d.config.max_avg_seconds > 0.0f && d.config.sample_rate > 0
        && d.config.frame_size > 0 && d.config.avg_frames > 0) {
        const double batch_seconds =
            static_cast<double>(d.config.avg_frames) *
            static_cast<double>(d.config.frame_size) /
            static_cast<double>(d.config.sample_rate);
        if (batch_seconds > 0.0) {
            const double batches =
                static_cast<double>(d.config.max_avg_seconds) / batch_seconds;
            d.max_avg_batches = std::max(1, static_cast<int>(std::ceil(batches)));
        }
    }

    d.initialized = true;
    return true;
}

void MultiSoundLocator::Reset() {
    if (!impl_ || !impl_->initialized) {
        return;
    }

    for (auto& pair : impl_->gcc_pairs) {
        pair.Reset();
    }
    std::fill(impl_->latest_tdoa.begin(), impl_->latest_tdoa.end(), 0.0f);
    std::fill(impl_->latest_pair_confidences.begin(),
        impl_->latest_pair_confidences.end(), 0.0f);
    impl_->result = MultiSoundLocatorResult{};
    impl_->avg_history.clear();
    impl_->average_x = 0.0f;
    impl_->average_y = 0.0f;
    impl_->sum_weights = 0.0;
    impl_->result_count = 0;
}

bool MultiSoundLocator::Process(
    const float* interleaved, size_t n_frames, int n_channels) {
    if (!impl_ || !impl_->initialized || !interleaved || n_frames == 0) {
        return false;
    }
    impl_->ValidateChannelCount(n_channels);

    // Steady-state zero-realloc per-channel buffer reuse (matches 2-ch
    // SoundLocator::Process buf_a/buf_b.resize() pattern; Initialize already
    // assigned n_channels empty vectors, so subsequent resize is no-op once
    // n_frames stabilizes).
    if (static_cast<int>(impl_->planar_buffers.size()) != impl_->n_channels) {
        impl_->planar_buffers.resize(impl_->n_channels);
    }
    for (int ch = 0; ch < impl_->n_channels; ++ch) {
        impl_->planar_buffers[ch].resize(n_frames);
    }
    for (size_t frame = 0; frame < n_frames; ++frame) {
        for (int ch = 0; ch < impl_->n_channels; ++ch) {
            impl_->planar_buffers[ch][frame] =
                interleaved[frame * impl_->n_channels + ch];
        }
    }

    if (static_cast<int>(impl_->channel_view.size()) != impl_->n_channels) {
        impl_->channel_view.resize(impl_->n_channels);
    }
    for (int ch = 0; ch < impl_->n_channels; ++ch) {
        impl_->channel_view[ch] = impl_->planar_buffers[ch].data();
    }
    return impl_->ProcessPlanar(impl_->channel_view.data(), n_frames);
}

bool MultiSoundLocator::Process(
    const float* const* channel_ptrs, size_t n_frames, int n_channels) {
    if (!impl_ || !impl_->initialized || n_frames == 0) {
        return false;
    }
    impl_->ValidateChannelCount(n_channels);
    return impl_->ProcessPlanar(channel_ptrs, n_frames);
}

bool MultiSoundLocator::Process(
    const int16_t* interleaved, size_t n_frames, int n_channels) {
    if (!impl_ || !impl_->initialized || !interleaved || n_frames == 0) {
        return false;
    }
    impl_->ValidateChannelCount(n_channels);

    constexpr float kScale = 1.0f / 32768.0f;
    const size_t total = n_frames * static_cast<size_t>(impl_->n_channels);
    impl_->int16_float_buffer.resize(total);
    for (size_t i = 0; i < total; ++i) {
        impl_->int16_float_buffer[i] = static_cast<float>(interleaved[i]) * kScale;
    }
    return Process(impl_->int16_float_buffer.data(), n_frames, n_channels);
}

MultiSoundLocatorResult MultiSoundLocator::GetResult() const {
    return impl_->result;
}

float MultiSoundLocator::GetAzimuth() const {
    return impl_->result.azimuth_deg;
}

float MultiSoundLocator::GetConfidence() const {
    return impl_->result.confidence;
}

bool MultiSoundLocator::IsValid() const {
    return impl_->result.valid;
}

float MultiSoundLocator::GetAverageAzimuth() const {
    if (impl_->result_count <= 0) {
        return impl_->result.azimuth_deg;
    }
    if (std::fabs(impl_->average_x) <= kEpsilon
        && std::fabs(impl_->average_y) <= kEpsilon) {
        return impl_->result.azimuth_deg;
    }
    return NormalizeAngle360(
        std::atan2(impl_->average_y, impl_->average_x)
        * 180.0f / static_cast<float>(M_PI));
}

int MultiSoundLocator::GetResultCount() const {
    return impl_->result_count;
}

float MultiSoundLocator::GetTDOA(int i, int j) const {
    const int idx = impl_->FindPairIndex(i, j);
    const float value = impl_->latest_tdoa[idx];
    return i < j ? value : -value;
}

int MultiSoundLocator::GetMaxDelaySamplesPair(int i, int j) const {
    const int idx = impl_->FindPairIndex(i, j);
    return impl_->gcc_pairs[idx].GetMaxDelaySamples();
}

const MultiSoundLocatorConfig& MultiSoundLocator::GetConfig() const {
    return impl_->config;
}

// ---------------------------------------------------------------------------
// [A1 / A3 / A5 / A6] new accessors
// ---------------------------------------------------------------------------

float MultiSoundLocator::GetClosureResidual() const {
    return impl_ ? impl_->result.closure_residual_sec : 0.0f;
}

float MultiSoundLocator::GetClosureSamples() const {
    if (!impl_) return 0.0f;
    return impl_->result.closure_residual_sec
         * static_cast<float>(impl_->config.sample_rate);
}

void MultiSoundLocator::GetPairConfidences(float* out, int n) const {
    if (!impl_ || !out) {
        throw std::invalid_argument(
            "GetPairConfidences requires a non-null output buffer");
    }
    const int expected = static_cast<int>(impl_->latest_pair_confidences.size());
    if (n != expected) {
        throw std::out_of_range(
            "GetPairConfidences buffer length " + std::to_string(n)
            + " does not match pair count " + std::to_string(expected));
    }
    for (int i = 0; i < n; ++i) {
        out[i] = impl_->latest_pair_confidences[i];
    }
}

int MultiSoundLocator::GetPairCount() const {
    return impl_ ? static_cast<int>(impl_->pairs.size()) : 0;
}

float MultiSoundLocator::GetQuality() const {
    return impl_ ? impl_->result.quality : 0.0f;
}

float MultiSoundLocator::GetAverageResultantLength() const {
    if (!impl_ || impl_->sum_weights <= 0.0) {
        return 0.0f;
    }
    const double r = std::sqrt(
        static_cast<double>(impl_->average_x) * impl_->average_x
        + static_cast<double>(impl_->average_y) * impl_->average_y);
    const double R = r / impl_->sum_weights;
    if (R < 0.0) return 0.0f;
    if (R > 1.0) return 1.0f;
    return static_cast<float>(R);
}

MultiSoundLocatorConfig MultiSoundLocator::CreateEquilateralTriangleConfig(
    float side_length_m) {
    MultiSoundLocatorConfig config;
    const float sqrt3 = std::sqrt(3.0f);
    config.microphones = {
        {side_length_m / sqrt3, 0.0f, 0.0f},
        {-side_length_m / (2.0f * sqrt3), side_length_m / 2.0f, 0.0f},
        {-side_length_m / (2.0f * sqrt3), -side_length_m / 2.0f, 0.0f},
    };
    config.azimuth_offset_deg = 0.0f;
    return config;
}

}  // namespace SpacemitAudio
