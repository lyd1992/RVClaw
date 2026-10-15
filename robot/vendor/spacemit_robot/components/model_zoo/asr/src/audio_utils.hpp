/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUDIO_UTILS_HPP
#define AUDIO_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace asr::audio_utils {

inline std::vector<float> resampleLinear(
    std::vector<float> audio, int source_sample_rate, int target_sample_rate) {
    if (source_sample_rate == target_sample_rate || audio.empty()) {
        return audio;
    }
    if (source_sample_rate <= 0 || target_sample_rate <= 0) {
        return {};
    }
    if (audio.size() == 1) {
        return audio;
    }

    const double ratio =
        static_cast<double>(target_sample_rate) / static_cast<double>(source_sample_rate);
    const size_t output_size =
        std::max<size_t>(1, static_cast<size_t>(std::llround(audio.size() * ratio)));
    const double source_step =
        static_cast<double>(source_sample_rate) / static_cast<double>(target_sample_rate);

    std::vector<float> resampled(output_size);
    const size_t last = audio.size() - 1;
    for (size_t i = 0; i < output_size; ++i) {
        const double source_pos = static_cast<double>(i) * source_step;
        size_t idx = static_cast<size_t>(source_pos);
        if (idx >= last) {
            resampled[i] = audio[last];
            continue;
        }
        const float frac = static_cast<float>(source_pos - static_cast<double>(idx));
        resampled[i] = audio[idx] * (1.0f - frac) + audio[idx + 1] * frac;
    }

    return resampled;
}

inline std::vector<float> normalizeSampleRate(
    std::vector<float> audio, int source_sample_rate, int target_sample_rate) {
    return resampleLinear(std::move(audio), source_sample_rate, target_sample_rate);
}

}  // namespace asr::audio_utils

#endif  // AUDIO_UTILS_HPP
