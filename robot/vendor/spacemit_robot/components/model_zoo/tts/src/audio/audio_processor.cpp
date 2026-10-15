/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "audio/audio_processor.hpp"

#include <cmath>
#include <cstdint>

#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tts {
namespace audio {

// =============================================================================
// RMS 计算
// =============================================================================

float calculateRMS(const std::vector<float>& audio) {
    if (audio.empty()) return 0.0f;

    float sum_squares = 0.0f;
    for (float sample : audio) {
        sum_squares += sample * sample;
    }
    return std::sqrt(sum_squares / audio.size());
}

// =============================================================================
// 动态压缩
// =============================================================================

std::vector<float> applyCompression(const std::vector<float>& audio,
                                    float threshold,
                                    float ratio) {
    std::vector<float> compressed = audio;

    for (float& sample : compressed) {
        float abs_sample = std::abs(sample);
        if (abs_sample > threshold) {
            // Apply compression above threshold
            float over_threshold = abs_sample - threshold;
            float compressed_over = over_threshold / ratio;
            float new_abs = threshold + compressed_over;
            sample = (sample < 0) ? -new_abs : new_abs;
        }
    }

    return compressed;
}

// =============================================================================
// 音频归一化
// =============================================================================

std::vector<float> normalizeRmsLevel(const std::vector<float>& audio,
    const AudioProcessConfig& config) {
    if (audio.empty() || !config.use_rms_norm) return audio;

    std::vector<float> processed(audio.begin(), audio.end());

    // RMS normalization for consistent perceived loudness.
    float current_rms = calculateRMS(processed);
    if (current_rms <= 0.0f) {
        return processed;
    }

    float scale = config.target_rms / current_rms;

    // Apply soft limiting to prevent noise amplification.
    const float max_gain = (config.max_gain > 0.0f) ? config.max_gain : 3.0f;
    scale = std::min(scale, max_gain);

    for (float& sample : processed) {
        sample *= scale;
    }

    // Soft clipping to prevent harsh distortion.
    for (float& sample : processed) {
        if (std::abs(sample) > 0.95f) {
            float sign = (sample < 0) ? -1.0f : 1.0f;
            float abs_val = std::abs(sample);
            // Soft knee compression near clipping.
            sample = sign * (0.95f + 0.05f * std::tanh((abs_val - 0.95f) * 20.0f));
        }
    }

    return processed;
}

std::vector<float> normalizeAudio(const std::vector<float>& audio,
    const AudioProcessConfig& config) {
    if (audio.empty()) return audio;

    // Create copy
    std::vector<float> processed;
    processed.reserve(audio.size());
    processed.assign(audio.begin(), audio.end());

    // Step 1: Apply dynamic range compression
    processed = applyCompression(processed, config.compression_threshold, config.compression_ratio);

    // Step 2: Apply normalization
    if (config.use_rms_norm) {
        processed = normalizeRmsLevel(processed, config);
    } else {
        // Fallback to peak normalization
        float max_amplitude = 0.0f;
        for (float sample : processed) {
            max_amplitude = std::max(max_amplitude, std::abs(sample));
        }

        if (max_amplitude > 0.0f) {
            float scale = 0.8f / max_amplitude;
            for (float& sample : processed) {
                sample *= scale;
            }
        }
    }

    return processed;
}

// =============================================================================
// 去爆音
// =============================================================================

std::vector<float> removeClicksAndPops(const std::vector<float>& audio) {
    if (audio.empty()) return audio;

    // Create copy
    std::vector<float> processed;
    processed.reserve(audio.size());
    processed.assign(audio.begin(), audio.end());

    // Step 1: Remove DC offset (average value should be zero)
    float dc_offset = 0.0f;
    for (float sample : processed) {
        dc_offset += sample;
    }
    dc_offset /= processed.size();

    // Only remove DC offset if it's significant (> 0.01)
    if (std::abs(dc_offset) > 0.01f) {
        for (float& sample : processed) {
            sample -= dc_offset;
        }
    }

    // Step 2: Apply very short fade-in at the beginning (2ms at 22050Hz = ~44 samples)
    const int fade_in_samples = std::min(44, static_cast<int>(processed.size() / 100));
    for (int i = 0; i < fade_in_samples && i < static_cast<int>(processed.size()); ++i) {
        float fade_factor = static_cast<float>(i) / fade_in_samples;
        // Use cosine fade for smoother transition
        fade_factor = 0.5f * (1.0f - std::cos(M_PI * fade_factor));
        processed[i] *= fade_factor;
    }

    // Step 3: Apply short fade-out at the end (5ms = ~110 samples)
    const int fade_out_samples = std::min(110, static_cast<int>(processed.size() / 50));
    for (int i = 0; i < fade_out_samples && i < static_cast<int>(processed.size()); ++i) {
        int idx = processed.size() - 1 - i;
        float fade_factor = static_cast<float>(i) / fade_out_samples;
        // Use cosine fade for smoother transition
        fade_factor = 0.5f * (1.0f - std::cos(M_PI * fade_factor));
        processed[idx] *= fade_factor;
    }

    // Step 4: Simple DC blocking filter (high-pass at 20Hz)
    if (processed.size() > 1) {
        const float cutoff = 0.999f;  // Very gentle high-pass
        float prev_input = 0.0f;
        float prev_output = 0.0f;

        for (size_t i = 0; i < processed.size(); ++i) {
            float current_input = processed[i];
            float current_output = cutoff * (prev_output + current_input - prev_input);
            processed[i] = current_output;
            prev_input = current_input;
            prev_output = current_output;
        }
    }

    // Step 5: Ensure the very last sample is zero
    if (!processed.empty()) {
        processed.back() = 0.0f;
    }

    return processed;
}

// =============================================================================
// 重采样
// =============================================================================

std::vector<float> resampleAudio(const std::vector<float>& audio,
    int src_rate,
    int dst_rate) {
    if (audio.empty() || src_rate == dst_rate || dst_rate <= 0) {
        return audio;
    }

    double ratio = static_cast<double>(dst_rate) / src_rate;
    size_t output_size = static_cast<size_t>(audio.size() * ratio);
    std::vector<float> resampled(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        double src_pos = i / ratio;
        size_t src_idx = static_cast<size_t>(src_pos);
        double frac = src_pos - src_idx;

        if (src_idx + 1 < audio.size()) {
            // Linear interpolation between two adjacent samples
            resampled[i] = static_cast<float>(
                audio[src_idx] * (1.0 - frac) + audio[src_idx + 1] * frac);
        } else if (src_idx < audio.size()) {
            resampled[i] = audio[src_idx];
        } else {
            resampled[i] = 0.0f;
        }
    }

    return resampled;
}

// =============================================================================
// 完整处理流程
// =============================================================================

std::vector<float> processAudio(const std::vector<float>& audio,
                                const AudioProcessConfig& config) {
    if (audio.empty()) return audio;

    // Step 1: Normalize
    std::vector<float> processed = normalizeAudio(audio, config);

    // Step 2: Remove clicks (optional)
    if (config.remove_clicks) {
        processed = removeClicksAndPops(processed);
    }

    return processed;
}

// =============================================================================
// 格式转换
// =============================================================================

std::vector<int16_t> floatToInt16(const std::vector<float>& audio) {
    std::vector<int16_t> result(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        float sample = audio[i];
        // Clamp to [-1.0, 1.0]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        result[i] = static_cast<int16_t>(sample * 32767.0f);
    }
    return result;
}

std::vector<float> int16ToFloat(const std::vector<int16_t>& audio) {
    std::vector<float> result(audio.size());
    for (size_t i = 0; i < audio.size(); ++i) {
        result[i] = static_cast<float>(audio[i]) / 32768.0f;
    }
    return result;
}

std::vector<uint8_t> floatToBytes(const std::vector<float>& audio) {
    auto int16_data = floatToInt16(audio);
    std::vector<uint8_t> result(int16_data.size() * 2);
    for (size_t i = 0; i < int16_data.size(); ++i) {
        result[i * 2] = static_cast<uint8_t>(int16_data[i] & 0xFF);
        result[i * 2 + 1] = static_cast<uint8_t>((int16_data[i] >> 8) & 0xFF);
    }
    return result;
}

}  // namespace audio
}  // namespace tts
