/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "audio_resampler.hpp"

#ifdef USE_LIBSAMPLERATE
#include <samplerate.h>
#endif

Resampler::Resampler(const Config& config)
    : config_(config)
    , ratio_(static_cast<double>(config.output_sample_rate) / config.input_sample_rate)
    , initialized_(false)
    , src_state_(nullptr)
{
}

Resampler::~Resampler() {
#ifdef USE_LIBSAMPLERATE
    if (src_state_) {
        src_delete(src_state_);
    }
#endif
    src_state_ = nullptr;
}

Resampler::Resampler(Resampler&& other) noexcept
    : config_(other.config_)
    , ratio_(other.ratio_)
    , initialized_(other.initialized_)
    , src_state_(other.src_state_)
{
    other.src_state_ = nullptr;
    other.initialized_ = false;
}

Resampler& Resampler::operator=(Resampler&& other) noexcept {
    if (this != &other) {
#ifdef USE_LIBSAMPLERATE
        if (src_state_) {
            src_delete(src_state_);
        }
#endif
        src_state_ = other.src_state_;
        other.src_state_ = nullptr;
        config_ = other.config_;
        ratio_ = other.ratio_;
        initialized_ = other.initialized_;
        other.initialized_ = false;
    }
    return *this;
}

bool Resampler::methodRequiresLibsamplerate(ResampleMethod method) {
    switch (method) {
        case ResampleMethod::LINEAR_UPSAMPLE:
        case ResampleMethod::LINEAR_DOWNSAMPLE:
            return false;
        default:
            return true;
    }
}

bool Resampler::initialize() {
    if (initialized_) {
        return true;
    }

    // Validate configuration
    if (config_.input_sample_rate <= 0 || config_.output_sample_rate <= 0) {
        std::cerr << "Resampler: Invalid sample rates" << std::endl;
        return false;
    }

    if (config_.channels <= 0) {
        std::cerr << "Resampler: Invalid channel count" << std::endl;
        return false;
    }

    // Check if method requires libsamplerate
    if (methodRequiresLibsamplerate(config_.method)) {
#ifdef USE_LIBSAMPLERATE
        int error = 0;
        src_state_ = src_new(methodToSrcType(config_.method), config_.channels, &error);
        if (src_state_ == nullptr || error != 0) {
            std::cerr << "Resampler: Failed to create libsamplerate state: "
                        << src_strerror(error) << std::endl;
            return false;
        }
#else
        // libsamplerate not available, fallback to linear interpolation
        std::cerr << "Resampler: libsamplerate not available, falling back to linear interpolation" << std::endl;
        config_.method = (ratio_ > 1.0) ? ResampleMethod::LINEAR_UPSAMPLE : ResampleMethod::LINEAR_DOWNSAMPLE;
#endif
    }

    initialized_ = true;
    return true;
}

void Resampler::reset() {
#ifdef USE_LIBSAMPLERATE
    if (src_state_) {
        src_reset(src_state_);
    }
#endif
}

std::vector<float> Resampler::process(const std::vector<float>& input) {
    return process(input.data(), input.size());
}

std::vector<float> Resampler::process(const float* input, size_t num_samples) {
    if (!initialized_) {
        if (!initialize()) {
            return {};
        }
    }

    if (num_samples == 0) {
        return {};
    }

    // Same rate - just copy
    if (config_.input_sample_rate == config_.output_sample_rate) {
        return std::vector<float>(input, input + num_samples);
    }

    // Select method
    switch (config_.method) {
        case ResampleMethod::LINEAR_UPSAMPLE:
            return linearUpsample(input, num_samples);

        case ResampleMethod::LINEAR_DOWNSAMPLE:
            return linearDownsample(input, num_samples);

#ifdef USE_LIBSAMPLERATE
        case ResampleMethod::SRC_SINC_BEST_QUALITY:
        case ResampleMethod::SRC_SINC_MEDIUM_QUALITY:
        case ResampleMethod::SRC_SINC_FASTEST:
        case ResampleMethod::SRC_ZERO_ORDER_HOLD:
        case ResampleMethod::SRC_LINEAR:
            return srcResample(input, num_samples, true);
#endif

        default:
            // Fallback to linear interpolation
            if (ratio_ > 1.0) {
                return linearUpsample(input, num_samples);
            } else {
                return linearDownsample(input, num_samples);
            }
    }
}

std::vector<float> Resampler::processStreaming(const std::vector<float>& input, bool end_of_input) {
    if (!initialized_) {
        if (!initialize()) {
            return {};
        }
    }

    if (input.empty() && !end_of_input) {
        return {};
    }

    // For linear methods, streaming is same as single-shot
    if (!methodRequiresLibsamplerate(config_.method)) {
        return process(input);
    }

#ifdef USE_LIBSAMPLERATE
    return srcResample(input.data(), input.size(), end_of_input);
#else
    return process(input);
#endif
}

std::vector<float> Resampler::linearUpsample(const float* input, size_t num_samples) {
#ifdef USE_RVV_OPTIMIZATION
    // Use RVV-optimized version if available at runtime
    static bool rvv_available = isRVVAvailable();
    static bool first_call = true;
    if (first_call) {
        if (rvv_available) {
            std::cout << "[Resampler] Using RVV-optimized linear upsample (m4 vectors)" << std::endl;
        } else {
            std::cout << "[Resampler] RVV not available, using scalar linear upsample" << std::endl;
        }
        first_call = false;
    }
    if (rvv_available) {
        return linearUpsample_RVV(input, num_samples);
    }
#endif

    // Fallback to scalar implementation
    size_t num_frames = num_samples / config_.channels;
    size_t output_frames = static_cast<size_t>(std::ceil(num_frames * ratio_));
    std::vector<float> output(output_frames * config_.channels);

    for (int ch = 0; ch < config_.channels; ++ch) {
        for (size_t i = 0; i < output_frames; ++i) {
            // Calculate source position
            double src_pos = i / ratio_;
            size_t src_idx = static_cast<size_t>(src_pos);
            double frac = src_pos - src_idx;

            // Clamp index
            if (src_idx >= num_frames - 1) {
                src_idx = num_frames - 2;
                frac = 1.0;
            }

            // Linear interpolation
            float sample0 = input[src_idx * config_.channels + ch];
            float sample1 = input[(src_idx + 1) * config_.channels + ch];
            output[i * config_.channels + ch] = static_cast<float>(
                sample0 * (1.0 - frac) + sample1 * frac);
        }
    }

    return output;
}

std::vector<float> Resampler::linearDownsample(const float* input, size_t num_samples) {
#ifdef USE_RVV_OPTIMIZATION
    // Use RVV-optimized version if available at runtime
    static bool rvv_available = isRVVAvailable();
    static bool first_call = true;
    if (first_call) {
        if (rvv_available) {
            std::cout << "[Resampler] Using RVV-optimized linear downsample (m4 vectors)" << std::endl;
        } else {
            std::cout << "[Resampler] RVV not available, using scalar linear downsample" << std::endl;
        }
        first_call = false;
    }
    if (rvv_available) {
        return linearDownsample_RVV(input, num_samples);
    }
#endif

    // Fallback to scalar implementation
    size_t num_frames = num_samples / config_.channels;
    size_t output_frames = static_cast<size_t>(std::ceil(num_frames * ratio_));
    std::vector<float> output(output_frames * config_.channels);

    // For downsampling, we use simple linear interpolation
    // For better quality, consider using an anti-aliasing filter
    for (int ch = 0; ch < config_.channels; ++ch) {
        for (size_t i = 0; i < output_frames; ++i) {
            double src_pos = i / ratio_;
            size_t src_idx = static_cast<size_t>(src_pos);
            double frac = src_pos - src_idx;

            // Clamp index
            if (src_idx >= num_frames - 1) {
                output[i * config_.channels + ch] = input[(num_frames - 1) * config_.channels + ch];
                continue;
            }

            // Linear interpolation
            float sample0 = input[src_idx * config_.channels + ch];
            float sample1 = input[(src_idx + 1) * config_.channels + ch];
            output[i * config_.channels + ch] = static_cast<float>(
                sample0 * (1.0 - frac) + sample1 * frac);
        }
    }

    return output;
}

#ifdef USE_LIBSAMPLERATE
std::vector<float> Resampler::srcResample(const float* input, size_t num_samples, bool end_of_input) {
    if (!src_state_) {
        std::cerr << "Resampler: libsamplerate not initialized" << std::endl;
        return {};
    }

    size_t num_frames = num_samples / config_.channels;
    size_t output_frames = static_cast<size_t>(std::ceil(num_frames * ratio_)) + 256;
    std::vector<float> output(output_frames * config_.channels);

    SRC_DATA src_data;
    src_data.data_in = input;
    src_data.input_frames = static_cast<long>(num_frames);  // NOLINT(runtime/int)
    src_data.data_out = output.data();
    src_data.output_frames = static_cast<long>(output_frames);  // NOLINT(runtime/int)
    src_data.src_ratio = ratio_;
    src_data.end_of_input = end_of_input ? 1 : 0;

    int error = src_process(src_state_, &src_data);
    if (error != 0) {
        std::cerr << "Resampler: libsamplerate error: " << src_strerror(error) << std::endl;
        return {};
    }

    // Resize to actual output size
    output.resize(src_data.output_frames_gen * config_.channels);
    return output;
}

int Resampler::methodToSrcType(ResampleMethod method) const {
    switch (method) {
        case ResampleMethod::SRC_SINC_BEST_QUALITY:
            return SRC_SINC_BEST_QUALITY;
        case ResampleMethod::SRC_SINC_MEDIUM_QUALITY:
            return SRC_SINC_MEDIUM_QUALITY;
        case ResampleMethod::SRC_SINC_FASTEST:
            return SRC_SINC_FASTEST;
        case ResampleMethod::SRC_ZERO_ORDER_HOLD:
            return SRC_ZERO_ORDER_HOLD;
        case ResampleMethod::SRC_LINEAR:
            return SRC_LINEAR;
        default:
            return SRC_SINC_MEDIUM_QUALITY;
    }
}
#endif  // USE_LIBSAMPLERATE

size_t Resampler::estimateOutputSize(size_t input_size, int input_rate, int output_rate) {
    double ratio = static_cast<double>(output_rate) / input_rate;
    return static_cast<size_t>(std::ceil(input_size * ratio)) + 256;
}

// ============== C API Implementation ==============

extern "C" {

struct ResamplerHandle {
    Resampler* resampler;
};

int resampler_has_libsamplerate(void) {
#ifdef USE_LIBSAMPLERATE
    return 1;
#else
    return 0;
#endif
}

static ResampleMethod methodCToCpp(ResampleMethodC method, int output_rate, int input_rate) {
    double ratio = static_cast<double>(output_rate) / input_rate;

    switch (method) {
        case RESAMPLE_METHOD_LINEAR:
            return (ratio > 1.0) ? ResampleMethod::LINEAR_UPSAMPLE : ResampleMethod::LINEAR_DOWNSAMPLE;

#ifdef USE_LIBSAMPLERATE
        case RESAMPLE_METHOD_SRC_BEST_QUALITY:
            return ResampleMethod::SRC_SINC_BEST_QUALITY;
        case RESAMPLE_METHOD_SRC_MEDIUM_QUALITY:
            return ResampleMethod::SRC_SINC_MEDIUM_QUALITY;
        case RESAMPLE_METHOD_SRC_FASTEST:
            return ResampleMethod::SRC_SINC_FASTEST;
        case RESAMPLE_METHOD_SRC_ZERO_ORDER_HOLD:
            return ResampleMethod::SRC_ZERO_ORDER_HOLD;
        case RESAMPLE_METHOD_SRC_LINEAR:
            return ResampleMethod::SRC_LINEAR;
#else
        // Fallback to linear when libsamplerate not available
        case RESAMPLE_METHOD_SRC_BEST_QUALITY:
        case RESAMPLE_METHOD_SRC_MEDIUM_QUALITY:
        case RESAMPLE_METHOD_SRC_FASTEST:
        case RESAMPLE_METHOD_SRC_ZERO_ORDER_HOLD:
        case RESAMPLE_METHOD_SRC_LINEAR:
            return (ratio > 1.0) ? ResampleMethod::LINEAR_UPSAMPLE : ResampleMethod::LINEAR_DOWNSAMPLE;
#endif

        default:
            return (ratio > 1.0) ? ResampleMethod::LINEAR_UPSAMPLE : ResampleMethod::LINEAR_DOWNSAMPLE;
    }
}

ResamplerHandle* resampler_create(int input_rate, int output_rate, int channels, ResampleMethodC method) {
    ResamplerHandle* handle = new (std::nothrow) ResamplerHandle();
    if (!handle) {
        return nullptr;
    }

    Resampler::Config config;
    config.input_sample_rate = input_rate;
    config.output_sample_rate = output_rate;
    config.channels = channels;
    config.method = methodCToCpp(method, output_rate, input_rate);

    handle->resampler = new (std::nothrow) Resampler(config);
    if (!handle->resampler) {
        delete handle;
        return nullptr;
    }

    if (!handle->resampler->initialize()) {
        delete handle->resampler;
        delete handle;
        return nullptr;
    }

    return handle;
}

void resampler_destroy(ResamplerHandle* handle) {
    if (handle) {
        delete handle->resampler;
        delete handle;
    }
}

void resampler_reset(ResamplerHandle* handle) {
    if (handle && handle->resampler) {
        handle->resampler->reset();
    }
}

int resampler_process(ResamplerHandle* handle,
                        const float* input, int input_samples,
                        float* output, int output_capacity) {
    if (!handle || !handle->resampler || !input || !output || input_samples <= 0) {
        return -1;
    }

    std::vector<float> result = handle->resampler->process(input, static_cast<size_t>(input_samples));

    if (static_cast<int>(result.size()) > output_capacity) {
        return -1;  // Output buffer too small
    }

    std::copy(result.begin(), result.end(), output);
    return static_cast<int>(result.size());
}

int resampler_estimate_output_size(int input_samples, int input_rate, int output_rate) {
    return static_cast<int>(Resampler::estimateOutputSize(
        static_cast<size_t>(input_samples), input_rate, output_rate));
}

int resample_simple(const float* input, int input_samples,
                    int input_rate, int output_rate,
                    int channels, ResampleMethodC method,
                    float* output, int output_capacity) {
    ResamplerHandle* handle = resampler_create(input_rate, output_rate, channels, method);
    if (!handle) {
        return -1;
    }

    int result = resampler_process(handle, input, input_samples, output, output_capacity);
    resampler_destroy(handle);
    return result;
}

}  // extern "C"
