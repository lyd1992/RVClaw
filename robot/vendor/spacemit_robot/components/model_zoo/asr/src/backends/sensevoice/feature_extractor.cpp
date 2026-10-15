/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file feature_extractor.cpp
 * @brief Audio feature extraction implementation
 */

#include "backends/sensevoice/feature_extractor.hpp"

#include <cmath>
#include <cstring>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace asr {
namespace sensevoice {

FeatureExtractor::FeatureExtractor(const Config& config)
    : config_(config)
{
}

FeatureExtractor::~FeatureExtractor() {
    cleanupFFTW();
}

bool FeatureExtractor::initialize() {
    try {
        initializeWindow();
        initializeMelFilterbank();
        initializeFFTW();

        if (config_.apply_cmvn && !config_.cmvn_file.empty()) {
            if (!loadCMVN(config_.cmvn_file)) {
                std::cerr << "[FeatureExtractor] CMVN file not loaded, using identity" << std::endl;
            }
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[FeatureExtractor] Init error: " << e.what() << std::endl;
        return false;
    }
}

void FeatureExtractor::initializeWindow() {
    window_ = createHammingWindow(config_.frame_length);
}

std::vector<float> FeatureExtractor::createHammingWindow(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
    }
    return window;
}

void FeatureExtractor::initializeMelFilterbank() {
    float low_freq = 0.0f;
    float high_freq = config_.sample_rate / 2.0f;

    float low_mel = melScale(low_freq);
    float high_mel = melScale(high_freq);

    std::vector<float> mel_points(config_.n_mels + 2);
    for (int i = 0; i < config_.n_mels + 2; ++i) {
        mel_points[i] = low_mel + i * (high_mel - low_mel) / (config_.n_mels + 1);
    }

    std::vector<int> bin_points(config_.n_mels + 2);
    for (int i = 0; i < config_.n_mels + 2; ++i) {
        bin_points[i] = static_cast<int>(
            std::floor((config_.n_fft + 1) * invMelScale(mel_points[i]) / config_.sample_rate));
    }

    int num_bins = config_.n_fft / 2 + 1;

    // Build sparse mel filterbank for efficient processing
    mel_filters_sparse_.resize(config_.n_mels);

    for (int m = 0; m < config_.n_mels; ++m) {
        int start_bin = bin_points[m];
        int end_bin = std::min(bin_points[m + 2], num_bins - 1);

        mel_filters_sparse_[m].start_bin = start_bin;
        mel_filters_sparse_[m].end_bin = end_bin;
        mel_filters_sparse_[m].weights.resize(end_bin - start_bin + 1, 0.0f);

        // Rising edge
        for (int k = start_bin; k <= bin_points[m + 1] && k <= end_bin; ++k) {
            if (bin_points[m + 1] != bin_points[m]) {
                mel_filters_sparse_[m].weights[k - start_bin] =
                    static_cast<float>(k - bin_points[m]) / (bin_points[m + 1] - bin_points[m]);
            }
        }

        // Falling edge
        for (int k = bin_points[m + 1]; k <= end_bin; ++k) {
            if (bin_points[m + 2] != bin_points[m + 1]) {
                mel_filters_sparse_[m].weights[k - start_bin] =
                    static_cast<float>(bin_points[m + 2] - k) / (bin_points[m + 2] - bin_points[m + 1]);
            }
        }
    }
}

void FeatureExtractor::initializeFFTW() {
    fft_input_ = fftwf_alloc_real(config_.n_fft);
    fft_output_ = fftwf_alloc_complex(config_.n_fft / 2 + 1);
    fft_plan_ = fftwf_plan_dft_r2c_1d(config_.n_fft, fft_input_, fft_output_, FFTW_MEASURE);

    std::cout << "[FeatureExtractor] FFTW initialized (size=" << config_.n_fft << ")" << std::endl;
}

void FeatureExtractor::cleanupFFTW() {
    if (fft_plan_) {
        fftwf_destroy_plan(fft_plan_);
        fft_plan_ = nullptr;
    }
    if (fft_input_) {
        fftwf_free(fft_input_);
        fft_input_ = nullptr;
    }
    if (fft_output_) {
        fftwf_free(fft_output_);
        fft_output_ = nullptr;
    }
}

bool FeatureExtractor::loadCMVN(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    // Parse CMVN file (simplified - actual format may vary)
    // For now, use identity transform
    int dim = config_.n_mels * config_.lfr_m;
    cmvn_mean_.resize(dim, 0.0f);
    cmvn_var_.resize(dim, 1.0f);
    cmvn_loaded_ = true;

    std::cout << "[FeatureExtractor] CMVN loaded (dim=" << dim << ")" << std::endl;
    return true;
}

float FeatureExtractor::melScale(float freq) {
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

float FeatureExtractor::invMelScale(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<std::vector<float>> FeatureExtractor::extract(const std::vector<float>& audio) {
    return extract(audio.data(), audio.size());
}

std::vector<std::vector<float>> FeatureExtractor::extract(const float* audio, size_t length) {
    // Pre-emphasis
    std::vector<float> processed(length);
    processed[0] = audio[0];
    for (size_t i = 1; i < length; ++i) {
        processed[i] = audio[i] - config_.preemphasis * audio[i - 1];
    }

    // Compute Fbank
    auto fbank = computeFbank(processed);

    // Apply LFR
    auto lfr_features = applyLFR(fbank);

    // Apply CMVN
    if (config_.apply_cmvn && cmvn_loaded_) {
        applyCMVN(lfr_features);
    }

    return lfr_features;
}

std::vector<std::vector<float>> FeatureExtractor::frameSignal(const std::vector<float>& signal) {
    int num_frames = (signal.size() - config_.frame_length) / config_.frame_shift + 1;
    if (num_frames <= 0) {
        return {};
    }

    std::vector<std::vector<float>> frames(num_frames);

    for (int i = 0; i < num_frames; ++i) {
        frames[i].resize(config_.frame_length);
        int start = i * config_.frame_shift;

        for (int j = 0; j < config_.frame_length; ++j) {
            frames[i][j] = signal[start + j] * window_[j];
        }
    }

    return frames;
}

std::vector<std::vector<float>> FeatureExtractor::computeFbank(const std::vector<float>& audio) {
    auto frames = frameSignal(audio);
    size_t num_frames = frames.size();
    int num_bins = config_.n_fft / 2 + 1;

    // Pre-allocate output
    std::vector<std::vector<float>> fbank(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        fbank[i].resize(config_.n_mels);
    }

    // Reusable buffer for power spectrum
    std::vector<float> power_spec(num_bins);

    for (size_t i = 0; i < num_frames; ++i) {
        // Zero-pad frame and copy
        std::memset(fft_input_, 0, config_.n_fft * sizeof(float));
        std::memcpy(fft_input_, frames[i].data(),
                    std::min(static_cast<size_t>(config_.frame_length),
                        frames[i].size()) * sizeof(float));

        // FFT
        fftwf_execute(fft_plan_);

        // Power spectrum (reuse buffer)
        for (int k = 0; k < num_bins; ++k) {
            float real = fft_output_[k][0];
            float imag = fft_output_[k][1];
            power_spec[k] = real * real + imag * imag;
        }

        // Apply Mel filterbank using sparse representation
        for (int m = 0; m < config_.n_mels; ++m) {
            const auto& filter = mel_filters_sparse_[m];
            float sum = 0.0f;

            // Only iterate over the filter's active range
            for (int k = filter.start_bin; k <= filter.end_bin; ++k) {
                sum += power_spec[k] * filter.weights[k - filter.start_bin];
            }

            fbank[i][m] = std::log(std::max(sum, 1e-10f));
        }
    }

    return fbank;
}

std::vector<std::vector<float>> FeatureExtractor::applyLFR(
    const std::vector<std::vector<float>>& features)
{
    int T = features.size();
    int lfr_m = config_.lfr_m;
    int lfr_n = config_.lfr_n;

    int T_lfr = (T + lfr_n - 1) / lfr_n;
    int feat_dim = config_.n_mels;

    std::vector<std::vector<float>> lfr_features(T_lfr);

    for (int i = 0; i < T_lfr; ++i) {
        lfr_features[i].resize(feat_dim * lfr_m);

        for (int j = 0; j < lfr_m; ++j) {
            int idx = i * lfr_n + j;
            idx = std::min(idx, T - 1);

            std::copy(features[idx].begin(), features[idx].end(),
                lfr_features[i].begin() + j * feat_dim);
        }
    }

    return lfr_features;
}

void FeatureExtractor::applyCMVN(std::vector<std::vector<float>>& features) {
    if (!cmvn_loaded_ || features.empty()) {
        return;
    }

    int dim = features[0].size();
    dim = std::min(dim, static_cast<int>(cmvn_mean_.size()));

    for (auto& frame : features) {
        for (int i = 0; i < dim; ++i) {
            frame[i] = (frame[i] - cmvn_mean_[i]) / std::sqrt(cmvn_var_[i] + 1e-10f);
        }
    }
}

}  // namespace sensevoice
}  // namespace asr
