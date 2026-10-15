/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_processor.hpp"

#include <fftw3.h>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

AudioProcessor::AudioProcessor(const Config& config) : config_(config) {
}

AudioProcessor::~AudioProcessor() {
    cleanupFFTW();
}

bool AudioProcessor::initialize() {
    initializeMelFilterbank();
    initializeWindow();
    initializeFFTW();

    if (!config_.cmvn_file.empty()) {
        loadCMVN(config_.cmvn_file);
    }

    return true;
}

void AudioProcessor::initializeFFTW() {
    // Pre-allocate FFTW plan and buffers for n_fft size
    fft_size_ = config_.n_fft;
    fft_input_ = fftwf_alloc_real(fft_size_);
    fft_output_ = fftwf_alloc_complex(fft_size_ / 2 + 1);  // r2c only needs N/2+1 outputs

    if (fft_input_ && fft_output_) {
        fft_plan_ = fftwf_plan_dft_r2c_1d(fft_size_, fft_input_, fft_output_, FFTW_MEASURE);
        std::cout << "[AudioProcessor] FFTW plan initialized (size=" << fft_size_ << ")" << std::endl;
    } else {
        std::cerr << "[AudioProcessor] Failed to allocate FFTW buffers" << std::endl;
    }
}

void AudioProcessor::cleanupFFTW() {
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
    fft_size_ = 0;
}

void AudioProcessor::initializeMelFilterbank() {
    // Create mel filterbank
    int num_filters = config_.n_mels;
    int fft_size = config_.n_fft / 2 + 1;

    mel_filterbank_.resize(num_filters);

    // Mel scale conversion
    auto hz_to_mel = [](float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };

    auto mel_to_hz = [](float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };

    float nyquist = config_.sample_rate / 2.0f;
    float mel_max = hz_to_mel(nyquist);

    // Create mel points
    std::vector<float> mel_points(num_filters + 2);
    for (int i = 0; i < num_filters + 2; ++i) {
        mel_points[i] = i * mel_max / (num_filters + 1);
    }

    // Convert back to Hz
    std::vector<float> hz_points(num_filters + 2);
    for (int i = 0; i < num_filters + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }

    // Convert Hz to FFT bin indices
    std::vector<int> bin_points(num_filters + 2);
    for (int i = 0; i < num_filters + 2; ++i) {
        bin_points[i] = static_cast<int>(
            std::floor((config_.n_fft + 1) * hz_points[i] / config_.sample_rate));
        bin_points[i] = std::min(bin_points[i], fft_size - 1);
    }

    // Create triangular filters
    for (int i = 0; i < num_filters; ++i) {
        mel_filterbank_[i].resize(fft_size, 0.0f);

        int start = bin_points[i];
        int center = bin_points[i + 1];
        int end = bin_points[i + 2];

        // Left side of triangle
        for (int j = start; j < center; ++j) {
            if (center != start) {
                mel_filterbank_[i][j] = static_cast<float>(j - start) / (center - start);
            }
        }

        // Right side of triangle
        for (int j = center; j < end; ++j) {
            if (end != center) {
                mel_filterbank_[i][j] = static_cast<float>(end - j) / (end - center);
            }
        }
    }
}

void AudioProcessor::initializeWindow() {
    window_ = createHammingWindow(config_.frame_length);
}

std::vector<float> AudioProcessor::createHammingWindow(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
    }
    return window;
}

void AudioProcessor::loadCMVN(const std::string& cmvn_file) {
    // In a real implementation, you would load CMVN statistics from file
    // For now, we'll initialize with dummy values
    cmvn_mean_.assign(config_.n_mels, 0.0f);
    cmvn_var_.assign(config_.n_mels, 1.0f);
    cmvn_loaded_ = true;

    std::cout << "CMVN loaded (dummy implementation)" << std::endl;
}

std::vector<std::vector<float>> AudioProcessor::extractFeatures(const std::vector<float>& audio) {
    return extractFeatures(audio.data(), audio.size());
}

std::vector<std::vector<float>> AudioProcessor::extractFeatures(const float* audio, size_t length) {
    // Preprocess audio in-place (single allocation)
    std::vector<float> preprocessed_audio;
    preprocessed_audio.reserve(length);
    preprocessed_audio.assign(audio, audio + length);

    // Apply preemphasis in-place
    if (config_.preemphasis > 0.0f) {
        for (size_t i = preprocessed_audio.size() - 1; i > 0; --i) {
            preprocessed_audio[i] = preprocessed_audio[i] - config_.preemphasis * preprocessed_audio[i - 1];
        }
    }

    // Extract fbank features
    auto fbank_features = computeFbank(preprocessed_audio);

    // Apply LFR (Low Frame Rate) if needed
    fbank_features = applyLFR(fbank_features);

    // Apply CMVN
    if (cmvn_loaded_) {
        applyCMVN(fbank_features);
    }

    return fbank_features;
}

std::vector<float> AudioProcessor::preprocess(const std::vector<float>& audio) {
    std::vector<float> result = audio;

    // Apply preemphasis
    if (config_.preemphasis > 0.0f) {
        for (size_t i = result.size() - 1; i > 0; --i) {
            result[i] = result[i] - config_.preemphasis * result[i - 1];
        }
    }

    return result;
}

std::vector<std::vector<float>> AudioProcessor::computeFbank(const std::vector<float>& audio) {
    // Frame the signal
    auto frames = frameSignal(audio);

    std::vector<std::vector<float>> fbank_features;
    fbank_features.reserve(frames.size());

    for (const auto& frame : frames) {
        // Apply window and pad to FFT size
        std::vector<float> windowed_frame(config_.n_fft, 0.0f);
        size_t copy_len = std::min(frame.size(), window_.size());
        for (size_t i = 0; i < copy_len; ++i) {
            windowed_frame[i] = frame[i] * window_[i];
        }

        // Compute FFT
        auto fft_result = fft(windowed_frame);

        // Compute power spectrum
        auto power_spectrum = computePowerSpectrum(fft_result);

        // Apply mel filterbank
        auto mel_features = applyMelFilterbank(power_spectrum);

        // Convert to log scale
        for (float& val : mel_features) {
            val = std::log(std::max(val, 1e-10f));
        }

        fbank_features.push_back(mel_features);
    }

    return fbank_features;
}

std::vector<std::vector<float>> AudioProcessor::frameSignal(const std::vector<float>& signal) {
    std::vector<std::vector<float>> frames;

    int num_frames = static_cast<int>(
        (signal.size() - config_.frame_length) / config_.frame_shift) + 1;
    frames.reserve(num_frames);

    for (int i = 0; i < num_frames; ++i) {
        int start = i * config_.frame_shift;
        int end = start + config_.frame_length;

        std::vector<float> frame;
        if (end <= static_cast<int>(signal.size())) {
            frame.assign(signal.begin() + start, signal.begin() + end);
        } else {
            frame.assign(signal.begin() + start, signal.end());
            frame.resize(config_.frame_length, 0.0f);  // Zero pad
        }

        frames.push_back(frame);
    }

    return frames;
}

std::vector<std::complex<float>> AudioProcessor::fft(const std::vector<float>& signal) {
    // Use pre-allocated FFTW plan and buffers for efficiency
    // This avoids the massive overhead of creating/destroying plans per frame
    int N = static_cast<int>(signal.size());

    // Check if we can use the pre-allocated plan
    if (fft_plan_ && fft_input_ && fft_output_ && N == fft_size_) {
        // Copy input data to pre-allocated buffer
        for (int i = 0; i < N; ++i) {
            fft_input_[i] = signal[i];
        }

        // Execute pre-allocated plan
        fftwf_execute(fft_plan_);

        // Convert to std::complex format
        std::vector<std::complex<float>> result(N / 2 + 1);
        for (int i = 0; i < N / 2 + 1; ++i) {
            result[i] = std::complex<float>(fft_output_[i][0], fft_output_[i][1]);
        }

        return result;
    }

    // Fallback: create temporary plan if size doesn't match (should rarely happen)
    std::cerr << "[AudioProcessor] Warning: FFT size mismatch, creating temporary plan (N="
                << N << ", expected=" << fft_size_ << ")" << std::endl;

    float* in = fftwf_alloc_real(N);
    fftwf_complex* out = fftwf_alloc_complex(N / 2 + 1);

    if (!in || !out) {
        if (in) fftwf_free(in);
        if (out) fftwf_free(out);
        return std::vector<std::complex<float>>(N / 2 + 1, std::complex<float>(0, 0));
    }

    // Copy input data
    for (int i = 0; i < N; ++i) {
        in[i] = signal[i];
    }

    // Create plan and execute
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftwf_execute(plan);

    // Convert to std::complex format
    std::vector<std::complex<float>> result(N / 2 + 1);
    for (int i = 0; i < N / 2 + 1; ++i) {
        result[i] = std::complex<float>(out[i][0], out[i][1]);
    }

    // Cleanup
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    return result;
}

std::vector<float> AudioProcessor::istft(const std::vector<std::complex<float>>& stft_data,
                                        int n_fft, int hop_length, int num_frames) {
    // Proper ISTFT implementation using FFTW3
    int fft_bins = n_fft / 2 + 1;
    int audio_length = (num_frames - 1) * hop_length + n_fft;

    std::vector<float> audio(audio_length, 0.0f);
    std::vector<float> window_sum(audio_length, 0.0f);

    // Create Hann window for overlap-add
    std::vector<float> window(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n_fft - 1)));
    }

    // FFTW setup for inverse FFT
    fftwf_complex* in = fftwf_alloc_complex(fft_bins);
    float* out = fftwf_alloc_real(n_fft);
    fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_fft, in, out, FFTW_ESTIMATE);

    // Process each frame
    for (int frame = 0; frame < num_frames; ++frame) {
        // Copy STFT frame data to FFTW input
        for (int bin = 0; bin < fft_bins; ++bin) {
            int stft_idx = frame * fft_bins + bin;
            if (stft_idx < static_cast<int>(stft_data.size())) {
                std::complex<float> val = stft_data[stft_idx];
                in[bin][0] = val.real();
                in[bin][1] = val.imag();
            } else {
                in[bin][0] = 0.0f;
                in[bin][1] = 0.0f;
            }
        }

        // Execute inverse FFT
        fftwf_execute(plan);

        // Apply window and overlap-add
        int start_pos = frame * hop_length;
        for (int i = 0; i < n_fft && start_pos + i < audio_length; ++i) {
            float windowed_sample = out[i] * window[i] / n_fft;
            audio[start_pos + i] += windowed_sample;
            window_sum[start_pos + i] += window[i] * window[i];
        }
    }

    // Normalize by window sum to avoid amplitude modulation
    for (int i = 0; i < audio_length; ++i) {
        if (window_sum[i] > 1e-6f) {
            audio[i] /= window_sum[i] / (n_fft * 0.5f);  // Normalization factor
        }
    }

    // Cleanup FFTW
    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    return audio;
}

std::vector<float> AudioProcessor::computePowerSpectrum(
    const std::vector<std::complex<float>>& fft_result) {
    std::vector<float> power_spectrum;
    power_spectrum.reserve(fft_result.size());

    for (const auto& val : fft_result) {
        float magnitude = std::abs(val);
        power_spectrum.push_back(magnitude * magnitude);
    }

    return power_spectrum;
}

std::vector<float> AudioProcessor::applyMelFilterbank(const std::vector<float>& power_spectrum) {
    std::vector<float> mel_features(config_.n_mels, 0.0f);

    for (int i = 0; i < config_.n_mels; ++i) {
        for (size_t j = 0; j < power_spectrum.size() && j < mel_filterbank_[i].size(); ++j) {
            mel_features[i] += power_spectrum[j] * mel_filterbank_[i][j];
        }
    }

    return mel_features;
}

std::vector<std::vector<float>> AudioProcessor::applyLFR(
    const std::vector<std::vector<float>>& features) {
    // LFR (Low Frame Rate) implementation
    // Based on config: lfr_m=7, lfr_n=6
    // Concatenate lfr_m consecutive frames, then downsample by lfr_n
    const int lfr_m = 7;  // number of frames to concatenate
    const int lfr_n = 6;  // downsample rate

    if (features.empty()) {
        return features;
    }

    std::vector<std::vector<float>> lfr_features;
    int feature_dim = features[0].size();

    // Process frames with LFR
    for (size_t i = 0; i < features.size(); i += lfr_n) {
        std::vector<float> lfr_frame;
        lfr_frame.reserve(feature_dim * lfr_m);

        // Concatenate lfr_m frames
        for (int j = 0; j < lfr_m; ++j) {
            size_t frame_idx = i + j;
            if (frame_idx < features.size()) {
                // Use existing frame
                lfr_frame.insert(lfr_frame.end(),
                    features[frame_idx].begin(), features[frame_idx].end());
            } else {
                // Pad with last frame if not enough frames
                size_t last_idx = features.size() - 1;
                lfr_frame.insert(lfr_frame.end(),
                    features[last_idx].begin(), features[last_idx].end());
            }
        }

        lfr_features.push_back(lfr_frame);
    }

    return lfr_features;
}

void AudioProcessor::applyCMVN(std::vector<std::vector<float>>& features) {
    if (!cmvn_loaded_ || features.empty()) {
        return;
    }

    // Apply CMVN normalization
    for (auto& frame : features) {
        for (size_t i = 0; i < frame.size() && i < cmvn_mean_.size(); ++i) {
            frame[i] = (frame[i] - cmvn_mean_[i]) / std::sqrt(cmvn_var_[i]);
        }
    }
}
