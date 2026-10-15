/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file feature_extractor.hpp
 * @brief Audio feature extraction for ASR (Fbank + LFR + CMVN)
 *
 * Extracts Mel-frequency filterbank features with Low Frame Rate (LFR)
 * stacking and Cepstral Mean Variance Normalization (CMVN).
 */

#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <fftw3.h>

#include <complex>
#include <string>
#include <vector>

namespace asr {
namespace sensevoice {

/**
 * @class FeatureExtractor
 * @brief Extracts acoustic features from audio for ASR
 *
 * Processing pipeline:
 * 1. Pre-emphasis filtering
 * 2. Framing with Hamming window
 * 3. FFT → Power spectrum
 * 4. Mel filterbank → Log Fbank features
 * 5. LFR (Low Frame Rate) stacking
 * 6. CMVN normalization
 */
class FeatureExtractor {
public:
    /**
     * @brief Configuration for feature extraction
     */
    struct Config {
        int sample_rate = 16000;        ///< Audio sample rate (Hz)
        int frame_length = 400;         ///< Frame length in samples (25ms)
        int frame_shift = 160;          ///< Frame shift in samples (10ms)
        int n_mels = 80;                ///< Number of Mel filterbanks
        int n_fft = 512;                ///< FFT size
        float preemphasis = 0.97f;      ///< Pre-emphasis coefficient

        bool apply_cmvn = true;         ///< Apply CMVN normalization
        std::string cmvn_file;          ///< Path to CMVN file (am.mvn)

        // LFR (Low Frame Rate) settings
        int lfr_m = 7;                  ///< LFR window size
        int lfr_n = 6;                  ///< LFR stride
    };

    explicit FeatureExtractor(const Config& config);
    ~FeatureExtractor();

    // Non-copyable
    FeatureExtractor(const FeatureExtractor&) = delete;
    FeatureExtractor& operator=(const FeatureExtractor&) = delete;

    /**
     * @brief Initialize the feature extractor
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Extract features from audio samples
     * @param audio Audio samples (float)
     * @return Feature matrix [frames x feature_dim]
     */
    std::vector<std::vector<float>> extract(const std::vector<float>& audio);

    /**
     * @brief Extract features from audio samples
     * @param audio Pointer to audio samples
     * @param length Number of samples
     * @return Feature matrix [frames x feature_dim]
     */
    std::vector<std::vector<float>> extract(const float* audio, size_t length);

    /**
     * @brief Get feature dimension after LFR
     */
    int getFeatureDim() const { return config_.n_mels * config_.lfr_m; }

private:
    Config config_;

    // CMVN parameters
    bool cmvn_loaded_ = false;
    std::vector<float> cmvn_mean_;
    std::vector<float> cmvn_var_;

    // Mel filterbank (dense version for compatibility)
    std::vector<std::vector<float>> mel_filterbank_;

    // Sparse mel filterbank for efficiency
    struct MelFilter {
        int start_bin;
        int end_bin;
        std::vector<float> weights;
    };
    std::vector<MelFilter> mel_filters_sparse_;

    // Window function
    std::vector<float> window_;

    // FFTW resources
    fftwf_plan fft_plan_ = nullptr;
    float* fft_input_ = nullptr;
    fftwf_complex* fft_output_ = nullptr;

    // Internal methods
    void initializeMelFilterbank();
    void initializeWindow();
    void initializeFFTW();
    void cleanupFFTW();
    bool loadCMVN(const std::string& path);

    // Processing steps
    std::vector<float> preprocess(const std::vector<float>& audio);
    std::vector<std::vector<float>> computeFbank(const std::vector<float>& audio);
    std::vector<std::vector<float>> applyLFR(const std::vector<std::vector<float>>& features);
    void applyCMVN(std::vector<std::vector<float>>& features);

    // FFT helpers
    std::vector<std::complex<float>> fft(const std::vector<float>& signal);
    std::vector<float> computePowerSpectrum(const std::vector<std::complex<float>>& fft_result);
    std::vector<float> applyMelFilterbank(const std::vector<float>& power_spectrum);

    // Utility
    float melScale(float freq);
    float invMelScale(float mel);
    std::vector<float> createHammingWindow(int size);
    std::vector<std::vector<float>> frameSignal(const std::vector<float>& signal);
};

}  // namespace sensevoice
}  // namespace asr

#endif  // FEATURE_EXTRACTOR_HPP
