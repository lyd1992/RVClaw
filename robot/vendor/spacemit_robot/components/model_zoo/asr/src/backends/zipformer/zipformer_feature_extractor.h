/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mel-fbank feature extractor for Zipformer CTC, using kaldi-native-fbank.
 * Adapted from github.com/k2-fsa/sherpa-onnx streaming zipformer demo.
 */

#ifndef ZIPFORMER_FEATURE_EXTRACTOR_H
#define ZIPFORMER_FEATURE_EXTRACTOR_H

#include <cstdint>
#include <memory>
#include <vector>

namespace zipformer {

class FeatureExtractor {
public:
    struct Config {
        int32_t sample_rate = 16000;
        int32_t feature_dim = 80;
        float frame_shift_ms = 10.0f;
        float frame_length_ms = 25.0f;
        float dither = 0.0f;
        bool snip_edges = false;
    };

    explicit FeatureExtractor(const Config& config);
    ~FeatureExtractor();

    void AcceptWaveform(int32_t sample_rate, const float* data, int32_t n);
    void InputFinished();

    int32_t NumFramesReady() const;
    int32_t FeatureDim() const;

    std::vector<float> GetFrames(int32_t frame_idx, int32_t n) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zipformer

#endif  // ZIPFORMER_FEATURE_EXTRACTOR_H
