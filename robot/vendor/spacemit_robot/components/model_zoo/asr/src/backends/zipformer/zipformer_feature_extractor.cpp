/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backends/zipformer/zipformer_feature_extractor.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "online-feature.h"

namespace zipformer {

class FeatureExtractor::Impl {
public:
    explicit Impl(const Config& config) : config_(config) {
        knf::FbankOptions opts;
        opts.frame_opts.samp_freq = config.sample_rate;
        opts.frame_opts.frame_shift_ms = config.frame_shift_ms;
        opts.frame_opts.frame_length_ms = config.frame_length_ms;
        opts.frame_opts.dither = config.dither;
        opts.frame_opts.snip_edges = config.snip_edges;
        opts.frame_opts.window_type = "povey";
        opts.mel_opts.num_bins = config.feature_dim;
        opts.mel_opts.low_freq = 20.0f;
        opts.mel_opts.high_freq = -400.0f;

        fbank_ = std::make_unique<knf::OnlineFbank>(opts);
    }

    void AcceptWaveform(int32_t sample_rate, const float* data, int32_t n) {
        if (sample_rate != config_.sample_rate) {
            float ratio = static_cast<float>(config_.sample_rate) / sample_rate;
            int32_t new_n = static_cast<int32_t>(n * ratio);
            std::vector<float> resampled(new_n);
            for (int32_t i = 0; i < new_n; ++i) {
                float src = i / ratio;
                int32_t i0 = static_cast<int32_t>(src);
                int32_t i1 = std::min(i0 + 1, n - 1);
                float f = src - i0;
                resampled[i] = data[i0] * (1.0f - f) + data[i1] * f;
            }
            fbank_->AcceptWaveform(config_.sample_rate, resampled.data(), new_n);
        } else {
            fbank_->AcceptWaveform(sample_rate, data, n);
        }
    }

    void InputFinished() { fbank_->InputFinished(); }

    int32_t NumFramesReady() const { return fbank_->NumFramesReady(); }

    int32_t FeatureDim() const { return config_.feature_dim; }

    std::vector<float> GetFrames(int32_t frame_idx, int32_t n) const {
        int32_t dim = config_.feature_dim;
        std::vector<float> features(n * dim);
        for (int32_t i = 0; i < n; ++i) {
            const float* frame = fbank_->GetFrame(frame_idx + i);
            std::copy(frame, frame + dim, features.data() + i * dim);
        }
        return features;
    }

private:
    Config config_;
    std::unique_ptr<knf::OnlineFbank> fbank_;
};

FeatureExtractor::FeatureExtractor(const Config& config)
        : impl_(std::make_unique<Impl>(config)) {}
FeatureExtractor::~FeatureExtractor() = default;

void FeatureExtractor::AcceptWaveform(int32_t sr, const float* data,
                                        int32_t n) {
    impl_->AcceptWaveform(sr, data, n);
}
void FeatureExtractor::InputFinished() { impl_->InputFinished(); }
int32_t FeatureExtractor::NumFramesReady() const {
    return impl_->NumFramesReady();
}
int32_t FeatureExtractor::FeatureDim() const { return impl_->FeatureDim(); }
std::vector<float> FeatureExtractor::GetFrames(int32_t idx, int32_t n) const {
    return impl_->GetFrames(idx, n);
}

}  // namespace zipformer
