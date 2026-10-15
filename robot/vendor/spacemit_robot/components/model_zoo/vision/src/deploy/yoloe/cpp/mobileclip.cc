/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Ported from yolo-world demo. MobileClip text encoder runs on CPU (the reference
 * demo did not enable the SpaceMIT EP for the text branch); text encoding is a
 * one-off per-vocabulary cost, so CPU is acceptable.
 */

#include "mobileclip.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include "vision_model_base.h"

namespace vision_deploy {

MobileClip::MobileClip(const std::string& text_model_path, const std::string& bpe_merges_path, int num_threads)
    : session_(nullptr) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_ = std::make_unique<Ort::Session>(
        vision_core::shared_ort_env(), text_model_path.c_str(), session_options);

    const size_t num_inputs = session_->GetInputCount();
    input_names_.resize(num_inputs);
    input_names_str_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator_);
        input_names_str_[i] = name.get();
        input_names_[i] = input_names_str_[i].c_str();
    }

    const size_t num_outputs = session_->GetOutputCount();
    output_names_.resize(num_outputs);
    output_names_str_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_str_[i] = name.get();
        output_names_[i] = output_names_str_[i].c_str();
    }

    tokenizer_ = std::make_unique<MobileClipTokenizer>(bpe_merges_path);
    sequence_len_ = 77;
}

MobileClip::~MobileClip() = default;

std::vector<std::vector<float>> MobileClip::encode(const std::vector<std::string>& texts) {
    std::vector<std::vector<float>> result;
    result.reserve(texts.size());

    const std::vector<int64_t> input_shape = {1, sequence_len_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (const auto& text : texts) {
        std::vector<int32_t> ids = tokenizer_->tokenize(text)[0];
        std::vector<int64_t> token(ids.begin(), ids.end());

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, token.data(), token.size(), input_shape.data(), input_shape.size());

        std::vector<Ort::Value> outputs = session_->Run(
            Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());

        const float* data = outputs[0].GetTensorData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t count = 1;
        for (size_t i = 1; i < shape.size(); ++i) {
            count *= static_cast<size_t>(shape[i]);
        }
        result.emplace_back(data, data + count);
    }
    return result;
}

}  // namespace vision_deploy
