/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backends/zipformer/zipformer_ctc_model.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef USE_SPACEMIT_EP
#include "spacemit_ort_env.h"
#endif

namespace zipformer {

CtcModel::CtcModel(const Config& config)
        : env_(ORT_LOGGING_LEVEL_WARNING, "zipformer_ctc") {
#ifdef USE_SPACEMIT_EP
    if (config.provider == "spacemit") {
        session_options_.SetIntraOpNumThreads(1);
    } else {
#endif
        session_options_.SetIntraOpNumThreads(config.num_threads);
#ifdef USE_SPACEMIT_EP
    }
#endif
    session_options_.SetInterOpNumThreads(config.num_threads);
    session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_SPACEMIT_EP
    if (config.provider == "spacemit") {
        std::unordered_map<std::string, std::string> ep_options = {
                {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(config.num_threads)}};
        Ort::Status status =
                Ort::SessionOptionsSpaceMITEnvInit(session_options_, ep_options);
        if (status.IsOK()) {
            std::cout << "[Zipformer] SpaceMIT EP initialized (threads="
                        << config.num_threads << ")" << std::endl;
        } else {
            std::cerr << "[Zipformer] SpaceMIT EP init failed: "
                        << status.GetErrorMessage() << ", fallback to CPU"
                        << std::endl;
        }
        const std::string& model =
                config.quantized_model_path.empty()
                        ? config.model_path
                        : config.quantized_model_path;
        session_ =
                std::make_unique<Ort::Session>(env_, model.c_str(), session_options_);
    } else {
#endif
        session_ = std::make_unique<Ort::Session>(
                env_, config.model_path.c_str(), session_options_);
#ifdef USE_SPACEMIT_EP
    }
#endif

    Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    allocator_ = std::make_unique<Ort::Allocator>(*session_, mem_info);

    InitNames();
    ReadMetadata();
}

CtcModel::~CtcModel() = default;

void CtcModel::InitNames() {
    Ort::AllocatorWithDefaultOptions alloc;

    size_t ni = session_->GetInputCount();
    size_t no = session_->GetOutputCount();

    input_names_.resize(ni);
    output_names_.resize(no);
    input_names_ptr_.resize(ni);
    output_names_ptr_.resize(no);

    for (size_t i = 0; i < ni; ++i) {
        auto name = session_->GetInputNameAllocated(i, alloc);
        input_names_[i] = name.get();
        input_names_ptr_[i] = input_names_[i].c_str();
    }
    for (size_t i = 0; i < no; ++i) {
        auto name = session_->GetOutputNameAllocated(i, alloc);
        output_names_[i] = name.get();
        output_names_ptr_[i] = output_names_[i].c_str();
    }
}

void CtcModel::ReadMetadata() {
    Ort::AllocatorWithDefaultOptions alloc;
    auto meta = session_->GetModelMetadata();
    auto keys = meta.GetCustomMetadataMapKeysAllocated(alloc);

    for (size_t i = 0; i < keys.size(); ++i) {
        std::string key = keys[i].get();
        auto val = meta.LookupCustomMetadataMapAllocated(key.c_str(), alloc);
        std::string val_str = val.get();

        if (key == "T") {
            chunk_size_ = std::stoi(val_str);
        } else if (key == "decode_chunk_len") {
            chunk_shift_ = std::stoi(val_str);
        }
    }

    auto out_info = session_->GetOutputTypeInfo(0);
    auto shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() >= 3 && shape[2] > 0) {
        vocab_size_ = static_cast<int32_t>(shape[2]);
    }

    std::cout << "[Zipformer] Model: T=" << chunk_size_
                << " decode_chunk_len=" << chunk_shift_
                << " vocab_size=" << vocab_size_ << std::endl;
}

std::vector<Ort::Value> CtcModel::CreateInitStates() {
    std::vector<Ort::Value> states;

    for (size_t i = 1; i < input_names_.size(); ++i) {
        auto info = session_->GetInputTypeInfo(i);
        auto tensor_info = info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        auto elem_type = tensor_info.GetElementType();

        for (auto& d : shape) {
            if (d < 0) d = 1;
        }

        size_t numel = 1;
        for (auto d : shape) numel *= d;

        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            auto tensor = Ort::Value::CreateTensor<int64_t>(
                    *allocator_, shape.data(), shape.size());
            std::fill_n(tensor.GetTensorMutableData<int64_t>(), numel, 0);
            states.push_back(std::move(tensor));
        } else {
            auto tensor = Ort::Value::CreateTensor<float>(
                    *allocator_, shape.data(), shape.size());
            std::fill_n(tensor.GetTensorMutableData<float>(), numel, 0.0f);
            states.push_back(std::move(tensor));
        }
    }
    return states;
}

std::vector<float> CtcModel::RunChunk(
        const std::vector<float>& chunk_features,
        std::vector<Ort::Value>& states,
        int32_t& processed_lens, int32_t& out_frames) {
    Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;

    std::array<int64_t, 3> x_shape{1, chunk_size_, 80};
    std::vector<float> x_data(chunk_features.begin(), chunk_features.end());
    x_data.resize(chunk_size_ * 80, 0.0f);

    inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_info, x_data.data(), x_data.size(), x_shape.data(), 3));

    for (auto& s : states) {
        inputs.push_back(std::move(s));
    }

    auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_ptr_.data(), inputs.data(),
            inputs.size(), output_names_ptr_.data(),
            output_names_.size());

    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    out_frames = static_cast<int32_t>(out_shape[1]);

    const float* log_probs = outputs[0].GetTensorData<float>();
    int32_t total = out_frames * vocab_size_;
    std::vector<float> result(log_probs, log_probs + total);

    states.clear();
    for (size_t i = 1; i < outputs.size(); ++i) {
        states.push_back(std::move(outputs[i]));
    }

    return result;
}

std::vector<float> CtcModel::Run(const std::vector<float>& features,
                                    int32_t num_frames, int32_t feature_dim) {
    auto states = CreateInitStates();

    int32_t pad = chunk_size_ - chunk_shift_;

    std::vector<float> all_log_probs;
    int32_t processed_lens = 0;
    output_frames_ = 0;

    int32_t offset = 0;
    while (offset < num_frames) {
        std::vector<float> chunk(chunk_size_ * feature_dim, 0.0f);
        int32_t src_start = offset - pad;

        for (int32_t i = 0; i < chunk_size_; ++i) {
            int32_t src_frame = src_start + i;
            if (src_frame >= 0 && src_frame < num_frames) {
                std::copy(features.data() + src_frame * feature_dim,
                            features.data() + (src_frame + 1) * feature_dim,
                            chunk.data() + i * feature_dim);
            }
        }

        int32_t out_frames = 0;
        auto log_probs = RunChunk(chunk, states, processed_lens, out_frames);

        all_log_probs.insert(all_log_probs.end(), log_probs.begin(),
                                log_probs.end());
        output_frames_ += out_frames;
        processed_lens += chunk_shift_;
        offset += chunk_shift_;
    }

    return all_log_probs;
}

}  // namespace zipformer
