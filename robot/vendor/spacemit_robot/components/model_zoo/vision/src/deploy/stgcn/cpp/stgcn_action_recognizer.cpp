/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stgcn_action_recognizer.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {
const std::vector<std::string> kClassNames = {
    "Standing", "Walking", "Sitting", "Lying Down",
    "Stand up", "Sit down", "Fall Down",
};
}  // namespace

std::unique_ptr<vision_core::BaseModel> StgcnActionRecognizer::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for StgcnActionRecognizer");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for StgcnActionRecognizer");
    }
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    return std::make_unique<StgcnActionRecognizer>(model_path, num_threads, lazy_load, provider);
}

StgcnActionRecognizer::StgcnActionRecognizer(const std::string& model_path,
                                                int num_threads,
                                                bool lazy_load,
                                                const std::string& provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider),
        class_names_(kClassNames) {
    if (!lazy_load) {
        load_model();
    }
}

void StgcnActionRecognizer::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

std::string StgcnActionRecognizer::get_class_name(int pred_class) const {
    if (pred_class >= 0 && pred_class < static_cast<int>(class_names_.size())) {
        return class_names_[static_cast<size_t>(pred_class)];
    }
    return std::to_string(pred_class);
}

vision_common::ActionResult StgcnActionRecognizer::predict(const std::vector<float>& pts,
                                                int image_width,
                                                int image_height) {
    if (pts.size() < static_cast<size_t>(kSequenceLength * kNumKeypoints * 3)) {
        throw std::runtime_error("StgcnActionRecognizer::predict pts size too small");
    }
    return predict(pts.data(), image_width, image_height);
}

vision_common::ActionResult StgcnActionRecognizer::predict(const float* pts,
                                                int image_width,
                                                int image_height) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t_total0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();

    const int T = kSequenceLength;
    const int V = kNumKeypoints;
    const int C = 3;
    const int V_plus = V + 1;  // 14 after adding center

    float width = static_cast<float>(image_width);
    float height = static_cast<float>(image_height);
    if (width <= 0.f) width = 1.f;
    if (height <= 0.f) height = 1.f;

    // Copy and normalize xy to [0,1], then scale to [-1,1]; keep score
    std::vector<float> pts_norm(static_cast<size_t>(T * V_plus * C), 0.f);
    for (int t = 0; t < T; ++t) {
        for (int v = 0; v < V; ++v) {
            size_t idx = static_cast<size_t>(t * V * C + v * C);
            float x = pts[idx + 0];
            float y = pts[idx + 1];
            float s = pts[idx + 2];
            pts_norm[t * V_plus * C + v * C + 0] = (x / width - 0.5f) * 2.f;
            pts_norm[t * V_plus * C + v * C + 1] = (y / height - 0.5f) * 2.f;
            pts_norm[t * V_plus * C + v * C + 2] = s;
        }
        // Center = (left_shoulder + right_shoulder) / 2, indices 1,2
        size_t i1 = static_cast<size_t>(t * V_plus * C + 1 * C);
        size_t i2 = static_cast<size_t>(t * V_plus * C + 2 * C);
        pts_norm[t * V_plus * C + V * C + 0] = (pts_norm[i1 + 0] + pts_norm[i2 + 0]) * 0.5f;
        pts_norm[t * V_plus * C + V * C + 1] = (pts_norm[i1 + 1] + pts_norm[i2 + 1]) * 0.5f;
        pts_norm[t * V_plus * C + V * C + 2] = (pts_norm[i1 + 2] + pts_norm[i2 + 2]) * 0.5f;
    }

    // pts_tensor (1, C, T, V_plus): layout batch, channel, time, node
    const int64_t pts_shape[] = {1, C, T, V_plus};
    size_t pts_size = static_cast<size_t>(1 * C * T * V_plus);
    std::vector<float> pts_tensor(pts_size);
    for (int t = 0; t < T; ++t) {
        for (int v = 0; v < V_plus; ++v) {
            for (int c = 0; c < C; ++c) {
                pts_tensor[static_cast<size_t>(c * T * V_plus + t * V_plus + v)] =
                    pts_norm[static_cast<size_t>(t * V_plus * C + v * C + c)];
            }
        }
    }

    // mot (1, 2, T-1, V_plus) = diff of xy along time
    const int T_mot = T - 1;
    std::vector<float> mot(static_cast<size_t>(1 * 2 * T_mot * V_plus), 0.f);
    for (int t = 0; t < T_mot; ++t) {
        for (int v = 0; v < V_plus; ++v) {
            for (int c = 0; c < 2; ++c) {
                float a = pts_tensor[static_cast<size_t>(c * T * V_plus + (t + 1) * V_plus + v)];
                float b = pts_tensor[static_cast<size_t>(c * T * V_plus + t * V_plus + v)];
                mot[static_cast<size_t>(c * T_mot * V_plus + t * V_plus + v)] = a - b;
            }
        }
    }
    // mot_padded (1, 2, T, V_plus): first time step zero
    const int64_t mot_shape[] = {1, 2, T, V_plus};
    size_t mot_padded_size = static_cast<size_t>(1 * 2 * T * V_plus);
    std::vector<float> mot_padded(mot_padded_size, 0.f);
    for (int t = 1; t < T; ++t) {
        for (int v = 0; v < V_plus; ++v) {
            for (int c = 0; c < 2; ++c) {
                mot_padded[static_cast<size_t>(c * T * V_plus + t * V_plus + v)] =
                    mot[static_cast<size_t>(c * T_mot * V_plus + (t - 1) * V_plus + v)];
            }
        }
    }

    std::vector<int64_t> pts_shape_vec(pts_shape, pts_shape + 4);
    std::vector<int64_t> mot_shape_vec(mot_shape, mot_shape + 4);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session_two_inputs(
        pts_tensor.data(), pts_shape_vec,
        mot_padded.data(), mot_shape_vec);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();

    if (outputs.empty()) {
        throw std::runtime_error("StgcnActionRecognizer::predict no output");
    }
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    size_t num_elem = static_cast<size_t>(tensor_info.GetElementCount());
    const float* out_data = outputs[0].GetTensorData<float>();
    if (out_data == nullptr) {
        throw std::runtime_error("StgcnActionRecognizer::predict output data is null");
    }
    std::vector<float> scores(out_data, out_data + num_elem);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t_total1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t_total1 - t_total0).count());

    // Create ActionResult
    vision_common::ActionResult result;
    result.class_scores = scores;

    // Find the action with highest score
    auto max_it = std::max_element(scores.begin(), scores.end());
    result.label = static_cast<int>(max_it - scores.begin());
    result.score = *max_it;

    return result;
}

vision_common::ActionResult StgcnActionRecognizer::infer_sequence(const float* pts,
    int image_width,
    int image_height) {
    return predict(pts, image_width, image_height);
}


std::vector<vision_core::InferIntent> StgcnActionRecognizer::supported_intents() const {
    return {vision_core::InferIntent::kInferSequence};
}

size_t StgcnActionRecognizer::expected_sequence_size() const {
    return static_cast<size_t>(kSequenceLength) * kNumKeypoints * 3;
}

std::vector<std::string> StgcnActionRecognizer::get_sequence_class_names() const {
    return get_class_names();
}

vision_core::InferResponse StgcnActionRecognizer::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kInferSequence);
    const auto* sequence_input = std::get_if<vision_core::SequenceInput>(&request.input);
    if (sequence_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "StgcnActionRecognizer expects SequenceInput";
        return response;
    }
    const size_t expected = static_cast<size_t>(kSequenceLength * kNumKeypoints * 3);
    if (sequence_input->pts.size() != expected) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "StgcnActionRecognizer expects " + std::to_string(expected) + " sequence values";
        return response;
    }
    vision_core::InferResponse response;
    response.results.emplace_back(
        predict(sequence_input->pts.data(), sequence_input->image_width, sequence_input->image_height));
    return response;
}

std::vector<vision_core::ModelCapability> StgcnActionRecognizer::get_capabilities() const {
    return {};
}

std::vector<Ort::Value> StgcnActionRecognizer::run_session_two_inputs(
    const float* pts_tensor_data,
    const std::vector<int64_t>& pts_shape,
    const float* mot_data,
    const std::vector<int64_t>& mot_shape) {
    ensure_model_loaded();
    if (input_node_names_.size() < 2) {
        throw std::runtime_error("StgcnActionRecognizer model must have 2 inputs (pts, mot)");
    }

    size_t pts_num = 1;
    for (size_t i = 0; i < pts_shape.size(); ++i) pts_num *= static_cast<size_t>(pts_shape[i]);
    size_t mot_num = 1;
    for (size_t i = 0; i < mot_shape.size(); ++i) mot_num *= static_cast<size_t>(mot_shape[i]);

    Ort::Value input_pts = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(pts_tensor_data),
        pts_num,
        pts_shape.data(),
        pts_shape.size());
    Ort::Value input_mot = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(mot_data),
        mot_num,
        mot_shape.data(),
        mot_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_pts));
    input_tensors.push_back(std::move(input_mot));

    return session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_node_names_.data(),
        output_node_names_.size());
}

static vision_core::ModelRegistrar<StgcnActionRecognizer> registrar("StgcnActionRecognizer");

}  // namespace vision_deploy
