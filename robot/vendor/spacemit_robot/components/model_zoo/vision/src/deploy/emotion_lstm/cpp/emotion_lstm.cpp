/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "emotion_lstm.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {
// Matches assets/labels/emotion.txt (Emo-AffectNet index order).
const std::vector<std::string> kClassNames = {
    "neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger",
};
}  // namespace

std::unique_ptr<vision_core::BaseModel> EmotionLstm::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for EmotionLstm");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for EmotionLstm");
    }
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    return std::make_unique<EmotionLstm>(model_path, num_threads, lazy_load, provider);
}

EmotionLstm::EmotionLstm(const std::string& model_path,
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

void EmotionLstm::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

std::vector<vision_core::InferIntent> EmotionLstm::supported_intents() const {
    return {vision_core::InferIntent::kInferSequence};
}

size_t EmotionLstm::expected_sequence_size() const {
    return static_cast<size_t>(kSeqLen) * kFeatureDim;
}

std::vector<std::string> EmotionLstm::get_sequence_class_names() const {
    return class_names_;
}

std::vector<vision_core::ModelCapability> EmotionLstm::get_capabilities() const {
    return {};
}

std::vector<Ort::Value> EmotionLstm::run_session_sequence(const float* data,
                                                            const std::vector<int64_t>& shape) {
    ensure_model_loaded();
    if (input_node_names_.empty()) {
        throw std::runtime_error("EmotionLstm model has no inputs");
    }
    size_t num = 1;
    for (int64_t d : shape) num *= static_cast<size_t>(d);

    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(data),
        num,
        shape.data(),
        shape.size());

    return session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input,
        1,
        output_node_names_.data(),
        output_node_names_.size());
}

vision_common::ActionResult EmotionLstm::predict(const float* feats) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t_total0 = std::chrono::steady_clock::now();

    // Single input tensor (1, kSeqLen, kFeatureDim). feats already flat [t][d].
    const std::vector<int64_t> shape = {1, kSeqLen, kFeatureDim};

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session_sequence(feats, shape);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    if (outputs.empty()) {
        throw std::runtime_error("EmotionLstm::predict no output");
    }
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    size_t num_elem = static_cast<size_t>(tensor_info.GetElementCount());
    const float* out_data = outputs[0].GetTensorData<float>();
    if (out_data == nullptr) {
        throw std::runtime_error("EmotionLstm::predict output data is null");
    }
    // Match Python EmotionLstm.infer: use raw output as class scores, argmax for label.
    std::vector<float> scores(out_data, out_data + num_elem);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t_total1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t_total1 - t_total0).count());

    vision_common::ActionResult result;
    result.class_scores = scores;
    auto max_it = std::max_element(scores.begin(), scores.end());
    result.label = static_cast<int>(max_it - scores.begin());
    result.score = *max_it;
    return result;
}

vision_core::InferResponse EmotionLstm::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kInferSequence);
    const auto* sequence_input = std::get_if<vision_core::SequenceInput>(&request.input);
    if (sequence_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "EmotionLstm expects SequenceInput";
        return response;
    }
    const size_t expected = expected_sequence_size();
    if (sequence_input->pts.size() != expected) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message =
            "EmotionLstm expects " + std::to_string(expected) + " sequence values";
        return response;
    }
    vision_core::InferResponse response;
    response.results.emplace_back(predict(sequence_input->pts.data()));
    return response;
}

static vision_core::ModelRegistrar<EmotionLstm> registrar("EmotionLstm");

}  // namespace vision_deploy
