/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "emotion_recognizer.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> EmotionRecognizer::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for EmotionRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for EmotionRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    bool feature_mode = vision_core::yaml_utils::getBool(default_params, "feature_mode", false);
    return std::make_unique<EmotionRecognizer>(model_path, num_threads, lazy_load, provider, feature_mode);
}

EmotionRecognizer::EmotionRecognizer(const std::string& model_path,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider,
                                    bool feature_mode)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        target_size_(224, 224),
        provider_(provider),
        feature_mode_(feature_mode) {
    if (!lazy_load) {
        load_model();
    }
}

void EmotionRecognizer::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat EmotionRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    // Resize to target size (224, 224)
    cv::Mat resized;
    cv::resize(image, resized, target_size_, 0, 0, cv::INTER_NEAREST);

    // blobFromImage: float conversion, mean subtraction, and HWC->CHW in one call
    // swapRB=false because emotion model expects BGR order
    return cv::dnn::blobFromImage(resized, 1.0, target_size_,
                                    cv::Scalar(91.4953, 103.8827, 131.0912),
                                    false, false, CV_32F);
}

vision_common::ClassificationResultList EmotionRecognizer::classify(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat inputTensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Run inference using base class method
    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(inputTensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Postprocess
    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::ClassificationResultList results = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


vision_common::EmbeddingResult EmotionRecognizer::infer_embedding(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat inputTensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(inputTensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Feature mode: return raw output vector (no argmax, no L2 normalization).
    const auto t_post0 = std::chrono::steady_clock::now();
    if (outputs.empty()) {
        throw std::runtime_error("EmotionRecognizer::infer_embedding: outputs is empty");
    }
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();
    size_t feature_size = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        feature_size *= static_cast<size_t>(dims[i]);
    }
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    if (output_data == nullptr) {
        throw std::runtime_error("EmotionRecognizer::infer_embedding: output_data is null");
    }
    vision_common::EmbeddingResult result;
    result.embedding.assign(output_data, output_data + feature_size);
    result.score = 1.0f;
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return result;
}


std::vector<vision_core::InferIntent> EmotionRecognizer::supported_intents() const {
    if (feature_mode_) {
        return {vision_core::InferIntent::kEmbed};
    }
    return {vision_core::InferIntent::kClassify};
}

vision_core::InferResponse EmotionRecognizer::Run(const vision_core::InferRequest& request) {
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "EmotionRecognizer expects ImageInput";
        return response;
    }

    vision_core::InferResponse response;
    if (request.intent == vision_core::InferIntent::kEmbed) {
        response.results.emplace_back(infer_embedding(image_input->image));
        return response;
    }

    // Default: classification
    vision_common::ClassificationResultList task_results = classify(image_input->image);
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> EmotionRecognizer::get_capabilities() const {
    return {};
}

vision_common::ClassificationResultList EmotionRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
    // Check if outputs is valid
    if (outputs.empty()) {
        throw std::runtime_error("Postprocess: outputs is empty");
    }

    // Extract logits from outputs
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    if (dims.empty()) {
        throw std::runtime_error("Postprocess: output tensor has no dimensions");
    }

    // Calculate total number of elements (skip batch dimension)
    size_t num_classes = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        if (dims[i] <= 0) {
            throw std::runtime_error("Postprocess: invalid dimension size");
        }
        num_classes *= static_cast<size_t>(dims[i]);
    }

    if (num_classes == 0) {
        throw std::runtime_error("Postprocess: num_classes is zero");
    }

    // Get output data
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    if (output_data == nullptr) {
        throw std::runtime_error("Postprocess: output_data is null");
    }

    // Copy class scores
    std::vector<float> class_scores(output_data, output_data + num_classes);

    if (class_scores.empty()) {
        throw std::runtime_error("Postprocess: class_scores is empty");
    }

    // Find emotion class (argmax) - same as Python: np.argmax(output)
    auto max_it = std::max_element(class_scores.begin(), class_scores.end());
    if (max_it == class_scores.end()) {
        throw std::runtime_error("Postprocess: failed to find max element");
    }
    int emotion_class = static_cast<int>(max_it - class_scores.begin());
    float emotion_score = *max_it;

    // Create ClassificationResult object
    vision_common::ClassificationResult result;
    result.score = emotion_score;
    result.label = emotion_class;
    result.class_scores = class_scores;

    return vision_common::ClassificationResultList{result};
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<EmotionRecognizer> registrar("EmotionRecognizer");

}  // namespace vision_deploy

