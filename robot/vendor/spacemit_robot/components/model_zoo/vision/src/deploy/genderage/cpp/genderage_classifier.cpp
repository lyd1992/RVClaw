/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "genderage_classifier.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>
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

std::unique_ptr<vision_core::BaseModel> GenderAgeClassifier::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for GenderAgeClassifier");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for GenderAgeClassifier");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    float input_mean = vision_core::yaml_utils::getFloat(default_params, "input_mean", 127.5f);
    float input_std = vision_core::yaml_utils::getFloat(default_params, "input_std", 128.0f);
    return std::make_unique<GenderAgeClassifier>(
        model_path, num_threads, lazy_load, provider, input_mean, input_std);
}

GenderAgeClassifier::GenderAgeClassifier(const std::string& model_path,
                                            int num_threads,
                                            bool lazy_load,
                                            const std::string& provider,
                                            float input_mean,
                                            float input_std)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider),
        input_mean_(input_mean),
        input_std_(input_std) {
    if (!lazy_load) {
        load_model();
    }
}

void GenderAgeClassifier::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat GenderAgeClassifier::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    cv::Mat resized;
    if (image.cols != target_size_.width || image.rows != target_size_.height) {
        cv::resize(image, resized, target_size_, 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    const float std_safe = (std::abs(input_std_) > 1e-6f) ? input_std_ : 1.0f;
    return cv::dnn::blobFromImage(rgb, 1.0 / std_safe,
                                target_size_,
                                cv::Scalar(input_mean_, input_mean_, input_mean_),
                                false, false, CV_32F);
}

vision_common::ClassificationResultList GenderAgeClassifier::classify(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat input_tensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(input_tensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::ClassificationResultList results = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> GenderAgeClassifier::supported_intents() const {
    return {vision_core::InferIntent::kClassify};
}

vision_core::InferResponse GenderAgeClassifier::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kClassify);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "GenderAgeClassifier expects ImageInput";
        return response;
    }

    vision_common::ClassificationResultList task_results = classify(image_input->image);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> GenderAgeClassifier::get_capabilities() const {
    return {};
}

vision_common::ClassificationResultList GenderAgeClassifier::postprocess(std::vector<Ort::Value>& outputs) {
    vision_common::ClassificationResult result;
    result.label = 0;
    result.score = 0.0f;
    result.class_scores = {0.0f, 0.0f, 0.0f};

    if (outputs.empty()) {
        return {result};
    }

    if (outputs.size() == 1) {
        const float* data = outputs[0].GetTensorData<float>();
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() >= 2 && shape[1] >= 3) {
            const float female = data[0];
            const float male = data[1];
            const float age_norm = data[2];
            result.label = (male > female) ? 1 : 0;
            result.score = std::max(female, male);
            const int age = std::clamp(static_cast<int>(std::lround(age_norm * 100.0f)), 1, 100);
            result.class_scores = {female, male, static_cast<float>(age)};
        }
        return {result};
    }

    if (outputs.size() >= 2 && outputs[0].IsTensor()) {
        const float* gender_data = outputs[0].GetTensorData<float>();
        auto gender_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        float female = 0.0f;
        float male = 0.0f;
        if (gender_shape.size() > 0) {
            const int num_gender = gender_shape.size() > 1 ? static_cast<int>(gender_shape[1]) : 2;
            if (num_gender >= 2) {
                female = gender_data[0];
                male = gender_data[1];
                result.label = (male > female) ? 1 : 0;
                result.score = std::max(female, male);
            }
        }

        int age = 0;
        if (outputs[1].IsTensor()) {
            const float* age_data = outputs[1].GetTensorData<float>();
            auto age_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            if (age_shape.size() > 1) {
                const int num_age_classes = static_cast<int>(age_shape[1]);
                int max_idx = 0;
                float max_val = age_data[0];
                for (int i = 1; i < num_age_classes; ++i) {
                    if (age_data[i] > max_val) {
                        max_val = age_data[i];
                        max_idx = i;
                    }
                }
                age = max_idx;
            }
        }
        result.class_scores = {female, male, static_cast<float>(age)};
    }

    return {result};
}

static vision_core::ModelRegistrar<GenderAgeClassifier> registrar("GenderAgeClassifier");

}  // namespace vision_deploy
