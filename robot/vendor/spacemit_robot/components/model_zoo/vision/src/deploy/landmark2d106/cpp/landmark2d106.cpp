/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "landmark2d106.h"

#include <cassert>
#include <chrono>
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

std::unique_ptr<vision_core::BaseModel> Landmark2d106::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for Landmark2d106");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for Landmark2d106");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    float input_mean = vision_core::yaml_utils::getFloat(default_params, "input_mean", 127.5f);
    float input_std = vision_core::yaml_utils::getFloat(default_params, "input_std", 128.0f);
    return std::make_unique<Landmark2d106>(
        model_path, num_threads, lazy_load, provider, input_mean, input_std);
}

Landmark2d106::Landmark2d106(const std::string& model_path,
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

void Landmark2d106::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() >= 4) {
        input_size_ = static_cast<int>(input_shape_[2]);
    }
    model_loaded_ = true;
}

cv::Mat Landmark2d106::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_size_, input_size_), 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    const float std_safe = (std::abs(input_std_) > 1e-6f) ? input_std_ : 1.0f;
    return cv::dnn::blobFromImage(rgb, 1.0 / std_safe,
                                cv::Size(input_size_, input_size_),
                                cv::Scalar(input_mean_, input_mean_, input_mean_),
                                false, false, CV_32F);
}

vision_common::PoseResultList Landmark2d106::estimate_pose(const cv::Mat& image,
                                                            float /*conf_threshold*/,
                                                            float /*iou_threshold*/) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const cv::Size face_size = image.size();
    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat input_tensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(input_tensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::PoseResultList results = postprocess(outputs, face_size);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> Landmark2d106::supported_intents() const {
    return {vision_core::InferIntent::kEstimatePose};
}

vision_core::InferResponse Landmark2d106::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kEstimatePose);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "Landmark2d106 expects ImageInput";
        return response;
    }

    vision_common::PoseResultList task_results = estimate_pose(image_input->image);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> Landmark2d106::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::PoseResultList Landmark2d106::postprocess(std::vector<Ort::Value>& outputs,
                                                        const cv::Size& face_size) {
    if (outputs.empty()) {
        throw std::runtime_error("Landmark2d106: empty outputs");
    }

    const float* landmarks_data = outputs[0].GetTensorData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_values = shape.size() >= 2 ? static_cast<int>(shape[1]) : 212;
    const int num_points = num_values / 2;

    vision_common::PoseResult det;
    det.bbox = vision_common::BoundingBox{0.0f, 0.0f,
                                        static_cast<float>(face_size.width),
                                        static_cast<float>(face_size.height)};
    det.score = 1.0f;
    det.label = 0;
    det.keypoints.reserve(static_cast<size_t>(num_points));
    for (int i = 0; i < num_points; ++i) {
        vision_common::KeyPoint kp;
        const float x_norm = landmarks_data[i * 2];
        const float y_norm = landmarks_data[i * 2 + 1];
        const float x_input = (x_norm + 1.0f) * (static_cast<float>(input_size_) * 0.5f);
        const float y_input = (y_norm + 1.0f) * (static_cast<float>(input_size_) * 0.5f);
        kp.x = x_input * static_cast<float>(face_size.width) / static_cast<float>(input_size_);
        kp.y = y_input * static_cast<float>(face_size.height) / static_cast<float>(input_size_);
        kp.visibility = 1.0f;
        det.keypoints.push_back(kp);
    }
    return {det};
}

static vision_core::ModelRegistrar<Landmark2d106> registrar("Landmark2d106");

}  // namespace vision_deploy
