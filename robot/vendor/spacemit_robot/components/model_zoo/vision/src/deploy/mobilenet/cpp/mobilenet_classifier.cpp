/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mobilenet_classifier.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>
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

// Read a 3-element float sequence from YAML (e.g. mean/std). Returns false if absent/invalid.
bool readScalar3(const YAML::Node& node, const std::string& key, cv::Scalar* out) {
    if (!node || !node[key] || !node[key].IsSequence() || node[key].size() < 3) {
        return false;
    }
    *out = cv::Scalar(node[key][0].as<float>(),
        node[key][1].as<float>(),
        node[key][2].as<float>());
    return true;
}

// Read a 2-element int sequence from YAML (e.g. resize_size).
bool readSize(const YAML::Node& node, const std::string& key, cv::Size* out) {
    if (!node || !node[key] || !node[key].IsSequence() || node[key].size() < 2) {
        return false;
    }
    *out = cv::Size(node[key][0].as<int>(), node[key][1].as<int>());
    return true;
}

// Map a human-readable interpolation name (from YAML) to an OpenCV flag.
int parseInterpolation(const std::string& name, int default_flag) {
    if (name == "bilinear" || name == "linear") return cv::INTER_LINEAR;
    if (name == "bicubic" || name == "cubic") return cv::INTER_CUBIC;
    if (name == "nearest") return cv::INTER_NEAREST;
    if (name == "area") return cv::INTER_AREA;
    return default_flag;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> MobileNetV1Classifier::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for MobileNetV1Classifier");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for MobileNetV1Classifier");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    cv::Size resize_size(256, 256);
    readSize(default_params, "resize_size", &resize_size);

    cv::Scalar mean(0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f);
    cv::Scalar std(0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f);
    cv::Scalar mean_norm, std_norm;
    if (readScalar3(default_params, "mean", &mean_norm)) {
        mean = mean_norm * 255.0;
    }
    if (readScalar3(default_params, "std", &std_norm)) {
        std = std_norm * 255.0;
    }

    bool center_crop = vision_core::yaml_utils::getBool(default_params, "center_crop", true);

    int interpolation = parseInterpolation(
        vision_core::yaml_utils::getString(default_params, "interpolation", "bilinear"),
        cv::INTER_LINEAR);

    return std::make_unique<MobileNetV1Classifier>(model_path, num_threads, lazy_load, provider,
        resize_size, mean, std, center_crop, interpolation);
}

MobileNetV1Classifier::MobileNetV1Classifier(
    const std::string& model_path,
    int num_threads,
    bool lazy_load,
    const std::string& provider,
    const cv::Size& resize_size,
    const cv::Scalar& mean,
    const cv::Scalar& std,
    bool center_crop,
    int interpolation)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        resize_size_(resize_size),
        mean_(mean),
        std_(std),
        center_crop_(center_crop),
        interpolation_(interpolation),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void MobileNetV1Classifier::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat MobileNetV1Classifier::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    cv::Mat blob = vision_common::preprocess_classification(
        image,
        std::make_pair(inputHeight, inputWidth),
        mean_,
        std_,
        resize_size_,
        center_crop_,
        interpolation_);

    return blob;
}

vision_common::ClassificationResultList MobileNetV1Classifier::classify(const cv::Mat& image) {
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

    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::ClassificationResultList results = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}

std::vector<vision_core::InferIntent> MobileNetV1Classifier::supported_intents() const {
    return {vision_core::InferIntent::kClassify};
}

vision_core::InferResponse MobileNetV1Classifier::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kClassify);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "MobileNetV1Classifier expects ImageInput";
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

std::vector<vision_core::ModelCapability> MobileNetV1Classifier::get_capabilities() const {
    return {};
}

vision_common::ClassificationResultList MobileNetV1Classifier::postprocess(std::vector<Ort::Value>& outputs) {
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    size_t num_classes = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        num_classes *= dims[i];
    }

    // Drop leading background class (index 0) for the 1001-class TF-Slim export,
    // so labels align with the 1000-class ImageNet file.
    size_t offset = (num_classes == 1001) ? 1 : 0;
    std::vector<float> class_scores(output_data + offset, output_data + num_classes);
    return vision_common::build_classification_top_k(std::move(class_scores), 5);
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<MobileNetV1Classifier> registrar("MobileNetV1Classifier");

}  // namespace vision_deploy

