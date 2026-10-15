/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "adaface_recognizer.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <variant>

#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> AdaFaceRecognizer::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for AdaFaceRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for AdaFaceRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<AdaFaceRecognizer>(model_path, num_threads, lazy_load, provider);
}

AdaFaceRecognizer::AdaFaceRecognizer(const std::string& model_path,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void AdaFaceRecognizer::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat AdaFaceRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    const int crop = std::min(image.cols, image.rows);
    const int crop_x = (image.cols - crop) / 2;
    const int crop_y = (image.rows - crop) / 2;
    cv::Mat cropped = image(cv::Rect(crop_x, crop_y, crop, crop));

    int input_width = static_cast<int>(input_shape_[3]);
    int input_height = static_cast<int>(input_shape_[2]);
    if (input_width <= 0 || input_height <= 0) {
        input_width = 112;
        input_height = 112;
    }

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);

    return cv::dnn::blobFromImage(resized, 1.0 / 127.5, cv::Size(input_width, input_height),
                                cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);
}

vision_common::EmbeddingResult AdaFaceRecognizer::infer_embedding(const cv::Mat& image) {
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
    std::vector<float> embedding = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    vision_common::EmbeddingResult result;
    result.embedding = std::move(embedding);
    result.score = 1.0f;
    return result;
}

std::vector<vision_core::InferIntent> AdaFaceRecognizer::supported_intents() const {
    return {vision_core::InferIntent::kEmbed};
}

vision_core::InferResponse AdaFaceRecognizer::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kEmbed);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "AdaFaceRecognizer expects ImageInput";
        return response;
    }
    vision_core::InferResponse response;
    response.results.emplace_back(infer_embedding(image_input->image));
    return response;
}

std::vector<vision_core::ModelCapability> AdaFaceRecognizer::get_capabilities() const {
    return {};
}

std::vector<float> AdaFaceRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    size_t embedding_size = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        embedding_size *= static_cast<size_t>(dims[i]);
    }

    std::vector<float> embedding(output_data, output_data + embedding_size);
    return vision_common::normalize_embedding(embedding);
}

static vision_core::ModelRegistrar<AdaFaceRecognizer> registrar("AdaFaceRecognizer");

}  // namespace vision_deploy
