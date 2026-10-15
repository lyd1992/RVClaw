/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arcface_recognizer.h"

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

std::unique_ptr<vision_core::BaseModel> ArcFaceRecognizer::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for ArcFaceRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for ArcFaceRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    float norm_std = vision_core::yaml_utils::getFloat(default_params, "norm_std", 127.5f);

    return std::make_unique<ArcFaceRecognizer>(model_path, num_threads, lazy_load, provider, norm_std);
}

ArcFaceRecognizer::ArcFaceRecognizer(const std::string& model_path,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider,
                                    float norm_std)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider),
        norm_std_(norm_std) {
    if (!lazy_load) {
        load_model();
    }
}

void ArcFaceRecognizer::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat ArcFaceRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    cv::Mat resized = image;
    if (image.cols != inputWidth || image.rows != inputHeight) {
        cv::resize(image, resized, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_LINEAR);
    }

    // Official ArcFace preprocess: direct resize, BGR->RGB, (x - 127.5) / norm_std.
    const float inv_std = 1.0f / norm_std_;
    return cv::dnn::blobFromImage(resized, inv_std,
                                    cv::Size(inputWidth, inputHeight),
                                    cv::Scalar(127.5, 127.5, 127.5),
                                    true, false, CV_32F);
}

vision_common::EmbeddingResult ArcFaceRecognizer::infer_embedding(const cv::Mat& image) {
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
    std::vector<float> embedding = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Create EmbeddingResult
    vision_common::EmbeddingResult result;
    result.embedding = std::move(embedding);
    result.score = 1.0f;  // Quality score (can be computed if needed)

    return result;
}


std::vector<vision_core::InferIntent> ArcFaceRecognizer::supported_intents() const {
    return {vision_core::InferIntent::kEmbed};
}

vision_core::InferResponse ArcFaceRecognizer::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kEmbed);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "ArcFaceRecognizer expects ImageInput";
        return response;
    }
    vision_core::InferResponse response;
    response.results.emplace_back(infer_embedding(image_input->image));
    return response;
}

std::vector<vision_core::ModelCapability> ArcFaceRecognizer::get_capabilities() const {
    return {};
}

std::vector<float> ArcFaceRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
    // Extract embedding from outputs
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    size_t embedding_size = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        embedding_size *= dims[i];
    }

    // Copy embedding data
    std::vector<float> embedding(output_data, output_data + embedding_size);

    // L2 normalize using common function
    embedding = vision_common::normalize_embedding(embedding);

    return embedding;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<ArcFaceRecognizer> registrar("ArcFaceRecognizer");

}  // namespace vision_deploy

