/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mobileclip2_encoder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <variant>

#include "spacemit_ort_env.h"  // NOLINT(build/include_order)
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {
constexpr int kImageSize = 256;
constexpr int kTextMaxLen = 77;
const cv::Scalar kClipMean(0.48145466 * 255.0, 0.4578275 * 255.0, 0.40821073 * 255.0);
const cv::Scalar kClipStd(0.26862954 * 255.0, 0.26130258 * 255.0, 0.27577711 * 255.0);
}  // namespace

std::unique_ptr<vision_core::BaseModel> Mobileclip2Encoder::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for Mobileclip2Encoder");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for Mobileclip2Encoder");
    }

    std::string text_model_path = vision_core::yaml_utils::getString(default_params, "text_model_path");
    if (text_model_path.empty()) {
        throw std::runtime_error("text_model_path not found in default_params for Mobileclip2Encoder");
    }
    std::string bpe_merges_path = vision_core::yaml_utils::getString(default_params, "bpe_merges_path");
    if (bpe_merges_path.empty()) {
        throw std::runtime_error("bpe_merges_path not found in default_params for Mobileclip2Encoder");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<Mobileclip2Encoder>(
        model_path, text_model_path, bpe_merges_path, num_threads, lazy_load, provider);
}

Mobileclip2Encoder::Mobileclip2Encoder(const std::string& image_model_path,
                                        const std::string& text_model_path,
                                        const std::string& bpe_merges_path,
                                        int num_threads,
                                        bool lazy_load,
                                        const std::string& provider)
    : BaseModel(image_model_path, lazy_load),
        text_model_path_(text_model_path),
        bpe_merges_path_(bpe_merges_path),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void Mobileclip2Encoder::init_text_session() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads_);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    text_session_ = std::make_unique<Ort::Session>(
        vision_core::shared_ort_env(), text_model_path_.c_str(), opts);

    const size_t num_inputs = text_session_->GetInputCount();
    text_input_names_.resize(num_inputs);
    text_input_names_str_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = text_session_->GetInputNameAllocated(i, allocator_);
        text_input_names_str_[i] = name.get();
        text_input_names_[i] = text_input_names_str_[i].c_str();
    }

    const size_t num_outputs = text_session_->GetOutputCount();
    text_output_names_.resize(num_outputs);
    text_output_names_str_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = text_session_->GetOutputNameAllocated(i, allocator_);
        text_output_names_str_[i] = name.get();
        text_output_names_[i] = text_output_names_str_[i].c_str();
    }
}

void Mobileclip2Encoder::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    init_text_session();
    tokenizer_ = std::make_unique<CLIPTokenizer>(bpe_merges_path_);
    sequence_len_ = kTextMaxLen;
    model_loaded_ = true;
}

cv::Mat Mobileclip2Encoder::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    int input_width = static_cast<int>(input_shape_[3]);
    int input_height = static_cast<int>(input_shape_[2]);
    if (input_width <= 0 || input_height <= 0) {
        input_width = kImageSize;
        input_height = kImageSize;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width, input_height), 0, 0, cv::INTER_LINEAR);
    // OpenCV images are BGR; CLIP mean/std expect RGB (same as reference stbi_load RGB).
    cv::Mat rgb;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } else if (resized.channels() == 4) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
    }

    // Per-channel (x - mean) / std into NCHW blob. Avoid merge→split round-trip;
    // blobFromImage cannot express different std per channel in one call.
    cv::Mat channels[3];
    cv::split(rgb, channels);
    const int plane = input_height * input_width;
    cv::Mat blob(1, 3 * plane, CV_32F);
    for (int c = 0; c < 3; ++c) {
        cv::Mat plane_f;
        const double inv_std = 1.0 / kClipStd[c];
        channels[c].convertTo(plane_f, CV_32F, inv_std, -kClipMean[c] * inv_std);
        plane_f.reshape(1, 1).copyTo(blob.colRange(c * plane, (c + 1) * plane));
    }
    return blob.reshape(1, {1, 3, input_height, input_width});
}

std::vector<float> Mobileclip2Encoder::postprocess_embedding(std::vector<Ort::Value>& outputs) {
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> dims = tensor_info.GetShape();

    size_t embedding_size = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        embedding_size *= static_cast<size_t>(dims[i]);
    }

    std::vector<float> embedding(output_data, output_data + embedding_size);
    return vision_common::normalize_embedding(embedding);
}

std::vector<float> Mobileclip2Encoder::run_vision(const cv::Mat& blob) {
    std::vector<Ort::Value> outputs = run_session(blob);
    return postprocess_embedding(outputs);
}

std::vector<float> Mobileclip2Encoder::run_text(const std::vector<int64_t>& ids) {
    ensure_model_loaded();
    std::array<int64_t, 2> shape = {1, sequence_len_};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(ids.data()), ids.size(), shape.data(), shape.size());

    std::vector<Ort::Value> outputs = text_session_->Run(
        Ort::RunOptions{nullptr}, text_input_names_.data(), &input_tensor, 1,
        text_output_names_.data(), text_output_names_.size());
    return postprocess_embedding(outputs);
}

vision_common::EmbeddingResult Mobileclip2Encoder::infer_embedding(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat blob = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<float> embedding = run_vision(blob);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    set_runtime_postprocess_ms(0.0);
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t_infer1 - t0).count());

    vision_common::EmbeddingResult result;
    result.embedding = std::move(embedding);
    result.score = 1.0f;
    return result;
}

vision_common::EmbeddingResult Mobileclip2Encoder::encode_text(const std::string& text) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    const std::vector<int32_t> ids32 = tokenizer_->tokenize(text, sequence_len_)[0];
    std::vector<int64_t> ids(ids32.begin(), ids32.end());
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<float> embedding = run_text(ids);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    set_runtime_postprocess_ms(0.0);
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t_infer1 - t0).count());

    vision_common::EmbeddingResult result;
    result.embedding = std::move(embedding);
    result.score = 1.0f;
    return result;
}

vision_core::InferResponse Mobileclip2Encoder::Run(const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent == vision_core::InferIntent::kEmbed) {
        const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
        if (image_input == nullptr) {
            response.ok = false;
            response.error_message = "Mobileclip2Encoder expects ImageInput for kEmbed";
            return response;
        }
        response.results.emplace_back(infer_embedding(image_input->image));
        return response;
    }
    if (request.intent == vision_core::InferIntent::kEmbedText) {
        const auto* text_input = std::get_if<vision_core::TextInput>(&request.input);
        if (text_input == nullptr) {
            response.ok = false;
            response.error_message = "Mobileclip2Encoder expects TextInput for kEmbedText";
            return response;
        }
        response.results.emplace_back(encode_text(text_input->text));
        return response;
    }
    response.ok = false;
    response.error_message = "Mobileclip2Encoder unsupported intent";
    return response;
}

std::vector<vision_core::InferIntent> Mobileclip2Encoder::supported_intents() const {
    return {vision_core::InferIntent::kEmbed, vision_core::InferIntent::kEmbedText};
}

std::vector<vision_core::ModelCapability> Mobileclip2Encoder::get_capabilities() const {
    return {};
}

static vision_core::ModelRegistrar<Mobileclip2Encoder> registrar("Mobileclip2Encoder");

}  // namespace vision_deploy
