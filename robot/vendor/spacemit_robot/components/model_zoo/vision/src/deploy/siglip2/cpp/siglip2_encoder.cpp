/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "siglip2_encoder.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <variant>

#include "spacemit_ort_env.h"  // NOLINT(build/include_order)
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {
constexpr int kImageSize = 224;
constexpr int kTextMaxLen = 64;
}  // namespace

std::unique_ptr<vision_core::BaseModel> Siglip2Encoder::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for Siglip2Encoder");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for Siglip2Encoder");
    }

    std::string text_model_path = vision_core::yaml_utils::getString(default_params, "text_model_path");
    if (text_model_path.empty()) {
        throw std::runtime_error("text_model_path not found in default_params for Siglip2Encoder");
    }
    std::string tokenizer_path = vision_core::yaml_utils::getString(default_params, "tokenizer_path");
    if (tokenizer_path.empty()) {
        throw std::runtime_error("tokenizer_path not found in default_params for Siglip2Encoder");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<Siglip2Encoder>(
        model_path, text_model_path, tokenizer_path, num_threads, lazy_load, provider);
}

Siglip2Encoder::Siglip2Encoder(const std::string& vision_model_path,
                                const std::string& text_model_path,
                                const std::string& tokenizer_path,
                                int num_threads,
                                bool lazy_load,
                                const std::string& provider)
    : BaseModel(vision_model_path, lazy_load),
        text_model_path_(text_model_path),
        tokenizer_path_(tokenizer_path),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void Siglip2Encoder::init_text_session() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads_);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider_ == "SpaceMITExecutionProvider") {
        Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(opts);
        if (!status.IsOK()) {
            std::cerr << "SpaceMIT EP init failed (SigLIP2 text): " << status.GetErrorMessage()
                << std::endl;
        }
    }

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

void Siglip2Encoder::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    init_text_session();
    tokenizer_ = std::make_unique<GemmaTokenizer>(tokenizer_path_);
    model_loaded_ = true;
}

cv::Mat Siglip2Encoder::preprocess(const cv::Mat& image) {
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
    return cv::dnn::blobFromImage(resized, 1.0 / 127.5, cv::Size(input_width, input_height),
                                cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);
}

std::vector<float> Siglip2Encoder::postprocess_embedding(std::vector<Ort::Value>& outputs) {
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

std::vector<float> Siglip2Encoder::run_vision(const cv::Mat& blob) {
    std::vector<Ort::Value> outputs = run_session(blob);
    return postprocess_embedding(outputs);
}

std::vector<float> Siglip2Encoder::run_text(const std::vector<int64_t>& ids) {
    ensure_model_loaded();
    std::array<int64_t, 2> shape = {1, kTextMaxLen};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(ids.data()), ids.size(), shape.data(), shape.size());

    std::vector<Ort::Value> outputs = text_session_->Run(
        Ort::RunOptions{nullptr}, text_input_names_.data(), &input_tensor, 1,
        text_output_names_.data(), text_output_names_.size());
    return postprocess_embedding(outputs);
}

vision_common::EmbeddingResult Siglip2Encoder::infer_embedding(const cv::Mat& image) {
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

vision_common::EmbeddingResult Siglip2Encoder::encode_text(const std::string& text) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    const std::vector<int64_t> ids = tokenizer_->encode(lower, kTextMaxLen);
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

vision_core::InferResponse Siglip2Encoder::Run(const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent == vision_core::InferIntent::kEmbed) {
        const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
        if (image_input == nullptr) {
            response.ok = false;
            response.error_message = "Siglip2Encoder expects ImageInput for kEmbed";
            return response;
        }
        response.results.emplace_back(infer_embedding(image_input->image));
        return response;
    }
    if (request.intent == vision_core::InferIntent::kEmbedText) {
        const auto* text_input = std::get_if<vision_core::TextInput>(&request.input);
        if (text_input == nullptr) {
            response.ok = false;
            response.error_message = "Siglip2Encoder expects TextInput for kEmbedText";
            return response;
        }
        response.results.emplace_back(encode_text(text_input->text));
        return response;
    }
    response.ok = false;
    response.error_message = "Siglip2Encoder unsupported intent";
    return response;
}

std::vector<vision_core::InferIntent> Siglip2Encoder::supported_intents() const {
    return {vision_core::InferIntent::kEmbed, vision_core::InferIntent::kEmbedText};
}

std::vector<vision_core::ModelCapability> Siglip2Encoder::get_capabilities() const {
    return {};
}

static vision_core::ModelRegistrar<Siglip2Encoder> registrar("Siglip2Encoder");

}  // namespace vision_deploy
