/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo_world_detector.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/dnn.hpp>

#include "image_processing.h"
#include "nms.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

// Parse prompts from YAML default_params. Accepts either a sequence
// (prompts: [a, b, c]) or a comma-separated scalar (prompts: "a,b,c").
std::vector<std::string> parsePrompts(const YAML::Node& default_params) {
    std::vector<std::string> out;
    if (!default_params || !default_params["prompts"]) {
        return out;
    }
    const YAML::Node& node = default_params["prompts"];
    if (node.IsSequence()) {
        for (const auto& item : node) {
            std::string s = item.as<std::string>();
            if (!s.empty()) out.push_back(s);
        }
    } else if (node.IsScalar()) {
        std::stringstream ss(node.as<std::string>());
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim spaces
            size_t a = tok.find_first_not_of(" \t");
            size_t b = tok.find_last_not_of(" \t");
            if (a != std::string::npos) out.push_back(tok.substr(a, b - a + 1));
        }
    }
    return out;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YoloWorldDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YoloWorldDetector");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YoloWorldDetector");
    }

    std::string clip_model_path = vision_core::yaml_utils::getString(default_params, "clip_model_path");
    if (clip_model_path.empty()) {
        throw std::runtime_error("clip_model_path not found in default_params for YoloWorldDetector");
    }
    std::string bpe_merges_path = vision_core::yaml_utils::getString(default_params, "bpe_merges_path");
    if (bpe_merges_path.empty()) {
        throw std::runtime_error("bpe_merges_path not found in default_params for YoloWorldDetector");
    }

    std::vector<std::string> default_prompts = parsePrompts(default_params);
    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YoloWorldDetector>(
        model_path, clip_model_path, bpe_merges_path, default_prompts,
        conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YoloWorldDetector::YoloWorldDetector(
    const std::string& model_path,
    const std::string& clip_model_path,
    const std::string& bpe_merges_path,
    const std::vector<std::string>& default_prompts,
    float conf_threshold,
    float iou_threshold,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        clip_model_path_(clip_model_path),
        bpe_merges_path_(bpe_merges_path),
        default_prompts_(default_prompts),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YoloWorldDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    // Detector session (image + text feature inputs).
    init_session(num_threads_, provider_);

    // Identify which input is the 4D image and which is the 3D text tensor.
    for (size_t i = 0; i < input_node_names_.size(); ++i) {
        Ort::TypeInfo info = session_->GetInputTypeInfo(i);
        std::vector<int64_t> dims = info.GetTensorTypeAndShapeInfo().GetShape();
        if (dims.size() == 4) {
            input_image_dims_ = dims;
            image_input_index_ = static_cast<int>(i);
        } else if (dims.size() == 3) {
            input_text_dims_ = dims;
            text_input_index_ = static_cast<int>(i);
        }
    }
    if (input_image_dims_.size() != 4 || input_text_dims_.size() != 3) {
        throw std::runtime_error(
            "YoloWorldDetector expects a 4D image input and a 3D text input");
    }

    // CLIP text encoder (CPU; one-off per-vocabulary cost).
    clip_ = std::make_unique<CLIP>(clip_model_path_, bpe_merges_path_, num_threads_);

    model_loaded_ = true;
}

void YoloWorldDetector::ensure_text_features(const std::vector<std::string>& prompts) {
    ensure_model_loaded();

    // Effective vocabulary: explicit prompts override; empty prompts always
    // mean "use the yaml default vocabulary" (not the last-used prompts), so
    // behavior isn't sticky across calls. The cache below still avoids
    // recomputation when the resolved vocabulary is unchanged.
    const std::vector<std::string>& vocab = !prompts.empty() ? prompts : default_prompts_;

    if (vocab.empty()) {
        throw std::runtime_error(
            "YoloWorldDetector: no prompts provided and no default vocabulary in config");
    }

    // Cache hit: same vocabulary already encoded -> reuse (no CLIP call).
    if (text_features_ready_ && vocab == cached_prompts_) {
        return;
    }

    const int model_num_classes = static_cast<int>(input_text_dims_[1]);
    const int feature_dim = static_cast<int>(input_text_dims_[2]);

    const int num_classes = static_cast<int>(vocab.size());
    if (num_classes > model_num_classes) {
        std::cerr << "YoloWorldDetector: " << num_classes << " prompts exceed model capacity "
            << model_num_classes << "; extra prompts are ignored" << std::endl;
    }

    // Encode via CLIP and pack into a {model_num_classes, feature_dim} tensor,
    // zero-padded to the model's fixed class capacity.
    std::vector<std::vector<float>> feats = clip_->encode(vocab);
    text_feature_data_.assign(static_cast<size_t>(model_num_classes) * feature_dim, 0.0f);
    const int copy_classes = std::min(num_classes, model_num_classes);
    for (int i = 0; i < copy_classes; ++i) {
        const std::vector<float>& v = feats[static_cast<size_t>(i)];
        const int n = std::min(static_cast<int>(v.size()), feature_dim);
        std::copy(v.begin(), v.begin() + n,
            text_feature_data_.begin() + static_cast<size_t>(i) * feature_dim);
    }

    cached_prompts_ = vocab;
    active_labels_.assign(vocab.begin(), vocab.begin() + copy_classes);
    text_features_ready_ = true;
}

void YoloWorldDetector::preprocess(const cv::Mat& image, cv::Mat& blob) {
    const int dst_h = static_cast<int>(input_image_dims_[2]);
    const int dst_w = static_cast<int>(input_image_dims_[3]);

    // Store the exact scale/offset that vision_common::letterbox applies, so
    // postprocess inverts the same transform (no drift). Mirrors letterbox():
    // r = min(...), new = round(orig*r), pad = (dst-new)/2, left/top = round(pad-0.1).
    const float r = std::min(static_cast<float>(dst_h) / static_cast<float>(image.rows),
        static_cast<float>(dst_w) / static_cast<float>(image.cols));
    const int new_w = static_cast<int>(std::round(image.cols * r));
    const int new_h = static_cast<int>(std::round(image.rows * r));
    letterbox_scale_ = r;
    letterbox_ox_ = static_cast<int>(std::round((dst_w - new_w) / 2.0f - 0.1));
    letterbox_oy_ = static_cast<int>(std::round((dst_h - new_h) / 2.0f - 0.1));

    // Reuse the shared letterbox (pad 114, same as the repo's YOLO family).
    cv::Mat padded = vision_common::letterbox(image, std::make_pair(dst_h, dst_w));

    blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(dst_w, dst_h),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}

vision_common::DetectionResultList YoloWorldDetector::postprocess(
    const float* output, int offset, int anchors,
    const cv::Size& orig_size, float conf_threshold, float iou_threshold) {
    // Invert exactly the transform applied in preprocess (shared scale/offsets),
    // so forward/inverse can't drift.
    const float ratio = letterbox_scale_;
    const float dw = static_cast<float>(letterbox_ox_);
    const float dh = static_cast<float>(letterbox_oy_);

    // Collect candidates (CN layout: output[c * anchors + j]) as DetectionResult,
    // then reuse the shared per-class NMS (vision_common::multi_class_nms).
    vision_common::DetectionResultList candidates;
    const int active_classes = static_cast<int>(active_labels_.size());

    for (int j = 0; j < anchors; ++j) {
        float max_score = -1.0f;
        int max_index = -1;
        for (int prob = 4; prob < offset; ++prob) {
            const float s = output[prob * anchors + j];
            if (s > max_score) {
                max_score = s;
                max_index = prob;
            }
        }
        if (max_score <= conf_threshold) {
            continue;
        }
        const int class_id = max_index - 4;
        if (class_id < 0 || class_id >= active_classes) {
            continue;  // padded/unused class slot
        }
        const float half_w = output[2 * anchors + j] / 2.0f;
        const float half_h = output[3 * anchors + j] / 2.0f;
        float x1 = (output[j] - half_w - dw) / ratio;
        float y1 = (output[anchors + j] - half_h - dh) / ratio;
        float x2 = (output[j] + half_w - dw) / ratio;
        float y2 = (output[anchors + j] + half_h - dh) / ratio;
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min(static_cast<float>(orig_size.width), x2);
        y2 = std::min(static_cast<float>(orig_size.height), y2);

        vision_common::DetectionResult det;
        det.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        det.score = max_score;
        det.label = class_id;
        candidates.push_back(det);
    }

    return vision_common::multi_class_nms(candidates, iou_threshold);
}

vision_common::DetectionResultList YoloWorldDetector::detect(
    const cv::Mat& image, float conf_threshold, float iou_threshold) {
    return detect_with_prompts(image, {}, conf_threshold, iou_threshold);
}

vision_common::DetectionResultList YoloWorldDetector::detect_with_prompts(
    const cv::Mat& image, const std::vector<std::string>& prompts,
    float conf_threshold, float iou_threshold) {
    if (image.empty()) {
        throw std::runtime_error("YoloWorldDetector: input image is empty");
    }
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_iou = iou_threshold > 0.0f ? iou_threshold : iou_threshold_;

    // Lazy text-feature cache (no CLIP call on steady-state / same prompts).
    ensure_text_features(prompts);

    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat image_blob;
    preprocess(image, image_blob);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Build the two input tensors in the model's declared input order.
    const std::vector<int64_t> image_shape = {1, 3, input_image_dims_[2], input_image_dims_[3]};
    const std::vector<int64_t> text_shape = {1, input_text_dims_[1], input_text_dims_[2]};

    // Ort::Value is move-only, so build then place each tensor at its declared
    // input index via move-assignment into a default-initialized vector.
    Ort::Value image_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, image_blob.ptr<float>(), image_blob.total(),
        image_shape.data(), image_shape.size());
    Ort::Value text_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, text_feature_data_.data(), text_feature_data_.size(),
        text_shape.data(), text_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(nullptr);
    inputs.emplace_back(nullptr);
    inputs[static_cast<size_t>(image_input_index_)] = std::move(image_tensor);
    inputs[static_cast<size_t>(text_input_index_)] = std::move(text_tensor);

    const auto t_inf0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_node_names_.data(), inputs.data(), inputs.size(),
        output_node_names_.data(), output_node_names_.size());
    const auto t_inf1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    const float* dets = outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> dets_dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int offset = static_cast<int>(dets_dims[1]);   // 4 + num_classes
    const int anchors = static_cast<int>(dets_dims[2]);
    vision_common::DetectionResultList results =
        postprocess(dets, offset, anchors, image.size(), use_conf, use_iou);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> YoloWorldDetector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

std::vector<vision_core::ModelCapability> YoloWorldDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_core::InferResponse YoloWorldDetector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kDetect);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YoloWorldDetector expects ImageInput";
        return response;
    }

    vision_common::DetectionResultList task_results = detect_with_prompts(
        image_input->image, request.params.prompts,
        request.params.conf_threshold, request.params.iou_threshold);

    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

static vision_core::ModelRegistrar<YoloWorldDetector> registrar("YoloWorldDetector");

}  // namespace vision_deploy
