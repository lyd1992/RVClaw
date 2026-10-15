/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo26_detector.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/dnn.hpp>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLO26Detector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLO26Detector");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLO26Detector");
    }
    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    return std::make_unique<YOLO26Detector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLO26Detector::YOLO26Detector(
    const std::string& model_path,
    float conf_threshold,
    float iou_threshold,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLO26Detector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat YOLO26Detector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();
    int input_width = static_cast<int>(input_shape_[3]);
    int input_height = static_cast<int>(input_shape_[2]);
    cv::Mat padded = vision_common::letterbox(image, std::make_pair(input_height, input_width));
    return cv::dnn::blobFromImage(
        padded, 1.0 / 255.0f, cv::Size(input_width, input_height),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}

vision_common::DetectionResultList YOLO26Detector::detect(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_iou = iou_threshold > 0.0f ? iou_threshold : iou_threshold_;

    const cv::Size orig_size = image.size();
    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat input_tensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(input_tensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::DetectionResultList results = postprocess(outputs, orig_size, use_conf, use_iou);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

vision_core::InferResponse YOLO26Detector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kDetect);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLO26Detector expects ImageInput";
        return response;
    }
    vision_common::DetectionResultList detections =
        detect(image_input->image, request.params.conf_threshold, request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(detections.size());
    for (auto& detection : detections) {
        response.results.emplace_back(std::move(detection));
    }
    return response;
}

std::vector<vision_core::InferIntent> YOLO26Detector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

std::vector<vision_core::ModelCapability> YOLO26Detector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::DetectionResultList YOLO26Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {
    (void)iou_threshold;
    if (outputs.empty()) {
        return {};
    }
    if (outputs.size() != 1) {
        throw std::runtime_error("YOLO26 expects single output tensor");
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);
    Ort::Value& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = info.GetShape();
    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected YOLO26 output shape (dims < 2)");
    }

    int64_t num_anchors = 0;
    int64_t features = 0;
    if (dims.size() == 3) {
        const int64_t dim1 = dims[1];
        const int64_t dim2 = dims[2];
        if (dim1 == dim2) {
            throw std::runtime_error("Unexpected YOLO26 output shape (channels == anchors)");
        }
        // only support [1, N, 6]
        if (dim2 != 6) {
            throw std::runtime_error("YOLO26 only supports [N,6] e2e output");
        }
        num_anchors = dim1;
        features = dim2;
    } else if (dims.size() == 2) {
        // support [N, 6]
        if (dims[1] != 6) {
            throw std::runtime_error("YOLO26 only supports [N,6] e2e output");
        }
        num_anchors = dims[0];
        features = dims[1];
    } else {
        throw std::runtime_error("Unexpected YOLO26 output rank");
    }

    const float* data = out.GetTensorData<float>();
    if (features != 6) {
        throw std::runtime_error("YOLO26 only supports [N,6] e2e output");
    }

    const float gain = std::min(
        static_cast<float>(input_height) / static_cast<float>(orig_size.height),
        static_cast<float>(input_width) / static_cast<float>(orig_size.width));
    const float pad_w = (static_cast<float>(input_width) - orig_size.width * gain) / 2.0f;
    const float pad_h = (static_cast<float>(input_height) - orig_size.height * gain) / 2.0f;
    const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
    const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));

    vision_common::DetectionResultList results;
    results.reserve(static_cast<size_t>(num_anchors));
    for (int64_t i = 0; i < num_anchors; ++i) {
        const float* row = data + i * features;
        const float score = row[4];
        if (score < conf_threshold) {
            continue;
        }

        int cls = static_cast<int>(std::round(row[5]));
        cls = std::max(0, cls);

        float x1 = (row[0] - pad_w) / gain;
        float y1 = (row[1] - pad_h) / gain;
        float x2 = (row[2] - pad_w) / gain;
        float y2 = (row[3] - pad_h) / gain;
        x1 = std::clamp(x1, 0.0f, max_x);
        y1 = std::clamp(y1, 0.0f, max_y);
        x2 = std::clamp(x2, 0.0f, max_x);
        y2 = std::clamp(y2, 0.0f, max_y);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        vision_common::DetectionResult r;
        r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        r.score = score;
        r.label = cls;
        results.push_back(r);
    }
    return results;
}

static vision_core::ModelRegistrar<YOLO26Detector> registrar("YOLO26Detector");

}  // namespace vision_deploy
