/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov5_gesture_detector.h"

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

std::unique_ptr<vision_core::BaseModel> YOLOv5GestureDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv5GestureDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv5GestureDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.4f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.5f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv5GestureDetector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv5GestureDetector::YOLOv5GestureDetector(const std::string& model_path,
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

void YOLOv5GestureDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat YOLOv5GestureDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    const int inputWidth = static_cast<int>(input_shape_[3]);   // width
    const int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // letterbox() in common converts BGR->RGB and pads to requested shape
    cv::Mat padded = vision_common::letterbox(image, std::make_pair(inputHeight, inputWidth));
    return cv::dnn::blobFromImage(
        padded,
        1.0 / 255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0),
        true,   // swapRB (already RGB, but keep consistent with other deploys)
        false,  // crop
        CV_32F);
}

vision_common::DetectionResultList YOLOv5GestureDetector::detect(
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
    cv::Mat inputTensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(inputTensor);
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


std::vector<vision_core::InferIntent> YOLOv5GestureDetector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

vision_core::InferResponse YOLOv5GestureDetector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kDetect);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLOv5GestureDetector expects ImageInput";
        return response;
    }

    vision_common::DetectionResultList task_results = detect(image_input->image, request.params.conf_threshold, request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> YOLOv5GestureDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::DetectionResultList YOLOv5GestureDetector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {

    if (outputs.empty()) {
        return {};
    }

    // Model input shape (letterbox output) for clipping before NMS
    const int inputHeight = static_cast<int>(input_shape_[2]);
    const int inputWidth = static_cast<int>(input_shape_[3]);

    // YOLOv5 ONNX typical output: (1, N, 5+nc) with xywh, obj, cls...
    Ort::Value& out0 = outputs[0];
    auto info = out0.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = info.GetShape();
    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected YOLOv5 output shape (dims < 2)");
    }

    // Flatten to [num_boxes, features]
    int64_t num_boxes = 0;
    int64_t features = 0;
    if (dims.size() == 3) {
        // (bs, N, F)
        num_boxes = dims[1];
        features = dims[2];
    } else if (dims.size() == 2) {
        // (N, F)
        num_boxes = dims[0];
        features = dims[1];
    } else {
        // Some exports may provide extra dims; keep it conservative.
        // Try interpret last dim as features, and the product of others (excluding batch) as num_boxes.
        features = dims.back();
        num_boxes = 1;
        for (size_t i = 0; i + 1 < dims.size(); ++i) {
            // skip batch dim if present
            if (i == 0 && dims[0] == 1) continue;
            num_boxes *= dims[i];
        }
    }

    if (features < 6) {
        throw std::runtime_error("Unexpected YOLOv5 output features (< 6)");
    }
    const int64_t num_classes = features - 5;

    const float* data = out0.GetTensorMutableData<float>();

    vision_common::DetectionResultList candidates;
    candidates.reserve(static_cast<size_t>(num_boxes));

    // Iterate proposals
    for (int64_t i = 0; i < num_boxes; ++i) {
        const float* p = data + i * features;

        const float obj = p[4];
        if (obj <= conf_threshold) {
            continue;
        }

        // find best class score (already multiplied by obj in python)
        float best_conf = 0.0f;
        int best_cls = -1;
        for (int64_t c = 0; c < num_classes; ++c) {
            float conf = p[5 + c] * obj;
            if (conf > best_conf) {
                best_conf = conf;
                best_cls = static_cast<int>(c);
            }
        }

        if (best_cls < 0 || best_conf <= conf_threshold) {
            continue;
        }

        // xywh -> xyxy
        float xywh[4] = {p[0], p[1], p[2], p[3]};
        float xyxy[4];
        vision_common::xywh2xyxy(xywh, xyxy);

        // Normalize + clip BEFORE NMS to avoid final clip making boxes collapse into overlaps.
        float x1 = std::min(xyxy[0], xyxy[2]);
        float y1 = std::min(xyxy[1], xyxy[3]);
        float x2 = std::max(xyxy[0], xyxy[2]);
        float y2 = std::max(xyxy[1], xyxy[3]);

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(inputWidth)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(inputWidth)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(inputHeight)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(inputHeight)));

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        vision_common::DetectionResult r;
        r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        r.score = best_conf;
        r.label = best_cls;
        candidates.push_back(r);
    }

    if (candidates.empty()) {
        return {};
    }

    // NMS (gesture detection uses class-agnostic NMS to avoid overlapped boxes across classes)
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates.size());
    scores.reserve(candidates.size());
    for (const auto& cand : candidates) {
        boxes.emplace_back(cand.bbox.x1, cand.bbox.y1,
            cand.bbox.x2 - cand.bbox.x1, cand.bbox.y2 - cand.bbox.y1);
        scores.push_back(cand.score);
    }
    std::vector<int> keep_indices = vision_common::nms(boxes, scores, iou_threshold);

    vision_common::DetectionResultList results;
    results.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        results.push_back(candidates[idx]);
    }

    // Scale back to original image size (letterbox-aware)
    const cv::Size input_shape(inputWidth, inputHeight);
    for (auto& r : results) {
        float coords[4] = {r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2};
        vision_common::scale_coords(input_shape, coords, orig_size);
        r.bbox.x1 = coords[0];
        r.bbox.y1 = coords[1];
        r.bbox.x2 = coords[2];
        r.bbox.y2 = coords[3];
    }

    return results;
}

// Self-registration: YAML class "...YOLOv5_GestureDetector" -> class_name "YOLOv5_GestureDetector"
static vision_core::ModelRegistrar<YOLOv5GestureDetector> registrar("YOLOv5_GestureDetector");

}  // namespace vision_deploy

