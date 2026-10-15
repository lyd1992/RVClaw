/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_pose_detector.h"

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

std::unique_ptr<vision_core::BaseModel> YOLOv8PoseDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8PoseDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8PoseDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    float point_confidence_threshold =
        vision_core::yaml_utils::getFloat(default_params, "point_confidence_threshold", 0.2f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8PoseDetector>(
        model_path,
        conf_threshold,
        iou_threshold,
        point_confidence_threshold,
        num_threads,
        lazy_load,
        provider);
}

YOLOv8PoseDetector::YOLOv8PoseDetector(const std::string& model_path,
                                        float conf_threshold,
                                        float iou_threshold,
                                        float point_confidence_threshold,
                                        int num_threads,
                                        bool lazy_load,
                                        const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        point_confidence_threshold_(point_confidence_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8PoseDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat YOLOv8PoseDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    // Use common letterbox function (similar to yolov8_detector.cpp)
    cv::Mat padded = vision_common::letterbox(image,
                                            std::make_pair(inputHeight, inputWidth),
                                            cv::Scalar(0, 0, 0));

    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}



vision_common::PoseResultList YOLOv8PoseDetector::estimate_pose(const cv::Mat& image,
                                                                float conf_threshold,
                                                                float iou_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_iou = iou_threshold > 0.0f ? iou_threshold : iou_threshold_;

    cv::Size orig_size = image.size();

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
    vision_common::PoseResultList results = postprocess(outputs, orig_size, use_conf, use_iou);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> YOLOv8PoseDetector::supported_intents() const {
    return {vision_core::InferIntent::kEstimatePose};
}

vision_core::InferResponse YOLOv8PoseDetector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kEstimatePose);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLOv8PoseDetector expects ImageInput";
        return response;
    }

    vision_common::PoseResultList task_results = estimate_pose(image_input->image, request.params.conf_threshold, request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> YOLOv8PoseDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::PoseResultList YOLOv8PoseDetector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {

    // Get output data
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();

    int offset = static_cast<int>(dets_dims[1]);  // 56
    int anchors = static_cast<int>(dets_dims[2]);  // 2100

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    float ratio = std::min(
        static_cast<float>(inputWidth) / static_cast<float>(orig_size.width),
        static_cast<float>(inputHeight) / static_cast<float>(orig_size.height));
    int unpad_w = static_cast<int>(std::round(orig_size.width * ratio));
    int unpad_h = static_cast<int>(std::round(orig_size.height * ratio));

    float dw = (inputWidth - unpad_w) / 2.0f;
    float dh = (inputHeight - unpad_h) / 2.0f;

    // First pass: extract all detections with confidence > threshold
    vision_common::PoseResultList objects;
    for (int j = 0; j < anchors; ++j) {
        if (output_data[4 * anchors + j] > conf_threshold) {
            // Decode box
            float half_width = output_data[2 * anchors + j] / 2;
            float half_height = output_data[3 * anchors + j] / 2;
            float x1 = (output_data[j] - half_width - dw) / ratio;
            float y1 = (output_data[anchors + j] - half_height - dh) / ratio;
            float x2 = (output_data[j] + half_width - dw) / ratio;
            float y2 = (output_data[anchors + j] + half_height - dh) / ratio;

            // Clamp to image bounds
            const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
            const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));
            x1 = std::clamp(x1, 0.0f, max_x);
            y1 = std::clamp(y1, 0.0f, max_y);
            x2 = std::clamp(x2, 0.0f, max_x);
            y2 = std::clamp(y2, 0.0f, max_y);
            if (x2 - x1 < 10.0f || y2 - y1 < 10.0f) {
                continue;
            }

            vision_common::PoseResult det;
            det.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
            det.score = output_data[4 * anchors + j];
            det.label = 0;  // Person class for pose detection

            // Decode keypoints and count visible ones
            det.keypoints.reserve(17);
            int visible_count = 0;
            for (int k = 0; k < 17; ++k) {
                vision_common::KeyPoint kp;
                kp.x = (output_data[(5 + k * 3) * anchors + j] - dw) / ratio;
                kp.y = (output_data[(5 + k * 3 + 1) * anchors + j] - dh) / ratio;
                kp.visibility = output_data[(5 + k * 3 + 2) * anchors + j];
                det.keypoints.push_back(kp);
                if (kp.visibility > point_confidence_threshold_) {
                    visible_count++;
                }
            }

            // Filter detections with too few visible keypoints (likely false positives)
            if (visible_count < 3) {
                continue;
            }

            objects.push_back(std::move(det));
        }
    }

    // NMS preserves keypoints since they're already in the PoseResult objects
    return nms(objects, iou_threshold);
}

float YOLOv8PoseDetector::calculate_iou(const vision_common::PoseResult& det1, const vision_common::PoseResult& det2) {
    // Use BoundingBox iou method
    return vision_common::iou(det1.bbox, det2.bbox);
}

vision_common::PoseResultList YOLOv8PoseDetector::nms(
    const vision_common::PoseResultList& dets,
    float iou_threshold) {
    if (dets.empty()) {
        return vision_common::PoseResultList();
    }

    // Convert Result to cv::Rect2f and scores for common nms function
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (const auto& det : dets) {
        boxes.push_back(cv::Rect2f(det.bbox.x1, det.bbox.y1, det.bbox.x2 - det.bbox.x1, det.bbox.y2 - det.bbox.y1));
        scores.push_back(det.score);
    }

    // Use common nms function
    std::vector<int> keep_indices = vision_common::nms(boxes, scores, iou_threshold);

    // Convert back to Result vector
    vision_common::PoseResultList final_dets;
    final_dets.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        final_dets.push_back(dets[idx]);
    }

    return final_dets;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8PoseDetector> registrar("YOLOv8PoseDetector");

}  // namespace vision_deploy

