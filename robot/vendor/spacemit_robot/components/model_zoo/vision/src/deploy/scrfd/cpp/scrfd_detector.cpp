/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scrfd_detector.h"

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

std::unique_ptr<vision_core::BaseModel> ScrfdDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for ScrfdDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for ScrfdDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.5f);
    float nms_threshold = vision_core::yaml_utils::getFloat(default_params, "nms_threshold", 0.4f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<ScrfdDetector>(
        model_path, conf_threshold, nms_threshold, num_threads, lazy_load, provider);
}

ScrfdDetector::ScrfdDetector(const std::string& model_path,
                            float conf_threshold,
                            float nms_threshold,
                            int num_threads,
                            bool lazy_load,
                            const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        nms_threshold_(nms_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void ScrfdDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() >= 4) {
        int h = static_cast<int>(input_shape_[2]);
        int w = static_cast<int>(input_shape_[3]);
        // Dynamic ONNX shapes are reported as -1; SCRFD det_10g uses 640x640.
        if (h <= 0) {
            h = 640;
        }
        if (w <= 0) {
            w = 640;
        }
        input_height_ = h;
        input_width_ = w;
        if (input_shape_[2] <= 0) {
            input_shape_[2] = h;
        }
        if (input_shape_[3] <= 0) {
            input_shape_[3] = w;
        }
    }
    generate_anchors();
    model_loaded_ = true;
}

void ScrfdDetector::generate_anchors() {
    const int steps[] = {8, 16, 32};
    const int min_sizes[][2] = {{16, 32}, {64, 128}, {256, 512}};

    anchors_.clear();
    for (int level = 0; level < 3; ++level) {
        const int step = steps[level];
        const int feature_h = input_height_ / step;
        const int feature_w = input_width_ / step;
        for (int i = 0; i < feature_h; ++i) {
            for (int j = 0; j < feature_w; ++j) {
                for (int k = 0; k < 2; ++k) {
                    Anchor anchor;
                    anchor.cx = static_cast<float>(j * step);
                    anchor.cy = static_cast<float>(i * step);
                    anchor.w = static_cast<float>(min_sizes[level][k]);
                    anchor.h = static_cast<float>(min_sizes[level][k]);
                    anchors_.push_back(anchor);
                }
            }
        }
    }
}

cv::Mat ScrfdDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    const float im_ratio = static_cast<float>(image.rows) / static_cast<float>(image.cols);
    const float model_ratio = static_cast<float>(input_height_) / static_cast<float>(input_width_);

    int new_width = 0;
    int new_height = 0;
    if (im_ratio > model_ratio) {
        new_height = input_height_;
        new_width = static_cast<int>(static_cast<float>(new_height) / im_ratio);
    } else {
        new_width = input_width_;
        new_height = static_cast<int>(static_cast<float>(new_width) * im_ratio);
    }

    if (input_width_ <= 0 || input_height_ <= 0 || image.cols <= 0 || image.rows <= 0) {
        throw std::runtime_error("SCRFD got invalid input size for resize");
    }
    new_width = std::max(1, std::min(new_width, input_width_));
    new_height = std::max(1, std::min(new_height, input_height_));

    last_det_scale_ = static_cast<float>(new_height) / static_cast<float>(image.rows);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    cv::Mat padded(input_height_, input_width_, image.type(), cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(0, 0, new_width, new_height)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    return cv::dnn::blobFromImage(rgb, 1.0 / 128.0,
                                cv::Size(input_width_, input_height_),
                                cv::Scalar(127.5, 127.5, 127.5),
                                false, false, CV_32F);
}

vision_common::PoseResultList ScrfdDetector::estimate_pose(const cv::Mat& image,
                                                            float conf_threshold,
                                                            float iou_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_nms = iou_threshold > 0.0f ? iou_threshold : nms_threshold_;

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
    vision_common::PoseResultList results =
        postprocess(outputs, orig_size, use_conf, use_nms, last_det_scale_);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> ScrfdDetector::supported_intents() const {
    return {vision_core::InferIntent::kEstimatePose};
}

vision_core::InferResponse ScrfdDetector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kEstimatePose);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "ScrfdDetector expects ImageInput";
        return response;
    }

    vision_common::PoseResultList task_results =
        estimate_pose(image_input->image, request.params.conf_threshold, request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> ScrfdDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::PoseResultList ScrfdDetector::postprocess(std::vector<Ort::Value>& outputs,
                                                        const cv::Size& orig_size,
                                                        float conf_threshold,
                                                        float nms_threshold,
                                                        float det_scale) {
    if (outputs.size() < 9) {
        throw std::runtime_error("ScrfdDetector expects 9 ONNX outputs");
    }

    const int feat_stride_fpn[3] = {8, 16, 32};
    const float* scores_data[3] = {
        outputs[0].GetTensorData<float>(),
        outputs[1].GetTensorData<float>(),
        outputs[2].GetTensorData<float>(),
    };
    const float* boxes_data[3] = {
        outputs[3].GetTensorData<float>(),
        outputs[4].GetTensorData<float>(),
        outputs[5].GetTensorData<float>(),
    };
    const float* landmarks_data[3] = {
        outputs[6].GetTensorData<float>(),
        outputs[7].GetTensorData<float>(),
        outputs[8].GetTensorData<float>(),
    };

    // Score tensors may be [N], [N,1], or [1,N]; use element count so both
    // InsightFace ([N,1]) and batched ([1,N]) layouts map to anchor count.
    const int num_anchors[3] = {
        static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetElementCount()),
        static_cast<int>(outputs[1].GetTensorTypeAndShapeInfo().GetElementCount()),
        static_cast<int>(outputs[2].GetTensorTypeAndShapeInfo().GetElementCount()),
    };

    const float scale_x = 1.0f / det_scale;
    const float scale_y = 1.0f / det_scale;
    const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
    const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));

    std::vector<vision_common::PoseResult> boxes;
    int anchor_idx = 0;
    for (int level = 0; level < 3; ++level) {
        for (int i = 0; i < num_anchors[level]; ++i) {
            const float score = scores_data[level][i];
            if (score < conf_threshold) {
                ++anchor_idx;
                continue;
            }

            const Anchor& anchor = anchors_[static_cast<size_t>(anchor_idx)];
            const float stride = static_cast<float>(feat_stride_fpn[level]);
            const float dx = boxes_data[level][i * 4 + 0] * stride;
            const float dy = boxes_data[level][i * 4 + 1] * stride;
            const float dw = boxes_data[level][i * 4 + 2] * stride;
            const float dh = boxes_data[level][i * 4 + 3] * stride;

            float x1 = (anchor.cx - dx) * scale_x;
            float y1 = (anchor.cy - dy) * scale_y;
            float x2 = (anchor.cx + dw) * scale_x;
            float y2 = (anchor.cy + dh) * scale_y;

            vision_common::PoseResult det;
            det.bbox.x1 = std::clamp(x1, 0.0f, max_x);
            det.bbox.y1 = std::clamp(y1, 0.0f, max_y);
            det.bbox.x2 = std::clamp(x2, 0.0f, max_x);
            det.bbox.y2 = std::clamp(y2, 0.0f, max_y);
            det.score = score;
            det.label = 0;
            det.keypoints.reserve(5);
            for (int j = 0; j < 5; ++j) {
                const float lx = landmarks_data[level][i * 10 + j * 2 + 0] * stride;
                const float ly = landmarks_data[level][i * 10 + j * 2 + 1] * stride;
                vision_common::KeyPoint kp;
                kp.x = std::clamp((anchor.cx + lx) * scale_x, 0.0f, max_x);
                kp.y = std::clamp((anchor.cy + ly) * scale_y, 0.0f, max_y);
                kp.visibility = 1.0f;
                det.keypoints.push_back(kp);
            }
            boxes.push_back(det);
            ++anchor_idx;
        }
    }

    return nms(boxes, nms_threshold);
}

std::vector<vision_common::PoseResult> ScrfdDetector::nms(std::vector<vision_common::PoseResult>& boxes,
                                                            float threshold) const {
    std::sort(boxes.begin(), boxes.end(),
        [](const vision_common::PoseResult& a, const vision_common::PoseResult& b) {
            return a.score > b.score;
        });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<vision_common::PoseResult> result;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        result.push_back(boxes[i]);
        const float area1 = (boxes[i].bbox.x2 - boxes[i].bbox.x1) * (boxes[i].bbox.y2 - boxes[i].bbox.y1);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            const float xx1 = std::max(boxes[i].bbox.x1, boxes[j].bbox.x1);
            const float yy1 = std::max(boxes[i].bbox.y1, boxes[j].bbox.y1);
            const float xx2 = std::min(boxes[i].bbox.x2, boxes[j].bbox.x2);
            const float yy2 = std::min(boxes[i].bbox.y2, boxes[j].bbox.y2);
            const float w = std::max(0.0f, xx2 - xx1);
            const float h = std::max(0.0f, yy2 - yy1);
            const float inter = w * h;
            const float area2 = (boxes[j].bbox.x2 - boxes[j].bbox.x1) * (boxes[j].bbox.y2 - boxes[j].bbox.y1);
            const float iou = inter / (area1 + area2 - inter + 1e-6f);
            if (iou > threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

static vision_core::ModelRegistrar<ScrfdDetector> registrar("ScrfdDetector");

}  // namespace vision_deploy
