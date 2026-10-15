/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov5_detector.h"

#include <algorithm>
#include <cassert>
#include <chrono>
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
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOv5Detector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv5Detector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv5Detector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv5Detector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv5Detector::YOLOv5Detector(const std::string& model_path,
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

void YOLOv5Detector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat YOLOv5Detector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int input_width = static_cast<int>(input_shape_[3]);
    int input_height = static_cast<int>(input_shape_[2]);

    cv::Mat padded = vision_common::letterbox(image, std::make_pair(input_height, input_width));
    return cv::dnn::blobFromImage(
        padded, 1.0f / 255.0f, cv::Size(input_width, input_height), cv::Scalar(0, 0, 0), true, false, CV_32F);
}

vision_common::DetectionResultList YOLOv5Detector::detect(
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


std::vector<vision_core::InferIntent> YOLOv5Detector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

vision_core::InferResponse YOLOv5Detector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kDetect);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLOv5Detector expects ImageInput";
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

std::vector<vision_core::ModelCapability> YOLOv5Detector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::DetectionResultList YOLOv5Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {
    if (outputs.empty()) {
        return {};
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);

    Ort::Value& out0 = outputs[0];
    auto info = out0.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = info.GetShape();
    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected YOLOv5 output shape (dims < 2)");
    }

    int64_t num_boxes = 0;
    int64_t features = 0;
    bool layout_cn = false;
    if (dims.size() == 3) {
        const int64_t dim1 = dims[1];
        const int64_t dim2 = dims[2];
        if ((dim1 <= 256 && dim2 > 256) || dim1 < dim2) {
            features = dim1;
            num_boxes = dim2;
            layout_cn = true;
        } else {
            num_boxes = dim1;
            features = dim2;
            layout_cn = false;
        }
    } else if (dims.size() == 2) {
        num_boxes = dims[0];
        features = dims[1];
    } else {
        features = dims.back();
        num_boxes = 1;
        for (size_t i = 0; i + 1 < dims.size(); ++i) {
            if (i == 0 && dims[0] == 1) {
                continue;
            }
            num_boxes *= dims[i];
        }
    }

    if (features < 5) {
        throw std::runtime_error("Unexpected YOLOv5 output features (< 5)");
    }

    const bool has_objectness = (features >= 85);
    const int64_t cls_start = has_objectness ? 5 : 4;
    const int64_t num_classes = features - cls_start;

    const float* data = out0.GetTensorData<float>();

    auto get_val = [&](int64_t box_idx, int64_t ch_idx) -> float {
        if (layout_cn) {
            return data[ch_idx * num_boxes + box_idx];
        }
        return data[box_idx * features + ch_idx];
    };

    bool need_sigmoid = false;
    bool need_scale_xywh = false;
    const int64_t probe_n = std::min<int64_t>(num_boxes, 200);
    float min_conf = 1e9f;
    float max_conf = -1e9f;
    float max_xywh = 0.0f;
    for (int64_t i = 0; i < probe_n; ++i) {
        for (int k = 0; k < 4; ++k) {
            max_xywh = std::max(max_xywh, std::fabs(get_val(i, k)));
        }
        if (has_objectness) {
            const float obj_raw = get_val(i, 4);
            min_conf = std::min(min_conf, obj_raw);
            max_conf = std::max(max_conf, obj_raw);
        }
        if (num_classes > 0) {
            const float cls_raw = get_val(i, cls_start);
            min_conf = std::min(min_conf, cls_raw);
            max_conf = std::max(max_conf, cls_raw);
        }
    }
    if (max_conf > 1.5f || min_conf < -0.1f) {
        need_sigmoid = true;
    }
    if (max_xywh <= 1.5f) {
        need_scale_xywh = true;
    }

    std::vector<vision_common::DetectionResult> candidates;
    candidates.reserve(static_cast<size_t>(num_boxes));

    for (int64_t i = 0; i < num_boxes; ++i) {
        float obj = 1.0f;
        if (has_objectness) {
            obj = get_val(i, 4);
            if (need_sigmoid) {
                obj = sigmoid(obj);
            }
            if (obj <= conf_threshold) {
                continue;
            }
        }

        float best_conf = 0.0f;
        int best_cls = -1;
        for (int64_t c = 0; c < num_classes; ++c) {
            float cls = get_val(i, cls_start + c);
            if (need_sigmoid) {
                cls = sigmoid(cls);
            }
            const float conf = has_objectness ? (cls * obj) : cls;
            if (conf > best_conf) {
                best_conf = conf;
                best_cls = static_cast<int>(c);
            }
        }

        if (best_cls < 0 || best_conf <= conf_threshold) {
            continue;
        }

        float xywh[4] = {get_val(i, 0), get_val(i, 1), get_val(i, 2), get_val(i, 3)};
        if (need_scale_xywh) {
            xywh[0] *= static_cast<float>(input_width);
            xywh[1] *= static_cast<float>(input_height);
            xywh[2] *= static_cast<float>(input_width);
            xywh[3] *= static_cast<float>(input_height);
        }

        float xyxy[4];
        vision_common::xywh2xyxy(xywh, xyxy);

        vision_common::DetectionResult r;
        r.bbox = vision_common::BoundingBox{xyxy[0], xyxy[1], xyxy[2], xyxy[3]};
        r.score = best_conf;
        r.label = best_cls;
        candidates.push_back(r);
    }

    if (candidates.empty()) {
        return {};
    }

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
        results.push_back(candidates[static_cast<size_t>(idx)]);
    }

    const cv::Size input_shape(input_width, input_height);
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

static vision_core::ModelRegistrar<YOLOv5Detector> registrar("YOLOv5Detector");

}  // namespace vision_deploy
