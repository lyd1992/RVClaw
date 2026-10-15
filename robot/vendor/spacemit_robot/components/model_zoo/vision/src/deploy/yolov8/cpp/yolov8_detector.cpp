/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_detector.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

constexpr int kBoxChannels = 4;

void collect_candidates_cn(
    const float* data,
    int64_t num_anchors,
    int64_t num_classes,
    float conf_threshold,
    std::vector<cv::Rect2f>& boxes,
    std::vector<float>& scores,
    std::vector<int>& labels) {
    const float* cx = data;
    const float* cy = data + num_anchors;
    const float* cw = data + 2 * num_anchors;
    const float* ch = data + 3 * num_anchors;

    std::vector<float> max_scores(static_cast<size_t>(num_anchors), -1.0f);
    std::vector<int> max_classes(static_cast<size_t>(num_anchors), -1);

    for (int64_t c = 0; c < num_classes; ++c) {
        const float* cls_row = data + (kBoxChannels + c) * num_anchors;
        for (int64_t i = 0; i < num_anchors; ++i) {
            const float s = cls_row[i];
            if (s > max_scores[static_cast<size_t>(i)]) {
                max_scores[static_cast<size_t>(i)] = s;
                max_classes[static_cast<size_t>(i)] = static_cast<int>(c);
            }
        }
    }

    boxes.clear();
    scores.clear();
    labels.clear();
    boxes.reserve(512);
    scores.reserve(512);
    labels.reserve(512);

    for (int64_t i = 0; i < num_anchors; ++i) {
        const size_t idx = static_cast<size_t>(i);
        if (max_scores[idx] < conf_threshold) {
            continue;
        }

        const float w = cw[i];
        const float h = ch[i];
        const float x1 = cx[i] - w * 0.5f;
        const float y1 = cy[i] - h * 0.5f;
        boxes.emplace_back(x1, y1, w, h);
        scores.push_back(max_scores[idx]);
        labels.push_back(max_classes[idx]);
    }
}

void collect_candidates_nc(
    const float* data,
    int64_t num_anchors,
    int64_t features,
    int64_t num_classes,
    float conf_threshold,
    std::vector<cv::Rect2f>& boxes,
    std::vector<float>& scores,
    std::vector<int>& labels) {
    boxes.clear();
    scores.clear();
    labels.clear();
    boxes.reserve(512);
    scores.reserve(512);
    labels.reserve(512);

    for (int64_t i = 0; i < num_anchors; ++i) {
        const float* row = data + i * features;

        float best_conf = 0.0f;
        int best_cls = -1;
        for (int64_t c = 0; c < num_classes; ++c) {
            const float s = row[kBoxChannels + c];
            if (s > best_conf) {
                best_conf = s;
                best_cls = static_cast<int>(c);
            }
        }
        if (best_cls < 0 || best_conf < conf_threshold) {
            continue;
        }

        const float w = row[2];
        const float h = row[3];
        const float x1 = row[0] - w * 0.5f;
        const float y1 = row[1] - h * 0.5f;
        boxes.emplace_back(x1, y1, w, h);
        scores.push_back(best_conf);
        labels.push_back(best_cls);
    }
}

vision_common::DetectionResultList nms_and_scale(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& labels,
    int num_classes,
    float iou_threshold,
    const cv::Size& input_shape,
    const cv::Size& orig_size) {
    if (boxes.empty()) {
        return {};
    }

    const float gain = std::min(
        static_cast<float>(input_shape.height) / static_cast<float>(orig_size.height),
        static_cast<float>(input_shape.width) / static_cast<float>(orig_size.width));
    const float pad_w = (static_cast<float>(input_shape.width) - orig_size.width * gain) / 2.0f;
    const float pad_h = (static_cast<float>(input_shape.height) - orig_size.height * gain) / 2.0f;
    const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
    const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));

    std::vector<std::vector<size_t>> by_class(static_cast<size_t>(num_classes));
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (labels[i] >= 0 && labels[i] < num_classes) {
            by_class[static_cast<size_t>(labels[i])].push_back(i);
        }
    }

    vision_common::DetectionResultList results;
    std::vector<cv::Rect2f> rects;
    std::vector<float> cls_scores;
    std::vector<int> keep;
    rects.reserve(boxes.size());
    cls_scores.reserve(boxes.size());

    for (const auto& cls_indices : by_class) {
        if (cls_indices.empty()) {
            continue;
        }

        rects.clear();
        cls_scores.clear();
        for (size_t idx : cls_indices) {
            rects.push_back(boxes[idx]);
            cls_scores.push_back(scores[idx]);
        }

        keep.clear();
        // Use the project's float NMS (same as the seg single-output path) to keep
        // sub-pixel box precision — rounding to integer cv::Rect skews IoU for small
        // boxes. Candidates were already conf-filtered in collect_candidates_*, so NMS
        // only applies the IoU criterion here.
        keep = vision_common::nms(rects, cls_scores, iou_threshold);

        for (int k : keep) {
            const size_t idx = cls_indices[static_cast<size_t>(k)];
            const cv::Rect2f& b = boxes[idx];
            float x1 = (b.x - pad_w) / gain;
            float y1 = (b.y - pad_h) / gain;
            float x2 = (b.x + b.width - pad_w) / gain;
            float y2 = (b.y + b.height - pad_h) / gain;

            x1 = std::clamp(x1, 0.0f, max_x);
            y1 = std::clamp(y1, 0.0f, max_y);
            x2 = std::clamp(x2, 0.0f, max_x);
            y2 = std::clamp(y2, 0.0f, max_y);

            vision_common::DetectionResult r;
            r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
            r.score = scores[idx];
            r.label = labels[idx];
            results.push_back(r);
        }
    }

    return results;
}

vision_common::DetectionResultList postprocess_single_output(
    Ort::Value& output,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold,
    int input_width,
    int input_height) {
    auto info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = info.GetShape();
    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected YOLOv8 output shape (dims < 2)");
    }

    int64_t num_anchors = 0;
    int64_t features = 0;
    bool layout_cn = false;
    if (dims.size() == 3) {
        // cn layout = [features, anchors]: features sit on the smaller trailing dim
        // (channels << anchors). Pure size comparison, matching the Python detector and
        // yolov8_seg's resolve_seg_layout. Equal dims are ambiguous — refuse rather than
        // silently mis-parsing box/score channels.
        const int64_t dim1 = dims[1];
        const int64_t dim2 = dims[2];
        if (dim1 == dim2) {
            throw std::runtime_error(
                "Unexpected YOLOv8 output shape (channels == anchors), "
                "cannot distinguish features from anchors");
        }
        if (dim1 < dim2) {
            features = dim1;
            num_anchors = dim2;
            layout_cn = true;
        } else {
            num_anchors = dim1;
            features = dim2;
            layout_cn = false;
        }
    } else if (dims.size() == 2) {
        num_anchors = dims[0];
        features = dims[1];
    } else {
        throw std::runtime_error("Unexpected YOLOv8 output rank");
    }

    if (features < 5) {
        throw std::runtime_error("Unexpected YOLOv8 output features (< 5)");
    }

    const int num_classes = static_cast<int>(features - kBoxChannels);
    const float* data = output.GetTensorData<float>();

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> labels;

    // Ultralytics exports box xywh in input-pixel scale, so no normalization rescale.
    if (layout_cn) {
        collect_candidates_cn(
            data, num_anchors, num_classes, conf_threshold,
            boxes, scores, labels);
    } else {
        collect_candidates_nc(
            data, num_anchors, features, num_classes, conf_threshold,
            boxes, scores, labels);
    }

    return nms_and_scale(
        boxes, scores, labels, num_classes, iou_threshold,
        cv::Size(input_width, input_height), orig_size);
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOv8Detector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8Detector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8Detector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8Detector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv8Detector::YOLOv8Detector(const std::string& model_path,
                                float conf_threshold,
                                float iou_threshold,
                                int num_threads,
                                bool lazy_load,
                                const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        num_classes_(0),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8Detector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (output_num_ >= 2) {
        Ort::TypeInfo score_type_info = session_->GetOutputTypeInfo(1);
        auto score_tensor_info = score_type_info.GetTensorTypeAndShapeInfo();
        auto score_dims = score_tensor_info.GetShape();
        if (score_dims.size() >= 2 && score_dims[1] > 0) {
            num_classes_ = static_cast<int>(score_dims[1]);
        }
    }
    if (num_classes_ <= 0) {
        num_classes_ = 80;
    }
    model_loaded_ = true;
}

cv::Mat YOLOv8Detector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);


    // Use common letterbox function (similar to Python implementation)
    cv::Mat padded = vision_common::letterbox(
        image,
        std::make_pair(inputHeight, inputWidth));


    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}



vision_common::DetectionResultList YOLOv8Detector::detect(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_iou = iou_threshold > 0.0f ? iou_threshold : iou_threshold_;

    cv::Size orig_size = image.size();
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

std::vector<vision_core::InferIntent> YOLOv8Detector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

vision_core::InferResponse YOLOv8Detector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kDetect);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLOv8Detector expects ImageInput";
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

std::vector<vision_core::ModelCapability> YOLOv8Detector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

void YOLOv8Detector::get_dets(
    const cv::Size& orig_size,
    const float* boxes,
    const float* scores,
    const float* score_sum,
    const std::vector<int64_t>& dims,
    int tensor_width,
    int tensor_height,
    float conf_threshold,
    vision_common::DetectionResultList& objects) {
    int grid_w = static_cast<int>(dims[2]);
    int grid_h = static_cast<int>(dims[3]);
    int anchors_per_branch = grid_w * grid_h;
    float scale_w = static_cast<float>(tensor_width) / static_cast<float>(grid_w);
    float scale_h = static_cast<float>(tensor_height) / static_cast<float>(grid_h);

    int orig_height = orig_size.height;
    int orig_width = orig_size.width;
    float scale2orign = std::min(
        static_cast<float>(tensor_height) / static_cast<float>(orig_width),
        static_cast<float>(tensor_width) / static_cast<float>(orig_height));
    int pad_h = static_cast<int>((tensor_width - orig_height * scale2orign) / 2);
    int pad_w = static_cast<int>((tensor_height - orig_width * scale2orign) / 2);

    for (int anchor_idx = 0; anchor_idx < anchors_per_branch; anchor_idx++) {
        if (score_sum[anchor_idx] < conf_threshold) {
            continue;
        }

        float max_score = -1.0f;
        int classId = -1;
        for (int class_idx = 0; class_idx < num_classes_; class_idx++) {
            size_t score_offset = class_idx * anchors_per_branch + anchor_idx;
            if ((scores[score_offset] > conf_threshold) && (scores[score_offset] > max_score)) {
                max_score = scores[score_offset];
                classId = class_idx;
            }
        }

        if (classId >= 0) {
            auto [x1, y1, x2, y2] = vision_common::dfl_decode(boxes, anchor_idx,
                anchors_per_branch, grid_h, scale_w, scale_h, scale2orign, pad_w, pad_h);

            vision_common::DetectionResult result;
            result.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
            result.label = classId;
            result.score = max_score;
            objects.push_back(result);
        }
    }
}

vision_common::DetectionResultList YOLOv8Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {
    if (outputs.empty()) {
        return {};
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);

    // outputs.size() == output_num_ (session output node count, fixed by the model graph):
    // 1 output  -> Ultralytics single-output export; otherwise -> multi-branch DFL heads.
    if (outputs.size() == 1) {
        return postprocess_single_output(
            outputs[0], orig_size, conf_threshold, iou_threshold,
            input_width, input_height);
    }

    vision_common::DetectionResultList objects;
    const size_t output_num = outputs.size();
    constexpr int kTensorsPerBranch = 3;
    const int inputWidth = static_cast<int>(input_shape_[3]);
    const int inputHeight = static_cast<int>(input_shape_[2]);

    for (int i = 0; i < static_cast<int>(output_num / kTensorsPerBranch); i++) {
        const float* boxes = outputs[i * kTensorsPerBranch].GetTensorData<float>();
        const float* branch_scores = outputs[i * kTensorsPerBranch + 1].GetTensorData<float>();
        const float* score_sum = outputs[i * kTensorsPerBranch + 2].GetTensorData<float>();
        const std::vector<int64_t> dims =
            outputs[i * kTensorsPerBranch].GetTensorTypeAndShapeInfo().GetShape();

        get_dets(orig_size, boxes, branch_scores, score_sum, dims,
            inputHeight, inputWidth, conf_threshold, objects);
    }

    return vision_common::multi_class_nms(objects, iou_threshold);
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8Detector> registrar("YOLOv8Detector");

}  // namespace vision_deploy

