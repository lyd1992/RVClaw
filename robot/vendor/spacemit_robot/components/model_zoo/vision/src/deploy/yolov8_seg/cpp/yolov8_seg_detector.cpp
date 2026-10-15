/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_seg_detector.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
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

// Resolve the single-output seg layout for output0 dims [dim1, dim2].
// cn layout = [features, anchors] (Ultralytics default); features sit on the smaller
// trailing dim (channels << anchors). Shared by load_model() and the postprocess path
// so both agree on which axis is features vs anchors. Equal dims are ambiguous.
struct SegLayout {
    bool cn;            // true if [features, anchors]
    int64_t features;
    int64_t anchors;
};

SegLayout resolve_seg_layout(int64_t dim1, int64_t dim2) {
    if (dim1 == dim2) {
        throw std::runtime_error(
            "YOLOv8-Seg: ambiguous output0 shape (channels == anchors), "
            "cannot distinguish features from anchors");
    }
    if (dim1 < dim2) {
        return SegLayout{true, dim1, dim2};
    }
    return SegLayout{false, dim2, dim1};
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOv8SegDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8SegDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8SegDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8SegDetector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv8SegDetector::YOLOv8SegDetector(const std::string& model_path,
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
        proto_channels_(0),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8SegDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);

    if (output_num_ == 2) {
        // Ultralytics export: output0 = [1, 4 + num_classes + proto_channels, num_anchors],
        // proto = [1, proto_channels, mask_h, mask_w].
        Ort::TypeInfo proto_type_info = session_->GetOutputTypeInfo(1);
        auto proto_dims = proto_type_info.GetTensorTypeAndShapeInfo().GetShape();
        // Expect 4-D [1, C, H, W]; reject e.g. a batch-less [C, H, W] where dims[1]
        // would be a spatial size rather than the channel count.
        if (proto_dims.size() != 4) {
            throw std::runtime_error(
                "YOLOv8-Seg: unexpected proto output rank (expected 4: [1, C, H, W])");
        }
        if (proto_dims[1] > 0) {
            proto_channels_ = static_cast<int>(proto_dims[1]);
        }
        if (proto_channels_ <= 0) {
            throw std::runtime_error(
                "YOLOv8-Seg: cannot read proto_channels from proto output shape");
        }

        Ort::TypeInfo out0_type_info = session_->GetOutputTypeInfo(0);
        auto out0_dims = out0_type_info.GetTensorTypeAndShapeInfo().GetShape();
        if (out0_dims.size() != 3) {
            throw std::runtime_error("YOLOv8-Seg: unexpected output0 rank (expected 3)");
        }
        // Shared layout resolver throws on ambiguous (features == anchors) shapes rather
        // than silently mis-deriving num_classes and shifting the mask-coefficient channels.
        const SegLayout layout = resolve_seg_layout(out0_dims[1], out0_dims[2]);
        const int derived = static_cast<int>(layout.features) - 4 - proto_channels_;
        if (derived <= 0) {
            throw std::runtime_error(
                "YOLOv8-Seg: derived num_classes <= 0 from output0 shape");
        }
        num_classes_ = derived;
    } else {
        // DFL multi-branch export:
        // [box0, score0, sum0, box1, score1, sum1, box2, score2, sum2, seg0, seg1, seg2, proto]
        // score output shape: [1, num_classes, h, w]
        if (output_num_ >= 2) {
            Ort::TypeInfo score_type_info = session_->GetOutputTypeInfo(1);
            auto score_dims = score_type_info.GetTensorTypeAndShapeInfo().GetShape();
            if (score_dims.size() >= 2 && score_dims[1] > 0) {
                num_classes_ = static_cast<int>(score_dims[1]);
            }
        }
        // proto output shape: [1, proto_channels, mask_h, mask_w]
        if (output_num_ >= 13) {
            Ort::TypeInfo proto_type_info = session_->GetOutputTypeInfo(12);
            auto proto_dims = proto_type_info.GetTensorTypeAndShapeInfo().GetShape();
            if (proto_dims.size() >= 2 && proto_dims[1] > 0) {
                proto_channels_ = static_cast<int>(proto_dims[1]);
            }
        }
    }

    // Default values if not found
    if (num_classes_ <= 0) {
        num_classes_ = 80;  // Default COCO classes
    }
    if (proto_channels_ <= 0) {
        proto_channels_ = 32;  // Default proto channels
    }

    model_loaded_ = true;
}

cv::Mat YOLOv8SegDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // Use letterbox preprocessing (same as Python)
    cv::Mat padded = vision_common::letterbox(
        image,
        std::make_pair(inputHeight, inputWidth));

    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}

vision_common::SegmentationResultList YOLOv8SegDetector::segment(
    const cv::Mat& image,
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
    vision_common::SegmentationResultList results = postprocess(outputs, orig_size, use_conf, use_iou);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> YOLOv8SegDetector::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

vision_core::InferResponse YOLOv8SegDetector::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kSegment);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YOLOv8SegDetector expects ImageInput";
        return response;
    }

    vision_common::SegmentationResultList task_results = segment(image_input->image, request.params.conf_threshold, request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> YOLOv8SegDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::SegmentationResultList YOLOv8SegDetector::postprocess(std::vector<Ort::Value>& outputs,
                                                                const cv::Size& orig_size,
                                                                float conf_threshold,
                                                                float iou_threshold) {
    // outputs.size() == output_num_ (session output node count, fixed by the model graph):
    // 2 outputs -> Ultralytics single-output seg export; otherwise -> multi-branch DFL heads.
    // Both paths must agree with load_model(), which keys num_classes_/proto_channels_ off
    // the same output_num_.
    if (outputs.size() == 2) {
        return postprocess_single_output_seg(outputs, orig_size, conf_threshold, iou_threshold);
    }

    // DFL path indexes outputs[0..8] (box/score/score_sum x3), outputs[9..11] (seg coeffs)
    // and outputs[12] (proto), so it requires the full 13-output export. Guard the lower
    // bound up front to fail clearly instead of reading out of bounds on a malformed model.
    if (outputs.size() < 13) {
        throw std::runtime_error(
            "YOLOv8-Seg: expected 2 (single-output) or 13 (DFL multi-branch) outputs, got "
            + std::to_string(outputs.size()));
    }

    // Temporary structure to hold results with mask coefficients before NMS
    struct TempSegResult {
        vision_common::SegmentationResult result;
        std::vector<float> mask_coeffs;
    };

    std::vector<TempSegResult> temp_objects;

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    // Process 3 branches (8x, 16x, 32x downsampling)
    for (int i = 0; i < 3; i++) {
        const float* boxes = outputs[i * 3].GetTensorMutableData<float>();
        const float* scores = outputs[i * 3 + 1].GetTensorMutableData<float>();
        const float* score_sum = outputs[i * 3 + 2].GetTensorMutableData<float>();
        const float* seg_part = outputs[9 + i].GetTensorMutableData<float>();
        std::vector<int64_t> dims = outputs[i * 3].GetTensorTypeAndShapeInfo().GetShape();

        int grid_w = static_cast<int>(dims[2]);
        int grid_h = static_cast<int>(dims[3]);
        int anchors_per_branch = grid_w * grid_h;
        float scale_w = static_cast<float>(inputWidth) / static_cast<float>(grid_w);
        float scale_h = static_cast<float>(inputHeight) / static_cast<float>(grid_h);

        int orig_height = orig_size.height;
        int orig_width = orig_size.width;
        float scale2orign = std::min(
            static_cast<float>(inputHeight) / static_cast<float>(orig_width),
            static_cast<float>(inputWidth) / static_cast<float>(orig_height));
        int pad_h = static_cast<int>((inputWidth - orig_height * scale2orign) / 2);
        int pad_w = static_cast<int>((inputHeight - orig_width * scale2orign) / 2);

        for (int anchor_idx = 0; anchor_idx < anchors_per_branch; anchor_idx++) {
            if (score_sum[anchor_idx] < conf_threshold) {
                continue;
            }

            float max_score = -1.0f;
            int classId = -1;
            for (int class_idx = 0; class_idx < num_classes_; class_idx++) {
                size_t score_offset = class_idx * anchors_per_branch + anchor_idx;
                if (scores[score_offset] > conf_threshold && scores[score_offset] > max_score) {
                    max_score = scores[score_offset];
                    classId = class_idx;
                }
            }

            if (classId >= 0) {
                auto [x1, y1, x2, y2] = vision_common::dfl_decode(boxes, anchor_idx, anchors_per_branch,
                    grid_w, scale_w, scale_h, scale2orign, pad_w, pad_h);

                TempSegResult temp;
                temp.result.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
                temp.result.label = classId;
                temp.result.score = max_score;
                temp.result.mask = nullptr;

                // Store mask coefficients
                temp.mask_coeffs.reserve(proto_channels_);
                for (int k = 0; k < proto_channels_; k++) {
                    temp.mask_coeffs.push_back(seg_part[k * anchors_per_branch + anchor_idx]);
                }

                temp_objects.push_back(temp);
            }
        }
    }

    // Extract just the results for NMS
    vision_common::SegmentationResultList objects;
    objects.reserve(temp_objects.size());
    for (const auto& temp : temp_objects) {
        objects.push_back(temp.result);
    }

    // Apply multi-class NMS
    vision_common::SegmentationResultList results = vision_common::multi_class_nms(objects, iou_threshold);

    // Process masks if we have results and proto output
    if (!results.empty() && outputs.size() >= 13) {
        const float* output_proto = outputs[12].GetTensorMutableData<float>();
        std::vector<int64_t> proto_dims = outputs[12].GetTensorTypeAndShapeInfo().GetShape();

        // Match NMS results back to temp_objects to get mask coefficients
        std::vector<std::vector<float>> mask_coeffs_list;
        mask_coeffs_list.reserve(results.size());

        for (const auto& res : results) {
            // Find matching temp object by bbox and score
            for (const auto& temp : temp_objects) {
                if (std::abs(temp.result.bbox.x1 - res.bbox.x1) < 0.01f &&
                    std::abs(temp.result.bbox.y1 - res.bbox.y1) < 0.01f &&
                    std::abs(temp.result.score - res.score) < 0.001f &&
                    temp.result.label == res.label) {
                    mask_coeffs_list.push_back(temp.mask_coeffs);
                    break;
                }
            }
        }

        // Process masks
        std::vector<std::shared_ptr<cv::Mat>> masks = _process_masks(
            output_proto, proto_dims, mask_coeffs_list, results, orig_size);

        // Assign masks
        for (size_t i = 0; i < results.size() && i < masks.size(); i++) {
            results[i].mask = masks[i];
        }
    }

    return results;
}


vision_common::SegmentationResultList YOLOv8SegDetector::postprocess_single_output_seg(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {
    Ort::Value& out0 = outputs[0];
    std::vector<int64_t> dims = out0.GetTensorTypeAndShapeInfo().GetShape();
    if (dims.size() != 3) {
        throw std::runtime_error("Unexpected YOLOv8-Seg single output rank (expected 3)");
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);

    std::vector<int64_t> proto_dims = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    // Expect 4-D [1, C, H, W] (same assumption as load_model); reject a batch-less
    // [C, H, W] where dims[1] would be a spatial size rather than the channel count.
    if (proto_dims.size() != 4 || proto_dims[1] <= 0) {
        throw std::runtime_error(
            "YOLOv8-Seg proto output shape invalid (expected 4: [1, C, H, W])");
    }
    const int num_masks = static_cast<int>(proto_dims[1]);

    // Same resolver as load_model() so feature/anchor axes are chosen identically.
    const SegLayout layout = resolve_seg_layout(dims[1], dims[2]);
    const bool layout_cn = layout.cn;
    const int64_t features = layout.features;
    const int64_t num_anchors = layout.anchors;

    const int num_classes = static_cast<int>(features) - 4 - num_masks;
    if (num_classes <= 0) {
        throw std::runtime_error("YOLOv8-Seg single output feature size too small");
    }

    const float* data = out0.GetTensorData<float>();
    auto at = [&](int c, int64_t i) -> float {
        return layout_cn ? data[static_cast<size_t>(c) * num_anchors + i]
            : data[static_cast<size_t>(i) * features + c];
    };

    // Collect candidates: max class score over [4, 4+num_classes), keep mask coeffs aligned.
    // Ultralytics exports box xywh in input-pixel scale, so no normalization rescale.
    std::vector<cv::Rect2f> cand_rects;   // xywh in input space, for NMS
    std::vector<float> cand_scores;
    std::vector<int> cand_labels;
    std::vector<std::vector<float>> cand_coeffs;
    cand_rects.reserve(256);
    cand_scores.reserve(256);
    cand_labels.reserve(256);
    cand_coeffs.reserve(256);

    for (int64_t i = 0; i < num_anchors; ++i) {
        float best = -1.0f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            const float s = at(4 + c, i);
            if (s > best) {
                best = s;
                best_cls = c;
            }
        }
        if (best_cls < 0 || best < conf_threshold) {
            continue;
        }

        const float w = at(2, i);
        const float h = at(3, i);
        const float x = at(0, i) - w * 0.5f;
        const float y = at(1, i) - h * 0.5f;
        cand_rects.emplace_back(x, y, w, h);
        cand_scores.push_back(best);
        cand_labels.push_back(best_cls);

        std::vector<float> coeffs;
        coeffs.reserve(num_masks);
        for (int k = 0; k < num_masks; ++k) {
            coeffs.push_back(at(4 + num_classes + k, i));
        }
        cand_coeffs.push_back(std::move(coeffs));
    }

    if (cand_rects.empty()) {
        return {};
    }

    // Per-class NMS; map kept indices straight back to results + mask coeffs.
    const float gain = std::min(
        static_cast<float>(input_height) / static_cast<float>(orig_size.height),
        static_cast<float>(input_width) / static_cast<float>(orig_size.width));
    const float pad_w = (static_cast<float>(input_width) - orig_size.width * gain) / 2.0f;
    const float pad_h = (static_cast<float>(input_height) - orig_size.height * gain) / 2.0f;
    const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
    const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));

    std::vector<std::vector<size_t>> by_class(static_cast<size_t>(num_classes));
    for (size_t i = 0; i < cand_rects.size(); ++i) {
        by_class[static_cast<size_t>(cand_labels[i])].push_back(i);
    }

    vision_common::SegmentationResultList results;
    std::vector<std::vector<float>> kept_coeffs;
    for (const auto& cls_indices : by_class) {
        if (cls_indices.empty()) {
            continue;
        }
        std::vector<cv::Rect2f> rects;
        std::vector<float> scores;
        rects.reserve(cls_indices.size());
        scores.reserve(cls_indices.size());
        for (size_t idx : cls_indices) {
            rects.push_back(cand_rects[idx]);
            scores.push_back(cand_scores[idx]);
        }

        std::vector<int> keep = vision_common::nms(rects, scores, iou_threshold);
        for (int k : keep) {
            const size_t idx = cls_indices[static_cast<size_t>(k)];
            const cv::Rect2f& b = cand_rects[idx];
            float x1 = (b.x - pad_w) / gain;
            float y1 = (b.y - pad_h) / gain;
            float x2 = (b.x + b.width - pad_w) / gain;
            float y2 = (b.y + b.height - pad_h) / gain;
            x1 = std::max(0.0f, std::min(x1, max_x));
            y1 = std::max(0.0f, std::min(y1, max_y));
            x2 = std::max(0.0f, std::min(x2, max_x));
            y2 = std::max(0.0f, std::min(y2, max_y));

            vision_common::SegmentationResult r;
            r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
            r.label = cand_labels[idx];
            r.score = cand_scores[idx];
            r.mask = nullptr;
            results.push_back(r);
            kept_coeffs.push_back(cand_coeffs[idx]);
        }
    }

    if (results.empty()) {
        return results;
    }

    // proto = outputs[1] = [1, proto_channels, mask_h, mask_w]
    const float* proto = outputs[1].GetTensorData<float>();
    std::vector<std::shared_ptr<cv::Mat>> masks =
        _process_masks(proto, proto_dims, kept_coeffs, results, orig_size);
    for (size_t i = 0; i < results.size() && i < masks.size(); ++i) {
        results[i].mask = masks[i];
    }

    return results;
}


std::vector<std::shared_ptr<cv::Mat>> YOLOv8SegDetector::_process_masks(
    const float* protos,
    const std::vector<int64_t>& proto_dims,
    const std::vector<std::vector<float>>& mask_coeffs,
    const vision_common::SegmentationResultList& results,
    const cv::Size& orig_shape) {
    if (results.empty() || mask_coeffs.empty()) {
        return std::vector<std::shared_ptr<cv::Mat>>();
    }

    int mask_dim = static_cast<int>(proto_dims[1]);
    int mask_h = static_cast<int>(proto_dims[2]);
    int mask_w = static_cast<int>(proto_dims[3]);
    int proto_size = mask_h * mask_w;
    int num_masks = static_cast<int>(mask_coeffs.size());

    // Manual matmul: [num_masks, mask_dim] x [mask_dim, proto_size] = [num_masks, proto_size]
    // Outer loop over k (channels) for cache-friendly sequential reads of proto rows.
    // ~7x faster than Eigen GEMM on RISC-V which lacks SIMD optimizations.
    std::vector<float> raw_masks(static_cast<size_t>(num_masks) * proto_size, 0.0f);
    for (int k = 0; k < mask_dim; ++k) {
        const float* proto_row = protos + k * proto_size;
        for (int i = 0; i < num_masks; ++i) {
            float c = mask_coeffs[i][k];
            float* out = raw_masks.data() + i * proto_size;
            for (int j = 0; j < proto_size; ++j) {
                out[j] += c * proto_row[j];
            }
        }
    }
    // Skip sigmoid: sigmoid(x) > 0.5 <==> x > 0, threshold on raw logits

    // Calculate crop parameters for letterbox padding removal
    int orig_h = orig_shape.height;
    int orig_w = orig_shape.width;
    float gain = std::min(static_cast<float>(mask_h) / orig_h, static_cast<float>(mask_w) / orig_w);
    float pad_w = (mask_w - orig_w * gain) / 2.0f;
    float pad_h = (mask_h - orig_h * gain) / 2.0f;

    int top = (pad_h > 0) ? static_cast<int>(std::round(pad_h - 0.1f)) : 0;
    int left = (pad_w > 0) ? static_cast<int>(std::round(pad_w - 0.1f)) : 0;
    int bottom = (pad_h > 0) ? (mask_h - static_cast<int>(std::round(pad_h + 0.1f))) : mask_h;
    int right = (pad_w > 0) ? (mask_w - static_cast<int>(std::round(pad_w + 0.1f))) : mask_w;

    top = std::max(0, std::min(top, mask_h - 1));
    left = std::max(0, std::min(left, mask_w - 1));
    bottom = std::max(top + 1, std::min(bottom, mask_h));
    right = std::max(left + 1, std::min(right, mask_w));

    const int crop_h = bottom - top;
    const int crop_w = right - left;
    const float scale_x = static_cast<float>(crop_w) / static_cast<float>(orig_w);
    const float scale_y = static_cast<float>(crop_h) / static_cast<float>(orig_h);

    std::vector<std::shared_ptr<cv::Mat>> masks_scaled;
    masks_scaled.reserve(num_masks);

    for (int i = 0; i < num_masks; ++i) {
        // Each mask is contiguous in [i * proto_size, (i+1) * proto_size).
        cv::Mat mask_full(mask_h, mask_w, CV_32F, raw_masks.data() + static_cast<size_t>(i) * proto_size);

        // Crop letterbox padding
        cv::Mat mask_cropped = mask_full(cv::Range(top, bottom), cv::Range(left, right));

        // Map bbox from original image coords to small-mask coords
        const auto& result = results[i];
        int bx1 = std::max(0, static_cast<int>(std::floor(result.bbox.x1 * scale_x)));
        int by1 = std::max(0, static_cast<int>(std::floor(result.bbox.y1 * scale_y)));
        int bx2 = std::min(crop_w, static_cast<int>(std::ceil(result.bbox.x2 * scale_x)));
        int by2 = std::min(crop_h, static_cast<int>(std::ceil(result.bbox.y2 * scale_y)));

        if (bx2 <= bx1 || by2 <= by1) {
            masks_scaled.push_back(std::make_shared<cv::Mat>(
                cv::Mat::zeros(orig_h, orig_w, CV_8U)));
            continue;
        }

        // Crop bbox region from small mask, resize only that region
        cv::Mat small_roi = mask_cropped(cv::Range(by1, by2), cv::Range(bx1, bx2));

        int ox1 = std::max(0, std::min(static_cast<int>(result.bbox.x1), orig_w - 1));
        int oy1 = std::max(0, std::min(static_cast<int>(result.bbox.y1), orig_h - 1));
        int ox2 = std::max(ox1 + 1, std::min(static_cast<int>(result.bbox.x2), orig_w));
        int oy2 = std::max(oy1 + 1, std::min(static_cast<int>(result.bbox.y2), orig_h));

        cv::Mat roi_resized;
        cv::resize(small_roi, roi_resized,
                    cv::Size(ox2 - ox1, oy2 - oy1), 0, 0, cv::INTER_LINEAR);

        // Threshold on raw logits: x > 0 <==> sigmoid(x) > 0.5
        cv::Mat roi_binary;
        cv::threshold(roi_resized, roi_binary, 0.0f, 255.0f, cv::THRESH_BINARY);
        cv::Mat roi_uint8;
        roi_binary.convertTo(roi_uint8, CV_8U);

        // Paste into full-size output mask
        cv::Mat mask_out = cv::Mat::zeros(orig_h, orig_w, CV_8U);
        roi_uint8.copyTo(mask_out(cv::Range(oy1, oy2), cv::Range(ox1, ox2)));

        masks_scaled.push_back(std::make_shared<cv::Mat>(mask_out));
    }

    return masks_scaled;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8SegDetector> registrar("YOLOv8SegDetector");

}  // namespace vision_deploy

