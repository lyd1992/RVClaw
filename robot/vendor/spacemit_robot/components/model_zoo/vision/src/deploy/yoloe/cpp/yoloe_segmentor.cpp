/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yoloe_segmentor.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
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

// Parse prompts from YAML default_params: sequence [a,b,c] or scalar "a,b,c".
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
            size_t a = tok.find_first_not_of(" \t");
            size_t b = tok.find_last_not_of(" \t");
            if (a != std::string::npos) out.push_back(tok.substr(a, b - a + 1));
        }
    }
    return out;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YoloeSegmentor::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YoloeSegmentor");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YoloeSegmentor");
    }

    std::string clip_model_path = vision_core::yaml_utils::getString(default_params, "clip_model_path");
    if (clip_model_path.empty()) {
        throw std::runtime_error("clip_model_path not found in default_params for YoloeSegmentor");
    }
    std::string bpe_merges_path = vision_core::yaml_utils::getString(default_params, "bpe_merges_path");
    if (bpe_merges_path.empty()) {
        throw std::runtime_error("bpe_merges_path not found in default_params for YoloeSegmentor");
    }

    std::vector<std::string> default_prompts = parsePrompts(default_params);
    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YoloeSegmentor>(
        model_path, clip_model_path, bpe_merges_path, default_prompts,
        conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YoloeSegmentor::YoloeSegmentor(
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

void YoloeSegmentor::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);

    // Identify the 4D image input and the 3D text input.
    for (size_t i = 0; i < input_node_names_.size(); ++i) {
        std::vector<int64_t> dims =
            session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (dims.size() == 4) {
            input_image_dims_ = dims;
            image_input_index_ = static_cast<int>(i);
        } else if (dims.size() == 3) {
            input_text_dims_ = dims;
            text_input_index_ = static_cast<int>(i);
        }
    }
    if (input_image_dims_.size() != 4 || input_text_dims_.size() != 3) {
        throw std::runtime_error("YoloeSegmentor expects a 4D image input and a 3D text input");
    }

    // Segmentation if the graph exposes a proto output (>=2 outputs).
    is_segment_ = (output_node_names_.size() >= 2);

    // Derive num_classes / num_mask_coeffs from the output layout.
    std::vector<int64_t> det_dims =
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (is_segment_) {
        std::vector<int64_t> proto_dims =
            session_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        if (proto_dims.size() == 4) {
            num_mask_coeffs_ = static_cast<int>(proto_dims[1]);
        }
        if (det_dims.size() == 3) {
            num_classes_ = static_cast<int>(det_dims[1]) - 4 - num_mask_coeffs_;
        }
        if (num_classes_ <= 0) {
            throw std::runtime_error("YoloeSegmentor: invalid seg output feature layout");
        }
    } else if (det_dims.size() == 3) {
        num_classes_ = static_cast<int>(det_dims[1]) - 4;
    }

    // MobileCLIP text encoder (CPU; one-off per-vocabulary cost).
    clip_ = std::make_unique<MobileClip>(clip_model_path_, bpe_merges_path_, num_threads_);

    model_loaded_ = true;
}

void YoloeSegmentor::ensure_text_features(const std::vector<std::string>& prompts) {
    ensure_model_loaded();

    // Empty prompts always mean "use the yaml default vocabulary" (not sticky).
    const std::vector<std::string>& vocab = !prompts.empty() ? prompts : default_prompts_;
    if (vocab.empty()) {
        throw std::runtime_error(
            "YoloeSegmentor: no prompts provided and no default vocabulary in config");
    }
    if (text_features_ready_ && vocab == cached_prompts_) {
        return;  // cache hit: no CLIP call
    }

    const int model_num_classes = static_cast<int>(input_text_dims_[1]);
    const int feature_dim = static_cast<int>(input_text_dims_[2]);
    const int num_classes = static_cast<int>(vocab.size());
    if (num_classes > model_num_classes) {
        std::cerr << "YoloeSegmentor: " << num_classes << " prompts exceed model capacity "
            << model_num_classes << "; extra prompts are ignored" << std::endl;
    }

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

void YoloeSegmentor::preprocess(const cv::Mat& image, cv::Mat& blob) {
    const int dst_h = static_cast<int>(input_image_dims_[2]);
    const int dst_w = static_cast<int>(input_image_dims_[3]);

    // Store the exact scale/offset that vision_common::letterbox applies, so
    // postprocess inverts the same transform (no drift). Mirrors letterbox().
    const float r = std::min(static_cast<float>(dst_h) / static_cast<float>(image.rows),
        static_cast<float>(dst_w) / static_cast<float>(image.cols));
    const int new_w = static_cast<int>(std::round(image.cols * r));
    const int new_h = static_cast<int>(std::round(image.rows * r));
    letterbox_scale_ = r;
    letterbox_ox_ = static_cast<int>(std::round((dst_w - new_w) / 2.0f - 0.1));
    letterbox_oy_ = static_cast<int>(std::round((dst_h - new_h) / 2.0f - 0.1));

    cv::Mat padded = vision_common::letterbox(image, std::make_pair(dst_h, dst_w));
    blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(dst_w, dst_h),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}

vision_common::SegmentationResultList YoloeSegmentor::postprocess_det(
    const float* det, int offset, int anchors,
    const cv::Size& orig_size, float conf_threshold, float iou_threshold) {
    (void)offset;
    const float ratio = letterbox_scale_;
    const float dw = static_cast<float>(letterbox_ox_);
    const float dh = static_cast<float>(letterbox_oy_);
    const int active_classes = static_cast<int>(active_labels_.size());

    vision_common::SegmentationResultList candidates;
    for (int j = 0; j < anchors; ++j) {
        float max_score = -1.0f;
        int max_index = -1;
        for (int c = 0; c < active_classes; ++c) {
            const float s = det[(4 + c) * anchors + j];
            if (s > max_score) {
                max_score = s;
                max_index = c;
            }
        }
        if (max_score <= conf_threshold || max_index < 0) {
            continue;
        }
        const float half_w = det[2 * anchors + j] / 2.0f;
        const float half_h = det[3 * anchors + j] / 2.0f;
        float x1 = std::max(0.0f, (det[j] - half_w - dw) / ratio);
        float y1 = std::max(0.0f, (det[anchors + j] - half_h - dh) / ratio);
        float x2 = std::min(static_cast<float>(orig_size.width), (det[j] + half_w - dw) / ratio);
        float y2 = std::min(static_cast<float>(orig_size.height), (det[anchors + j] + half_h - dh) / ratio);

        vision_common::SegmentationResult r;
        r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        r.score = max_score;
        r.label = max_index;
        r.mask = nullptr;
        candidates.push_back(r);
    }
    vision_common::SegmentationResultList kept = vision_common::multi_class_nms(candidates, iou_threshold);
    if (max_det_ > 0 && static_cast<int>(kept.size()) > max_det_) {
        kept.resize(static_cast<size_t>(max_det_));
    }
    return kept;
}

vision_common::SegmentationResultList YoloeSegmentor::postprocess_seg(
    const float* det, int offset, int anchors,
    const float* proto, const std::vector<int64_t>& proto_dims,
    const cv::Size& orig_size, float conf_threshold, float iou_threshold) {
    (void)offset;
    const float ratio = letterbox_scale_;
    const float dw = static_cast<float>(letterbox_ox_);
    const float dh = static_cast<float>(letterbox_oy_);
    auto at = [&](int c, int j) -> float { return det[c * anchors + j]; };

    // Only argmax over the ACTIVE prompt count: open-vocabulary fills just the
    // first N text-feature slots, the rest are zero-padded and would otherwise
    // produce false positives. The mask-coefficient offset, however, must use
    // the model's full num_classes_ because that is the tensor channel layout
    // (4 + num_classes_ + num_mask_coeffs_).
    const int active_classes = std::min(static_cast<int>(active_labels_.size()), num_classes_);

    // Collect candidates (with mask coeffs), keyed for post-NMS coeff lookup.
    vision_common::SegmentationResultList candidates;
    std::vector<std::vector<float>> cand_coeffs;
    candidates.reserve(256);
    cand_coeffs.reserve(256);
    for (int j = 0; j < anchors; ++j) {
        float best = -1.0f;
        int best_cls = -1;
        for (int c = 0; c < active_classes; ++c) {
            const float s = at(4 + c, j);
            if (s > best) {
                best = s;
                best_cls = c;
            }
        }
        if (best_cls < 0 || best <= conf_threshold) {
            continue;
        }
        const float half_w = at(2, j) / 2.0f;
        const float half_h = at(3, j) / 2.0f;
        float x1 = std::max(0.0f, (at(0, j) - half_w - dw) / ratio);
        float y1 = std::max(0.0f, (at(1, j) - half_h - dh) / ratio);
        float x2 = std::min(static_cast<float>(orig_size.width), (at(0, j) + half_w - dw) / ratio);
        float y2 = std::min(static_cast<float>(orig_size.height), (at(1, j) + half_h - dh) / ratio);

        vision_common::SegmentationResult r;
        r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        r.score = best;
        r.label = best_cls;
        r.mask = nullptr;
        candidates.push_back(r);

        std::vector<float> coeffs;
        coeffs.reserve(static_cast<size_t>(num_mask_coeffs_));
        for (int k = 0; k < num_mask_coeffs_; ++k) {
            coeffs.push_back(at(4 + num_classes_ + k, j));
        }
        cand_coeffs.push_back(std::move(coeffs));
    }
    if (candidates.empty()) {
        return {};
    }

    // NMS on boxes. multi_class_nms returns copies (no indices), so recover each
    // kept box's mask coefficients by matching on the full box + score + label
    // (a candidate is uniquely identified by all four coords + score, so this is
    // robust; the earlier version omitted y2 and could mis-match).
    vision_common::SegmentationResultList kept = vision_common::multi_class_nms(candidates, iou_threshold);
    if (max_det_ > 0 && static_cast<int>(kept.size()) > max_det_) {
        kept.resize(static_cast<size_t>(max_det_));
    }
    std::vector<std::vector<float>> kept_coeffs;
    kept_coeffs.reserve(kept.size());
    for (const auto& res : kept) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& c = candidates[i];
            if (c.label == res.label && c.score == res.score &&
                c.bbox.x1 == res.bbox.x1 && c.bbox.y1 == res.bbox.y1 &&
                c.bbox.x2 == res.bbox.x2 && c.bbox.y2 == res.bbox.y2) {
                kept_coeffs.push_back(cand_coeffs[i]);
                break;
            }
        }
    }

    if (!kept.empty() && kept_coeffs.size() == kept.size()) {
        process_masks(proto, proto_dims, kept_coeffs, kept, orig_size);
    }
    return kept;
}

void YoloeSegmentor::process_masks(
    const float* proto, const std::vector<int64_t>& proto_dims,
    const std::vector<std::vector<float>>& mask_coeffs,
    vision_common::SegmentationResultList& objects,
    const cv::Size& orig_shape) {
    if (objects.empty() || mask_coeffs.empty() || proto_dims.size() != 4) {
        return;
    }
    const int mask_dim = static_cast<int>(proto_dims[1]);
    const int mask_h = static_cast<int>(proto_dims[2]);
    const int mask_w = static_cast<int>(proto_dims[3]);
    const int proto_size = mask_h * mask_w;
    const int num_masks = static_cast<int>(mask_coeffs.size());

    // masks = coeffs (num_masks x mask_dim) @ proto (mask_dim x proto_size)
    std::vector<float> raw_masks(static_cast<size_t>(num_masks) * proto_size, 0.0f);
    for (int k = 0; k < mask_dim; ++k) {
        const float* proto_row = proto + k * proto_size;
        for (int i = 0; i < num_masks; ++i) {
            const float c = mask_coeffs[static_cast<size_t>(i)][static_cast<size_t>(k)];
            float* out = raw_masks.data() + static_cast<size_t>(i) * proto_size;
            for (int j = 0; j < proto_size; ++j) {
                out[j] += c * proto_row[j];
            }
        }
    }

    const int orig_h = orig_shape.height;
    const int orig_w = orig_shape.width;
    const float gain = std::min(static_cast<float>(mask_h) / orig_h, static_cast<float>(mask_w) / orig_w);
    const float pad_w = (mask_w - orig_w * gain) / 2.0f;
    const float pad_h = (mask_h - orig_h * gain) / 2.0f;

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

    for (int i = 0; i < num_masks; ++i) {
        cv::Mat mask_full(mask_h, mask_w, CV_32F, raw_masks.data() + static_cast<size_t>(i) * proto_size);
        cv::Mat mask_cropped = mask_full(cv::Range(top, bottom), cv::Range(left, right));

        const vision_common::SegmentationResult& obj = objects[static_cast<size_t>(i)];
        const int bx1 = std::max(0, static_cast<int>(std::floor(obj.bbox.x1 * scale_x)));
        const int by1 = std::max(0, static_cast<int>(std::floor(obj.bbox.y1 * scale_y)));
        const int bx2 = std::min(crop_w, static_cast<int>(std::ceil(obj.bbox.x2 * scale_x)));
        const int by2 = std::min(crop_h, static_cast<int>(std::ceil(obj.bbox.y2 * scale_y)));

        cv::Mat mask_out = cv::Mat::zeros(orig_h, orig_w, CV_8U);
        if (bx2 > bx1 && by2 > by1) {
            cv::Mat small_roi = mask_cropped(cv::Range(by1, by2), cv::Range(bx1, bx2));
            const int ox1 = std::max(0, std::min(static_cast<int>(obj.bbox.x1), orig_w - 1));
            const int oy1 = std::max(0, std::min(static_cast<int>(obj.bbox.y1), orig_h - 1));
            const int ox2 = std::max(ox1 + 1, std::min(static_cast<int>(obj.bbox.x2), orig_w));
            const int oy2 = std::max(oy1 + 1, std::min(static_cast<int>(obj.bbox.y2), orig_h));

            cv::Mat roi_resized;
            cv::resize(small_roi, roi_resized, cv::Size(ox2 - ox1, oy2 - oy1), 0, 0, cv::INTER_LINEAR);
            cv::Mat roi_binary;
            cv::threshold(roi_resized, roi_binary, 0.0f, 255.0f, cv::THRESH_BINARY);
            cv::Mat roi_uint8;
            roi_binary.convertTo(roi_uint8, CV_8U);
            roi_uint8.copyTo(mask_out(cv::Range(oy1, oy2), cv::Range(ox1, ox2)));
        }
        objects[static_cast<size_t>(i)].mask = std::make_shared<cv::Mat>(mask_out);
    }
}

vision_common::SegmentationResultList YoloeSegmentor::segment(
    const cv::Mat& image, float conf_threshold, float iou_threshold) {
    return segment_with_prompts(image, {}, conf_threshold, iou_threshold);
}

vision_common::SegmentationResultList YoloeSegmentor::segment_with_prompts(
    const cv::Mat& image, const std::vector<std::string>& prompts,
    float conf_threshold, float iou_threshold) {
    if (image.empty()) {
        throw std::runtime_error("YoloeSegmentor: input image is empty");
    }
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const float use_conf = conf_threshold > 0.0f ? conf_threshold : conf_threshold_;
    const float use_iou = iou_threshold > 0.0f ? iou_threshold : iou_threshold_;

    ensure_text_features(prompts);

    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat image_blob;
    preprocess(image, image_blob);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const std::vector<int64_t> image_shape = {1, 3, input_image_dims_[2], input_image_dims_[3]};
    const std::vector<int64_t> text_shape = {1, input_text_dims_[1], input_text_dims_[2]};

    // Ort::Value is move-only; place each tensor at its declared input index.
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
    const float* det = outputs[0].GetTensorMutableData<float>();
    std::vector<int64_t> det_dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int offset = static_cast<int>(det_dims[1]);
    const int anchors = static_cast<int>(det_dims[2]);

    vision_common::SegmentationResultList results;
    if (is_segment_) {
        const float* proto = outputs[1].GetTensorMutableData<float>();
        std::vector<int64_t> proto_dims = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        results = postprocess_seg(det, offset, anchors, proto, proto_dims, image.size(), use_conf, use_iou);
    } else {
        results = postprocess_det(det, offset, anchors, image.size(), use_conf, use_iou);
    }
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> YoloeSegmentor::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

std::vector<vision_core::ModelCapability> YoloeSegmentor::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_core::InferResponse YoloeSegmentor::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kSegment);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "YoloeSegmentor expects ImageInput";
        return response;
    }

    vision_common::SegmentationResultList task_results = segment_with_prompts(
        image_input->image, request.params.prompts,
        request.params.conf_threshold, request.params.iou_threshold);

    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

static vision_core::ModelRegistrar<YoloeSegmentor> registrar("YoloeSegmentor");

}  // namespace vision_deploy
