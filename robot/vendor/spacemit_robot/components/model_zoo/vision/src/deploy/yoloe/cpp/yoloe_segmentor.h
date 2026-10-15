/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOE_SEGMENTOR_H
#define YOLOE_SEGMENTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

#include "mobileclip.hpp"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief YOLOE open-vocabulary instance segmentation.
 *
 * Two ONNX models: a MobileCLIP text encoder (prompts -> text features) and the
 * YOLOE detector/segmentor (image + text features -> boxes + mask coeffs + proto).
 * Text prompts come from config (default vocabulary) or per-inference via
 * InferParams::prompts; text features are lazily computed and cached, keyed by
 * the prompt list (same prompts reuse the cache, no CLIP call).
 *
 * Output layout (CN): det {1, 4 + num_classes + num_mask_coeffs, anchors} plus a
 * proto {1, num_mask_coeffs, mh, mw}. When the model exposes >=2 outputs it is
 * treated as segmentation; a single-output export degrades to detection (empty masks).
 */
class YoloeSegmentor : public vision_core::BaseModel, public vision_core::ISegmentationModel {
public:
    YoloeSegmentor(
        const std::string& model_path,
        const std::string& clip_model_path,
        const std::string& bpe_merges_path,
        const std::vector<std::string>& default_prompts,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f,
        int num_threads = 4,
        bool lazy_load = false,
        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YoloeSegmentor() = default;

    void load_model() override;

    vision_common::SegmentationResultList segment(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;

    // Segment with an explicit prompt vocabulary (empty -> use yaml default).
    vision_common::SegmentationResultList segment_with_prompts(
        const cv::Mat& image,
        const std::vector<std::string>& prompts,
        float conf_threshold,
        float iou_threshold);

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Runtime labels from the active text prompts (open-vocabulary).
    std::vector<std::string> get_dynamic_class_names() const override { return active_labels_; }

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    void ensure_text_features(const std::vector<std::string>& prompts);
    void preprocess(const cv::Mat& image, cv::Mat& blob);

    // Segmentation postprocess: proto x coeffs -> per-instance mask.
    vision_common::SegmentationResultList postprocess_seg(
        const float* det, int offset, int anchors,
        const float* proto, const std::vector<int64_t>& proto_dims,
        const cv::Size& orig_size, float conf_threshold, float iou_threshold);

    // Plain detection postprocess (single-output export, no masks).
    vision_common::SegmentationResultList postprocess_det(
        const float* det, int offset, int anchors,
        const cv::Size& orig_size, float conf_threshold, float iou_threshold);

    void process_masks(
        const float* proto, const std::vector<int64_t>& proto_dims,
        const std::vector<std::vector<float>>& mask_coeffs,
        vision_common::SegmentationResultList& objects,
        const cv::Size& orig_size);

    std::string clip_model_path_;
    std::string bpe_merges_path_;
    std::vector<std::string> default_prompts_;
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    std::string provider_;

    std::unique_ptr<MobileClip> clip_;

    std::vector<int64_t> input_image_dims_;   // {1,3,H,W}
    std::vector<int64_t> input_text_dims_;    // {1,num_classes,feat_dim}
    int image_input_index_ = 0;
    int text_input_index_ = 1;

    bool is_segment_ = false;
    int num_classes_ = 0;
    int num_mask_coeffs_ = 32;
    int max_det_ = 300;

    // Text-feature cache keyed by prompt list.
    std::vector<std::string> cached_prompts_;
    std::vector<std::string> active_labels_;
    std::vector<float> text_feature_data_;    // {num_classes*feat_dim}, padded to capacity
    bool text_features_ready_ = false;

    // Letterbox state from last preprocess (reused by postprocess; no drift).
    float letterbox_scale_ = 1.0f;
    int letterbox_ox_ = 0;
    int letterbox_oy_ = 0;
};

}  // namespace vision_deploy

#endif  // YOLOE_SEGMENTOR_H
