/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLO_WORLD_DETECTOR_H
#define YOLO_WORLD_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

#include "clip.hpp"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief YOLO-World open-vocabulary detector.
 *
 * Two ONNX models: a CLIP text encoder (prompts -> text features) and the
 * YOLO-World detector (image + text features -> boxes). Text prompts come from
 * config (default vocabulary) or per-inference via InferParams::prompts. Text
 * features are lazily computed and cached, keyed by the prompt list: identical
 * prompts reuse the cache (no CLIP call), changed prompts trigger re-encoding.
 * Steady-state inference with an unchanged vocabulary skips CLIP.
 */
class YoloWorldDetector : public vision_core::BaseModel, public vision_core::IDetectionModel {
public:
    YoloWorldDetector(
        const std::string& model_path,
        const std::string& clip_model_path,
        const std::string& bpe_merges_path,
        const std::vector<std::string>& default_prompts,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.45f,
        int num_threads = 4,
        bool lazy_load = false,
        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YoloWorldDetector() = default;

    void load_model() override;

    vision_common::DetectionResultList detect(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;

    // Detect with an explicit prompt vocabulary (empty -> use yaml default).
    vision_common::DetectionResultList detect_with_prompts(
        const cv::Mat& image,
        const std::vector<std::string>& prompts,
        float conf_threshold,
        float iou_threshold);

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Active vocabulary (labels for the last-used prompts). Used for drawing.
    const std::vector<std::string>& active_labels() const { return active_labels_; }

    // Runtime labels from the active text prompts (open-vocabulary), so
    // VisionService::GetClassNames()/Draw() show prompt words, not "Class N".
    std::vector<std::string> get_dynamic_class_names() const override { return active_labels_; }

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    // Lazily (re)compute cached text features for the given prompts.
    void ensure_text_features(const std::vector<std::string>& prompts);

    void preprocess(const cv::Mat& image, cv::Mat& blob);
    vision_common::DetectionResultList postprocess(
        const float* output, int offset, int anchors,
        const cv::Size& orig_size, float conf_threshold, float iou_threshold);

    std::string clip_model_path_;
    std::string bpe_merges_path_;
    std::vector<std::string> default_prompts_;
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    std::string provider_;

    std::unique_ptr<CLIP> clip_;

    // Detector session inputs/outputs.
    std::vector<int64_t> input_image_dims_;   // {1,3,H,W}
    std::vector<int64_t> input_text_dims_;    // {1,num_classes,feat_dim}
    int image_input_index_ = 0;
    int text_input_index_ = 1;

    // Text-feature cache keyed by prompt list.
    std::vector<std::string> cached_prompts_;
    std::vector<std::string> active_labels_;
    std::vector<float> text_feature_data_;    // {num_classes*feat_dim}, padded to model capacity
    bool text_features_ready_ = false;

    // Letterbox state from last preprocess (reused by postprocess so the
    // forward/inverse transform can't drift).
    float letterbox_scale_ = 1.0f;
    int letterbox_ox_ = 0;
    int letterbox_oy_ = 0;
};

}  // namespace vision_deploy

#endif  // YOLO_WORLD_DETECTOR_H
