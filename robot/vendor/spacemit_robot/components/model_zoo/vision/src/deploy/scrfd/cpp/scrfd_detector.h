/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SCRFD_DETECTOR_H
#define SCRFD_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief SCRFD face detector (buffalo_l det_10g).
 *
 * Returns vision::Pose with bbox + 5 facial keypoints per face.
 */
class ScrfdDetector : public vision_core::BaseModel, public vision_core::IPoseModel {
public:
    ScrfdDetector(const std::string& model_path,
                    float conf_threshold = 0.5f,
                    float nms_threshold = 0.4f,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    ~ScrfdDetector() override = default;

    void load_model() override;

    cv::Mat preprocess(const cv::Mat& image);

    vision_common::PoseResultList estimate_pose(const cv::Mat& image,
                                                float conf_threshold = -1.0f,
                                                float iou_threshold = -1.0f) override;

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    vision_common::PoseResultList postprocess(std::vector<Ort::Value>& outputs,
                                                const cv::Size& orig_size,
                                                float conf_threshold,
                                                float nms_threshold,
                                                float det_scale);

private:
    struct Anchor {
        float cx = 0.0f;
        float cy = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
    };

    void generate_anchors();
    std::vector<vision_common::PoseResult> nms(std::vector<vision_common::PoseResult>& boxes,
                                                float threshold) const;

    float conf_threshold_;
    float nms_threshold_;
    int num_threads_;
    std::string provider_;
    int input_width_ = 640;
    int input_height_ = 640;
    std::vector<Anchor> anchors_;
    float last_det_scale_ = 1.0f;
};

}  // namespace vision_deploy

#endif  // SCRFD_DETECTOR_H
