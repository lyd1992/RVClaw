/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LANDMARK2D106_H
#define LANDMARK2D106_H

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
 * @brief 106-point 2D facial landmark model (buffalo_l 2d106det).
 *
 * Expects a face crop (typically SCRFD bbox crop). Input is resized to 192x192.
 */
class Landmark2d106 : public vision_core::BaseModel, public vision_core::IPoseModel {
public:
    Landmark2d106(const std::string& model_path,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider",
                    float input_mean = 127.5f,
                    float input_std = 128.0f);

    ~Landmark2d106() override = default;

    void load_model() override;

    cv::Mat preprocess(const cv::Mat& image);

    vision_common::PoseResultList estimate_pose(const cv::Mat& image,
                                                float conf_threshold = -1.0f,
                                                float iou_threshold = -1.0f) override;

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    vision_common::PoseResultList postprocess(std::vector<Ort::Value>& outputs, const cv::Size& face_size);

private:
    int num_threads_;
    std::string provider_;
    float input_mean_ = 127.5f;
    float input_std_ = 128.0f;
    int input_size_ = 192;
};

}  // namespace vision_deploy

#endif  // LANDMARK2D106_H
