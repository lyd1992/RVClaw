/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV8_POSE_DETECTOR_H
#define YOLOV8_POSE_DETECTOR_H

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
 * @brief YOLOv8-Pose Keypoint Detection Model
 * 
 * Detects persons and their 17 keypoints.
 */
class YOLOv8PoseDetector : public vision_core::BaseModel, public vision_core::IPoseModel {
public:
    YOLOv8PoseDetector(const std::string& model_path,
                        float conf_threshold = 0.25f,
                        float iou_threshold = 0.45f,
                        float point_confidence_threshold = 0.2f,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YOLOv8PoseDetector() = default;

    /**
     * @brief Load YOLOv8-Pose ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for YOLOv8-Pose inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    vision_common::PoseResultList estimate_pose(const cv::Mat& image,
                                                float conf_threshold = -1.0f,
                                                float iou_threshold = -1.0f) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    vision_common::PoseResultList postprocess(std::vector<Ort::Value>& outputs,
        const cv::Size& orig_size,
        float conf_threshold,
        float iou_threshold);

private:
    float conf_threshold_;
    float iou_threshold_;
    float point_confidence_threshold_;
    int num_threads_;
    std::string provider_;

    float calculate_iou(const vision_common::PoseResult& det1, const vision_common::PoseResult& det2);
    vision_common::PoseResultList nms(const vision_common::PoseResultList& dets, float iou_threshold);
};

}  // namespace vision_deploy

#endif  // YOLOV8_POSE_DETECTOR_H
