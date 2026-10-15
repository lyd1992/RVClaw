/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

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
 * @brief YOLOv8 Object Detector
 *
 * Auto-selects postprocess by ONNX output count:
 * - 1 output: Ultralytics export, e.g. [1, 84, 8400]
 * - 6+ outputs: multi-branch DFL heads (boxes / scores / score_sum per scale)
 */
class YOLOv8Detector : public vision_core::BaseModel, public vision_core::IDetectionModel {
public:
    YOLOv8Detector(const std::string& model_path,
                    float conf_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YOLOv8Detector() = default;

    /**
     * @brief Load YOLOv8 ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for YOLOv8 inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    vision_common::DetectionResultList detect(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    vision_common::DetectionResultList postprocess(
        std::vector<Ort::Value>& outputs,
        const cv::Size& orig_size,
        float conf_threshold,
        float iou_threshold);

private:
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    int num_classes_;
    std::string provider_;

    void get_dets(
        const cv::Size& orig_size,
        const float* boxes,
        const float* scores,
        const float* score_sum,
        const std::vector<int64_t>& dims,
        int tensor_width,
        int tensor_height,
        float conf_threshold,
        vision_common::DetectionResultList& objects);
};

}  // namespace vision_deploy

#endif  // YOLOV8_DETECTOR_H
