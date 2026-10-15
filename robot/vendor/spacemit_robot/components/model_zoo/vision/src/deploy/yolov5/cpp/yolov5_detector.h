/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV5_DETECTOR_H
#define YOLOV5_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class YOLOv5Detector : public vision_core::BaseModel, public vision_core::IDetectionModel {
public:
    YOLOv5Detector(const std::string& model_path,
                    float conf_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    ~YOLOv5Detector() override = default;

    void load_model() override;
    cv::Mat preprocess(const cv::Mat& image);
    vision_common::DetectionResultList detect(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    vision_common::DetectionResultList postprocess(
        std::vector<Ort::Value>& outputs,
        const cv::Size& orig_size,
        float conf_threshold,
        float iou_threshold);

private:
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // YOLOV5_DETECTOR_H
