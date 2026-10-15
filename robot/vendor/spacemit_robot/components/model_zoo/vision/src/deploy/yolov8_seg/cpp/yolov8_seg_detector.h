/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV8_SEG_DETECTOR_H
#define YOLOV8_SEG_DETECTOR_H

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
 * @brief YOLOv8-Seg Instance Segmentation Detector
 * 
 * Detects objects and generates segmentation masks using YOLOv8-Seg model.
 */
class YOLOv8SegDetector : public vision_core::BaseModel, public vision_core::ISegmentationModel {
public:
    YOLOv8SegDetector(const std::string& model_path,
                        float conf_threshold = 0.25f,
                        float iou_threshold = 0.45f,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YOLOv8SegDetector() = default;

    /**
     * @brief Load YOLOv8-Seg ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for YOLOv8-Seg inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    vision_common::SegmentationResultList segment(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    vision_common::SegmentationResultList postprocess(
        std::vector<Ort::Value>& outputs,
        const cv::Size& orig_size,
        float conf_threshold,
        float iou_threshold);

private:
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    int num_classes_;
    int proto_channels_;
    std::string provider_;

    /** @brief Ultralytics single-output seg path: output0 = [1, 4+nc+nm, anchors], proto. */
    vision_common::SegmentationResultList postprocess_single_output_seg(
        std::vector<Ort::Value>& outputs,
        const cv::Size& orig_size,
        float conf_threshold,
        float iou_threshold);

    std::vector<std::shared_ptr<cv::Mat>> _process_masks(
        const float* protos,
        const std::vector<int64_t>& proto_dims,
        const std::vector<std::vector<float>>& mask_coeffs,
        const vision_common::SegmentationResultList& results,
        const cv::Size& orig_shape);
};

}  // namespace vision_deploy

#endif  // YOLOV8_SEG_DETECTOR_H
