/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PP_LITESEG_H
#define PP_LITESEG_H

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
 * @brief PaddleSeg PP-LiteSeg semantic segmentation (Cityscapes-style ONNX).
 *
 * Pre/post-processing aligned with pp-liteseg/python/utils.py (RGB, scale+pad, mean/std 0.5).
 */
class PPLiteSeg : public vision_core::BaseModel, public vision_core::ISegmentationModel {
public:
    PPLiteSeg(const std::string& model_path,
                int num_threads,
                int num_classes,
                bool lazy_load,
                const std::string& provider);

    ~PPLiteSeg() override = default;

    void load_model() override;

    vision_common::SegmentationResultList segment(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    cv::Mat preprocess(const cv::Mat& image, int& valid_h, int& valid_w);

    cv::Mat postprocess_to_label_map(std::vector<Ort::Value>& outputs,
                                        int origin_h,
                                        int origin_w,
                                        int valid_h,
                                        int valid_w);

    vision_common::SegmentationResultList split_semantic_masks(const cv::Mat& label_u8);

    int num_threads_;
    int num_classes_;
    std::string provider_;
    float mean_val_;
    float std_val_;
};

}  // namespace vision_deploy

#endif  // PP_LITESEG_H
