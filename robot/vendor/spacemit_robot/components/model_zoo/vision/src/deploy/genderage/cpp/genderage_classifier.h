/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GENDERAGE_CLASSIFIER_H
#define GENDERAGE_CLASSIFIER_H

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
 * @brief InsightFace gender/age classifier.
 *
 * Expects an aligned 112x112 face; internally resizes to 96x96.
 */
class GenderAgeClassifier : public vision_core::BaseModel, public vision_core::IClassificationModel {
public:
    GenderAgeClassifier(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider",
                        float input_mean = 127.5f,
                        float input_std = 128.0f);

    ~GenderAgeClassifier() override = default;

    void load_model() override;

    cv::Mat preprocess(const cv::Mat& image);

    vision_common::ClassificationResultList classify(const cv::Mat& image) override;

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    vision_common::ClassificationResultList postprocess(std::vector<Ort::Value>& outputs);

private:
    int num_threads_;
    std::string provider_;
    float input_mean_ = 127.5f;
    float input_std_ = 128.0f;
    cv::Size target_size_{96, 96};
};

}  // namespace vision_deploy

#endif  // GENDERAGE_CLASSIFIER_H
