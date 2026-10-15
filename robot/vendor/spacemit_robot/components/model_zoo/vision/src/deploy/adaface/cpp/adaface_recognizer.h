/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ADAFACE_RECOGNIZER_H
#define ADAFACE_RECOGNIZER_H

#include <memory>
#include <string>
#include <vector>

#include "embedding_utils.h"
#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class AdaFaceRecognizer : public vision_core::BaseModel, public vision_core::IEmbeddingModel {
public:
    AdaFaceRecognizer(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    ~AdaFaceRecognizer() override = default;

    void load_model() override;
    cv::Mat preprocess(const cv::Mat& image);
    vision_common::EmbeddingResult infer_embedding(const cv::Mat& image) override;
    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    static float compute_similarity(const std::vector<float>& embedding1,
                                    const std::vector<float>& embedding2) {
        return vision_common::compute_similarity(embedding1, embedding2);
    }

    std::vector<float> postprocess(std::vector<Ort::Value>& outputs);

private:
    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // ADAFACE_RECOGNIZER_H
