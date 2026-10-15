/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARCFACE_RECOGNIZER_H
#define ARCFACE_RECOGNIZER_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"
#include "embedding_utils.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief ArcFace Face Recognition Model
 * 
 * Extracts face embeddings for face recognition tasks.
 * The model should receive aligned face images as input.
 */
class ArcFaceRecognizer : public vision_core::BaseModel, public vision_core::IEmbeddingModel {
public:
    ArcFaceRecognizer(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider",
                        float norm_std = 127.5f);

    virtual ~ArcFaceRecognizer() = default;

    /**
     * @brief Load ArcFace ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess face image for ArcFace inference
     * @param image Input face image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    /**
     * @brief Run inference and return face embedding
     * @param image Input face image in BGR format
     * @return Face embedding result with normalized vector (128 dimensions)
     */
    vision_common::EmbeddingResult infer_embedding(const cv::Mat& image) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /**
     * @brief Compute similarity between two face embeddings
     * @param embedding1 First face embedding
     * @param embedding2 Second face embedding
     * @return Similarity score (-1 to 1, higher means more similar)
     */
    static float compute_similarity(const std::vector<float>& embedding1,
                                    const std::vector<float>& embedding2) {
        return vision_common::compute_similarity(embedding1, embedding2);
    }

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    std::vector<float> postprocess(std::vector<Ort::Value>& outputs);

private:
    int num_threads_;
    std::string provider_;
    float norm_std_ = 127.5f;
};

}  // namespace vision_deploy

#endif  // ARCFACE_RECOGNIZER_H

