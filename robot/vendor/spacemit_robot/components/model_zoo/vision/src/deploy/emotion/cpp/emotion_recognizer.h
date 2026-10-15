/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EMOTION_RECOGNIZER_H
#define EMOTION_RECOGNIZER_H

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
 * @brief Emotion Recognizer
 *
 * Classifies emotions in face images.
 * Note: This model expects pre-cropped face images, not full images.
 *
 * Feature mode: when constructed with feature_mode=true, the model acts as a
 * feature extractor (kEmbed) instead of a classifier (kClassify). Inference
 * then returns the raw output vector (e.g. 512-d) without argmax or L2 norm.
 * This is used as the backbone for dynamic (LSTM) emotion recognition.
 */
class EmotionRecognizer : public vision_core::BaseModel,
    public vision_core::IClassificationModel,
    public vision_core::IEmbeddingModel {
public:
    EmotionRecognizer(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider",
                        bool feature_mode = false);

    virtual ~EmotionRecognizer() = default;

    /**
     * @brief Load Emotion ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess face image for emotion recognition
     * @param image Input face image in BGR format (should be cropped face)
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    vision_common::ClassificationResultList classify(const cv::Mat& image) override;

    /**
     * @brief Feature mode: return the raw output vector (no argmax, no L2 norm).
     * @param image Input face image in BGR format (cropped face)
     * @return EmbeddingResult holding the raw feature vector
     */
    vision_common::EmbeddingResult infer_embedding(const cv::Mat& image) override;

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    vision_common::ClassificationResultList postprocess(std::vector<Ort::Value>& outputs);

private:
    int num_threads_;
    cv::Size target_size_;
    std::string provider_;
    bool feature_mode_;
};

}  // namespace vision_deploy

#endif  // EMOTION_RECOGNIZER_H
