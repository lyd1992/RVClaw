/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EMOTION_LSTM_H
#define EMOTION_LSTM_H

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
 * @brief Dynamic Emotion LSTM classifier (C++)
 *
 * Stateless: input is a flattened feature sequence of kSeqLen * kFeatureDim
 * floats (10 * 512 = 5120), output is 7 emotion-class probabilities.
 *
 * The 10-frame sliding window of ResNet50 features is maintained by the
 * caller (application layer). Feature extraction is done separately by
 * EmotionRecognizer in feature mode.
 *
 * Reuses the kInferSequence intent / SequenceInput / ActionResult path
 * (same as STGCN), so VisionService::Infer with a sequence request drives it.
 */
class EmotionLstm : public vision_core::BaseModel {
public:
    static constexpr int kSeqLen = 10;
    static constexpr int kFeatureDim = 512;
    static constexpr int kNumClasses = 7;

    EmotionLstm(const std::string& model_path,
                int num_threads = 4,
                bool lazy_load = false,
                const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~EmotionLstm() = default;

    void load_model() override;

    /**
     * @brief Run LSTM on a feature sequence.
     * @param feats Flat array of kSeqLen * kFeatureDim floats, layout [t][d].
     * @return ActionResult with 7-class probabilities (class_scores) + argmax label.
     */
    vision_common::ActionResult predict(const float* feats);

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    size_t expected_sequence_size() const override;

    std::vector<std::string> get_sequence_class_names() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    int num_threads_;
    std::string provider_;
    std::vector<std::string> class_names_;

    /** Run ONNX with a single input tensor of shape {1, kSeqLen, kFeatureDim}. */
    std::vector<Ort::Value> run_session_sequence(
        const float* data,
        const std::vector<int64_t>& shape);
};

}  // namespace vision_deploy

#endif  // EMOTION_LSTM_H
