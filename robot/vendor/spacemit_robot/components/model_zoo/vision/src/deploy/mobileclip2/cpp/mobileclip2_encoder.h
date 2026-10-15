/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MOBILECLIP2_ENCODER_H
#define MOBILECLIP2_ENCODER_H

#include <memory>
#include <string>
#include <vector>

#include "clip_tokenizer.hpp"
#include "embedding_utils.h"
#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class Mobileclip2Encoder : public vision_core::BaseModel, public vision_core::IEmbeddingModel {
public:
    Mobileclip2Encoder(const std::string& image_model_path,
                        const std::string& text_model_path,
                        const std::string& bpe_merges_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    ~Mobileclip2Encoder() override = default;

    void load_model() override;
    cv::Mat preprocess(const cv::Mat& image);
    vision_common::EmbeddingResult infer_embedding(const cv::Mat& image) override;
    vision_common::EmbeddingResult encode_text(const std::string& text);
    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    std::vector<float> postprocess_embedding(std::vector<Ort::Value>& outputs);
    std::vector<float> run_vision(const cv::Mat& blob);
    std::vector<float> run_text(const std::vector<int64_t>& ids);
    void init_text_session();

    std::string text_model_path_;
    std::string bpe_merges_path_;
    int num_threads_;
    std::string provider_;
    int sequence_len_ = 77;

    std::unique_ptr<Ort::Session> text_session_;
    std::unique_ptr<CLIPTokenizer> tokenizer_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> text_input_names_;
    std::vector<const char*> text_output_names_;
    std::vector<std::string> text_input_names_str_;
    std::vector<std::string> text_output_names_str_;
};

}  // namespace vision_deploy

#endif  // MOBILECLIP2_ENCODER_H
