/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Streaming CTC model wrapper with cache state management and SpaceMIT EP.
 */

#ifndef ZIPFORMER_CTC_MODEL_H
#define ZIPFORMER_CTC_MODEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace zipformer {

class CtcModel {
public:
    struct Config {
        std::string model_path;
        std::string quantized_model_path;
        std::string provider = "cpu";
        int32_t num_threads = 2;
    };

    explicit CtcModel(const Config& config);
    ~CtcModel();

    std::vector<float> Run(const std::vector<float>& features,
                            int32_t num_frames, int32_t feature_dim);

    int32_t VocabSize() const { return vocab_size_; }
    int32_t ChunkSize() const { return chunk_size_; }
    int32_t ChunkShift() const { return chunk_shift_; }
    int32_t OutputFrames() const { return output_frames_; }

private:
    void InitNames();
    void ReadMetadata();
    std::vector<Ort::Value> CreateInitStates();
    std::vector<float> RunChunk(const std::vector<float>& chunk_features,
                                std::vector<Ort::Value>& states,
                                int32_t& processed_lens, int32_t& out_frames);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Allocator> allocator_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptr_;
    std::vector<const char*> output_names_ptr_;

    int32_t vocab_size_ = 2000;
    int32_t chunk_size_ = 45;
    int32_t chunk_shift_ = 32;
    int32_t output_frames_ = 0;
};

}  // namespace zipformer

#endif  // ZIPFORMER_CTC_MODEL_H
