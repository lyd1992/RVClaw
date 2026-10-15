/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CLIP text encoder (ported from yolo-world demo). Encodes text prompts into
 * per-prompt feature vectors via a tokenizer + CLIP text ONNX model.
 */

#ifndef CLIP_HPP
#define CLIP_HPP

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

#include "clip_tokenizer.hpp"

namespace vision_deploy {

class CLIP {
public:
    // text_model_path: CLIP text-encoder ONNX; bpe_merges_path: BPE merges table.
    CLIP(const std::string& text_model_path, const std::string& bpe_merges_path, int num_threads = 4);
    ~CLIP();

    // Encode each prompt into a feature vector. Result[i] is the embedding of texts[i].
    std::vector<std::vector<float>> encode(const std::vector<std::string>& texts);

private:
    std::unique_ptr<CLIPTokenizer> tokenizer_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    int sequence_len_ = 77;
};

}  // namespace vision_deploy

#endif  // CLIP_HPP
