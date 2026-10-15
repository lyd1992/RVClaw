/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MobileClip text encoder (ported from yolo-world demo). Encodes text prompts into
 * per-prompt feature vectors via a tokenizer + MobileClip text ONNX model.
 */

#ifndef MOBILECLIP_HPP
#define MOBILECLIP_HPP

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

#include "mobileclip_tokenizer.hpp"

namespace vision_deploy {

class MobileClip {
public:
    // text_model_path: MobileClip text-encoder ONNX; bpe_merges_path: BPE merges table.
    MobileClip(const std::string& text_model_path, const std::string& bpe_merges_path, int num_threads = 4);
    ~MobileClip();

    // Encode each prompt into a feature vector. Result[i] is the embedding of texts[i].
    std::vector<std::vector<float>> encode(const std::vector<std::string>& texts);

private:
    std::unique_ptr<MobileClipTokenizer> tokenizer_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    int sequence_len_ = 77;
};

}  // namespace vision_deploy

#endif  // MOBILECLIP_HPP
