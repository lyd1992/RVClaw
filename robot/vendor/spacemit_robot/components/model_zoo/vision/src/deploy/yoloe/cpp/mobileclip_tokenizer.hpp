/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CLIP BPE tokenizer (ported from yolo-world demo; bpe_merges path is now a
 * constructor argument instead of a hardcoded relative path).
 */

#ifndef MOBILECLIP_TOKENIZER_HPP
#define MOBILECLIP_TOKENIZER_HPP

#include <memory>
#include <string>
#include <vector>

namespace vision_deploy {

class MobileClipTokenizer {
public:
    // bpe_merges_path: path to the BPE merges table (required).
    explicit MobileClipTokenizer(const std::string& bpe_merges_path);
    ~MobileClipTokenizer();

    // Main tokenization function that matches Python clip.tokenize().
    std::vector<std::vector<int32_t>> tokenize(const std::string& text, int context_length = 77);

    // Encode text to token IDs (without padding/special tokens).
    std::vector<int32_t> encode(const std::string& text);

    // Decode token IDs back to text.
    std::string decode(const std::vector<int32_t>& tokens);

    // Vocabulary size.
    size_t vocab_size() const;

    // Special token IDs.
    int32_t start_token_id() const { return 49406; }
    int32_t end_token_id() const { return 49407; }
    int32_t pad_token_id() const { return 0; }

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

}  // namespace vision_deploy

#endif  // MOBILECLIP_TOKENIZER_HPP
