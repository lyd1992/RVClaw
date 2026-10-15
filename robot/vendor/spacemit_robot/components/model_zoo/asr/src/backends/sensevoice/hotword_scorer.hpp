/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOTWORD_SCORER_HPP
#define HOTWORD_SCORER_HPP

#include <string>
#include <unordered_map>
#include <vector>

namespace asr {
namespace sensevoice {

class Tokenizer;

class HotwordScorer {
public:
    HotwordScorer() = default;

    void build(const std::vector<std::string>& hotwords,
            Tokenizer& tokenizer, float boost);

    void reset();

    void advanceFrame(int emitted_token);

    void applyBias(const float* frame_logits, int vocab_size,
            int* best_token, float* best_score) const;

    bool empty() const { return nodes_.empty(); }

private:
    void addPath(const std::vector<int>& token_ids);
    struct TrieNode {
        std::unordered_map<int, int> children;
        bool is_end = false;
    };

    std::vector<TrieNode> nodes_;
    float boost_ = 1.0f;
    int vocab_size_ = 0;

    // Active trie positions at current frame (node indices)
    std::vector<int> active_positions_;
};

}  // namespace sensevoice
}  // namespace asr

#endif  // HOTWORD_SCORER_HPP
