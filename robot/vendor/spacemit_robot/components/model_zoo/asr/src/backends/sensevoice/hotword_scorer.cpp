/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hotword_scorer.hpp"

#include <string>
#include <utility>
#include <vector>

#include "tokenizer.hpp"

namespace asr {
namespace sensevoice {

static std::vector<std::string> splitUTF8Chars(const std::string& text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 1;
        unsigned char c = text[i];
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len <= text.size()) {
            chars.push_back(text.substr(i, len));
        }
        i += len;
    }
    return chars;
}

static std::vector<int> encodeGreedy(const std::string& word,
        Tokenizer& tokenizer) {
    std::vector<int> tokens;
    auto chars = splitUTF8Chars(word);
    int unk_id = tokenizer.getUnkId();

    size_t pos = 0;
    while (pos < chars.size()) {
        int best_id = -1;
        size_t best_len = 0;
        for (size_t len = chars.size() - pos; len > 0; --len) {
            std::string substr;
            for (size_t j = pos; j < pos + len; ++j) {
                substr += chars[j];
            }
            int tid = tokenizer.tokenToId(substr);
            if (tid != unk_id) {
                best_id = tid;
                best_len = len;
                break;
            }
        }
        if (best_len > 0) {
            tokens.push_back(best_id);
            pos += best_len;
        } else {
            pos++;
        }
    }
    return tokens;
}

void HotwordScorer::addPath(const std::vector<int>& token_ids) {
    if (token_ids.empty()) return;
    int node = 0;
    for (int tid : token_ids) {
        auto it = nodes_[node].children.find(tid);
        if (it == nodes_[node].children.end()) {
            int new_node = static_cast<int>(nodes_.size());
            nodes_[node].children[tid] = new_node;
            nodes_.emplace_back();
            node = new_node;
        } else {
            node = it->second;
        }
    }
    nodes_[node].is_end = true;
}

void HotwordScorer::build(const std::vector<std::string>& hotwords,
        Tokenizer& tokenizer, float boost) {
    nodes_.clear();
    active_positions_.clear();
    boost_ = boost;
    vocab_size_ = static_cast<int>(tokenizer.getVocabSize());

    nodes_.emplace_back();  // root node at index 0

    for (const auto& word : hotwords) {
        auto char_tokens = tokenizer.encode(word);
        addPath(char_tokens);

        auto subword_tokens = encodeGreedy(word, tokenizer);
        if (subword_tokens != char_tokens) {
            addPath(subword_tokens);
        }
    }

    reset();
}

void HotwordScorer::reset() {
    active_positions_.clear();
    active_positions_.push_back(0);  // always start from root
}

void HotwordScorer::advanceFrame(int emitted_token) {
    if (emitted_token < 0) return;

    std::vector<int> next_positions;
    next_positions.push_back(0);  // root is always active

    for (int pos : active_positions_) {
        auto it = nodes_[pos].children.find(emitted_token);
        if (it != nodes_[pos].children.end()) {
            int child = it->second;
            next_positions.push_back(child);
        }
    }

    active_positions_ = std::move(next_positions);
}

void HotwordScorer::applyBias(const float* frame_logits, int vocab_size,
        int* best_token, float* best_score) const {
    for (int pos : active_positions_) {
        if (pos == 0) continue;
        for (const auto& kv : nodes_[pos].children) {
            int tid = kv.first;
            if (tid >= 0 && tid < vocab_size) {
                float boosted = frame_logits[tid] + boost_;
                if (boosted > *best_score) {
                    *best_score = boosted;
                    *best_token = tid;
                }
            }
        }
    }
}

}  // namespace sensevoice
}  // namespace asr
