/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gemma_tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace vision_deploy {

namespace {

std::string read_str(std::ifstream& f) {
    uint16_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    if (len > 0) {
        f.read(s.data(), len);
    }
    return s;
}

std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if ((c & 0x80) == 0x00) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        chars.push_back(s.substr(i, static_cast<size_t>(len)));
        i += static_cast<size_t>(len);
    }
    return chars;
}

}  // namespace

GemmaTokenizer::GemmaTokenizer(const std::string& bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("GemmaTokenizer: cannot open tokenizer: " + bin_path);
    }

    uint32_t vocab_size = 0;
    f.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    id_to_token_.resize(vocab_size);
    token_to_id_.reserve(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        id_to_token_[i] = read_str(f);
        token_to_id_[id_to_token_[i]] = static_cast<int32_t>(i);
    }

    uint32_t merge_count = 0;
    f.read(reinterpret_cast<char*>(&merge_count), sizeof(merge_count));
    merge_rank_.reserve(merge_count);
    for (uint32_t rank = 0; rank < merge_count; ++rank) {
        const std::string a = read_str(f);
        const std::string b = read_str(f);
        merge_rank_[a + '\x00' + b] = static_cast<int32_t>(rank);
    }
}

std::vector<int32_t> GemmaTokenizer::bpe_segment(const std::string& seg) const {
    auto chars = split_utf8(seg);
    if (chars.empty()) {
        return {};
    }

    std::vector<std::string> tokens = chars;
    while (tokens.size() > 1) {
        int best_rank = INT32_MAX;
        int best_i = -1;
        for (int i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
            const std::string key = tokens[static_cast<size_t>(i)] + '\x00' +
                                    tokens[static_cast<size_t>(i) + 1];
            const auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i == -1) {
            break;
        }
        tokens[static_cast<size_t>(best_i)] += tokens[static_cast<size_t>(best_i) + 1];
        tokens.erase(tokens.begin() + best_i + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens) {
        const auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
            continue;
        }
        for (unsigned char byte : tok) {
            char buf[8];
            snprintf(buf, sizeof(buf), "<0x%02X>", byte);
            const auto it2 = token_to_id_.find(buf);
            if (it2 != token_to_id_.end()) {
                ids.push_back(it2->second);
            }
        }
    }
    return ids;
}

std::vector<int64_t> GemmaTokenizer::encode(const std::string& text, int max_len) const {
    static const std::string kSpaceMark = "\xe2\x96\x81";
    std::string normalized;
    normalized.reserve(text.size() + 32);
    for (char ch : text) {
        if (ch == ' ') {
            normalized += kSpaceMark;
        } else {
            normalized += ch;
        }
    }

    const auto ids32 = bpe_segment(normalized);
    std::vector<int64_t> out(static_cast<size_t>(max_len), kPadId);
    int pos = 0;
    for (int32_t id : ids32) {
        if (pos >= max_len - 1) {
            break;
        }
        out[static_cast<size_t>(pos++)] = id;
    }
    if (pos < max_len) {
        out[static_cast<size_t>(pos)] = kEosId;
    }
    return out;
}

}  // namespace vision_deploy
