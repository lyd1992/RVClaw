/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Ported from yolo-world demo. Only change: BPE merges path is passed in
 * (constructor arg) rather than the hardcoded "../../data/bpe_merges.txt".
 */

#include "mobileclip_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vision_deploy {

namespace {
// Hash function for string pairs used by the BPE rank table.
struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};
}  // namespace

class MobileClipTokenizer::Impl {
public:
    explicit Impl(const std::string& bpe_merges_path);

    std::vector<int32_t> encode(const std::string& text);
    std::string decode(const std::vector<int32_t>& tokens);
    size_t vocab_size() const { return encoder.size(); }

private:
    std::unordered_map<std::string, int32_t> encoder;
    std::unordered_map<int32_t, std::string> decoder;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks;
    std::unordered_map<uint8_t, std::string> byte_encoder;
    std::unordered_map<std::string, uint8_t> byte_decoder;
    std::unordered_map<std::string, std::string> cache;
    std::regex pat;
    std::string bpe_merges_path_;

    void initialize_byte_encoder();
    void load_vocabulary();
    std::string bpe(const std::string& token);
    std::vector<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word);
    std::string basic_clean(const std::string& text);
    std::string whitespace_clean(const std::string& text);
    std::string lowercase(const std::string& text);
};

MobileClipTokenizer::Impl::Impl(const std::string& bpe_merges_path)
    : pat(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]+)",
        std::regex::icase),
        bpe_merges_path_(bpe_merges_path) {
    initialize_byte_encoder();
    load_vocabulary();
    cache["<|startoftext|>"] = "<|startoftext|>";
    cache["<|endoftext|>"] = "<|endoftext|>";
}

void MobileClipTokenizer::Impl::initialize_byte_encoder() {
    std::vector<int> bs;
    for (int i = 33; i <= 126; i++) bs.push_back(i);
    for (int i = 161; i <= 172; i++) bs.push_back(i);
    for (int i = 174; i <= 255; i++) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        if (cs[i] < 128) {
            byte_encoder[bs[i]] = std::string(1, static_cast<char>(cs[i]));
        } else {
            std::string result;
            if (cs[i] < 0x800) {
                result += static_cast<char>(0xC0 | (cs[i] >> 6));
                result += static_cast<char>(0x80 | (cs[i] & 0x3F));
            } else {
                result += static_cast<char>(0xE0 | (cs[i] >> 12));
                result += static_cast<char>(0x80 | ((cs[i] >> 6) & 0x3F));
                result += static_cast<char>(0x80 | (cs[i] & 0x3F));
            }
            byte_encoder[bs[i]] = result;
        }
        byte_decoder[byte_encoder[bs[i]]] = bs[i];
    }
}

void MobileClipTokenizer::Impl::load_vocabulary() {
    std::ifstream merges_file(bpe_merges_path_);
    if (!merges_file.is_open()) {
        throw std::runtime_error("MobileClipTokenizer: could not open bpe_merges file: " + bpe_merges_path_);
    }

    std::string line;
    int rank = 0;
    while (std::getline(merges_file, line)) {
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string first, second;
            if (iss >> first >> second) {
                bpe_ranks[{first, second}] = rank++;
            }
        }
    }
    merges_file.close();

    std::vector<std::string> vocab;
    std::vector<int> byte_order;
    for (int i = 33; i <= 126; i++) byte_order.push_back(i);
    for (int i = 161; i <= 172; i++) byte_order.push_back(i);
    for (int i = 174; i <= 255; i++) byte_order.push_back(i);
    for (int b = 0; b < 256; b++) {
        if (std::find(byte_order.begin(), byte_order.end(), b) == byte_order.end()) {
            byte_order.push_back(b);
        }
    }

    for (int b : byte_order) {
        vocab.push_back(byte_encoder[b]);
    }
    size_t base_size = vocab.size();
    for (size_t i = 0; i < base_size; i++) {
        vocab.push_back(vocab[i] + "</w>");
    }

    std::vector<std::pair<std::pair<std::string, std::string>, int>> sorted_ranks;
    for (const auto& pair : bpe_ranks) {
        sorted_ranks.push_back(pair);
    }
    std::sort(sorted_ranks.begin(), sorted_ranks.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& pair : sorted_ranks) {
        vocab.push_back(pair.first.first + pair.first.second);
    }

    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");

    for (size_t i = 0; i < vocab.size(); i++) {
        encoder[vocab[i]] = static_cast<int32_t>(i);
        decoder[static_cast<int32_t>(i)] = vocab[i];
    }
}

std::vector<std::pair<std::string, std::string>> MobileClipTokenizer::Impl::get_pairs(
    const std::vector<std::string>& word) {
    std::vector<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2) return pairs;
    for (size_t i = 0; i < word.size() - 1; i++) {
        pairs.push_back({word[i], word[i + 1]});
    }
    return pairs;
}

std::string MobileClipTokenizer::Impl::bpe(const std::string& token) {
    if (cache.find(token) != cache.end()) {
        return cache[token];
    }

    std::vector<std::string> word;
    for (size_t i = 0; i + 1 < token.length(); i++) {
        word.push_back(std::string(1, token[i]));
    }
    if (!token.empty()) {
        word.push_back(std::string(1, token.back()) + "</w>");
    }

    auto pairs = get_pairs(word);
    if (pairs.empty()) {
        return token + "</w>";
    }

    while (true) {
        int min_rank = std::numeric_limits<int>::max();
        std::pair<std::string, std::string> bigram;
        bool found = false;
        for (const auto& pair : pairs) {
            auto it = bpe_ranks.find(pair);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                bigram = pair;
                found = true;
            }
        }
        if (!found) break;

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == bigram.first && word[i + 1] == bigram.second) {
                new_word.push_back(bigram.first + bigram.second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;
        if (word.size() == 1) break;
        pairs = get_pairs(word);
    }

    std::string result;
    for (size_t i = 0; i < word.size(); i++) {
        if (i > 0) result += " ";
        result += word[i];
    }
    cache[token] = result;
    return result;
}

std::string MobileClipTokenizer::Impl::basic_clean(const std::string& text) {
    return text;
}

std::string MobileClipTokenizer::Impl::whitespace_clean(const std::string& text) {
    std::string result;
    bool prev_space = true;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }
    size_t first = result.find_first_not_of(' ');
    size_t last = result.find_last_not_of(' ');
    if (first == std::string::npos) return "";
    return result.substr(first, last - first + 1);
}

std::string MobileClipTokenizer::Impl::lowercase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

std::vector<int32_t> MobileClipTokenizer::Impl::encode(const std::string& text) {
    std::vector<int32_t> bpe_tokens;
    std::string cleaned = lowercase(whitespace_clean(basic_clean(text)));

    std::regex word_regex(R"(\S+)");
    auto words_begin = std::sregex_iterator(cleaned.begin(), cleaned.end(), word_regex);
    auto words_end = std::sregex_iterator();

    for (auto it = words_begin; it != words_end; ++it) {
        std::string token = it->str();
        std::string byte_token;
        for (unsigned char c : token) {
            byte_token += byte_encoder[c];
        }
        std::string bpe_result = bpe(byte_token);
        std::istringstream iss(bpe_result);
        std::string bpe_token;
        while (iss >> bpe_token) {
            auto found = encoder.find(bpe_token);
            if (found != encoder.end()) {
                bpe_tokens.push_back(found->second);
            }
        }
    }
    return bpe_tokens;
}

std::string MobileClipTokenizer::Impl::decode(const std::vector<int32_t>& tokens) {
    std::string text;
    for (int32_t token : tokens) {
        auto it = decoder.find(token);
        if (it != decoder.end()) {
            text += it->second;
        }
    }
    std::string result;
    for (const auto& ch : text) {
        auto it = byte_decoder.find(std::string(1, ch));
        if (it != byte_decoder.end()) {
            result += static_cast<char>(it->second);
        }
    }
    size_t pos = 0;
    while ((pos = result.find("</w>", pos)) != std::string::npos) {
        result.replace(pos, 4, " ");
        pos += 1;
    }
    return result;
}

MobileClipTokenizer::MobileClipTokenizer(const std::string& bpe_merges_path)
    : pImpl(std::make_unique<Impl>(bpe_merges_path)) {}

MobileClipTokenizer::~MobileClipTokenizer() = default;

std::vector<std::vector<int32_t>> MobileClipTokenizer::tokenize(const std::string& text, int context_length) {
    auto tokens = pImpl->encode(text);
    std::vector<int32_t> result;
    result.push_back(start_token_id());
    result.insert(result.end(), tokens.begin(), tokens.end());
    result.push_back(end_token_id());

    if (result.size() > static_cast<size_t>(context_length)) {
        result.resize(context_length);
        result.back() = end_token_id();
    }
    while (result.size() < static_cast<size_t>(context_length)) {
        result.push_back(pad_token_id());
    }
    return {result};
}

std::vector<int32_t> MobileClipTokenizer::encode(const std::string& text) {
    return pImpl->encode(text);
}

std::string MobileClipTokenizer::decode(const std::vector<int32_t>& tokens) {
    return pImpl->decode(tokens);
}

size_t MobileClipTokenizer::vocab_size() const {
    return pImpl->vocab_size();
}

}  // namespace vision_deploy
