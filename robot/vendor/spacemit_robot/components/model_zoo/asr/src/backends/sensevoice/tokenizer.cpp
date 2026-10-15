/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file tokenizer.cpp
 * @brief Token decoder implementation for SenseVoice
 */

#include "backends/sensevoice/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace asr {
namespace sensevoice {
namespace {

std::string asciiLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return value;
}

std::string normalizeEmotionToken(const std::string& token) {
    if (token.size() <= 4 || token.rfind("<|", 0) != 0 ||
        token.substr(token.size() - 2) != "|>") {
        return "";
    }

    std::string value = asciiLower(token.substr(2, token.size() - 4));
    if (value == "emo_unknown") {
        return "unknown";
    }
    if (value == "happy" || value == "sad" || value == "angry" ||
        value == "neutral" || value == "fearful" ||
        value == "disgusted" || value == "surprised" ||
        value == "other") {
        return value;
    }
    return "";
}

}  // namespace

Tokenizer::Tokenizer(const Config& config)
    : config_(config)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
{
}

Tokenizer::~Tokenizer() {
    shutdown();
}

bool Tokenizer::initialize() {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Tokenizer");

        if (!loadVocabulary()) {
            std::cerr << "[Tokenizer] Failed to load vocabulary" << std::endl;
            return false;
        }

        if (!config_.decoder_model_path.empty()) {
            // Optional decoder - don't fail if it doesn't load
            initializeDecoder();
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Tokenizer] Init error: " << e.what() << std::endl;
        return false;
    }
}

bool Tokenizer::loadVocabulary() {
    if (config_.vocab_file.empty()) {
        std::cerr << "[Tokenizer] No vocabulary file specified" << std::endl;
        return false;
    }

    std::ifstream file(config_.vocab_file);
    if (!file.is_open()) {
        std::cerr << "[Tokenizer] Cannot open: " << config_.vocab_file << std::endl;
        return false;
    }

    std::string line;
    int id = 0;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Format: "token\tscore" or just "token"
            size_t tab_pos = line.find('\t');
            std::string token = (tab_pos != std::string::npos)
                ? line.substr(0, tab_pos)
                : line;

            id_to_token_[id] = token;
            token_to_id_[token] = id;
            ++id;
        }
    }

    vocab_size_ = id_to_token_.size();
    std::cout << "[Tokenizer] Loaded " << vocab_size_ << " tokens" << std::endl;

    return vocab_size_ > 0;
}

bool Tokenizer::initializeDecoder() {
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.DisableCpuMemArena();
        options.DisableMemPattern();

        decoder_session_ = std::make_unique<Ort::Session>(
            *env_, config_.decoder_model_path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = decoder_session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = decoder_session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.release());
        }

        size_t num_outputs = decoder_session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = decoder_session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.release());
        }

        std::cout << "[Tokenizer] ONNX decoder loaded" << std::endl;
        return true;
    } catch (const std::exception& e) {
        // Decoder is optional
        return false;
    }
}

void Tokenizer::shutdown() {
    cleanupDecoder();
}

void Tokenizer::cleanupDecoder() {
    decoder_session_.reset();
    env_.reset();

    for (auto name : input_names_) {
        delete[] name;
    }
    input_names_.clear();

    for (auto name : output_names_) {
        delete[] name;
    }
    output_names_.clear();
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) {
    // Simple vocabulary-based decoding
    std::vector<std::string> tokens;
    tokens.reserve(token_ids.size());

    for (int id : token_ids) {
        std::string token = idToToken(id);
        if (!token.empty() && token != "<blank>" && token != "<unk>") {
            tokens.push_back(token);
        }
    }

    std::string result = joinTokens(tokens);
    return postProcess(result);
}

Tokenizer::DecodeResult Tokenizer::decodeWithMetadata(
    const std::vector<int>& token_ids) {
    // Simple vocabulary-based decoding
    std::vector<std::string> tokens;
    tokens.reserve(token_ids.size());
    DecodeResult decoded;

    for (int id : token_ids) {
        std::string token = idToToken(id);
        if (!token.empty() && token != "<blank>" && token != "<unk>") {
            if (decoded.emotion.empty()) {
                decoded.emotion = normalizeEmotionToken(token);
            }
            tokens.push_back(token);
        }
    }

    std::string result = joinTokens(tokens);
    decoded.text = postProcess(result);
    return decoded;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    auto chars = splitUTF8(text);

    for (const auto& ch : chars) {
        ids.push_back(tokenToId(ch));
    }

    return ids;
}

std::string Tokenizer::idToToken(int id) const {
    auto it = id_to_token_.find(id);
    return (it != id_to_token_.end()) ? it->second : "<unk>";
}

int Tokenizer::tokenToId(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : unk_id_;
}

std::string Tokenizer::postProcess(const std::string& text) {
    std::string result = text;

    // Static regex objects (expensive to construct)
    static const std::regex re_special("<\\|[^|]*\\|>");
    static const std::regex re_scores("-?\\d+\\.\\d+");
    // static const std::regex re_numbers("\\d+");
    static const std::regex re_questions("\\?\\s*\\?");
    static const std::regex re_underscore("▁");
    static const std::regex re_whitespace("\\s+");

    result = std::regex_replace(result, re_special, "");
    result = std::regex_replace(result, re_scores, "");
    // result = std::regex_replace(result, re_numbers, "");
    result = std::regex_replace(result, re_questions, "");
    result = std::regex_replace(result, re_underscore, " ");
    result = std::regex_replace(result, re_whitespace, " ");

    // Trim
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");

    if (start == std::string::npos) {
        return "";
    }

    return result.substr(start, end - start + 1);
}

std::vector<std::string> Tokenizer::splitUTF8(const std::string& text) {
    std::vector<std::string> result;

    size_t i = 0;
    while (i < text.length()) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;

        if (i + len <= text.length()) {
            std::string ch = text.substr(i, len);
            if (ch != " " && ch != "\t" && ch != "\n") {
                result.push_back(ch);
            }
        }

        i += len;
    }

    return result;
}

std::string Tokenizer::joinTokens(const std::vector<std::string>& tokens) {
    std::string result;
    for (const auto& token : tokens) {
        result += token;
    }
    return result;
}

}  // namespace sensevoice
}  // namespace asr
