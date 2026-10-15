/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/matcha/matcha_zh_backend.hpp"

#include <cstdint>

#include <algorithm>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cppjieba/Jieba.hpp"
#include "backends/matcha/tts_model_downloader.hpp"
#include "text/number_utils.hpp"
#include "text/text_utils.hpp"

namespace fs = std::filesystem;

namespace tts {

// =============================================================================
// 构造与析构
// =============================================================================

MatchaZhBackend::MatchaZhBackend()
    : MatchaBackend(BackendType::MATCHA_ZH) {
}

MatchaZhBackend::~MatchaZhBackend() {
    shutdownLanguageSpecific();
}

// =============================================================================
// MatchaBackend 纯虚方法实现
// =============================================================================

std::string MatchaZhBackend::getModelSubdir() const {
    return "matcha-icefall-zh-baker";
}

bool MatchaZhBackend::usesBlankTokens() const {
    return true;
}

ErrorInfo MatchaZhBackend::initializeLanguageSpecific(const TtsConfig& config) {
    try {
        // 初始化 Jieba
        initializeJieba();

        // 加载词典
        const auto& internal_config = getInternalConfig();
        if (!internal_config.lexicon_path.empty() &&
            fs::exists(internal_config.lexicon_path)) {
            loadLexicon(internal_config.lexicon_path);
        } else {
            std::cout << "Warning: Lexicon file not found. Continuing without lexicon." << std::endl;
        }

        return ErrorInfo::ok();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
            std::string("Failed to initialize Chinese backend: ") + e.what());
    }
}

void MatchaZhBackend::shutdownLanguageSpecific() {
    jieba_.reset();
    lexicon_.clear();
}

ErrorInfo MatchaZhBackend::updateLexicon(
    const std::vector<LexiconEntry>& entries) {
    for (const auto& e : entries) {
        lexicon_[e.word] = e.phoneme;
    }
    return ErrorInfo::ok();
}

// =============================================================================
// 文本转 Token IDs (中文)
// =============================================================================

std::vector<int64_t> MatchaZhBackend::textToTokenIds(const std::string& text) {
    std::vector<int64_t> token_ids;

    if (!jieba_) {
        std::cerr << "Error: Jieba not initialized" << std::endl;
        return token_ids;
    }

    // Step 1: 替换标点符号 (遵循 sherpa-onnx 模式)
    std::string processed_text = text;
    std::regex punct_re1("：|、|；");
    processed_text = std::regex_replace(processed_text, punct_re1, "，");
    std::regex punct_re2("[.]");
    processed_text = std::regex_replace(processed_text, punct_re2, "。");
    std::regex punct_re3("[?]");
    processed_text = std::regex_replace(processed_text, punct_re3, "？");
    std::regex punct_re4("[!]");
    processed_text = std::regex_replace(processed_text, punct_re4, "！");

    // Step 2: Jieba 分词
    std::vector<std::string> words;
    jieba_->Cut(processed_text, words, true);

    // Step 3: 移除冗余空格和标点 (遵循 sherpa-onnx)
    std::vector<std::string> cleaned_words;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i == 0) {
            cleaned_words.push_back(words[i]);
        } else if (words[i] == " ") {
            if (cleaned_words.back() == " " || text::isPunctuation(cleaned_words.back())) {
                continue;
            } else {
                cleaned_words.push_back(words[i]);
            }
        } else if (text::isPunctuation(words[i])) {
            if (cleaned_words.back() == " " || text::isPunctuation(cleaned_words.back())) {
                continue;
            } else {
                cleaned_words.push_back(words[i]);
            }
        } else {
            cleaned_words.push_back(words[i]);
        }
    }

    // Step 4: 将词转换为 token IDs
    for (const auto& word : cleaned_words) {
        auto word_ids = convertWordToIds(word);
        if (!word_ids.empty()) {
            token_ids.insert(token_ids.end(), word_ids.begin(), word_ids.end());
        }
    }

    return token_ids;
}

// =============================================================================
// 私有方法
// =============================================================================

void MatchaZhBackend::initializeJieba() {
    // 确保 cppjieba 已克隆到 ~/.cache
    TTSModelDownloader downloader;
    if (!downloader.ensureCppJieba()) {
        throw std::runtime_error("Failed to download cppjieba dictionary.");
    }

    std::string jieba_dir = downloader.getCppJiebaPath();
    std::cout << "Using cppjieba dictionary at: " << jieba_dir << std::endl;

    std::string dict_path = jieba_dir + "/jieba.dict.utf8";
    std::string hmm_path = jieba_dir + "/hmm_model.utf8";
    std::string user_dict = jieba_dir + "/user.dict.utf8";
    std::string idf_path = jieba_dir + "/idf.utf8";
    std::string stop_words = jieba_dir + "/stop_words.utf8";

    jieba_ = std::make_unique<cppjieba::Jieba>(
        dict_path, hmm_path, user_dict, idf_path, stop_words);
}

void MatchaZhBackend::loadLexicon(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open lexicon file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string word;
        iss >> word;

        std::string phonemes;
        std::getline(iss, phonemes);

        // 去除首尾空白
        size_t start = phonemes.find_first_not_of(" \t");
        if (start != std::string::npos) {
            size_t end = phonemes.find_last_not_of(" \t");
            phonemes = phonemes.substr(start, end - start + 1);
        }

        if (!word.empty() && !phonemes.empty()) {
            lexicon_[word] = phonemes;
        }
    }

    std::cout << "Loaded " << lexicon_.size() << " entries from lexicon." << std::endl;
}

std::vector<int64_t> MatchaZhBackend::convertWordToIds(const std::string& word) {
    const auto& token_to_id = getTokenToIdMap();

    // Skip words containing English letters — zh model cannot pronounce them
    auto chars_check = text::splitUtf8(word);
    bool has_english = std::any_of(chars_check.begin(), chars_check.end(),
        [](const std::string& c) { return text::isEnglishLetter(c); });
    if (has_english) {
        return {};
    }

    // 转小写查找
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);

    // 1. 尝试在词典中查找
    if (!lexicon_.empty()) {
        auto lex_it = lexicon_.find(lower_word);
        if (lex_it != lexicon_.end()) {
            return convertPhonemesToIds(lex_it->second);
        }
    }

    // 2. 尝试直接 token 查找
    auto token_it = token_to_id.find(word);
    if (token_it != token_to_id.end()) {
        return {token_it->second};
    }

    // 3. 处理标点符号
    if (text::isPunctuation(word)) {
        std::string punct_token = mapPunctuation(word);
        if (!punct_token.empty()) {
            auto punct_it = token_to_id.find(punct_token);
            if (punct_it != token_to_id.end()) {
                return {punct_it->second};
            }
        }
    }

    // 4. 字符级回退
    std::vector<int64_t> result;
    std::vector<std::string> chars = text::splitUtf8(word);

    for (const auto& char_str : chars) {
        if (!lexicon_.empty()) {
            auto char_lex_it = lexicon_.find(char_str);
            if (char_lex_it != lexicon_.end()) {
                auto char_ids = convertPhonemesToIds(char_lex_it->second);
                result.insert(result.end(), char_ids.begin(), char_ids.end());
            } else {
                auto char_token_it = token_to_id.find(char_str);
                if (char_token_it != token_to_id.end()) {
                    result.push_back(char_token_it->second);
                }
            }
        } else {
            auto char_token_it = token_to_id.find(char_str);
            if (char_token_it != token_to_id.end()) {
                result.push_back(char_token_it->second);
            }
        }
    }

    return result;
}

std::vector<int64_t> MatchaZhBackend::convertPhonemesToIds(const std::string& phonemes) {
    const auto& token_to_id = getTokenToIdMap();
    std::vector<int64_t> ids;
    std::istringstream iss(phonemes);
    std::string phone;

    while (iss >> phone) {
        auto token_it = token_to_id.find(phone);
        if (token_it != token_to_id.end()) {
            ids.push_back(token_it->second);
        } else {
            // 尝试回退映射
            std::string mapped_phone = mapPhoneme(phone);
            if (mapped_phone != phone) {
                auto mapped_it = token_to_id.find(mapped_phone);
                if (mapped_it != token_to_id.end()) {
                    ids.push_back(mapped_it->second);
                }
            }
        }
    }

    return ids;
}

std::string MatchaZhBackend::mapPunctuation(const std::string& punct) {
    return text::mapPunctuation(punct, getTokenToIdMap());
}

std::string MatchaZhBackend::mapPhoneme(const std::string& phone) {
    // 处理常见的拼音不匹配
    static std::unordered_map<std::string, std::string> phoneme_mapping = {
        {"shei2", "she2"},
        {"cei2", "ce2"},
        {"den1", "de1"},
        {"den2", "de2"},
        {"den3", "de3"},
        {"den4", "de4"},
        {"kei2", "ke2"},
        {"kei3", "ke3"},
        {"nei1", "ne1"},
        {"pou1", "po1"},
        {"pou2", "po2"},
        {"pou3", "po3"},
        {"yo1", "yo"},
        {"m2", "m"},
        {"n2", "n"},
        {"ng2", "ng"},
        {"hm", "hm1"},
    };

    auto it = phoneme_mapping.find(phone);
    if (it != phoneme_mapping.end()) {
        return it->second;
    }

    // 如果没有直接映射，尝试去除或更改声调
    if (phone.length() > 1) {
        char last_char = phone.back();
        if (last_char >= '1' && last_char <= '4') {
            return phone.substr(0, phone.length() - 1);
        } else {
            return phone + "1";
        }
    }

    return phone;
}

}  // namespace tts
