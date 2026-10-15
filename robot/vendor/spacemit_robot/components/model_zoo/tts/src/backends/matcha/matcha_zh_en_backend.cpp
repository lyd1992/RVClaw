/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/matcha/matcha_zh_en_backend.hpp"

#include <cpp-pinyin/G2pglobal.h>
#include <cpp-pinyin/Pinyin.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/matcha/tts_model_downloader.hpp"
#include "text/number_utils.hpp"
#include "text/phoneme_utils.hpp"
#include "text/text_utils.hpp"

namespace fs = std::filesystem;

namespace tts {

// =============================================================================
// 构造与析构
// =============================================================================

MatchaZhEnBackend::MatchaZhEnBackend()
    : MatchaBackend(BackendType::MATCHA_ZH_EN)
    , espeak_initialized_(false) {
}

MatchaZhEnBackend::~MatchaZhEnBackend() {
    shutdownLanguageSpecific();
}

// =============================================================================
// MatchaBackend 纯虚方法实现
// =============================================================================

std::string MatchaZhEnBackend::getModelSubdir() const {
    return "matcha-icefall-zh-en";
}

bool MatchaZhEnBackend::usesBlankTokens() const {
    return false;  // 关键差异: zh-en 不使用 blank tokens
}

ErrorInfo MatchaZhEnBackend::initializeLanguageSpecific(const TtsConfig& config) {
    try {
        // 检查 espeak-ng
        if (!checkEspeakNgAvailable()) {
            return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
                "espeak-ng is required for zh-en TTS but not available.");
        }
        espeak_initialized_ = true;

        // 初始化 cpp-pinyin
        initializePinyin();

        std::cout << "Info: zh-en bilingual mode initialized successfully." << std::endl;
        return ErrorInfo::ok();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
            std::string("Failed to initialize zh-en backend: ") + e.what());
    }
}

void MatchaZhEnBackend::shutdownLanguageSpecific() {
    pinyin_converter_.reset();
    espeak_initialized_ = false;
    lexicon_.clear();
}

ErrorInfo MatchaZhEnBackend::updateLexicon(
    const std::vector<LexiconEntry>& entries) {
    for (const auto& e : entries) {
        lexicon_[e.word] = {e.phoneme, e.locale};
    }
    return ErrorInfo::ok();
}

// =============================================================================
// 初始化
// =============================================================================

void MatchaZhEnBackend::initializePinyin() {
    // 确保 cpp-pinyin 已克隆到 ~/.cache
    TTSModelDownloader downloader;
    if (!downloader.ensureCppPinyin()) {
        throw std::runtime_error("Failed to download cpp-pinyin dictionary.");
    }

    std::string pinyin_dict_dir = downloader.getCppPinyinPath();
    std::cout << "Using cpp-pinyin dictionary at: " << pinyin_dict_dir << std::endl;

    // 设置词典路径
    Pinyin::setDictionaryPath(fs::path(pinyin_dict_dir));

    // 创建 Pinyin 转换器
    pinyin_converter_ = std::make_unique<Pinyin::Pinyin>();

    std::cout << "cpp-pinyin initialized successfully." << std::endl;
}

void MatchaZhEnBackend::initializeEspeak() {
    espeak_initialized_ = checkEspeakNgAvailable();
}

// =============================================================================
// 文本转 Token IDs (中英混合)
// =============================================================================

std::vector<int64_t> MatchaZhEnBackend::textToTokenIds(const std::string& text) {
    return processZhEnText(text);
}

std::vector<int64_t> MatchaZhEnBackend::processZhEnText(const std::string& text) {
    // Lexicon lookup: scan text for exact matches before character-level processing
    if (!lexicon_.empty()) {
        for (const auto& [word, val] : lexicon_) {
            size_t pos = text.find(word);
            if (pos != std::string::npos) {
                std::vector<int64_t> result;
                std::string before = text.substr(0, pos);
                std::string after = text.substr(pos + word.length());
                if (!before.empty()) {
                    auto ids = processZhEnText(before);
                    result.insert(result.end(), ids.begin(), ids.end());
                }
                // locale routing: en → espeak IPA, zh → pinyin tokens
                std::vector<int64_t> lex_ids;
                if (val.locale == "en") {
                    lex_ids = processEnglishToIds(val.phoneme);
                } else {
                    lex_ids = convertPhonemesToIds(val.phoneme);
                }
                result.insert(result.end(), lex_ids.begin(), lex_ids.end());
                if (!after.empty()) {
                    auto ids = processZhEnText(after);
                    result.insert(result.end(), ids.begin(), ids.end());
                }
                return result;
            }
        }
    }

    std::vector<int64_t> token_ids;
    const auto& token_to_id = getTokenToIdMap();
    std::vector<std::string> chars = text::splitUtf8(text);

    // 段级语言判定: 整段无中文字符时交给 espeak-ng, 避免数字被 processArabicNumeralToIds 强行读中文。
    // 匹配 kokoro_phonemizer.cpp:287 的先规范化再分段模式, 与业界段级路由共识一致。
    bool has_chinese = std::any_of(chars.begin(), chars.end(),
        [](const std::string& c) { return text::isChineseChar(c); });
    if (!has_chinese) {
        return processEnglishToIds(text);
    }

    size_t i = 0;
    while (i < chars.size()) {
        if (text::isChineseChar(chars[i])) {
            // 收集连续的中文字符
            std::string chinese_part;
            while (i < chars.size() && text::isChineseChar(chars[i])) {
                chinese_part += chars[i];
                i++;
            }
            // 转换为拼音并获取 IDs
            auto ids = processChineseToPinyinIds(chinese_part);
            token_ids.insert(token_ids.end(), ids.begin(), ids.end());

        } else if (text::isEnglishLetter(chars[i])) {
            // 收集连续的英文单词和空格
            std::string english_part;
            while (i < chars.size()) {
                if (text::isEnglishLetter(chars[i])) {
                    english_part += chars[i];
                    i++;
                } else if (chars[i] == " ") {
                    english_part += chars[i];
                    i++;
                } else {
                    break;
                }
            }
            // 去除尾部空格
            while (!english_part.empty() && english_part.back() == ' ') {
                english_part.pop_back();
            }

            // 处理英文短语，检查罗马数字
            if (!english_part.empty()) {
                std::istringstream iss(english_part);
                std::string word;
                std::vector<std::string> words;
                while (iss >> word) {
                    words.push_back(word);
                }

                std::string non_roman_buffer;
                for (size_t w = 0; w < words.size(); ++w) {
                    if (text::isRomanNumeral(words[w])) {
                        // 先处理缓冲的非罗马数字单词
                        if (!non_roman_buffer.empty()) {
                            auto ids = processEnglishToIds(non_roman_buffer);
                            token_ids.insert(token_ids.end(), ids.begin(), ids.end());
                            non_roman_buffer.clear();
                        }
                        // 处理罗马数字
                        auto ids = processRomanNumeralToIds(words[w]);
                        token_ids.insert(token_ids.end(), ids.begin(), ids.end());
                    } else {
                        if (!non_roman_buffer.empty()) {
                            non_roman_buffer += " ";
                        }
                        non_roman_buffer += words[w];
                    }
                }
                // 处理剩余的非罗马数字单词
                if (!non_roman_buffer.empty()) {
                    auto ids = processEnglishToIds(non_roman_buffer);
                    token_ids.insert(token_ids.end(), ids.begin(), ids.end());
                }
            }

        } else if (text::isDigit(chars[i])) {
            // 收集连续的数字（包括小数点）
            std::string num_part;
            while (i < chars.size() && (text::isDigit(chars[i]) || chars[i] == ".")) {
                num_part += chars[i];
                i++;
            }
            // 转换阿拉伯数字为中文读音
            auto ids = processArabicNumeralToIds(num_part);
            token_ids.insert(token_ids.end(), ids.begin(), ids.end());

        } else {
            // 处理标点和其他字符
            std::string ch = chars[i];
            // 映射中文标点到 ASCII
            if (ch == "，") ch = ",";
            else if (ch == "。") ch = ".";
            else if (ch == "！") ch = "!";
            else if (ch == "？") ch = "?";

            auto it = token_to_id.find(ch);
            if (it != token_to_id.end()) {
                token_ids.push_back(it->second);
            } else {
                token_ids.push_back(1);  // 默认未知 token
            }
            i++;
        }
    }

    return token_ids;
}

std::vector<int64_t> MatchaZhEnBackend::processChineseToPinyinIds(const std::string& chinese_text) {
    std::vector<int64_t> ids;
    const auto& token_to_id = getTokenToIdMap();

    if (!pinyin_converter_) {
        return ids;
    }

    // 使用 cpp-pinyin 转换
    Pinyin::PinyinResVector result = pinyin_converter_->hanziToPinyin(
        chinese_text,
        Pinyin::ManTone::Style::TONE3,  // 声调数字在末尾 (zhong1)
        Pinyin::Error::Default,
        false,  // candidates
        false,  // v_to_u
        true);  // neutral_tone_with_five (轻声用5)

    // 转换每个拼音为 token ID
    for (const auto& res : result) {
        std::string pinyin = res.pinyin;
        auto it = token_to_id.find(pinyin);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            // 尝试小写
            std::string lower_pinyin = pinyin;
            std::transform(lower_pinyin.begin(), lower_pinyin.end(),
                lower_pinyin.begin(), ::tolower);
            auto lower_it = token_to_id.find(lower_pinyin);
            if (lower_it != token_to_id.end()) {
                ids.push_back(lower_it->second);
            } else {
                ids.push_back(1);  // 默认未知 token
            }
        }
    }

    return ids;
}

std::vector<int64_t> MatchaZhEnBackend::processEnglishToIds(const std::string& english_text) {
    std::vector<int64_t> ids;
    const auto& token_to_id = getTokenToIdMap();

    // 获取 IPA
    std::string ipa = processEnglishTextToPhonemes(english_text);
    if (ipa.empty()) {
        return ids;
    }

    // 转换为 gruut en-us 格式
    std::string gruut_ipa = text::convertToGruutEnUs(ipa);

    // 转换每个字符为 token ID
    std::vector<std::string> ipa_chars = text::splitUtf8(gruut_ipa);
    for (const auto& ch : ipa_chars) {
        if (ch.empty()) continue;

        auto it = token_to_id.find(ch);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        }
        // 静默跳过未知 token
    }

    return ids;
}

std::vector<int64_t> MatchaZhEnBackend::processArabicNumeralToIds(const std::string& numStr) {
    // 处理小数
    size_t dotPos = numStr.find('.');
    if (dotPos != std::string::npos) {
        std::string intPart = numStr.substr(0, dotPos);
        std::string decPart = numStr.substr(dotPos + 1);

        std::string chinese;
        if (!intPart.empty()) {
            int64_t intVal = std::stoll(intPart);
            chinese = text::intToChineseReading(intVal);
        } else {
            chinese = "零";
        }
        chinese += "点";

        // 逐位读取小数部分
        static const char* digits[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
        for (char c : decPart) {
            if (c >= '0' && c <= '9') {
                chinese += digits[c - '0'];
            }
        }
        return processChineseToPinyinIds(chinese);
    } else {
        // 整数
        try {
            int64_t value = std::stoll(numStr);
            std::string chinese = text::intToChineseReading(value);
            return processChineseToPinyinIds(chinese);
        } catch (...) {
            // 数字太大，逐位读取
            static const char* digits[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
            std::string chinese;
            for (char c : numStr) {
                if (c >= '0' && c <= '9') {
                    chinese += digits[c - '0'];
                }
            }
            return processChineseToPinyinIds(chinese);
        }
    }
}

std::vector<int64_t> MatchaZhEnBackend::processRomanNumeralToIds(const std::string& roman) {
    int value = text::romanToInt(roman);
    std::string chinese = text::intToChineseReading(value);
    return processChineseToPinyinIds(chinese);
}

std::vector<int64_t> MatchaZhEnBackend::convertPhonemesToIds(const std::string& phonemes) {
    const auto& token_to_id = getTokenToIdMap();
    std::vector<int64_t> ids;
    std::istringstream iss(phonemes);
    std::string phone;

    while (iss >> phone) {
        auto it = token_to_id.find(phone);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            std::string mapped = mapPhoneme(phone);
            if (mapped != phone) {
                auto mapped_it = token_to_id.find(mapped);
                if (mapped_it != token_to_id.end()) {
                    ids.push_back(mapped_it->second);
                }
            }
        }
    }

    return ids;
}

std::string MatchaZhEnBackend::mapPhoneme(const std::string& phone) {
    if (phone.length() > 1) {
        char last = phone.back();
        if (last >= '1' && last <= '5') {
            return phone.substr(0, phone.length() - 1);
        } else {
            return phone + "1";
        }
    }
    return phone;
}

}  // namespace tts
