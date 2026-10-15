/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef MATCHA_ZH_EN_BACKEND_HPP
#define MATCHA_ZH_EN_BACKEND_HPP

#include <cstdint>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/matcha/matcha_backend.hpp"

// Forward declaration for cpp-pinyin
namespace Pinyin {
class Pinyin;
}  // namespace Pinyin

namespace tts {

// =============================================================================
// Matcha Chinese-English Bilingual Backend
// =============================================================================
//
// 中英混合 TTS 后端实现。
// 使用 cpp-pinyin 处理中文，espeak-ng 处理英文。
//
// 模型: matcha-icefall-zh-en
// 采样率: 16000 Hz
// 特点: 不使用 blank tokens (与 zh/en 后端不同)
//

class MatchaZhEnBackend : public MatchaBackend {
public:
    MatchaZhEnBackend();
    ~MatchaZhEnBackend() override;

protected:
    // -------------------------------------------------------------------------
    // MatchaBackend 纯虚方法实现
    // -------------------------------------------------------------------------

    std::vector<int64_t> textToTokenIds(const std::string& text) override;
    std::string getModelSubdir() const override;
    bool usesBlankTokens() const override;
    ErrorInfo initializeLanguageSpecific(const TtsConfig& config) override;
    void shutdownLanguageSpecific() override;
    ErrorInfo updateLexicon(
        const std::vector<LexiconEntry>& entries) override;

private:
    // -------------------------------------------------------------------------
    // 中英混合文本处理
    // -------------------------------------------------------------------------

    /// @brief 初始化 cpp-pinyin
    void initializePinyin();

    /// @brief 初始化 espeak-ng
    void initializeEspeak();

    /// @brief 处理中英混合文本
    std::vector<int64_t> processZhEnText(const std::string& text);

    /// @brief 将中文转换为拼音并获取 token IDs
    std::vector<int64_t> processChineseToPinyinIds(const std::string& chinese_text);

    /// @brief 将英文转换为 IPA 并获取 token IDs
    std::vector<int64_t> processEnglishToIds(const std::string& english_text);

    /// @brief 处理阿拉伯数字转中文读音
    std::vector<int64_t> processArabicNumeralToIds(const std::string& numStr);

    /// @brief 处理罗马数字转中文读音
    std::vector<int64_t> processRomanNumeralToIds(const std::string& roman);

    /// @brief 将拼音序列转换为 token IDs
    std::vector<int64_t> convertPhonemesToIds(const std::string& phonemes);

    /// @brief 处理拼音映射 (处理未知拼音)
    std::string mapPhoneme(const std::string& phone);

    // -------------------------------------------------------------------------
    // 成员变量
    // -------------------------------------------------------------------------

    std::unique_ptr<Pinyin::Pinyin> pinyin_converter_;
    bool espeak_initialized_ = false;

    struct LexiconValue {
        std::string phoneme;
        std::string locale;
    };
    std::unordered_map<std::string, LexiconValue> lexicon_;
};

}  // namespace tts

#endif  // MATCHA_ZH_EN_BACKEND_HPP
