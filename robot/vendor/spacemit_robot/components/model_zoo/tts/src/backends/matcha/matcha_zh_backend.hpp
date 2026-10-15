/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef MATCHA_ZH_BACKEND_HPP
#define MATCHA_ZH_BACKEND_HPP

#include <cstdint>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/matcha/matcha_backend.hpp"

// Forward declaration for Jieba
namespace cppjieba {
class Jieba;
}  // namespace cppjieba

namespace tts {

// =============================================================================
// Matcha Chinese Backend
// =============================================================================
//
// 中文 TTS 后端实现。
// 使用 Jieba 分词 + Lexicon 查表转换为拼音序列。
//
// 模型: matcha-icefall-zh-baker
// 采样率: 22050 Hz
// 特点: 使用 blank tokens
//

class MatchaZhBackend : public MatchaBackend {
public:
    MatchaZhBackend();
    ~MatchaZhBackend() override;

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
    // 中文文本处理
    // -------------------------------------------------------------------------

    /// @brief 初始化 Jieba 分词器
    void initializeJieba();

    /// @brief 加载词典
    void loadLexicon(const std::string& path);

    /// @brief 将分词后的词转换为 token IDs
    std::vector<int64_t> convertWordToIds(const std::string& word);

    /// @brief 将拼音序列转换为 token IDs
    std::vector<int64_t> convertPhonemesToIds(const std::string& phonemes);

    /// @brief 映射标点符号
    std::string mapPunctuation(const std::string& punct);

    /// @brief 处理拼音映射 (处理未知拼音)
    std::string mapPhoneme(const std::string& phone);

    // -------------------------------------------------------------------------
    // 成员变量
    // -------------------------------------------------------------------------

    std::unique_ptr<cppjieba::Jieba> jieba_;
    std::unordered_map<std::string, std::string> lexicon_;
};

}  // namespace tts

#endif  // MATCHA_ZH_BACKEND_HPP
