/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef MATCHA_EN_BACKEND_HPP
#define MATCHA_EN_BACKEND_HPP

#include <cstddef>
#include <cstdint>

#include <string>
#include <unordered_map>
#include <vector>

#include "backends/matcha/matcha_backend.hpp"

namespace tts {

// =============================================================================
// Matcha English Backend
// =============================================================================
//
// 英文 TTS 后端实现。
// 使用 espeak-ng 生成 IPA 音素，转换为 Gruut US 格式。
//
// 模型: matcha-icefall-en_US-ljspeech
// 采样率: 22050 Hz
// 特点: 使用 blank tokens
//

class MatchaEnBackend : public MatchaBackend {
public:
    MatchaEnBackend();
    ~MatchaEnBackend() override;

    ErrorInfo synthesize(const std::string& text, SynthesisResult& result) override;

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
    // 英文文本处理
    // -------------------------------------------------------------------------

    /// @brief 初始化 espeak-ng
    void initializeEspeak();

    /// @brief 应用外部发音词典后转换英文文本
    std::vector<int64_t> processEnglishWithLexicon(const std::string& text);

    /// @brief 将纯英文文本转换为 Matcha English token IDs，不添加起止 token
    std::vector<int64_t> processPlainEnglishToIds(const std::string& text);

    /// @brief 将 Matcha English token 字符串追加为 token IDs
    void appendTokenString(
        const std::string& token_text,
        std::vector<int64_t>* token_ids) const;

    /// @brief 对分段音频边界应用短淡入淡出，避免拼接静音时爆音
    void smoothSegmentBoundary(
        std::vector<float>* audio,
        int sample_rate) const;

    /// @brief 检查 lexicon 命中是否位于英文词边界
    bool isLexiconWordMatch(
        const std::string& text,
        size_t pos,
        size_t len) const;

    // -------------------------------------------------------------------------
    // 成员变量
    // -------------------------------------------------------------------------

    bool espeak_initialized_ = false;
    std::unordered_map<std::string, std::string> lexicon_;
};

}  // namespace tts

#endif  // MATCHA_EN_BACKEND_HPP
