/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TEXT_NORMALIZER_HPP
#define TEXT_NORMALIZER_HPP

/**
 * TextNormalizer - 文本规范化模块
 *
 * 将数字、日期时间、电话号码、货币等特殊格式转换为可读文本。
 *
 * 实现策略:
 *   - 中文 (ZH / AUTO): 走 kaldifst + sherpa-onnx 预编译 FST 规则 (date → phone → number),
 *     FST 前做轻量预处理补盲点 (千分位, -/°C 等符号)。
 *   - 英文 (EN): identity 透传。MatchaEn / Kokoro 的英文通道依赖 espeak-ng,
 *     espeak-ng 内置 TN 已经处理得足够好, 再插一道中间态 TN 反而徒增差错。
 *
 * FST 文件缺失或加载失败时, 自动退化为 identity + 预处理, 不阻塞 TTS 合成。
 */

#include <memory>
#include <string>

namespace tts {
namespace text {

class FstNormalizer;

// =============================================================================
// Language (语言)
// =============================================================================

enum class Language {
    ZH,         // 中文
    EN,         // 英文 (identity, 由下游 espeak-ng 处理)
    AUTO        // 自动检测 (目前等价于 ZH; 英文字符 FST 会原样透传)
};

// =============================================================================
// TextNormalizer (文本规范化器)
// =============================================================================

class TextNormalizer {
public:
    TextNormalizer();
    ~TextNormalizer();

    TextNormalizer(const TextNormalizer&) = delete;
    TextNormalizer& operator=(const TextNormalizer&) = delete;

    /// @brief 规范化文本
    /// @param text 输入文本
    /// @param lang 目标语言 (ZH/EN/AUTO)
    /// @return 规范化后的文本, 异常时原文返回
    std::string normalize(const std::string& text, Language lang = Language::AUTO);

    /// @brief 设置默认语言 (lang=AUTO 时生效)
    void setDefaultLanguage(Language lang);

    /// @brief 获取默认语言
    Language getDefaultLanguage() const { return default_lang_; }

private:
    /// @brief 延迟加载中文 FST (首次 ZH/AUTO 调用时触发, 失败后不再重试)
    void ensureZhLoaded();

    Language default_lang_ = Language::AUTO;

    std::unique_ptr<FstNormalizer> zh_fst_;
    bool zh_load_attempted_ = false;  // 避免加载失败后反复重试
};

// =============================================================================
// 便捷函数
// =============================================================================

/// @brief 规范化文本 (全局单例)
std::string normalizeText(const std::string& text, Language lang = Language::AUTO);

}  // namespace text
}  // namespace tts

#endif  // TEXT_NORMALIZER_HPP
