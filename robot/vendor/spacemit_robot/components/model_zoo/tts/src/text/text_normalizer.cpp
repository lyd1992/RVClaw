/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "text/text_normalizer.hpp"

#include <cstdlib>

#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "backends/matcha/tts_model_downloader.hpp"
#include "text/fst_normalizer.hpp"

namespace fs = std::filesystem;

namespace tts {
namespace text {

namespace {

// FST 规则应用顺序: date → phone → number。
// phone 必须在 number 之前, 否则 11 位手机号会被 number 读成 "一百三十八亿..."。
constexpr const char* kZhFstOrder[] = {"date.fst", "phone.fst", "number.fst"};

/// @brief 拿 FST 缓存目录 (~/.cache/models/tts/text_norm/v1/zh/)
std::string getZhFstDir() {
    const char* home = std::getenv("HOME");
    std::string base = home ? std::string(home) + "/.cache" : std::string("./.cache");
    return base + "/models/tts/text_norm/v1/zh/";
}

/// @brief 快速判断文本里是否含中文字符 (UTF-8 下 CJK 的前缀字节在 0xE4..0xE9)。
///        仅用于 Language::AUTO 分支早退; 精度够用。
bool hasChineseByte(const std::string& text) {
    for (unsigned char c : text) {
        // CJK Unified Ideographs (U+4E00..U+9FFF) UTF-8 前缀字节范围
        if (c >= 0xE4 && c <= 0xE9) return true;
    }
    return false;
}

/// @brief 补 FST 的盲点, 在送 kaldifst 之前做一次轻量替换。
///        不碰纯阿拉伯数字, 只处理符号 (千分位 ',', 裸 '-', '℃' 等)。
std::string preprocessZh(const std::string& text) {
    std::string out;
    out.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // 千分位逗号: "1,234" → "1234"。要求两侧都是 ASCII 数字。
        if (c == ',' && i > 0 && i + 1 < text.size()) {
            char prev = text[i - 1];
            char next = text[i + 1];
            if (prev >= '0' && prev <= '9' && next >= '0' && next <= '9') {
                // 吞掉逗号
                continue;
            }
        }

        // 负号: 数字前的裸 '-' → "负"。要求 '-' 后紧跟数字且前面不是数字/字母 (避免拆 "COVID-19")。
        if (c == '-' && i + 1 < text.size()) {
            char next = text[i + 1];
            bool next_is_digit = (next >= '0' && next <= '9');
            bool prev_is_alnum = false;
            if (i > 0) {
                unsigned char prev = static_cast<unsigned char>(text[i - 1]);
                prev_is_alnum = (prev >= '0' && prev <= '9') ||
                                (prev >= 'A' && prev <= 'Z') ||
                                (prev >= 'a' && prev <= 'z');
            }
            if (next_is_digit && !prev_is_alnum) {
                out += "负";
                continue;
            }
        }

        // 摄氏度符号 (U+2103, UTF-8: E2 84 83) → "摄氏度"
        if (c == 0xE2 && i + 2 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0x84 &&
            static_cast<unsigned char>(text[i + 2]) == 0x83) {
            out += "摄氏度";
            i += 2;
            continue;
        }

        // 华氏度符号 (U+2109, UTF-8: E2 84 89) → "华氏度"
        if (c == 0xE2 && i + 2 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0x84 &&
            static_cast<unsigned char>(text[i + 2]) == 0x89) {
            out += "华氏度";
            i += 2;
            continue;
        }

        // 人民币符号 ¥ (U+00A5, UTF-8: C2 A5) 和 ￥ (U+FFE5, UTF-8: EF BF A5) → "元"
        // 注: FST 期望 "数字+元", 所以把符号放到数字"之后" 需要上下文感知, 这里简单丢到前面
        //     让 FST 读, 实际效果是"元X元"之类重复, 待之后优化; 先保简单。
        // TODO(tts): 上下文感知的货币符号归一化
        if (c == 0xC2 && i + 1 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0xA5) {
            // 跳过半角 ¥
            ++i;
            continue;
        }
        if (c == 0xEF && i + 2 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0xBF &&
            static_cast<unsigned char>(text[i + 2]) == 0xA5) {
            // 跳过全角 ￥
            i += 2;
            continue;
        }

        out += text[i];
    }
    return out;
}

}  // namespace

// =============================================================================
// 构造与析构
// =============================================================================

TextNormalizer::TextNormalizer() = default;
TextNormalizer::~TextNormalizer() = default;

// =============================================================================
// 主入口
// =============================================================================

std::string TextNormalizer::normalize(const std::string& text, Language lang) {
    if (text.empty()) return text;

    Language effective_lang = (lang == Language::AUTO) ? default_lang_ : lang;
    if (effective_lang == Language::AUTO) {
        effective_lang = hasChineseByte(text) ? Language::ZH : Language::EN;
    }

    // 英文路径: identity 透传。下游 espeak-ng 会处理。
    if (effective_lang == Language::EN) {
        return text;
    }

    // 中文路径: 预处理 + FST
    std::string preprocessed = preprocessZh(text);

    ensureZhLoaded();
    if (!zh_fst_ || !zh_fst_->Ready()) {
        // FST 不可用时退化为 identity + 预处理, 合成仍能跑
        return preprocessed;
    }

    return zh_fst_->Apply(preprocessed);
}

void TextNormalizer::setDefaultLanguage(Language lang) {
    default_lang_ = lang;
}

// =============================================================================
// 延迟加载中文 FST
// =============================================================================

void TextNormalizer::ensureZhLoaded() {
    if (zh_load_attempted_) return;
    zh_load_attempted_ = true;

    // 1) 检查/下载 FST 文件
    TTSModelDownloader downloader;
    if (!downloader.ensureTextNormFiles("zh")) {
        std::cerr << "[TextNormalizer] Failed to ensure FST files for zh, "
            << "falling back to identity normalization" << std::endl;
        return;
    }

    // 2) 加载到 FstNormalizer
    std::string dir = getZhFstDir();
    std::vector<std::string> paths;
    paths.reserve(sizeof(kZhFstOrder) / sizeof(kZhFstOrder[0]));
    for (const auto* name : kZhFstOrder) {
        paths.push_back(dir + name);
    }

    zh_fst_ = std::make_unique<FstNormalizer>();
    if (!zh_fst_->Load(paths)) {
        std::cerr << "[TextNormalizer] Failed to load FST files from " << dir
            << ", falling back to identity normalization" << std::endl;
        zh_fst_.reset();
    }
}

// =============================================================================
// 便捷函数
// =============================================================================

std::string normalizeText(const std::string& text, Language lang) {
    static TextNormalizer normalizer;
    return normalizer.normalize(text, lang);
}

}  // namespace text
}  // namespace tts
