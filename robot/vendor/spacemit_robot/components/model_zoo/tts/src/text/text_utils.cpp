/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "text/text_utils.hpp"

#include <cstdint>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tts {
namespace text {

// =============================================================================
// UTF-8 字符串处理
// =============================================================================

std::vector<std::string> splitUtf8(const std::string& str) {
    std::vector<std::string> result;
    for (size_t i = 0; i < str.length();) {
        int char_len = 1;
        unsigned char c = str[i];

        // Determine UTF-8 character length
        if ((c & 0x80) == 0) {
            char_len = 1;  // ASCII
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;  // 2-byte UTF-8
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;  // 3-byte UTF-8
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;  // 4-byte UTF-8
        }

        if (i + char_len <= str.length()) {
            result.push_back(str.substr(i, char_len));
        }
        i += char_len;
    }
    return result;
}

// =============================================================================
// 字符类型判断
// =============================================================================

bool isChinese(unsigned char ch) {
    // Simple check for Chinese characters using UTF-8 byte patterns
    // Chinese characters typically start with bytes in range 0xE4-0xE9
    return ch >= 0xE4 && ch <= 0xE9;
}

bool isChineseChar(const std::string& ch) {
    if (ch.length() != 3) return false;
    unsigned char c0 = ch[0];
    unsigned char c1 = ch[1];
    // CJK Unified Ideographs: U+4E00 to U+9FFF (3-byte UTF-8: E4 B8 80 to E9 BF BF)
    if (c0 >= 0xE4 && c0 <= 0xE9) {
        if (c0 == 0xE4 && c1 < 0xB8) return false;
        if (c0 == 0xE9 && c1 > 0xBF) return false;
        return true;
    }
    return false;
}

bool containsChinese(const std::string& text) {
    for (size_t i = 0; i < text.length(); i++) {
        unsigned char ch = static_cast<unsigned char>(text[i]);
        if (isChinese(ch)) {
            return true;
        }
    }
    return false;
}

bool isEnglishLetter(const std::string& ch) {
    if (ch.length() != 1) return false;
    char c = ch[0];
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool isDigit(const std::string& ch) {
    if (ch.length() != 1) return false;
    char c = ch[0];
    return c >= '0' && c <= '9';
}

// =============================================================================
// 标点符号处理
// =============================================================================

bool isPunctuation(const std::string& s) {
    static const std::unordered_set<std::string> puncts = {
        ",", ".", "!", "?", ":", "\"", "'", "，",
        "。", "！", "？", """, """, "'", "'", "；", "、",
        "—", "–", "…", "-", "(", ")", "（", "）",
        "[", "]", "【", "】", "{", "}", "《", "》"
    };
    return puncts.count(s) > 0;
}

std::string mapChinesePunctToAscii(const std::string& punct) {
    if (punct == "！") return "!";
    if (punct == "？") return "?";
    if (punct == "，") return ",";
    if (punct == "。") return ".";
    if (punct == "：") return ":";
    if (punct == "；") return ";";
    if (punct == "、") return ",";
    if (punct == "'") return "'";
    if (punct == "'") return "'";
    if (punct == "—") return "-";  // em-dash to hyphen
    if (punct == "–") return "-";  // en-dash to hyphen
    if (punct == "…") return "...";  // ellipsis
    return punct;  // Return original if no mapping
}

std::string mapPunctuation(const std::string& punct,
    const std::unordered_map<std::string, int64_t>& token_to_id) {
    // Try to find the punctuation directly in tokens first
    auto direct_it = token_to_id.find(punct);
    if (direct_it != token_to_id.end()) {
        return punct;
    }

    // Try Chinese to ASCII mapping
    std::string ascii_punct = mapChinesePunctToAscii(punct);
    if (ascii_punct != punct) {
        auto ascii_it = token_to_id.find(ascii_punct);
        if (ascii_it != token_to_id.end()) {
            return ascii_punct;
        }
    }

    // Try to find common pause tokens for major punctuations
    if (punct == "。" || punct == "！" || punct == "？" ||
        punct == "." || punct == "!" || punct == "?") {
        if (token_to_id.count("sil")) return "sil";
        if (token_to_id.count("sp")) return "sp";
        if (token_to_id.count("<eps>")) return "<eps>";
    }

    return "";  // No mapping found
}

}  // namespace text
}  // namespace tts
