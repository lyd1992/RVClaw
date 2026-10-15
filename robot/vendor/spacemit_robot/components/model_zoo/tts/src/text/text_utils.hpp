/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TEXT_UTILS_HPP
#define TEXT_UTILS_HPP

/**
 * TextUtils - 文本处理工具模块
 *
 * 提供 UTF-8 字符串处理、标点符号处理、字符类型判断等功能。
 */

#include <cstdint>

#include <string>
#include <unordered_map>
#include <vector>

namespace tts {
namespace text {

// =============================================================================
// UTF-8 字符串处理
// =============================================================================

/**
 * @brief 将 UTF-8 字符串分割为单个字符
 * @param str UTF-8 编码的字符串
 * @return 每个 UTF-8 字符组成的向量
 */
std::vector<std::string> splitUtf8(const std::string& str);

// =============================================================================
// 字符类型判断
// =============================================================================

/**
 * @brief 判断单字节是否为中文字符的起始字节
 * @param ch 字节
 * @return true 如果是中文字符起始字节
 */
bool isChinese(unsigned char ch);

/**
 * @brief 判断 UTF-8 字符是否为中文字符 (CJK Unified Ideographs)
 * @param ch UTF-8 编码的单个字符
 * @return true 如果是中文字符
 */
bool isChineseChar(const std::string& ch);

/**
 * @brief 判断字符串是否包含中文字符
 * @param text 要检查的字符串
 * @return true 如果包含中文字符
 */
bool containsChinese(const std::string& text);

/**
 * @brief 判断是否为英文字母
 * @param ch UTF-8 编码的单个字符
 * @return true 如果是 A-Z 或 a-z
 */
bool isEnglishLetter(const std::string& ch);

/**
 * @brief 判断是否为数字
 * @param ch UTF-8 编码的单个字符
 * @return true 如果是 0-9
 */
bool isDigit(const std::string& ch);

// =============================================================================
// 标点符号处理
// =============================================================================

/**
 * @brief 判断是否为标点符号
 * @param s 要检查的字符串
 * @return true 如果是标点符号
 */
bool isPunctuation(const std::string& s);

/**
 * @brief 映射中文标点符号到 ASCII 标点
 * @param punct 标点符号
 * @param token_to_id token 映射表 (用于检查 token 是否存在)
 * @return 映射后的标点符号，如果没有映射则返回空字符串
 */
std::string mapPunctuation(
        const std::string& punct,
        const std::unordered_map<std::string, int64_t>& token_to_id);

/**
 * @brief 简单的中文标点到 ASCII 标点映射
 * @param punct 中文标点符号
 * @return 对应的 ASCII 标点符号
 */
std::string mapChinesePunctToAscii(const std::string& punct);

}  // namespace text
}  // namespace tts

#endif  // TEXT_UTILS_HPP
