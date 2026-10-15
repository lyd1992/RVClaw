/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef NUMBER_UTILS_HPP
#define NUMBER_UTILS_HPP

/**
 * NumberUtils - 数字处理工具模块
 *
 * 提供数字转中文、罗马数字处理等功能。
 */

#include <cstdint>

#include <string>

namespace tts {
namespace text {

// =============================================================================
// 中文数字转换
// =============================================================================

/**
 * @brief 将整数转换为中文读法
 * @param num 要转换的整数
 * @return 中文读法字符串，如 123 -> "一百二十三"
 *
 * 支持范围：负数到万亿级别
 * 特殊处理：
 * - 0 -> "零"
 * - 负数前缀 "负"
 * - 正确处理"零"的插入（如 101 -> "一百零一"）
 * - 10-19 的简化读法（如 12 -> "十二" 而非 "一十二"）
 */
std::string intToChineseReading(int64_t num);

// =============================================================================
// 罗马数字处理
// =============================================================================

/**
 * @brief 判断字符是否为罗马数字字符
 * @param c 要检查的字符
 * @return true 如果是 I, V, X, L, C, D, M 之一
 */
bool isRomanNumeralChar(char c);

/**
 * @brief 判断字符串是否为有效的罗马数字
 * @param str 要检查的字符串
 * @return true 如果是有效的罗马数字（至少2个字符）
 *
 * 注意：要求至少2个字符，避免将单字母如 "I" 误判为罗马数字
 */
bool isRomanNumeral(const std::string& str);

/**
 * @brief 将罗马数字转换为整数
 * @param roman 罗马数字字符串
 * @return 对应的整数值
 *
 * 支持的罗马数字：I(1), V(5), X(10), L(50), C(100), D(500), M(1000)
 * 支持减法规则：如 IV=4, IX=9, XL=40, XC=90, CD=400, CM=900
 */
int romanToInt(const std::string& roman);

}  // namespace text
}  // namespace tts

#endif  // NUMBER_UTILS_HPP
