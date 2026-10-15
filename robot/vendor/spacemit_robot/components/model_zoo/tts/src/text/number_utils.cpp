/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "text/number_utils.hpp"

#include <cctype>

#include <string>
#include <unordered_map>

namespace tts {
namespace text {

// =============================================================================
// Chinese number conversion
// =============================================================================

std::string intToChineseReading(int64_t num) {
    if (num == 0) return "零";
    if (num < 0) return "负" + intToChineseReading(-num);

    static const char* digits[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};

    std::string result;

    // Handle numbers up to trillion
    if (num >= 1000000000000LL) {
        result += intToChineseReading(num / 1000000000000LL) + "万亿";
        num %= 1000000000000LL;
        if (num > 0 && num < 100000000000LL) result += "零";
    }
    if (num >= 100000000LL) {
        result += intToChineseReading(num / 100000000LL) + "亿";
        num %= 100000000LL;
        if (num > 0 && num < 10000000LL) result += "零";
    }
    if (num >= 10000LL) {
        result += intToChineseReading(num / 10000LL) + "万";
        num %= 10000LL;
        if (num > 0 && num < 1000LL) result += "零";
    }
    if (num >= 1000LL) {
        result += std::string(digits[num / 1000LL]) + "千";
        num %= 1000LL;
        if (num > 0 && num < 100LL) result += "零";
    }
    if (num >= 100LL) {
        result += std::string(digits[num / 100LL]) + "百";
        num %= 100LL;
        if (num > 0 && num < 10LL) result += "零";
    }
    if (num >= 10LL) {
        if (num / 10LL != 1 || result.length() > 0) {
            result += digits[num / 10LL];
        }
        result += "十";
        num %= 10LL;
    }
    if (num > 0) {
        result += digits[num];
    }

    return result;
}

// =============================================================================
// Roman numeral handling
// =============================================================================

bool isRomanNumeralChar(char c) {
    c = std::toupper(c);
    return c == 'I' || c == 'V' || c == 'X' || c == 'L' || c == 'C' || c == 'D' || c == 'M';
}

bool isRomanNumeral(const std::string& str) {
    // Require at least 2 characters to avoid treating single letters like "I" as Roman numerals
    if (str.length() < 2) return false;
    for (char c : str) {
        if (!isRomanNumeralChar(c)) return false;
    }
    return true;
}

int romanToInt(const std::string& roman) {
    std::unordered_map<char, int> values = {
        {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50},
        {'C', 100}, {'D', 500}, {'M', 1000}
    };

    int result = 0;
    for (size_t i = 0; i < roman.length(); i++) {
        char c = std::toupper(roman[i]);
        int value = values[c];

        if (i + 1 < roman.length()) {
            char next = std::toupper(roman[i + 1]);
            int next_value = values[next];
            if (value < next_value) {
                result -= value;
            } else {
                result += value;
            }
        } else {
            result += value;
        }
    }
    return result;
}

}  // namespace text
}  // namespace tts
