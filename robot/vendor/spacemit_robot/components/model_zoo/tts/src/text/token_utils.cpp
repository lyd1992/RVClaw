/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "text/token_utils.hpp"

#include <cstdint>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tts {
namespace text {

// =============================================================================
// Token 映射读取
// =============================================================================

std::unordered_map<std::string, int64_t> readTokenToIdMap(const std::string& path) {
    std::unordered_map<std::string, int64_t> token_to_id;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open tokens file: " + path);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string token;
            int64_t id;

            if (iss >> token >> id) {
                // Format: "token_name token_id"
                token_to_id[token] = id;
            } else {
                // Fallback: use line number as ID (0-indexed)
                token_to_id[line] = line_num - 1;
            }
        }
    }

    return token_to_id;
}

std::unordered_map<std::string, int64_t> readZhEnTokenToIdMap(const std::string& path) {
    std::unordered_map<std::string, int64_t> token_to_id;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open tokens file: " + path);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        // For zh-en model: line number = token ID (1-indexed)
        // Handle special case: line with only a space character
        if (line == " ") {
            token_to_id[" "] = line_num;
            continue;
        }

        if (!line.empty()) {
            // Trim whitespace for normal tokens
            size_t start = line.find_first_not_of(" \t\r\n");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                std::string token = line.substr(start, end - start + 1);
                token_to_id[token] = line_num;
            }
        }
    }

    return token_to_id;
}

// =============================================================================
// 词典读取
// =============================================================================

std::unordered_map<std::string, std::string> readLexicon(const std::string& path) {
    std::unordered_map<std::string, std::string> lexicon;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open lexicon file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string word = line.substr(0, space_pos);
                std::string phones = line.substr(space_pos + 1);
                lexicon[word] = phones;
            }
        }
    }

    return lexicon;
}

}  // namespace text
}  // namespace tts
