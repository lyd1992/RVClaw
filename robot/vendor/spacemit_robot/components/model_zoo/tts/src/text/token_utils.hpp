/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TOKEN_UTILS_HPP
#define TOKEN_UTILS_HPP

/**
 * TokenUtils - Token/词典读取工具模块
 *
 * 提供 token 映射表、词典文件的读取功能。
 */

#include <cstdint>

#include <string>
#include <unordered_map>

namespace tts {
namespace text {

// =============================================================================
// Token 映射读取
// =============================================================================

/**
 * @brief 读取 token 到 ID 的映射文件
 * @param path 映射文件路径
 * @return token 到 ID 的映射表
 * @throws std::runtime_error 如果文件无法打开
 *
 * 文件格式支持两种：
 * 1. "token_name token_id" - 空格分隔的 token 和 ID
 * 2. 每行一个 token，行号作为 ID (0-indexed)
 */
std::unordered_map<std::string, int64_t> readTokenToIdMap(const std::string& path);

/**
 * @brief 读取 zh-en 模型的 token 到 ID 映射文件
 * @param path 映射文件路径 (vocab_tts.txt)
 * @return token 到 ID 的映射表
 * @throws std::runtime_error 如果文件无法打开
 *
 * zh-en 模型特殊格式：行号 = token ID (1-indexed)
 * 特殊处理空格字符行
 */
std::unordered_map<std::string, int64_t> readZhEnTokenToIdMap(const std::string& path);

// =============================================================================
// 词典读取
// =============================================================================

/**
 * @brief 读取词典文件
 * @param path 词典文件路径
 * @return 词到音素的映射表
 * @throws std::runtime_error 如果文件无法打开
 *
 * 文件格式：每行 "word phonemes"，空格分隔词和音素序列
 */
std::unordered_map<std::string, std::string> readLexicon(const std::string& path);

}  // namespace text
}  // namespace tts

#endif  // TOKEN_UTILS_HPP
