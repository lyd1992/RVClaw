/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_HPP
#define ASR_HPP

// =============================================================================
// ASR Framework - 主包含文件
// =============================================================================
//
// 只需包含此文件即可使用完整的ASR框架
//
//   #include <asr/asr.hpp>
//
// 快速开始:
//
//   // 1. 创建引擎
//   asr::ASREngine engine;
//
//   // 2. 配置 (使用SenseVoice本地模型)
//   auto config = asr::ASRConfig::sensevoice("~/.cache/models/asr/sensevoice");
//
//   // 3. 初始化
//   auto err = engine.initialize(config);
//   if (!err.isOk()) {
//       std::cerr << "Init failed: " << err.message << std::endl;
//       return -1;
//   }
//
//   // 4. 识别
//   auto result = engine.recognize(audio_data, sample_count);
//   std::cout << "Result: " << result.getText() << std::endl;
//

#include <string>

// 核心类型
#include "asr_types.hpp"

// 配置
#include "asr_config.hpp"

// 回调接口
#include "asr_callback.hpp"

// 主引擎
#include "asr_engine.hpp"

// 后端接口 (通常不需要直接使用)
#include "backends/asr_backend.hpp"

namespace asr {

// =============================================================================
// 版本信息
// =============================================================================

constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 2;

inline std::string getVersionString() {
    return std::to_string(VERSION_MAJOR) + "." +
        std::to_string(VERSION_MINOR) + "." +
        std::to_string(VERSION_PATCH);
}

}  // namespace asr

#endif  // ASR_HPP
