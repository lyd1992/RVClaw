/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SENSEVOICE_HPP
#define SENSEVOICE_HPP

/**
 * @file sensevoice.hpp
 * @brief SenseVoice backend unified header
 *
 * Include this header to use all SenseVoice components.
 */

// Core components
#include "sensevoice_model.hpp"
#include "feature_extractor.hpp"
#include "tokenizer.hpp"
#include "model_loader.hpp"

namespace asr {
namespace sensevoice {

/**
 * @brief SenseVoice backend version
 */
constexpr const char* VERSION = "1.0.2";

/**
 * @brief Default model directory
 */
constexpr const char* DEFAULT_MODEL_DIR = "~/.cache/models/asr/sensevoice";

/**
 * @brief Required sample rate
 */
constexpr int REQUIRED_SAMPLE_RATE = 16000;

}  // namespace sensevoice
}  // namespace asr

#endif  // SENSEVOICE_HPP
