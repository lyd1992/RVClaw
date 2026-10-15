/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/kokoro/kokoro_voice_manager.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace tts {

bool KokoroVoiceManager::loadVoice(const std::string& voice_path) {
    std::ifstream file(voice_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[KokoroVoiceManager] Failed to open voice file: " << voice_path << std::endl;
        return false;
    }

    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = static_cast<size_t>(file_size) / sizeof(float);
    if (num_floats == 0 || num_floats % STYLE_DIM != 0) {
        std::cerr << "[KokoroVoiceManager] Invalid voice file size: " << file_size
            << " bytes (not divisible by " << STYLE_DIM * sizeof(float) << ")" << std::endl;
        return false;
    }

    style_data_.resize(num_floats);
    file.read(reinterpret_cast<char*>(style_data_.data()), file_size);

    if (!file.good()) {
        std::cerr << "[KokoroVoiceManager] Failed to read voice file: " << voice_path << std::endl;
        style_data_.clear();
        return false;
    }

    num_rows_ = static_cast<int>(num_floats / STYLE_DIM);
    std::cout << "[KokoroVoiceManager] Loaded voice: " << voice_path
        << " (" << num_rows_ << " style vectors)" << std::endl;

    return true;
}

std::vector<float> KokoroVoiceManager::getStyleVector(int token_len) const {
    if (style_data_.empty()) {
        return std::vector<float>(STYLE_DIM, 0.0f);
    }

    // Index by token length, clamped to valid range
    int row = std::min(token_len, num_rows_ - 1);
    row = std::max(row, 0);

    size_t offset = static_cast<size_t>(row) * STYLE_DIM;
    return std::vector<float>(style_data_.begin() + offset,
        style_data_.begin() + offset + STYLE_DIM);
}

}  // namespace tts
