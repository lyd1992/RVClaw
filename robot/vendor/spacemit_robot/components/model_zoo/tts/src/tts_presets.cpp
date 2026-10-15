/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tts_service.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// SpacemiT::TtsConfig presets (public API)
// =============================================================================

namespace SpacemiT {

static const std::map<std::string, std::function<TtsConfig()>>& getPublicPresets() {
    static const std::map<std::string, std::function<TtsConfig()>> presets = {
        {"matcha_zh", []() {
            TtsConfig config;
            config.backend = BackendType::MATCHA_ZH;
            config.model = "matcha-icefall-zh-baker";
            config.model_dir = "~/.cache/models/tts/matcha-tts";
            config.sample_rate = 22050;
            return config;
        }},
        {"matcha_en", []() {
            TtsConfig config;
            config.backend = BackendType::MATCHA_EN;
            config.model = "matcha-icefall-en_US-ljspeech";
            config.model_dir = "~/.cache/models/tts/matcha-tts";
            config.sample_rate = 22050;
            return config;
        }},
        {"matcha_zh_en", []() {
            TtsConfig config;
            config.backend = BackendType::MATCHA_ZH_EN;
            config.model = "matcha-icefall-zh-en";
            config.model_dir = "~/.cache/models/tts/matcha-tts";
            config.sample_rate = 16000;
            return config;
        }},
        {"kokoro", []() {
            TtsConfig config;
            config.backend = BackendType::KOKORO;
            config.model = "kokoro-v1.0.q";
            config.model_dir = "~/.cache/models/tts/kokoro-tts";
            config.voice = "default";
            config.sample_rate = 24000;
            return config;
        }},
    };
    return presets;
}

TtsConfig TtsConfig::Preset(const std::string& name) {
    const auto& presets = getPublicPresets();
    auto it = presets.find(name);
    if (it == presets.end()) {
        throw std::invalid_argument("Unknown TTS preset: '" + name + "'");
    }
    return it->second();
}

std::vector<std::string> TtsConfig::AvailablePresets() {
    const auto& presets = getPublicPresets();
    std::vector<std::string> names;
    names.reserve(presets.size());
    for (const auto& [name, _] : presets) {
        names.push_back(name);
    }
    return names;
}

}  // namespace SpacemiT
