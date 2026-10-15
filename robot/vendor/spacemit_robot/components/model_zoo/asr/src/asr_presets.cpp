/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asr_service.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpacemiT {

static const std::map<std::string, std::function<AsrConfig()>>& getPresets() {
    static const std::map<std::string, std::function<AsrConfig()>> presets = {
        {"sensevoice", []() {
            AsrConfig config;
            config.engine = "sensevoice";
            config.model_dir = "~/.cache/models/asr/sensevoice";
            return config;
        }},
        {"qwen3-asr", []() {
            AsrConfig config;
            config.engine = "qwen3-asr";
            config.model_dir = "";
            config.provider = "cpu";
            return config;
        }},
        {"zipformer", []() {
            AsrConfig config;
            config.engine = "zipformer";
            config.model_dir = "~/.cache/models/asr/zipformer";
            return config;
        }},
    };
    return presets;
}

AsrConfig AsrConfig::Preset(const std::string& name) {
    const auto& presets = getPresets();
    auto it = presets.find(name);
    if (it == presets.end()) {
        throw std::invalid_argument("Unknown ASR preset: '" + name + "'");
    }
    return it->second();
}

std::vector<std::string> AsrConfig::AvailablePresets() {
    const auto& presets = getPresets();
    std::vector<std::string> names;
    names.reserve(presets.size());
    for (const auto& [name, _] : presets) {
        names.push_back(name);
    }
    return names;
}

}  // namespace SpacemiT
