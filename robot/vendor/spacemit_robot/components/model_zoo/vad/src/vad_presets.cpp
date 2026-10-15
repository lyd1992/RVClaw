/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vad_service.h"

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace SpacemiT {

static const std::map<std::string, std::function<VadConfig()>>& getPresets() {
    static const std::map<std::string, std::function<VadConfig()>> presets = {
        {"silero", []() {
            VadConfig config;
            config.backend = VadBackendType::SILERO;
            config.model_dir = "~/.cache/models/vad/silero";
            return config;
        }},
        {"energy", []() {
            VadConfig config;
            config.backend = VadBackendType::ENERGY;
            return config;
        }},
    };
    return presets;
}

VadConfig VadConfig::Preset(const std::string& name) {
    const auto& presets = getPresets();
    auto it = presets.find(name);
    if (it == presets.end()) {
        throw std::invalid_argument("Unknown VAD preset: '" + name + "'");
    }
    return it->second();
}

std::vector<std::string> VadConfig::AvailablePresets() {
    const auto& presets = getPresets();
    std::vector<std::string> names;
    names.reserve(presets.size());
    for (const auto& [name, _] : presets) {
        names.push_back(name);
    }
    return names;
}

}  // namespace SpacemiT
