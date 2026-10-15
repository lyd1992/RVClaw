/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_MODEL_CONFIG_H
#define VISION_MODEL_CONFIG_H

#include <string>

#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

namespace vision_core::yaml_utils {

inline std::string getString(const YAML::Node& node,
                            const std::string& key,
                            const std::string& default_val = "") {
    if (node && node[key]) {
        return node[key].as<std::string>();
    }
    return default_val;
}

inline float getFloat(const YAML::Node& node,
                        const std::string& key,
                        float default_val = 0.0f) {
    if (node && node[key]) {
        return node[key].as<float>();
    }
    return default_val;
}

inline int getInt(const YAML::Node& node,
                    const std::string& key,
                    int default_val = 0) {
    if (node && node[key]) {
        return node[key].as<int>();
    }
    return default_val;
}

inline bool getBool(const YAML::Node& node,
                    const std::string& key,
                    bool default_val = false) {
    if (node && node[key]) {
        return node[key].as<bool>();
    }
    return default_val;
}

inline std::string getProvider(const YAML::Node& config,
                                const std::string& default_provider = "SpaceMITExecutionProvider") {
    if (config && config["default_params"] && config["default_params"]["providers"]) {
        auto providers = config["default_params"]["providers"];
        if (providers.IsSequence() && providers.size() > 0) {
            return providers[0].as<std::string>();
        }
    }
    return default_provider;
}

}  // namespace vision_core::yaml_utils

#endif  // VISION_MODEL_CONFIG_H
