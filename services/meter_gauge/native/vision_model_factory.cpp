/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_model_factory.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <filesystem>  // NOLINT(build/c++17)
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

namespace vision_core {

ModelFactory& ModelFactory::getInstance() {
    static ModelFactory instance;
    return instance;
}

bool ModelFactory::registerModel(const std::string& class_name, CreatorFunc creator) {
    if (class_name.empty() || !creator) {
        return false;
    }
    creators_[class_name] = std::move(creator);
    return true;
}

namespace {
    std::filesystem::path findProjectRootFromConfig(const std::filesystem::path& config_file) {
        namespace fs = std::filesystem;
        fs::path p = fs::absolute(config_file).parent_path();
        while (!p.empty()) {
            if (fs::exists(p / "examples") && fs::exists(p / "src")) {
                return p;
            }
            auto parent = p.parent_path();
            if (parent == p) {
                break;
            }
            p = parent;
        }
        return {};
    }

    YAML::Node loadYamlConfigFromPath(const std::string& config_file,
                                        const std::string& model_path_override) {
        if (!std::filesystem::exists(config_file)) {
            throw std::runtime_error("Config file not found: " + config_file);
        }
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            if (!model_path_override.empty()) {
                config["model_path"] = model_path_override;
            } else if (config["model_path"]) {
                const std::string model_path = config["model_path"].as<std::string>();
                config["model_path"] = resolveResourcePath(model_path, config_file);
            }
            return config;
        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Failed to load config file " + config_file + ": " + e.what());
        }
    }

    std::string getString(const YAML::Node& node, const std::string& key, const std::string& default_val = "") {
        if (node[key]) {
            return node[key].as<std::string>();
        }
        return default_val;
    }

    std::string getClassName(const std::string& class_path) {
        auto pos = class_path.find_last_of('.');
        if (pos == std::string::npos || pos + 1 >= class_path.size()) {
            return class_path;
        }
        return class_path.substr(pos + 1);
    }
}  // namespace

std::unique_ptr<BaseModel> ModelFactory::createModelFromConfigPath(
    const std::string& config_path,
    const std::string& model_path_override,
    bool lazy_load,
    YAML::Node* preloaded_config) {
    YAML::Node config;
    if (preloaded_config != nullptr) {
        config = *preloaded_config;
        if (!model_path_override.empty()) {
            config["model_path"] = model_path_override;
        } else if (config["model_path"]) {
            config["model_path"] = resolveResourcePath(config["model_path"].as<std::string>(), config_path);
        }
    } else {
        config = loadYamlConfigFromPath(config_path, model_path_override);
    }

    std::string class_path = getString(config, "class");
    if (class_path.empty()) {
        throw std::runtime_error("class not found in config file: " + config_path);
    }
    std::string class_name = getClassName(class_path);

    auto it = creators_.find(class_name);
    if (it == creators_.end()) {
        throw std::runtime_error("Unsupported model class: " + class_name + " (config: " + config_path + ")");
    }

    // Resolve any default_params entry whose key ends with "_path" (e.g.
    // clip_model_path, bpe_merges_path) like model_path: expand ~ and resolve
    // repo-relative paths against the config file location, so auxiliary
    // resource paths are robust regardless of the process cwd.
    YAML::Node dp = config["default_params"];
    if (dp && dp.IsMap()) {
        const std::string suffix = "_path";
        for (auto entry = dp.begin(); entry != dp.end(); ++entry) {
            const std::string key = entry->first.as<std::string>();
            if (key.size() >= suffix.size() &&
                key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0 &&
                entry->second.IsScalar()) {
                const std::string raw = entry->second.as<std::string>();
                if (!raw.empty()) {
                    dp[key] = resolveResourcePath(raw, config_path);
                }
            }
        }
    }

    return it->second(config, lazy_load);
}

static std::string expand_tilde(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home || home[0] == '\0') return path;
    if (path.size() == 1 || path[1] == '/') return std::string(home) + path.substr(1);
    return path;
}

std::string resolveResourcePath(const std::string& raw_path, const std::string& config_file) {
    namespace fs = std::filesystem;
    if (raw_path.empty()) return raw_path;
    std::string path = expand_tilde(raw_path);
    fs::path raw(path);
    if (raw.is_absolute()) return raw.string();
    if (fs::exists(raw)) return raw.string();
    fs::path project_root = findProjectRootFromConfig(config_file);
    if (!project_root.empty()) {
        fs::path candidate = project_root / path;
        if (fs::exists(candidate)) return candidate.string();
    }
    return path;
}

std::unique_ptr<BaseModel> createModelFromConfigPath(
    const std::string& config_path,
    const std::string& model_path_override,
    bool lazy_load,
    YAML::Node* preloaded_config) {
    return ModelFactory::getInstance().createModelFromConfigPath(
        config_path, model_path_override, lazy_load, preloaded_config);
}

}  // namespace vision_core
