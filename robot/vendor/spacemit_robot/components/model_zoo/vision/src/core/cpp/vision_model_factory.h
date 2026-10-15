/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_MODEL_FACTORY_H
#define VISION_MODEL_FACTORY_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_core {

class ModelFactory {
public:
    using CreatorFunc = std::function<std::unique_ptr<BaseModel>(const YAML::Node& config,
                                                                bool lazy_load)>;

    static ModelFactory& getInstance();

    bool registerModel(const std::string& class_name, CreatorFunc creator);

    std::unique_ptr<BaseModel> createModelFromConfigPath(const std::string& config_path,
                                                        const std::string& model_path_override,
                                                        bool lazy_load = false,
                                                        YAML::Node* preloaded_config = nullptr);

private:
    ModelFactory() = default;
    ~ModelFactory() = default;
    ModelFactory(const ModelFactory&) = delete;
    ModelFactory& operator=(const ModelFactory&) = delete;
    ModelFactory(ModelFactory&&) = delete;
    ModelFactory& operator=(ModelFactory&&) = delete;

    std::unordered_map<std::string, CreatorFunc> creators_;
};

template <typename T>
class ModelRegistrar {
public:
    explicit ModelRegistrar(const std::string& class_name) {
        ModelFactory::getInstance().registerModel(
            class_name,
            [](const YAML::Node& config, bool lazy_load) -> std::unique_ptr<BaseModel> {
                return T::create(config, lazy_load);
            });
    }
};

std::unique_ptr<BaseModel> createModelFromConfigPath(const std::string& config_path,
                                                    const std::string& model_path_override = "",
                                                    bool lazy_load = false,
                                                    YAML::Node* preloaded_config = nullptr);

std::string resolveResourcePath(const std::string& raw_path, const std::string& config_file);

}  // namespace vision_core

#endif  // VISION_MODEL_FACTORY_H
