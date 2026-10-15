/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/tts_backend.hpp"

#include <memory>
#include <vector>

#include "backends/kokoro/kokoro_backend.hpp"
#include "backends/matcha/matcha_backend.hpp"
#include "backends/matcha/matcha_en_backend.hpp"
#include "backends/matcha/matcha_zh_backend.hpp"
#include "backends/matcha/matcha_zh_en_backend.hpp"

namespace tts {

// =============================================================================
// TtsBackendFactory 实现
// =============================================================================

std::unique_ptr<ITtsBackend> TtsBackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::MATCHA_ZH:
            return std::make_unique<MatchaZhBackend>();

        case BackendType::MATCHA_EN:
            return std::make_unique<MatchaEnBackend>();

        case BackendType::MATCHA_ZH_EN:
            return std::make_unique<MatchaZhEnBackend>();

        case BackendType::KOKORO:
            return std::make_unique<KokoroBackend>();

        case BackendType::COSYVOICE:
        case BackendType::VITS:
        case BackendType::PIPER:
        case BackendType::CUSTOM:
            // 这些后端尚未实现
            return nullptr;

        default:
            return nullptr;
    }
}

bool TtsBackendFactory::isAvailable(BackendType type) {
    switch (type) {
        case BackendType::MATCHA_ZH:
        case BackendType::MATCHA_EN:
        case BackendType::MATCHA_ZH_EN:
        case BackendType::KOKORO:
            return true;

        default:
            return false;
    }
}

std::vector<BackendType> TtsBackendFactory::getAvailableBackends() {
    return {
        BackendType::MATCHA_ZH,
        BackendType::MATCHA_EN,
        BackendType::MATCHA_ZH_EN,
        BackendType::KOKORO
    };
}

}  // namespace tts
