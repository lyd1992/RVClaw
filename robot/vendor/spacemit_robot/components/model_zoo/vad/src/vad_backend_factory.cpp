/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * VAD Backend Factory Implementation
 */

#include <iostream>
#include <memory>
#include <vector>

#include "backends/vad_backend.hpp"
#include "backends/silero/silero_backend.hpp"

namespace vad {

std::unique_ptr<IVadBackend> VadBackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::SILERO:
            return std::make_unique<SileroBackend>();

        case BackendType::ENERGY:
            // TODO(spacemit): Implement EnergyBackend
            std::cerr << "Energy backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::WEBRTC:
            // TODO(spacemit): Implement WebRTCBackend
            std::cerr << "WebRTC backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::FSMN:
            // TODO(spacemit): Implement FSMNBackend
            std::cerr << "FSMN backend not yet implemented" << std::endl;
            return nullptr;

        case BackendType::CUSTOM:
            std::cerr << "Custom backend requires user implementation" << std::endl;
            return nullptr;

        default:
            std::cerr << "Unknown backend type: " << static_cast<int>(type) << std::endl;
            return nullptr;
    }
}

bool VadBackendFactory::isAvailable(BackendType type) {
    switch (type) {
        case BackendType::SILERO:
            return true;  // Always available if ONNX Runtime is present

        case BackendType::ENERGY:
        case BackendType::WEBRTC:
        case BackendType::FSMN:
        case BackendType::CUSTOM:
            return false;  // Not yet implemented

        default:
            return false;
    }
}

std::vector<BackendType> VadBackendFactory::getAvailableBackends() {
    std::vector<BackendType> backends;
    backends.push_back(BackendType::SILERO);
    // Add more as they become available
    return backends;
}

int VadBackendFactory::getDefaultSampleRate(BackendType type) {
    switch (type) {
        case BackendType::SILERO:
            return 16000;
        case BackendType::WEBRTC:
            return 16000;
        default:
            return 16000;
    }
}

int VadBackendFactory::getRecommendedFrameSize(BackendType type) {
    switch (type) {
        case BackendType::SILERO:
            return 512;  // 32ms at 16kHz
        case BackendType::WEBRTC:
            return 480;  // 30ms at 16kHz
        default:
            return 512;
    }
}

}  // namespace vad
