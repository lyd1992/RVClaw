/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * VadEngine Implementation
 *
 * SpacemiT::VadEngine 的实现，封装内部后端。
 */

#include "vad_service.h"
#include "backends/vad_backend.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace SpacemiT {

// =============================================================================
// Type Conversion Helpers
// =============================================================================

namespace {

// Convert SpacemiT::VadBackendType to vad::BackendType
vad::BackendType toInternalBackendType(VadBackendType type) {
    switch (type) {
        case VadBackendType::SILERO: return vad::BackendType::SILERO;
        case VadBackendType::ENERGY: return vad::BackendType::ENERGY;
        case VadBackendType::WEBRTC: return vad::BackendType::WEBRTC;
        case VadBackendType::FSMN:   return vad::BackendType::FSMN;
        case VadBackendType::CUSTOM: return vad::BackendType::CUSTOM;
        default:                     return vad::BackendType::SILERO;
    }
}

// Convert vad::VadState to SpacemiT::VadState
VadState toPublicState(vad::VadState state) {
    switch (state) {
        case vad::VadState::SILENCE:      return VadState::SILENCE;
        case vad::VadState::SPEECH_START: return VadState::SPEECH_START;
        case vad::VadState::SPEECH:       return VadState::SPEECH;
        case vad::VadState::SPEECH_END:   return VadState::SPEECH_END;
        default:                          return VadState::SILENCE;
    }
}

// Convert SpacemiT::VadConfig to vad::VadConfig
vad::VadConfig toInternalConfig(const VadConfig& config) {
    vad::VadConfig internal;
    internal.backend = toInternalBackendType(config.backend);
    internal.model_dir = config.model_dir;
    internal.sample_rate = config.sample_rate;
    internal.window_size = config.window_size;
    internal.trigger_threshold = config.trigger_threshold;
    internal.stop_threshold = config.stop_threshold;
    internal.min_speech_duration_ms = config.min_speech_duration_ms;
    internal.min_silence_duration_ms = config.min_silence_duration_ms;
    internal.smoothing_window = config.smoothing_window;
    internal.use_smoothing = config.use_smoothing;
    internal.num_threads = config.num_threads;
    return internal;
}

}  // anonymous namespace

// =============================================================================
// VadResult Implementation (Pimpl)
// =============================================================================

struct VadResult::Impl {
    float probability = 0.0f;
    float smoothed_probability = 0.0f;
    VadState state = VadState::SILENCE;
    bool is_speech = false;
    int64_t timestamp_ms = 0;
    int64_t speech_start_ms = -1;
    int64_t speech_end_ms = -1;
    int speech_duration_ms = 0;
    int processing_time_ms = 0;
    bool success = true;
    std::string code;
    std::string message;
};

VadResult::VadResult() : impl_(std::make_unique<Impl>()) {}
VadResult::~VadResult() = default;
VadResult::VadResult(VadResult&&) noexcept = default;
VadResult& VadResult::operator=(VadResult&&) noexcept = default;

float VadResult::GetProbability() const { return impl_->probability; }
float VadResult::GetSmoothedProbability() const { return impl_->smoothed_probability; }
bool VadResult::IsSpeech() const { return impl_->is_speech; }
VadState VadResult::GetState() const { return impl_->state; }
bool VadResult::IsSpeechStart() const { return impl_->state == VadState::SPEECH_START; }
bool VadResult::IsSpeechEnd() const { return impl_->state == VadState::SPEECH_END; }
int64_t VadResult::GetTimestampMs() const { return impl_->timestamp_ms; }
int64_t VadResult::GetSpeechStartMs() const { return impl_->speech_start_ms; }
int64_t VadResult::GetSpeechEndMs() const { return impl_->speech_end_ms; }
int VadResult::GetSpeechDurationMs() const { return impl_->speech_duration_ms; }
int VadResult::GetProcessingTimeMs() const { return impl_->processing_time_ms; }
bool VadResult::IsSuccess() const { return impl_->success; }
std::string VadResult::GetCode() const { return impl_->code; }
std::string VadResult::GetMessage() const { return impl_->message; }

// =============================================================================
// Callback Adapter
// =============================================================================

class CallbackAdapter : public vad::IVadCallback {
public:
    explicit CallbackAdapter(std::shared_ptr<VadEngineCallback> callback)
        : callback_(callback) {}

    void onStart() override {
        if (callback_) callback_->OnOpen();
    }

    void onComplete() override {
        if (callback_) callback_->OnComplete();
    }

    void onClose() override {
        if (callback_) callback_->OnClose();
    }

    void onResult(const vad::DetectionResult& result) override {
        if (!callback_) return;

        auto vadResult = std::make_shared<VadResult>();
        vadResult->impl_->probability = result.probability;
        vadResult->impl_->smoothed_probability = result.smoothed_probability;
        vadResult->impl_->state = toPublicState(result.state);
        vadResult->impl_->is_speech = result.is_speech;
        vadResult->impl_->timestamp_ms = result.timestamp_ms;
        vadResult->impl_->speech_start_ms = result.speech_start_ms;
        vadResult->impl_->speech_end_ms = result.speech_end_ms;
        vadResult->impl_->speech_duration_ms = result.speech_duration_ms;
        vadResult->impl_->processing_time_ms = result.processing_time_ms;
        vadResult->impl_->success = true;

        callback_->OnEvent(vadResult);
    }

    void onSpeechStart(int64_t timestamp_ms) override {
        if (callback_) callback_->OnSpeechStart(timestamp_ms);
    }

    void onSpeechEnd(int64_t timestamp_ms, int duration_ms) override {
        if (callback_) callback_->OnSpeechEnd(timestamp_ms, duration_ms);
    }

    void onError(const vad::ErrorInfo& error) override {
        if (callback_) {
            callback_->OnError(error.message + ": " + error.detail);
        }
    }

private:
    std::shared_ptr<VadEngineCallback> callback_;
};

// =============================================================================
// VadEngine Implementation (Pimpl)
// =============================================================================

struct VadEngine::Impl {
    VadConfig config;
    std::unique_ptr<vad::IVadBackend> backend;
    std::unique_ptr<CallbackAdapter> callback_adapter;
    std::shared_ptr<VadEngineCallback> user_callback;
    std::mutex mutex;
    bool initialized = false;
    bool streaming = false;
    VadState current_state = VadState::SILENCE;
    float last_probability = 0.0f;
};

VadEngine::VadEngine(VadBackendType backend, const std::string& model_dir)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = VadConfig();
    impl_->config.backend = backend;
    if (!model_dir.empty()) {
        impl_->config.model_dir = model_dir;
    }

    // Create and initialize backend
    auto internal_config = toInternalConfig(impl_->config);
    impl_->backend = vad::VadBackendFactory::create(toInternalBackendType(backend));
    if (impl_->backend) {
        auto err = impl_->backend->initialize(internal_config);
        impl_->initialized = err.isOk();
    }
}

VadEngine::VadEngine(const VadConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;

    // Create and initialize backend
    auto internal_config = toInternalConfig(config);
    impl_->backend = vad::VadBackendFactory::create(toInternalBackendType(config.backend));
    if (impl_->backend) {
        auto err = impl_->backend->initialize(internal_config);
        impl_->initialized = err.isOk();
    }
}

VadEngine::~VadEngine() {
    // Don't use mutex lock in destructor - it may already be destroyed
    // Clear callback first to avoid calling into possibly-destroyed Python objects
    if (impl_) {
        impl_->user_callback.reset();
        impl_->callback_adapter.reset();
        if (impl_->backend) {
            impl_->backend->setCallback(nullptr);
            if (impl_->streaming) {
                impl_->backend->stopStream();
            }
        }
        impl_->streaming = false;
    }
}

std::shared_ptr<VadResult> VadEngine::Detect(const std::vector<float>& audio, int sample_rate) {
    return Detect(audio.data(), audio.size(), sample_rate);
}

std::shared_ptr<VadResult> VadEngine::Detect(const float* data, size_t num_samples, int sample_rate) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    auto result = std::make_shared<VadResult>();

    if (!impl_->backend || !impl_->initialized) {
        result->impl_->success = false;
        result->impl_->code = "NOT_INITIALIZED";
        result->impl_->message = "VAD engine not initialized";
        return result;
    }

    vad::AudioChunk chunk = vad::AudioChunk::fromFloat(data, num_samples, sample_rate);
    vad::DetectionResult detection_result;

    auto err = impl_->backend->detect(chunk, detection_result);
    if (!err.isOk()) {
        result->impl_->success = false;
        result->impl_->code = std::to_string(static_cast<int>(err.code));
        result->impl_->message = err.message;
        return result;
    }

    result->impl_->probability = detection_result.probability;
    result->impl_->smoothed_probability = detection_result.smoothed_probability;
    result->impl_->state = toPublicState(detection_result.state);
    result->impl_->is_speech = detection_result.is_speech;
    result->impl_->timestamp_ms = detection_result.timestamp_ms;
    result->impl_->speech_start_ms = detection_result.speech_start_ms;
    result->impl_->speech_end_ms = detection_result.speech_end_ms;
    result->impl_->speech_duration_ms = detection_result.speech_duration_ms;
    result->impl_->processing_time_ms = detection_result.processing_time_ms;
    result->impl_->success = true;

    impl_->current_state = result->impl_->state;
    impl_->last_probability = result->impl_->smoothed_probability;

    return result;
}

void VadEngine::SetCallback(std::shared_ptr<VadEngineCallback> callback) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->user_callback = callback;
    impl_->callback_adapter = std::make_unique<CallbackAdapter>(callback);
    if (impl_->backend) {
        impl_->backend->setCallback(impl_->callback_adapter.get());
    }
}

bool VadEngine::Start() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->backend || !impl_->initialized) {
        return false;
    }

    if (impl_->streaming) {
        return true;  // Already streaming
    }

    auto err = impl_->backend->startStream();
    if (err.isOk()) {
        impl_->streaming = true;
        if (impl_->callback_adapter) {
            impl_->callback_adapter->onStart();
        }
        return true;
    }

    return false;
}

void VadEngine::SendAudioFrame(const std::vector<float>& data) {
    SendAudioFrame(data.data(), data.size());
}

void VadEngine::SendAudioFrame(const float* data, size_t num_samples) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->backend || !impl_->streaming) {
        return;
    }

    vad::AudioChunk chunk = vad::AudioChunk::fromFloat(data, num_samples, impl_->config.sample_rate);
    impl_->backend->feedAudio(chunk);
}

void VadEngine::Stop() {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (!impl_->backend || !impl_->streaming) {
        return;
    }

    impl_->backend->stopStream();
    impl_->streaming = false;

    if (impl_->callback_adapter) {
        impl_->callback_adapter->onComplete();
        impl_->callback_adapter->onClose();
    }
}

void VadEngine::Reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->backend) {
        impl_->backend->reset();
    }
    impl_->current_state = VadState::SILENCE;
    impl_->last_probability = 0.0f;
}

VadState VadEngine::GetCurrentState() const {
    return impl_->current_state;
}

bool VadEngine::IsInSpeech() const {
    return impl_->current_state == VadState::SPEECH_START ||
            impl_->current_state == VadState::SPEECH;
}

bool VadEngine::IsInitialized() const {
    return impl_->initialized;
}

bool VadEngine::IsStreaming() const {
    return impl_->streaming;
}

void VadEngine::SetTriggerThreshold(float threshold) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config.trigger_threshold = threshold;
    if (impl_->backend) {
        impl_->backend->setThresholds(threshold, impl_->config.stop_threshold);
    }
}

void VadEngine::SetStopThreshold(float threshold) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->config.stop_threshold = threshold;
    if (impl_->backend) {
        impl_->backend->setThresholds(impl_->config.trigger_threshold, threshold);
    }
}

VadConfig VadEngine::GetConfig() const {
    return impl_->config;
}

std::string VadEngine::GetEngineName() const {
    if (impl_->backend) {
        return impl_->backend->getName();
    }
    return "Unknown";
}

VadBackendType VadEngine::GetBackendType() const {
    return impl_->config.backend;
}

float VadEngine::GetLastProbability() const {
    return impl_->last_probability;
}

}  // namespace SpacemiT
