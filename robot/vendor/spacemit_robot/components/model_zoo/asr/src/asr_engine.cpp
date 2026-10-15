/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asr_engine.hpp"

#include <iostream>
#include <chrono>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>

#include "backends/sensevoice/sensevoice_backend.hpp"

namespace asr {

// =============================================================================
// ASREngine Implementation
// =============================================================================

ASREngine::ASREngine() = default;

ASREngine::~ASREngine() {
    shutdown();
}

// Move semantics - need manual implementation due to atomic/mutex members
ASREngine::ASREngine(ASREngine&& other) noexcept
    : config_(std::move(other.config_))
    , backend_(std::move(other.backend_))
    , owned_callback_(std::move(other.owned_callback_))
    , callback_(other.callback_)
    , initialized_(other.initialized_.load())
    , streaming_(other.streaming_.load())
    , last_request_id_(std::move(other.last_request_id_))
    , first_packet_latency_ms_(other.first_packet_latency_ms_)
    , last_packet_latency_ms_(other.last_packet_latency_ms_)
    , last_error_(std::move(other.last_error_))
{
    other.callback_ = nullptr;
    other.initialized_.store(false);
    other.streaming_.store(false);
}

ASREngine& ASREngine::operator=(ASREngine&& other) noexcept {
    if (this != &other) {
        shutdown();

        config_ = std::move(other.config_);
        backend_ = std::move(other.backend_);
        owned_callback_ = std::move(other.owned_callback_);
        callback_ = other.callback_;
        initialized_.store(other.initialized_.load());
        streaming_.store(other.streaming_.load());
        last_request_id_ = std::move(other.last_request_id_);
        first_packet_latency_ms_ = other.first_packet_latency_ms_;
        last_packet_latency_ms_ = other.last_packet_latency_ms_;
        last_error_ = std::move(other.last_error_);

        other.callback_ = nullptr;
        other.initialized_.store(false);
        other.streaming_.store(false);
    }
    return *this;
}

ErrorInfo ASREngine::initialize(const ASRConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (initialized_.load()) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Engine already initialized");
    }

    // Validate config
    auto validation_error = ConfigValidator::validate(config);
    if (!validation_error.isOk()) {
        last_error_ = validation_error;
        return validation_error;
    }

    config_ = config;

    // Create backend
    auto err = createBackend();
    if (!err.isOk()) {
        last_error_ = err;
        return err;
    }

    // Initialize backend
    err = backend_->initialize(config_);
    if (!err.isOk()) {
        last_error_ = err;
        backend_.reset();
        return err;
    }

    // Set callback if available
    if (callback_) {
        backend_->setCallback(callback_);
    }

    initialized_.store(true);
    std::cout << "[ASREngine] Initialized with backend: " << backend_->getName() << std::endl;

    return ErrorInfo::ok();
}

void ASREngine::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (streaming_.load()) {
        // Stop streaming first
        if (backend_) {
            backend_->stopStream();
        }
        streaming_.store(false);
    }

    if (backend_) {
        backend_->shutdown();
        backend_.reset();
    }

    initialized_.store(false);
}

bool ASREngine::isInitialized() const {
    return initialized_.load();
}

ErrorInfo ASREngine::createBackend() {
    backend_ = ASRBackendFactory::create(config_.backend);
    if (!backend_) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            "Failed to create backend: " + std::string(backendTypeToString(config_.backend)));
    }
    return ErrorInfo::ok();
}

void ASREngine::setCallback(std::unique_ptr<IASRCallback> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    owned_callback_ = std::move(callback);
    callback_ = owned_callback_.get();
    if (backend_) {
        backend_->setCallback(callback_);
    }
}

void ASREngine::setCallback(IASRCallback* callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    owned_callback_.reset();  // Release owned callback if any
    callback_ = callback;
    if (backend_) {
        backend_->setCallback(callback_);
    }
}

// =============================================================================
// Offline Recognition API
// =============================================================================

RecognitionResult ASREngine::recognize(const int16_t* audio, size_t samples) {
    auto chunk = AudioChunk::fromPCM16(audio, samples, config_.sample_rate, config_.channels);
    return recognize(chunk);
}

RecognitionResult ASREngine::recognize(const float* audio, size_t samples) {
    auto chunk = AudioChunk::fromPCMFloat(audio, samples, config_.sample_rate, config_.channels);
    return recognize(chunk);
}

RecognitionResult ASREngine::recognize(const AudioChunk& chunk) {
    RecognitionResult result;
    result.request_id = generateRequestId();

    if (!initialized_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Engine not initialized");
        if (callback_) {
            callback_->onError(last_error_);
        }
        return result;
    }

    if (streaming_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::ALREADY_STARTED,
            "Cannot use offline recognition while streaming");
        if (callback_) {
            callback_->onError(last_error_);
        }
        return result;
    }

    // Notify start
    if (callback_) {
        callback_->onStart();
    }

    auto start_time = std::chrono::steady_clock::now();

    // Call backend
    auto err = backend_->recognize(chunk, result);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    result.processing_time_ms = duration.count();
    last_request_id_ = result.request_id;

    if (!err.isOk()) {
        last_error_ = err;
        if (callback_) {
            callback_->onError(err);
            callback_->onClose();
        }
        return result;
    }

    updateLatencyMetrics(result);

    // Notify result
    if (callback_) {
        callback_->onResult(result);
        callback_->onComplete();
        callback_->onClose();
    }

    return result;
}

RecognitionResult ASREngine::recognizeFile(const std::string& file_path) {
    RecognitionResult result;
    result.request_id = generateRequestId();

    if (!initialized_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Engine not initialized");
        if (callback_) {
            callback_->onError(last_error_);
        }
        return result;
    }

    // Notify start
    if (callback_) {
        callback_->onStart();
    }

    auto start_time = std::chrono::steady_clock::now();

    // Call backend
    auto err = backend_->recognizeFile(file_path, result);

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    result.processing_time_ms = duration.count();
    last_request_id_ = result.request_id;

    if (!err.isOk()) {
        last_error_ = err;
        if (callback_) {
            callback_->onError(err);
            callback_->onClose();
        }
        return result;
    }

    updateLatencyMetrics(result);

    // Notify result
    if (callback_) {
        callback_->onResult(result);
        callback_->onComplete();
        callback_->onClose();
    }

    return result;
}

// =============================================================================
// Streaming API
// =============================================================================

ErrorInfo ASREngine::start() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!initialized_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Engine not initialized");
        return last_error_;
    }

    if (streaming_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Already streaming");
        return last_error_;
    }

    if (!backend_->supportsStreaming()) {
        last_error_ = ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
            "Backend does not support streaming");
        return last_error_;
    }

    auto err = backend_->startStream();
    if (!err.isOk()) {
        last_error_ = err;
        return err;
    }

    streaming_.store(true);
    last_request_id_ = generateRequestId();

    return ErrorInfo::ok();
}

ErrorInfo ASREngine::sendAudio(const int16_t* audio, size_t samples) {
    auto chunk = AudioChunk::fromPCM16(audio, samples, config_.sample_rate, config_.channels);
    return sendAudio(chunk);
}

ErrorInfo ASREngine::sendAudio(const float* audio, size_t samples) {
    auto chunk = AudioChunk::fromPCMFloat(audio, samples, config_.sample_rate, config_.channels);
    return sendAudio(chunk);
}

ErrorInfo ASREngine::sendAudio(const AudioChunk& chunk) {
    if (!streaming_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::NOT_STARTED, "Streaming not started");
        return last_error_;
    }

    auto err = backend_->feedAudio(chunk);
    if (!err.isOk()) {
        last_error_ = err;
    }
    return err;
}

ErrorInfo ASREngine::stop() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!streaming_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::NOT_STARTED, "Streaming not started");
        return last_error_;
    }

    auto err = backend_->stopStream();
    streaming_.store(false);

    if (!err.isOk()) {
        last_error_ = err;
    }

    return err;
}

ErrorInfo ASREngine::flush() {
    // No lock needed - flushStream handles its own synchronization
    if (!streaming_.load()) {
        last_error_ = ErrorInfo::error(ErrorCode::NOT_STARTED, "Streaming not started");
        return last_error_;
    }

    auto err = backend_->flushStream();
    if (!err.isOk()) {
        last_error_ = err;
    }

    return err;
}

bool ASREngine::isStreaming() const {
    return streaming_.load();
}

// =============================================================================
// Dynamic Configuration
// =============================================================================

ErrorInfo ASREngine::updateHotwords(const std::vector<std::string>& hotwords) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Engine not initialized");
    }
    return backend_->updateHotwords(hotwords);
}

ErrorInfo ASREngine::setLanguage(Language language) {
    if (!initialized_.load()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Engine not initialized");
    }
    config_.language = language;
    return backend_->setLanguage(language);
}

// =============================================================================
// Status & Info
// =============================================================================

BackendType ASREngine::getBackendType() const {
    if (backend_) {
        return backend_->getType();
    }
    return config_.backend;
}

std::string ASREngine::getBackendName() const {
    if (backend_) {
        return backend_->getName();
    }
    return backendTypeToString(config_.backend);
}

// =============================================================================
// Static Methods
// =============================================================================

std::vector<BackendType> ASREngine::getAvailableBackends() {
    return ASRBackendFactory::getAvailableBackends();
}

bool ASREngine::isBackendAvailable(BackendType type) {
    return ASRBackendFactory::isAvailable(type);
}

std::string ASREngine::getVersion() {
    return "1.0.2";
}

// =============================================================================
// Private Helpers
// =============================================================================

void ASREngine::updateLatencyMetrics(const RecognitionResult& result) {
    first_packet_latency_ms_ = result.first_result_latency_ms;
    last_packet_latency_ms_ = result.final_result_latency_ms;
}

std::string ASREngine::generateRequestId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex = "0123456789abcdef";

    std::stringstream ss;
    ss << "asr-";
    for (int i = 0; i < 8; ++i) {
        ss << hex[dis(gen)];
    }
    ss << "-";
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    ss << std::hex << ms;

    return ss.str();
}

// =============================================================================
// Quick Namespace Functions
// =============================================================================

namespace quick {

std::string recognize(const float* audio, size_t samples, const std::string& model_dir) {
    ASREngine engine;

    std::string dir = model_dir.empty() ? "~/.cache/models/asr/sensevoice" : model_dir;
    auto config = ASRConfig::sensevoice(dir);

    auto err = engine.initialize(config);
    if (!err.isOk()) {
        std::cerr << "[quick::recognize] Init failed: " << err.message << std::endl;
        return "";
    }

    auto result = engine.recognize(audio, samples);
    return result.getText();
}

std::string recognizeFile(const std::string& file_path, const std::string& model_dir) {
    ASREngine engine;

    std::string dir = model_dir.empty() ? "~/.cache/models/asr/sensevoice" : model_dir;
    auto config = ASRConfig::sensevoice(dir);

    auto err = engine.initialize(config);
    if (!err.isOk()) {
        std::cerr << "[quick::recognizeFile] Init failed: " << err.message << std::endl;
        return "";
    }

    auto result = engine.recognizeFile(file_path);
    return result.getText();
}

}  // namespace quick

}  // namespace asr
