/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/kokoro/kokoro_backend.hpp"

#include <fcntl.h>
#include <unistd.h>

#if defined(__linux__)
#include <malloc.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "audio/audio_processor.hpp"
#include "backends/kokoro/kokoro_model_downloader.hpp"
#include "text/text_normalizer.hpp"

namespace fs = std::filesystem;

namespace {

void releaseUnusedHeapMemory() noexcept {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

std::vector<float> smoothNarrowImpulses(const std::vector<float>& audio) {
    if (audio.size() < 3) {
        return audio;
    }

    std::vector<float> smoothed(audio);
    for (size_t i = 1; i + 1 < audio.size(); ++i) {
        smoothed[i] = 0.25f * audio[i - 1] + 0.5f * audio[i] + 0.25f * audio[i + 1];
    }
    return smoothed;
}

}  // namespace

namespace tts {

// =============================================================================
// Construction / Destruction
// =============================================================================

KokoroBackend::KokoroBackend()
    : initialized_(false)
    , current_speed_(1.0f) {
}

KokoroBackend::~KokoroBackend() {
    shutdown();
}

// =============================================================================
// Lifecycle
// =============================================================================

ErrorInfo KokoroBackend::initialize(const TtsConfig& config) {
    if (initialized_) {
        return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Backend already initialized");
    }

    config_ = config;

    // Initialize cpp-pinyin for the phonemizer
    try {
        phonemizer_.initPinyin();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG,
            std::string("Failed to initialize cpp-pinyin: ") + e.what());
    }

    // Resolve model directory
    std::string model_dir = getModelDir();

    // Auto-download model and voice files if needed
    std::string voice_name = config.voice.empty() ? "default" : config.voice;
    KokoroModelDownloader downloader;
    if (!downloader.ensureModelsExist(voice_name)) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
            "Failed to download Kokoro models to: " + model_dir);
    }

    // Load voice
    std::string voice_path = model_dir + "/voices/" + voice_name + ".bin";
    if (!fs::exists(voice_path)) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
            "Kokoro voice file not found at: " + voice_path + "\n"
            "Please download a voice file and place it at:\n"
            "  " + model_dir + "/voices/" + voice_name + ".bin");
    }

    if (!voice_manager_.loadVoice(voice_path)) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
            "Failed to load voice file: " + voice_path);
    }

    try {
        // Initialize ONNX Runtime (suppress stderr warnings)
        std::string model_path = model_dir + "/" + KokoroModelDownloader::MODEL_FILE;
        int stderr_fd = dup(STDERR_FILENO);
        int devnull_fd = open("/dev/null", O_WRONLY);
        dup2(devnull_fd, STDERR_FILENO);

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "KokoroBackend");

        // Restore stderr
        dup2(stderr_fd, STDERR_FILENO);
        close(stderr_fd);
        close(devnull_fd);

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config.num_threads > 0 ? config.num_threads : 2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        #if defined(__riscv) || defined(__riscv__)
        session_options.DisableMemPattern();
        session_options.DisableCpuMemArena();
        #endif

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

        // Warm up with a small inference
        if (config.enable_warmup) {
            std::cout << "[Kokoro] Warming up model..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<int64_t> small_tokens = {0, 43, 56, 0};  // pad, 'a', 'n', pad
            auto style = voice_manager_.getStyleVector(static_cast<int>(small_tokens.size()));
            runInference(small_tokens, style, 1.0f);

            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[Kokoro] Model warmed up in " << dur.count() << "ms" << std::endl;
        }

        initialized_ = true;
        current_speed_ = config.speech_rate;

        std::cout << "[Kokoro] Using voice: " << voice_name << std::endl;
        std::cout << "[Kokoro] Backend initialized successfully" << std::endl;
        return ErrorInfo::ok();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
            std::string("Failed to initialize Kokoro model: ") + e.what());
    }
}

void KokoroBackend::shutdown() {
    if (initialized_) {
        session_.reset();
        env_.reset();
        initialized_ = false;
    }
    releaseUnusedHeapMemory();
}

bool KokoroBackend::isInitialized() const {
    return initialized_;
}

// =============================================================================
// Backend Info
// =============================================================================

BackendType KokoroBackend::getType() const {
    return BackendType::KOKORO;
}

std::string KokoroBackend::getName() const {
    return "Kokoro-TTS v1.0 (Chinese/English)";
}

std::string KokoroBackend::getVersion() const {
    return "1.0.0";
}

bool KokoroBackend::supportsStreaming() const {
    return false;
}

int KokoroBackend::getNumSpeakers() const {
    return 1;
}

int KokoroBackend::getSampleRate() const {
    return SAMPLE_RATE;
}

// =============================================================================
// Synthesis
// =============================================================================

ErrorInfo KokoroBackend::synthesize(const std::string& text, SynthesisResult& result) {
    if (!initialized_) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Backend not initialized");
    }

    if (text.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_TEXT, "Empty text");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Step 1: Convert to token IDs via phonemizer (handles normalization internally)
        std::vector<int64_t> token_ids = phonemizer_.textToTokenIds(text);

        if (token_ids.empty()) {
            result.audio = AudioChunk::fromFloat({}, SAMPLE_RATE, true);
            result.success = true;
            return ErrorInfo::ok();
        }

        // Step 2: Get style vector based on token length
        auto style_vector = voice_manager_.getStyleVector(static_cast<int>(token_ids.size()));

        // Step 3: Run ONNX inference
        // Kokoro uses inverse speed: 1.0/speech_rate
        float kokoro_speed = 1.0f / current_speed_;
        std::vector<float> audio_samples = runInference(token_ids, style_vector, kokoro_speed);

        if (audio_samples.empty()) {
            result.audio = AudioChunk::fromFloat({}, SAMPLE_RATE, true);
            result.success = true;
            return ErrorInfo::ok();
        }

        // Step 4: Audio post-processing
        audio::AudioProcessConfig audio_config;
        audio_config.target_rms = config_.target_rms;
        audio_config.compression_ratio = config_.compression_ratio;
        audio_config.use_rms_norm = config_.use_rms_norm;
        audio_config.remove_clicks = config_.remove_clicks;

        audio_samples = audio::processAudio(audio_samples, audio_config);
        if (config_.remove_clicks) {
            audio_samples = smoothNarrowImpulses(audio_samples);
        }

        // Record timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Fill result
        result.audio = AudioChunk::fromFloat(audio_samples, SAMPLE_RATE, true);
        result.audio_duration_ms = result.audio.getDurationMs();
        result.processing_time_ms = duration.count();
        result.calculateRTF();
        result.success = true;

        // Add sentence info
        SentenceInfo sentence;
        sentence.text = text;
        sentence.begin_time_ms = 0;
        sentence.end_time_ms = result.audio_duration_ms;
        sentence.is_final = true;
        result.sentences.push_back(sentence);

        // Trigger callback if set
        if (callback_) {
            notifyAudioChunk(result.audio);
        }

        return ErrorInfo::ok();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::SYNTHESIS_FAILED,
            std::string("Kokoro synthesis failed: ") + e.what());
    }
}

ErrorInfo KokoroBackend::setSpeed(float speed) {
    if (speed <= 0.0f || speed > 10.0f) {
        return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Speed must be between 0.1 and 10.0");
    }
    current_speed_ = speed;
    return ErrorInfo::ok();
}

// =============================================================================
// Private Methods
// =============================================================================

std::string KokoroBackend::getModelDir() const {
    std::string model_dir = config_.model_dir;
    if (model_dir.empty()) {
        model_dir = "~/.cache/models/tts/kokoro-tts";
    }
    // Expand ~
    if (!model_dir.empty() && model_dir[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            model_dir = std::string(home) + model_dir.substr(1);
        }
    }
    return model_dir;
}

std::vector<float> KokoroBackend::runInference(
    const std::vector<int64_t>& token_ids,
    const std::vector<float>& style_vector,
    float speed) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input 1: input_ids [1, seq_len]
    std::vector<int64_t> ids_shape = {1, static_cast<int64_t>(token_ids.size())};
    auto ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(token_ids.data()),
        token_ids.size(),
        ids_shape.data(),
        ids_shape.size());

    // Input 2: style [1, 256]
    std::vector<int64_t> style_shape = {1, KokoroVoiceManager::STYLE_DIM};
    auto style_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(style_vector.data()),
        style_vector.size(),
        style_shape.data(),
        style_shape.size());

    // Input 3: speed [1]
    std::vector<float> speed_data = {speed};
    std::vector<int64_t> speed_shape = {1};
    auto speed_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        speed_data.data(),
        speed_data.size(),
        speed_shape.data(),
        speed_shape.size());

    // Run inference
    const char* input_names[] = {"input_ids", "style", "speed"};
    const char* output_names[] = {"waveform"};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(ids_tensor));
    input_tensors.push_back(std::move(style_tensor));
    input_tensors.push_back(std::move(speed_tensor));

    std::lock_guard<std::mutex> lock(inference_mutex_);
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 3,
        output_names, 1);

    // Extract audio output [1, num_samples]
    float* audio_data = output_tensors[0].GetTensorMutableData<float>();
    auto audio_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t num_samples = 1;
    for (auto dim : audio_shape) {
        num_samples *= static_cast<size_t>(dim);
    }

    return std::vector<float>(audio_data, audio_data + num_samples);
}

}  // namespace tts
