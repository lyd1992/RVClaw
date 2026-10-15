/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Silero VAD Backend Implementation
 */

#include "silero_backend.hpp"

#include <curl/curl.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace vad {

// LSTM hidden size for Silero VAD
static constexpr size_t LSTM_HIDDEN_SIZE = 128;
static constexpr size_t LSTM_NUM_LAYERS = 2;
static constexpr size_t STATE_SIZE = LSTM_NUM_LAYERS * 1 * LSTM_HIDDEN_SIZE;  // (2, 1, 128)

SileroBackend::SileroBackend()
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
}

SileroBackend::~SileroBackend() {
    shutdown();
}

void SileroBackend::shutdown() {
    // Clear vectors that may reference session data first
    input_names_.clear();
    output_names_.clear();
    input_names_str_.clear();
    output_names_str_.clear();
    state_h_.clear();
    state_c_.clear();
    context_.clear();
    prob_history_.clear();

    // Release ONNX resources in correct order: session before env
    session_.reset();
    env_.reset();
}

std::string SileroBackend::expandPath(const std::string& path) {
    if (path.empty()) return path;

    std::string result = path;
    if (result[0] == '~') {
        const char* home = getenv("HOME");
        if (home) {
            result = std::string(home) + result.substr(1);
        }
    }
    return result;
}

std::string SileroBackend::findModelPath(const std::string& model_dir) {
    std::string dir = expandPath(model_dir);

    // Try common model file names
    std::vector<std::string> candidates = {
        dir + "/silero_vad.onnx",
        dir + "/silero_vad_v5.onnx",
        dir + "/model.onnx",
        dir  // If model_dir is actually a file path
    };

    for (const auto& path : candidates) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            return path;
        }
    }

    // Try default paths
    const char* home = getenv("HOME");
    if (home) {
        std::vector<std::string> default_paths = {
            std::string(home) + "/.cache/models/vad/silero/silero_vad.onnx",
            std::string(home) + "/.cache/silero/silero_vad.onnx",
            std::string(home) + "/.cache/silero_vad/silero_vad.onnx",
            std::string(home) + "/.cache/sensevoice/silero_vad.onnx",
        };

        for (const auto& path : default_paths) {
            struct stat st;
            if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                return path;
            }
        }
    }

    return "";
}

static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    size_t total_size = size * nmemb;
    file->write(static_cast<const char*>(contents), total_size);
    return total_size;
}

static int progressCallback(void* /*clientp*/, double dltotal, double dlnow,
    double /*ultotal*/, double /*ulnow*/) {
    if (dltotal > 0) {
        int progress = static_cast<int>((dlnow / dltotal) * 100);
        double dl_mb = dlnow / (1024.0 * 1024.0);
        double total_mb = dltotal / (1024.0 * 1024.0);
        std::cout << "\r[SileroVAD] Downloading: " << progress << "% ("
            << std::fixed << std::setprecision(1) << dl_mb << "/"
            << total_mb << " MB)" << std::flush;
    }
    return 0;
}

bool SileroBackend::downloadModel(const std::string& dest_dir) {
    static const std::string kModelUrl =
        "https://archive.spacemit.com/spacemit-ai/model_zoo/vad/silero/silero_vad.onnx";
    std::string dir = expandPath(dest_dir);
    std::string dest_path = dir + "/silero_vad.onnx";

    // Create directory if not exists (mkdir -p equivalent)
    for (size_t pos = 1; pos <= dir.size(); ++pos) {
        if (pos == dir.size() || dir[pos] == '/') {
            std::string sub = dir.substr(0, pos);
            if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                std::cerr << "[SileroVAD] Failed to create directory " << sub
                    << ": " << strerror(errno) << std::endl;
                return false;
            }
        }
    }

    // Skip if already exists
    struct stat st;
    if (stat(dest_path.c_str(), &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
        return true;
    }

    std::cout << "[SileroVAD] Model not found, downloading from " << kModelUrl << std::endl;

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[SileroVAD] Failed to initialize CURL" << std::endl;
        return false;
    }

    std::ofstream file(dest_path, std::ios::binary);
    if (!file) {
        std::cerr << "[SileroVAD] Failed to open file for writing: " << dest_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, kModelUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "silero-vad/1.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    CURLcode res = curl_easy_perform(curl);
    int64_t http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);
    file.close();

    if (res != CURLE_OK) {
        std::cerr << "\n[SileroVAD] Download failed: " << curl_easy_strerror(res) << std::endl;
        std::remove(dest_path.c_str());
        return false;
    }

    if (http_code != 200) {
        std::cerr << "\n[SileroVAD] HTTP error " << http_code << " for " << kModelUrl << std::endl;
        std::remove(dest_path.c_str());
        return false;
    }

    // Verify file size > 0
    if (stat(dest_path.c_str(), &st) != 0 || st.st_size <= 0) {
        std::cerr << "\n[SileroVAD] Downloaded file is empty" << std::endl;
        std::remove(dest_path.c_str());
        return false;
    }

    std::cout << "\n[SileroVAD] Model downloaded to " << dest_path << std::endl;
    return true;
}

ErrorInfo SileroBackend::initialize(const VadConfig& config) {
    config_ = config;

    // Find model path
    model_path_ = findModelPath(config.model_dir);
    if (model_path_.empty()) {
        // Try to download the model
        std::string download_dir = config.model_dir.empty() ? "~/.cache/models/vad/silero" : config.model_dir;
        if (!downloadModel(download_dir)) {
            return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
                "Silero VAD model not found and download failed",
                "Searched in: " + config.model_dir);
        }
        model_path_ = findModelPath(config.model_dir);
        if (model_path_.empty()) {
            return ErrorInfo::error(ErrorCode::MODEL_NOT_FOUND,
                "Silero VAD model not found after download",
                "Searched in: " + config.model_dir);
        }
    }

    // Validate sample rate
    if (config.sample_rate != 16000 && config.sample_rate != 8000) {
        return ErrorInfo::error(ErrorCode::UNSUPPORTED_SAMPLE_RATE,
            "Unsupported sample rate",
            "Silero VAD supports 8000 or 16000 Hz, got: " + std::to_string(config.sample_rate));
    }

    // Initialize ONNX Runtime
    if (!initializeSession()) {
        return ErrorInfo::error(ErrorCode::BACKEND_ERROR,
            "Failed to initialize ONNX session",
            "Model path: " + model_path_);
    }

    // Reset state
    reset();

    return ErrorInfo::ok();
}

bool SileroBackend::initializeSession() {
    try {
        // Suppress stderr temporarily to avoid ONNX schema warnings
        int stderr_fd = dup(STDERR_FILENO);
        int devnull_fd = open("/dev/null", O_WRONLY);
        if (devnull_fd >= 0) {
            dup2(devnull_fd, STDERR_FILENO);
        }

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SileroVAD");

        // Restore stderr
        if (stderr_fd >= 0) {
            dup2(stderr_fd, STDERR_FILENO);
            close(stderr_fd);
        }
        if (devnull_fd >= 0) {
            close(devnull_fd);
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config_.num_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), session_options);

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session_->GetInputCount();
        input_names_str_.clear();
        input_names_.clear();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_.push_back(std::string(name.get()));
        }
        for (const auto& name : input_names_str_) {
            input_names_.push_back(name.c_str());
        }

        size_t num_outputs = session_->GetOutputCount();
        output_names_str_.clear();
        output_names_.clear();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_.push_back(std::string(name.get()));
        }
        for (const auto& name : output_names_str_) {
            output_names_.push_back(name.c_str());
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX session: " << e.what() << std::endl;
        return false;
    }
}

void SileroBackend::reset() {
    // Initialize LSTM states
    state_h_.assign(STATE_SIZE, 0.0f);
    state_c_.assign(STATE_SIZE, 0.0f);
    context_.assign(config_.context_size, 0.0f);
    prob_history_.clear();

    // Reset streaming state
    current_state_ = VadState::SILENCE;
    speech_start_time_ = -1;
    frame_count_ = 0;
}

float SileroBackend::runInference(const float* audio, size_t length) {
    if (!session_) {
        return 0.0f;
    }

    try {
        // Prepare input audio (pad or truncate to window_size)
        std::vector<float> input_audio(config_.window_size);
        size_t copy_len = std::min(length, static_cast<size_t>(config_.window_size));
        std::copy(audio, audio + copy_len, input_audio.begin());
        if (copy_len < static_cast<size_t>(config_.window_size)) {
            std::fill(input_audio.begin() + copy_len, input_audio.end(), 0.0f);
        }

        // Concatenate context and current audio
        std::vector<float> x(config_.context_size + config_.window_size);
        std::copy(context_.begin(), context_.end(), x.begin());
        std::copy(input_audio.begin(), input_audio.end(), x.begin() + config_.context_size);

        // Update context for next frame
        std::copy(x.end() - config_.context_size, x.end(), context_.begin());

        // Prepare input tensors
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(x.size())};
        std::vector<int64_t> sr_shape = {1};
        std::vector<int64_t> state_shape = {LSTM_NUM_LAYERS, 1, LSTM_HIDDEN_SIZE};

        std::vector<int64_t> sr_value = {config_.sample_rate};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, x.data(), x.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, state_h_.data(), state_h_.size(), state_shape.data(), state_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, sr_value.data(), sr_value.size(), sr_shape.data(), sr_shape.size()));

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), input_tensors.data(), input_tensors.size(),
            output_names_.data(), output_names_.size());

        // Get probability
        float* prob_data = output_tensors[0].GetTensorMutableData<float>();
        float prob = prob_data[0];

        // Update LSTM states
        if (output_tensors.size() > 2) {
            float* new_state_h = output_tensors[1].GetTensorMutableData<float>();
            std::copy(new_state_h, new_state_h + state_h_.size(), state_h_.begin());

            float* new_state_c = output_tensors[2].GetTensorMutableData<float>();
            std::copy(new_state_c, new_state_c + state_c_.size(), state_c_.begin());
        } else if (output_tensors.size() > 1) {
            float* new_state_h = output_tensors[1].GetTensorMutableData<float>();
            std::copy(new_state_h, new_state_h + state_h_.size(), state_h_.begin());
        }

        return prob;
    } catch (const std::exception& e) {
        std::cerr << "Silero VAD inference error: " << e.what() << std::endl;
        return 0.0f;
    }
}

float SileroBackend::applySmoothing(float prob) {
    if (!config_.use_smoothing) {
        return prob;
    }

    prob_history_.push_back(prob);
    while (prob_history_.size() > static_cast<size_t>(config_.smoothing_window)) {
        prob_history_.pop_front();
    }

    return std::accumulate(prob_history_.begin(), prob_history_.end(), 0.0f)
            / prob_history_.size();
}

VadState SileroBackend::updateState(float smoothed_prob, DetectionResult& result) {
    VadState new_state = current_state_;

    bool is_speech = smoothed_prob >= config_.trigger_threshold;
    bool is_silence = smoothed_prob < config_.stop_threshold;

    // Frame duration in ms
    int frame_duration_ms = (config_.window_size * 1000) / config_.sample_rate;
    int64_t current_time_ms = frame_count_ * frame_duration_ms;

    switch (current_state_) {
        case VadState::SILENCE:
            if (is_speech) {
                new_state = VadState::SPEECH_START;
                speech_start_time_ = current_time_ms;
                result.speech_start_ms = speech_start_time_;
            }
            break;

        case VadState::SPEECH_START:
            if (is_speech) {
                new_state = VadState::SPEECH;
            } else if (is_silence) {
                new_state = VadState::SILENCE;
                speech_start_time_ = -1;
            }
            break;

        case VadState::SPEECH:
            if (is_silence) {
                new_state = VadState::SPEECH_END;
                result.speech_end_ms = current_time_ms;
                result.speech_duration_ms = static_cast<int>(current_time_ms - speech_start_time_);
            }
            break;

        case VadState::SPEECH_END:
            if (is_speech) {
                new_state = VadState::SPEECH_START;
                speech_start_time_ = current_time_ms;
                result.speech_start_ms = speech_start_time_;
            } else {
                new_state = VadState::SILENCE;
                speech_start_time_ = -1;
            }
            break;
    }

    current_state_ = new_state;
    return new_state;
}

ErrorInfo SileroBackend::detect(const AudioChunk& audio, DetectionResult& result) {
    if (!session_) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED,
            "Backend not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Run inference
    float prob = runInference(audio.data, audio.num_samples);
    float smoothed_prob = applySmoothing(prob);

    // Update state
    VadState state = updateState(smoothed_prob, result);

    auto end_time = std::chrono::steady_clock::now();
    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    // Fill result
    result.probability = prob;
    result.smoothed_probability = smoothed_prob;
    result.state = state;
    result.is_speech = (state == VadState::SPEECH_START || state == VadState::SPEECH);
    result.processing_time_ms = static_cast<int>(processing_time);

    // Calculate timestamp
    int frame_duration_ms = (config_.window_size * 1000) / config_.sample_rate;
    result.timestamp_ms = frame_count_ * frame_duration_ms;

    frame_count_++;

    return ErrorInfo::ok();
}

}  // namespace vad
