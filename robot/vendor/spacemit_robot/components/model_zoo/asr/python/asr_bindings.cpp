/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * ASR Python Bindings
 *
 * Provides Python interface to the ASR engine using pybind11.
 * Supports numpy arrays for audio data and Python callbacks.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>
#include <string>
#include <utility>

#include "asr_engine.hpp"
#include "asr_config.hpp"
#include "asr_callback.hpp"

namespace py = pybind11;

// =============================================================================
// Python Callback Wrapper
// =============================================================================

class PyASRCallback : public asr::IASRCallback {
public:
    using ResultCallback = std::function<void(const asr::RecognitionResult&)>;
    using ErrorCallback = std::function<void(const asr::ErrorInfo&)>;
    using SimpleCallback = std::function<void()>;

    void setOnResult(ResultCallback cb) { on_result_ = std::move(cb); }
    void setOnError(ErrorCallback cb) { on_error_ = std::move(cb); }
    void setOnStart(SimpleCallback cb) { on_start_ = std::move(cb); }
    void setOnComplete(SimpleCallback cb) { on_complete_ = std::move(cb); }
    void setOnClose(SimpleCallback cb) { on_close_ = std::move(cb); }

    void onResult(const asr::RecognitionResult& result) override {
        if (on_result_) {
            py::gil_scoped_acquire acquire;
            on_result_(result);
        }
    }

    void onError(const asr::ErrorInfo& error) override {
        if (on_error_) {
            py::gil_scoped_acquire acquire;
            on_error_(error);
        }
    }

    void onStart() override {
        if (on_start_) {
            py::gil_scoped_acquire acquire;
            on_start_();
        }
    }

    void onComplete() override {
        if (on_complete_) {
            py::gil_scoped_acquire acquire;
            on_complete_();
        }
    }

    void onClose() override {
        if (on_close_) {
            py::gil_scoped_acquire acquire;
            on_close_();
        }
    }

private:
    ResultCallback on_result_;
    ErrorCallback on_error_;
    SimpleCallback on_start_;
    SimpleCallback on_complete_;
    SimpleCallback on_close_;
};

// =============================================================================
// Python Module Definition
// =============================================================================

PYBIND11_MODULE(_spacemit_asr, m) {
    m.doc() = "ASR (Automatic Speech Recognition) Python bindings";

    // -------------------------------------------------------------------------
    // Enums
    // -------------------------------------------------------------------------

    py::enum_<asr::Language>(m, "Language", "Language codes for ASR")
        .value("AUTO", asr::Language::AUTO, "Auto-detect language")
        .value("ZH", asr::Language::ZH, "Chinese")
        .value("EN", asr::Language::EN, "English")
        .value("JA", asr::Language::JA, "Japanese")
        .value("KO", asr::Language::KO, "Korean")
        .value("YUE", asr::Language::YUE, "Cantonese")
        .export_values();

    py::enum_<asr::AudioFormat>(m, "AudioFormat", "Audio format types")
        .value("PCM_S16LE", asr::AudioFormat::PCM_S16LE, "16-bit signed PCM")
        .value("PCM_F32LE", asr::AudioFormat::PCM_F32LE, "32-bit float PCM")
        .value("WAV", asr::AudioFormat::WAV, "WAV file format")
        .value("MP3", asr::AudioFormat::MP3, "MP3 format")
        .value("OPUS", asr::AudioFormat::OPUS, "Opus format")
        .value("AAC", asr::AudioFormat::AAC, "AAC format")
        .export_values();

    py::enum_<asr::RecognitionMode>(m, "RecognitionMode", "Recognition modes")
        .value("OFFLINE", asr::RecognitionMode::OFFLINE, "Offline batch processing")
        .value("STREAMING", asr::RecognitionMode::STREAMING, "Real-time streaming")
        .export_values();

    py::enum_<asr::BackendType>(m, "BackendType", "ASR backend types")
        .value("SENSEVOICE", asr::BackendType::SENSEVOICE, "SenseVoice (local ONNX)")
        .value("FUNASR", asr::BackendType::FUNASR, "FunASR (cloud)")
        .value("WHISPER", asr::BackendType::WHISPER, "Whisper")
        .value("PARAFORMER", asr::BackendType::PARAFORMER, "Paraformer")
        .value("QWEN3_ASR", asr::BackendType::QWEN3_ASR, "Qwen3-ASR (llama-server HTTP)")
        .value("ZIPFORMER", asr::BackendType::ZIPFORMER, "Zipformer CTC (local ONNX)")
        .value("CUSTOM", asr::BackendType::CUSTOM, "Custom backend")
        .export_values();

    py::enum_<asr::ErrorCode>(m, "ErrorCode", "Error codes")
        .value("OK", asr::ErrorCode::OK)
        .value("INVALID_CONFIG", asr::ErrorCode::INVALID_CONFIG)
        .value("MODEL_NOT_FOUND", asr::ErrorCode::MODEL_NOT_FOUND)
        .value("UNSUPPORTED_FORMAT", asr::ErrorCode::UNSUPPORTED_FORMAT)
        .value("UNSUPPORTED_SAMPLE_RATE", asr::ErrorCode::UNSUPPORTED_SAMPLE_RATE)
        .value("NOT_INITIALIZED", asr::ErrorCode::NOT_INITIALIZED)
        .value("ALREADY_STARTED", asr::ErrorCode::ALREADY_STARTED)
        .value("NOT_STARTED", asr::ErrorCode::NOT_STARTED)
        .value("INFERENCE_FAILED", asr::ErrorCode::INFERENCE_FAILED)
        .value("TIMEOUT", asr::ErrorCode::TIMEOUT)
        .value("NETWORK_ERROR", asr::ErrorCode::NETWORK_ERROR)
        .value("CONNECTION_FAILED", asr::ErrorCode::CONNECTION_FAILED)
        .value("AUTH_FAILED", asr::ErrorCode::AUTH_FAILED)
        .value("INTERNAL_ERROR", asr::ErrorCode::INTERNAL_ERROR)
        .value("OUT_OF_MEMORY", asr::ErrorCode::OUT_OF_MEMORY)
        .export_values();

    // -------------------------------------------------------------------------
    // Error Info
    // -------------------------------------------------------------------------

    py::class_<asr::ErrorInfo>(m, "ErrorInfo", "Error information")
        .def(py::init<>())
        .def_readonly("code", &asr::ErrorInfo::code)
        .def_readonly("message", &asr::ErrorInfo::message)
        .def_readonly("detail", &asr::ErrorInfo::detail)
        .def("is_ok", &asr::ErrorInfo::isOk, "Check if no error")
        .def("__bool__", &asr::ErrorInfo::isOk)
        .def("__repr__", [](const asr::ErrorInfo& e) {
            if (e.isOk()) return std::string("ErrorInfo(OK)");
            return "ErrorInfo(" + e.message + ")";
        });

    // -------------------------------------------------------------------------
    // Word Result
    // -------------------------------------------------------------------------

    py::class_<asr::WordResult>(m, "WordResult", "Word-level recognition result")
        .def(py::init<>())
        .def_readonly("text", &asr::WordResult::text)
        .def_readonly("begin_time_ms", &asr::WordResult::begin_time_ms)
        .def_readonly("end_time_ms", &asr::WordResult::end_time_ms)
        .def_readonly("confidence", &asr::WordResult::confidence)
        .def_readonly("punctuation", &asr::WordResult::punctuation)
        .def("__repr__", [](const asr::WordResult& w) {
            return "WordResult('" + w.text + "', " +
                    std::to_string(w.begin_time_ms) + "-" +
                    std::to_string(w.end_time_ms) + "ms)";
        });

    // -------------------------------------------------------------------------
    // Sentence Result
    // -------------------------------------------------------------------------

    py::class_<asr::SentenceResult>(m, "SentenceResult", "Sentence-level recognition result")
        .def(py::init<>())
        .def_readonly("text", &asr::SentenceResult::text)
        .def_readonly("begin_time_ms", &asr::SentenceResult::begin_time_ms)
        .def_readonly("end_time_ms", &asr::SentenceResult::end_time_ms)
        .def_readonly("confidence", &asr::SentenceResult::confidence)
        .def_readonly("is_final", &asr::SentenceResult::is_final)
        .def_readonly("words", &asr::SentenceResult::words)
        .def_readonly("detected_language", &asr::SentenceResult::detected_language)
        .def_readonly("emotion", &asr::SentenceResult::emotion)
        .def("__repr__", [](const asr::SentenceResult& s) {
            return "SentenceResult('" + s.text + "')";
        });

    // -------------------------------------------------------------------------
    // Recognition Result
    // -------------------------------------------------------------------------

    py::class_<asr::RecognitionResult>(m, "RecognitionResult", "Recognition result")
        .def(py::init<>())
        .def_readonly("request_id", &asr::RecognitionResult::request_id)
        .def_readonly("sentences", &asr::RecognitionResult::sentences)
        .def_readonly("audio_duration_ms", &asr::RecognitionResult::audio_duration_ms)
        .def_readonly("processing_time_ms", &asr::RecognitionResult::processing_time_ms)
        .def_readonly("rtf", &asr::RecognitionResult::rtf)
        .def_readonly("first_result_latency_ms", &asr::RecognitionResult::first_result_latency_ms)
        .def_readonly("final_result_latency_ms", &asr::RecognitionResult::final_result_latency_ms)
        .def("get_text", &asr::RecognitionResult::getText, "Get full text")
        .def("is_empty", &asr::RecognitionResult::isEmpty, "Check if result is empty")
        .def_property_readonly("text", &asr::RecognitionResult::getText, "Full text property")
        .def("__str__", &asr::RecognitionResult::getText)
        .def("__repr__", [](const asr::RecognitionResult& r) {
            return "RecognitionResult('" + r.getText() + "', rtf=" +
                    std::to_string(r.rtf) + ")";
        })
        .def("__bool__", [](const asr::RecognitionResult& r) {
            return !r.isEmpty();
        });

    // -------------------------------------------------------------------------
    // ASR Config
    // -------------------------------------------------------------------------

    py::class_<asr::ASRConfig>(m, "ASRConfig", "ASR configuration")
        .def(py::init<>())
        // Constructor with model_dir - creates SenseVoice config
        .def(py::init([](const std::string& model_dir) {
            return asr::ASRConfig::sensevoice(model_dir);
        }), py::arg("model_dir"), "Create SenseVoice configuration with model directory")
        // Backend
        .def_readwrite("backend", &asr::ASRConfig::backend)
        // Model paths
        .def_readwrite("model_path", &asr::ASRConfig::model_path)
        .def_readwrite("config_path", &asr::ASRConfig::config_path)
        .def_readwrite("vocab_path", &asr::ASRConfig::vocab_path)
        .def_readwrite("decoder_path", &asr::ASRConfig::decoder_path)
        // Model options
        .def_readwrite("use_quantized", &asr::ASRConfig::use_quantized)
        .def_readwrite("num_threads", &asr::ASRConfig::num_threads)
        // Audio format
        .def_readwrite("audio_format", &asr::ASRConfig::audio_format)
        .def_readwrite("sample_rate", &asr::ASRConfig::sample_rate)
        .def_readwrite("channels", &asr::ASRConfig::channels)
        // Recognition mode
        .def_readwrite("mode", &asr::ASRConfig::mode)
        // Language
        .def_readwrite("language", &asr::ASRConfig::language)
        .def_readwrite("language_hints", &asr::ASRConfig::language_hints)
        // VAD
        .def_readwrite("vad_enabled", &asr::ASRConfig::vad_enabled)
        .def_readwrite("vad_silence_threshold_ms", &asr::ASRConfig::vad_silence_threshold_ms)
        .def_readwrite("vad_threshold", &asr::ASRConfig::vad_threshold)
        // Punctuation
        .def_readwrite("punctuation_enabled", &asr::ASRConfig::punctuation_enabled)
        .def_readwrite("itn_enabled", &asr::ASRConfig::itn_enabled)
        // Hotwords
        .def_readwrite("hotwords", &asr::ASRConfig::hotwords)
        .def_readwrite("hotword_boost", &asr::ASRConfig::hotword_boost)
        // Streaming
        .def_readwrite("chunk_size_ms", &asr::ASRConfig::chunk_size_ms)
        .def_readwrite("return_partial_results", &asr::ASRConfig::return_partial_results)
        .def_readwrite("return_word_timestamps", &asr::ASRConfig::return_word_timestamps)
        .def_readwrite("enable_emotion", &asr::ASRConfig::enable_emotion)
        // Performance
        .def_readwrite("timeout_ms", &asr::ASRConfig::timeout_ms)
        // Static factory methods
        .def_static("sensevoice", &asr::ASRConfig::sensevoice,
                    py::arg("model_dir") = "~/.cache/models/asr/sensevoice",
                    "Create SenseVoice configuration")
        .def_static("zipformer", &asr::ASRConfig::zipformer,
                    py::arg("model_dir") = "~/.cache/models/asr/zipformer",
                    "Create Zipformer CTC configuration")
        .def_static("funasr_cloud", &asr::ASRConfig::funasrCloud,
                    py::arg("api_key"),
                    py::arg("model_id") = "fun-asr-realtime",
                    "Create FunASR cloud configuration")
        .def_static("qwen3_asr", [](const std::string& endpoint,
                                    const std::string& model,
                                    int timeout) {
            asr::ASRConfig config;
            config.backend = asr::BackendType::QWEN3_ASR;
            config.extra_params["endpoint"] = endpoint;
            config.extra_params["model"] = model;
            config.extra_params["timeout"] = std::to_string(timeout);
            return config;
        }, py::arg("endpoint") = "http://127.0.0.1:8063/v1/chat/completions",
            py::arg("model") = "qwen3-asr", py::arg("timeout") = 60,
            "Create Qwen3-ASR llama-server configuration")
        // Builder methods
        .def("with_streaming", &asr::ASRConfig::withStreaming,
            py::arg("chunk_ms") = 100,
            "Enable streaming mode")
        .def("with_language", &asr::ASRConfig::withLanguage,
            py::arg("language"),
            "Set language")
        .def("without_vad", &asr::ASRConfig::withoutVAD, "Disable VAD")
        .def("with_word_timestamps", &asr::ASRConfig::withWordTimestamps,
            "Enable word timestamps")
        .def("with_emotion", &asr::ASRConfig::withEmotion,
            py::arg("enabled") = true,
            "Enable emotion labels")
        .def_readwrite("extra_params", &asr::ASRConfig::extra_params,
            "Extra backend-specific parameters (e.g. provider)");

    // -------------------------------------------------------------------------
    // ASR Engine
    // -------------------------------------------------------------------------

    py::class_<asr::ASREngine>(m, "ASREngine", "Main ASR engine class")
        .def(py::init<>())
        // Lifecycle
        .def("initialize", &asr::ASREngine::initialize,
            py::arg("config"),
            "Initialize the engine with configuration")
        .def("shutdown", &asr::ASREngine::shutdown, "Shutdown the engine")
        .def("is_initialized", &asr::ASREngine::isInitialized,
            "Check if engine is initialized")
        .def_property_readonly("config", &asr::ASREngine::getConfig,
            "Get current configuration")

        // Offline recognition with numpy arrays
        // Note: Release GIL during inference to allow audio callbacks to run
        .def("recognize", [](asr::ASREngine& self, py::array_t<float> audio) {
            auto buf = audio.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Audio array must be 1-dimensional");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            size_t size = buf.size;
            // Release GIL during ONNX inference
            py::gil_scoped_release release;
            return self.recognize(ptr, size);
        }, py::arg("audio"), "Recognize float audio array")

        .def("recognize", [](asr::ASREngine& self, py::array_t<int16_t> audio) {
            auto buf = audio.request();
            if (buf.ndim != 1) {
                throw std::runtime_error("Audio array must be 1-dimensional");
            }
            int16_t* ptr = static_cast<int16_t*>(buf.ptr);
            size_t size = buf.size;
            // Release GIL during ONNX inference
            py::gil_scoped_release release;
            return self.recognize(ptr, size);
        }, py::arg("audio"), "Recognize int16 audio array")

        .def("recognize_file", [](asr::ASREngine& self, const std::string& file_path) {
            // Release GIL during file loading and inference
            py::gil_scoped_release release;
            return self.recognizeFile(file_path);
        }, py::arg("file_path"), "Recognize audio file")

        // Streaming API
        .def("set_callback", [](asr::ASREngine& self, std::shared_ptr<PyASRCallback> cb) {
            // Store callback to prevent garbage collection
            static std::shared_ptr<PyASRCallback> stored_callback;
            stored_callback = cb;
            self.setCallback(cb.get());
        }, py::arg("callback"), "Set streaming callback")
        .def("start", &asr::ASREngine::start, "Start streaming session")
        .def("send_audio", [](asr::ASREngine& self, py::array_t<float> audio) {
            auto buf = audio.request();
            return self.sendAudio(static_cast<float*>(buf.ptr), buf.size);
        }, py::arg("audio"), "Send audio data in streaming mode (float array)")
        .def("send_audio_frame", [](asr::ASREngine& self, py::bytes data) {
            // PCM S16LE bytes -> float array
            std::string str = data;
            const int16_t* ptr = reinterpret_cast<const int16_t*>(str.data());
            size_t samples = str.size() / 2;
            return self.sendAudio(ptr, samples);
        }, py::arg("data"), "Send audio frame (PCM S16LE bytes)")
        .def("stop", &asr::ASREngine::stop, "Stop streaming session")
        .def("flush", &asr::ASREngine::flush,
            "Flush buffer and recognize immediately (keeps session active)")
        .def("is_streaming", &asr::ASREngine::isStreaming,
            "Check if streaming is active")

        // Status
        .def_property_readonly("last_request_id", &asr::ASREngine::getLastRequestId)
        .def_property_readonly("last_error", &asr::ASREngine::getLastError)
        .def_property_readonly("backend_type", &asr::ASREngine::getBackendType)
        .def_property_readonly("backend_name", &asr::ASREngine::getBackendName)

        // Dynamic config
        .def("update_hotwords", &asr::ASREngine::updateHotwords,
            py::arg("hotwords"), "Update hotword list")
        .def("set_language", &asr::ASREngine::setLanguage,
            py::arg("language"), "Set recognition language")

        // Static methods
        .def_static("get_available_backends", &asr::ASREngine::getAvailableBackends,
                    "Get list of available backends")
        .def_static("is_backend_available", &asr::ASREngine::isBackendAvailable,
                    py::arg("backend"), "Check if backend is available")
        .def_static("get_version", &asr::ASREngine::getVersion,
                    "Get library version");

    // -------------------------------------------------------------------------
    // Python Callback
    // -------------------------------------------------------------------------

    py::class_<PyASRCallback, std::shared_ptr<PyASRCallback>>(m, "ASRCallback",
            "Callback for streaming recognition results")
        .def(py::init<>())
        .def("on_result", &PyASRCallback::setOnResult,
            py::arg("callback"), "Set result callback")
        .def("on_error", &PyASRCallback::setOnError,
            py::arg("callback"), "Set error callback")
        .def("on_start", &PyASRCallback::setOnStart,
            py::arg("callback"), "Set start callback")
        .def("on_complete", &PyASRCallback::setOnComplete,
            py::arg("callback"), "Set complete callback")
        .def("on_close", &PyASRCallback::setOnClose,
            py::arg("callback"), "Set close callback");

    // -------------------------------------------------------------------------
    // Quick Functions
    // -------------------------------------------------------------------------

    m.def("recognize_file", &asr::quick::recognizeFile,
            py::arg("file_path"),
            py::arg("model_dir") = "",
            "Quick recognition of audio file");

    m.def("recognize", [](py::array_t<float> audio, const std::string& model_dir) {
        auto buf = audio.request();
        return asr::quick::recognize(static_cast<float*>(buf.ptr), buf.size, model_dir);
    }, py::arg("audio"), py::arg("model_dir") = "",
        "Quick recognition of float audio array");

    // -------------------------------------------------------------------------
    // Module Info
    // -------------------------------------------------------------------------

    m.attr("__version__") = asr::ASREngine::getVersion();
    m.attr("__author__") = "muggle";
}
