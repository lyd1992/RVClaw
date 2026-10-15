/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tts_service.h"

namespace py = pybind11;

// =============================================================================
// PyTtsCallback - Python 回调包装类
// =============================================================================

/**
 * @brief Python 回调适配器
 *
 * 继承 TtsResultCallback，将 C++ 回调桥接到 Python 函数
 * 关键点：在调用 Python 回调时获取 GIL
 */
class PyTtsCallback : public SpacemiT::TtsResultCallback {
public:
    using OpenCallback = std::function<void()>;
    using EventCallback = std::function<void(std::shared_ptr<SpacemiT::TtsEngineResult>)>;
    using CompleteCallback = std::function<void()>;
    using ErrorCallback = std::function<void(const std::string&)>;
    using CloseCallback = std::function<void()>;

    PyTtsCallback() = default;
    ~PyTtsCallback() override = default;

    // 设置回调
    void setOnOpen(OpenCallback cb) { on_open_ = std::move(cb); }
    void setOnEvent(EventCallback cb) { on_event_ = std::move(cb); }
    void setOnComplete(CompleteCallback cb) { on_complete_ = std::move(cb); }
    void setOnError(ErrorCallback cb) { on_error_ = std::move(cb); }
    void setOnClose(CloseCallback cb) { on_close_ = std::move(cb); }

    // 重写基类虚函数
    void OnOpen() override {
        if (on_open_) {
            py::gil_scoped_acquire acquire;  // 获取 GIL
            on_open_();
        }
    }

    void OnEvent(std::shared_ptr<SpacemiT::TtsEngineResult> result) override {
        if (on_event_) {
            py::gil_scoped_acquire acquire;  // 获取 GIL
            on_event_(result);
        }
    }

    void OnComplete() override {
        if (on_complete_) {
            py::gil_scoped_acquire acquire;  // 获取 GIL
            on_complete_();
        }
    }

    void OnError(const std::string& message) override {
        if (on_error_) {
            py::gil_scoped_acquire acquire;  // 获取 GIL
            on_error_(message);
        }
    }

    void OnClose() override {
        if (on_close_) {
            py::gil_scoped_acquire acquire;  // 获取 GIL
            on_close_();
        }
    }

private:
    OpenCallback on_open_;
    EventCallback on_event_;
    CompleteCallback on_complete_;
    ErrorCallback on_error_;
    CloseCallback on_close_;
};

// =============================================================================
// pybind11 模块定义
// =============================================================================

PYBIND11_MODULE(_spacemit_tts, m) {
    m.doc() = "SpacemiT TTS - Text-To-Speech Engine Python bindings";

    // =========================================================================
    // 枚举类型
    // =========================================================================

    // =========================================================================
    // PronunciationEntry
    // =========================================================================

    py::class_<SpacemiT::PronunciationEntry>(m, "PronunciationEntry",
        "Custom pronunciation entry (word -> phoneme)")
        .def(py::init<>())
        .def(py::init([](const std::string& word, const std::string& phoneme,
                const std::string& locale) {
            SpacemiT::PronunciationEntry e;
            e.word = word;
            e.phoneme = phoneme;
            e.locale = locale;
            return e;
        }), py::arg("word"), py::arg("phoneme"), py::arg("locale") = "zh")
        .def_readwrite("word", &SpacemiT::PronunciationEntry::word)
        .def_readwrite("phoneme", &SpacemiT::PronunciationEntry::phoneme)
        .def_readwrite("locale", &SpacemiT::PronunciationEntry::locale)
        .def("__repr__", [](const SpacemiT::PronunciationEntry& e) {
            return "<PronunciationEntry '" + e.word + "' -> '" + e.phoneme +
                "' locale=" + e.locale + ">";
        });

    // =========================================================================
    // 枚举类型
    // =========================================================================

    py::enum_<SpacemiT::AudioFormat>(m, "AudioFormat", "Audio format types")
        .value("PCM", SpacemiT::AudioFormat::PCM, "Raw PCM data")
        .value("WAV", SpacemiT::AudioFormat::WAV, "WAV file format")
        .value("MP3", SpacemiT::AudioFormat::MP3, "MP3 format (reserved)")
        .value("OGG", SpacemiT::AudioFormat::OGG, "OGG format (reserved)")
        .export_values();

    py::enum_<SpacemiT::BackendType>(m, "BackendType", "TTS backend types")
        .value("MATCHA_ZH", SpacemiT::BackendType::MATCHA_ZH,
            "Matcha Chinese (22050Hz)")
        .value("MATCHA_EN", SpacemiT::BackendType::MATCHA_EN,
            "Matcha English (22050Hz)")
        .value("MATCHA_ZH_EN", SpacemiT::BackendType::MATCHA_ZH_EN,
            "Matcha Chinese-English bilingual (16000Hz)")
        .value("COSYVOICE", SpacemiT::BackendType::COSYVOICE, "CosyVoice (reserved)")
        .value("VITS", SpacemiT::BackendType::VITS, "VITS (reserved)")
        .value("PIPER", SpacemiT::BackendType::PIPER, "Piper TTS (reserved)")
        .value("KOKORO", SpacemiT::BackendType::KOKORO, "Kokoro TTS (reserved)")
        .export_values();

    // =========================================================================
    // TtsConfig - 配置结构
    // =========================================================================

    py::class_<SpacemiT::TtsConfig>(m, "TtsConfig", "TTS engine configuration")
        .def(py::init<>(), "Create default configuration")

        // 字段（readwrite）
        .def_readwrite("backend", &SpacemiT::TtsConfig::backend, "Backend type")
        .def_readwrite("model", &SpacemiT::TtsConfig::model, "Model name")
        .def_readwrite("model_dir", &SpacemiT::TtsConfig::model_dir, "Model directory path")
        .def_readwrite("voice", &SpacemiT::TtsConfig::voice, "Voice name")
        .def_readwrite("speaker_id", &SpacemiT::TtsConfig::speaker_id, "Speaker ID (multi-speaker models)")
        .def_readwrite("format", &SpacemiT::TtsConfig::format, "Output audio format")
        .def_readwrite("sample_rate", &SpacemiT::TtsConfig::sample_rate, "Output sample rate (Hz)")
        .def_readwrite("volume", &SpacemiT::TtsConfig::volume, "Volume [0, 100]")
        .def_readwrite("speech_rate", &SpacemiT::TtsConfig::speech_rate, "Speech rate (>1.0 fast, <1.0 slow)")
        .def_readwrite("pitch", &SpacemiT::TtsConfig::pitch, "Pitch")
        .def_readwrite("target_rms", &SpacemiT::TtsConfig::target_rms, "Target RMS level")
        .def_readwrite("compression_ratio", &SpacemiT::TtsConfig::compression_ratio, "Compression ratio")
        .def_readwrite("use_rms_norm", &SpacemiT::TtsConfig::use_rms_norm, "Use RMS normalization")
        .def_readwrite("remove_clicks", &SpacemiT::TtsConfig::remove_clicks, "Remove clicks")
        .def_readwrite("num_threads", &SpacemiT::TtsConfig::num_threads, "Number of inference threads")
        .def_readwrite("enable_warmup", &SpacemiT::TtsConfig::enable_warmup, "Enable warmup on startup")
        .def_readwrite("provider", &SpacemiT::TtsConfig::provider, "Provider: auto, cpu, spacemit")
        .def_readwrite("lexicon", &SpacemiT::TtsConfig::lexicon, "Custom pronunciation lexicon")

        // 静态工厂方法
        .def_static("preset", &SpacemiT::TtsConfig::Preset,
            py::arg("name"),
            "Create configuration from preset name (e.g. 'matcha_zh', 'matcha_en', 'matcha_zh_en', 'kokoro')")
        .def_static("available_presets", &SpacemiT::TtsConfig::AvailablePresets,
            "Get list of available preset names")

        // Builder 方法（链式调用）
        .def("withSpeed", &SpacemiT::TtsConfig::withSpeed,
            py::arg("speed"),
            "Set speech rate (chainable)")
        .def("withSpeaker", &SpacemiT::TtsConfig::withSpeaker,
            py::arg("id"),
            "Set speaker ID (chainable)")
        .def("withVolume", &SpacemiT::TtsConfig::withVolume,
            py::arg("vol"),
            "Set volume (chainable)")
        .def("withProvider", &SpacemiT::TtsConfig::withProvider,
            py::arg("provider"),
            "Set provider: auto, cpu, spacemit (chainable)")

        // 字符串表示
        .def("__repr__", [](const SpacemiT::TtsConfig& config) {
            return "<TtsConfig backend=" + std::to_string(static_cast<int>(config.backend)) +
                " model='" + config.model + "'" +
                " provider='" + config.provider + "'" +
                " sample_rate=" + std::to_string(config.sample_rate) + ">";
        });

    // =========================================================================
    // TtsEngineResult - 合成结果
    // =========================================================================

    py::class_<SpacemiT::TtsEngineResult, std::shared_ptr<SpacemiT::TtsEngineResult>>(
        m, "TtsEngineResult", "TTS synthesis result")

        // 音频数据获取
        .def("get_audio_data", &SpacemiT::TtsEngineResult::GetAudioData,
            "Get audio as raw bytes (uint8[])")
        .def("get_audio_float", &SpacemiT::TtsEngineResult::GetAudioFloat,
            "Get audio as float32 array [-1.0, 1.0]")
        .def("get_audio_int16", &SpacemiT::TtsEngineResult::GetAudioInt16,
            "Get audio as int16 array (PCM)")

        // 元信息
        .def("get_timestamp", &SpacemiT::TtsEngineResult::GetTimestamp,
            "Get timestamp information (JSON string)")
        .def("get_response", &SpacemiT::TtsEngineResult::GetResponse,
            "Get full response (JSON string)")
        .def("get_request_id", &SpacemiT::TtsEngineResult::GetRequestId,
            "Get request ID")

        // 状态检查
        .def("is_success", &SpacemiT::TtsEngineResult::IsSuccess,
            "Check if synthesis succeeded")
        .def("get_code", &SpacemiT::TtsEngineResult::GetCode,
            "Get error code")
        .def("get_message", &SpacemiT::TtsEngineResult::GetMessage,
            "Get error message")
        .def("is_empty", &SpacemiT::TtsEngineResult::IsEmpty,
            "Check if result is empty (no audio)")
        .def("is_sentence_end", &SpacemiT::TtsEngineResult::IsSentenceEnd,
            "Check if this is the end of a sentence (streaming mode)")

        // 性能指标
        .def("get_sample_rate", &SpacemiT::TtsEngineResult::GetSampleRate,
            "Get sample rate (Hz)")
        .def("get_duration_ms", &SpacemiT::TtsEngineResult::GetDurationMs,
            "Get audio duration (milliseconds)")
        .def("get_processing_time_ms", &SpacemiT::TtsEngineResult::GetProcessingTimeMs,
            "Get processing time (milliseconds)")
        .def("get_rtf", &SpacemiT::TtsEngineResult::GetRTF,
            "Get Real-Time Factor (processing_time / audio_duration)")

        // 文件操作
        .def("save_to_file", &SpacemiT::TtsEngineResult::SaveToFile,
            py::arg("file_path"),
            "Save audio to file")

        // Python 魔术方法
        .def("__bool__", [](const SpacemiT::TtsEngineResult& r) {
            return !r.IsEmpty();
        }, "Check if result has audio content")
        .def("__repr__", [](const SpacemiT::TtsEngineResult& r) {
            return "<TtsEngineResult " +
                std::string(r.IsSuccess() ? "success" : "failed") +
                " duration=" + std::to_string(r.GetDurationMs()) + "ms" +
                " rtf=" + std::to_string(r.GetRTF()) + ">";
        });

    // =========================================================================
    // TtsResultCallback - 基类（先绑定基类）
    // =========================================================================

    py::class_<SpacemiT::TtsResultCallback, std::shared_ptr<SpacemiT::TtsResultCallback>>(
        m, "TtsResultCallbackBase", "Base callback interface for TTS")
        .def("OnOpen", &SpacemiT::TtsResultCallback::OnOpen)
        .def("OnEvent", &SpacemiT::TtsResultCallback::OnEvent)
        .def("OnComplete", &SpacemiT::TtsResultCallback::OnComplete)
        .def("OnError", &SpacemiT::TtsResultCallback::OnError)
        .def("OnClose", &SpacemiT::TtsResultCallback::OnClose);

    // =========================================================================
    // PyTtsCallback - 回调包装（派生类，声明继承关系）
    // =========================================================================

    py::class_<PyTtsCallback, SpacemiT::TtsResultCallback, std::shared_ptr<PyTtsCallback>>(
        m, "TtsCallback", "Callback for streaming synthesis")
        .def(py::init<>(), "Create callback object")

        // 设置回调函数
        .def("on_open", &PyTtsCallback::setOnOpen,
            py::arg("callback"),
            "Set callback for connection open")
        .def("on_event", &PyTtsCallback::setOnEvent,
            py::arg("callback"),
            "Set callback for audio events (main callback)")
        .def("on_complete", &PyTtsCallback::setOnComplete,
            py::arg("callback"),
            "Set callback for completion")
        .def("on_error", &PyTtsCallback::setOnError,
            py::arg("callback"),
            "Set callback for errors")
        .def("on_close", &PyTtsCallback::setOnClose,
            py::arg("callback"),
            "Set callback for connection close");

    // =========================================================================
    // TtsEngine - 主引擎
    // =========================================================================

    py::class_<SpacemiT::TtsEngine>(m, "TtsEngine", "TTS Engine - Main synthesis interface")
        // 构造函数
        .def(py::init<SpacemiT::BackendType, const std::string&>(),
            py::arg("backend") = SpacemiT::BackendType::MATCHA_ZH_EN,
            py::arg("model_dir") = "",
            "Create TTS engine with backend type")
        .def(py::init<const SpacemiT::TtsConfig&>(),
            py::arg("config"),
            "Create TTS engine with configuration")

        // 非流式调用（阻塞）- 释放 GIL
        .def("call", [](SpacemiT::TtsEngine& self, const std::string& text) {
            py::gil_scoped_release release;  // 释放 GIL，允许其他 Python 线程运行
            return self.Call(text);
        }, py::arg("text"),
            "Synthesize text (blocking, releases GIL)")

        .def("call_with_config", [](SpacemiT::TtsEngine& self,
            const std::string& text,
            const SpacemiT::TtsConfig& config) {
            py::gil_scoped_release release;
            return self.Call(text, config);
        }, py::arg("text"), py::arg("config"),
            "Synthesize text with custom config (blocking, releases GIL)")

        .def("call_to_file", [](SpacemiT::TtsEngine& self,
            const std::string& text,
            const std::string& file_path) {
            py::gil_scoped_release release;
            return self.CallToFile(text, file_path);
        }, py::arg("text"), py::arg("file_path"),
            "Synthesize text and save to file (blocking, releases GIL)")

        // 流式调用
        .def("streaming_call", &SpacemiT::TtsEngine::StreamingCall,
            py::arg("text"),
            py::arg("callback"),
            py::arg("config") = SpacemiT::TtsConfig(),
            "Streaming synthesis with callback")

        // 注意：DuplexStream 暂不绑定（需要额外的包装类）

        // 动态配置
        .def("set_speed", &SpacemiT::TtsEngine::SetSpeed,
            py::arg("speed"),
            "Set speech rate (>1.0 fast, <1.0 slow)")
        .def("set_speaker", &SpacemiT::TtsEngine::SetSpeaker,
            py::arg("speaker_id"),
            "Set speaker ID")
        .def("set_volume", &SpacemiT::TtsEngine::SetVolume,
            py::arg("volume"),
            "Set volume [0, 100]")
        .def("update_lexicon", &SpacemiT::TtsEngine::UpdateLexicon,
            py::arg("entries"),
            "Update custom pronunciation lexicon")
        .def("get_config", &SpacemiT::TtsEngine::GetConfig,
            "Get current configuration")
        .def("shutdown", [](SpacemiT::TtsEngine& self) {
            py::gil_scoped_release release;
            self.Shutdown();
        }, "Release native backend resources")

        // 辅助方法
        .def("is_initialized", &SpacemiT::TtsEngine::IsInitialized,
            "Check if engine is initialized")
        .def("get_engine_name", &SpacemiT::TtsEngine::GetEngineName,
            "Get engine type name")
        .def("get_backend_type", &SpacemiT::TtsEngine::GetBackendType,
            "Get backend type")
        .def("get_num_speakers", &SpacemiT::TtsEngine::GetNumSpeakers,
            "Get number of supported speakers")
        .def("get_sample_rate", &SpacemiT::TtsEngine::GetSampleRate,
            "Get output sample rate (Hz)")
        .def("get_last_request_id", &SpacemiT::TtsEngine::GetLastRequestId,
            "Get last request ID")

        // 字符串表示
        .def("__repr__", [](const SpacemiT::TtsEngine& engine) {
            return "<TtsEngine backend=" + engine.GetEngineName() +
                " sample_rate=" + std::to_string(engine.GetSampleRate()) + "Hz" +
                " initialized=" + (engine.IsInitialized() ? "true" : "false") + ">";
        });

    // =========================================================================
    // 模块级属性
    // =========================================================================

    m.attr("__version__") = "1.0.4";
    m.attr("__author__") = "muggle";
    m.attr("__doc__") = "SpaceTTS - Text-To-Speech Engine Python bindings";
}
