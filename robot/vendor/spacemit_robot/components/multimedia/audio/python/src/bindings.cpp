/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <string>

#include "audio_base.hpp"

namespace py = pybind11;

// ============================================================================
// Module-level functions
// ============================================================================

void py_init(int sample_rate, int channels, int chunk_size,
            int capture_device, int player_device) {
    SpacemitAudio::Init(sample_rate, channels, chunk_size,
                    capture_device, player_device);
}

py::dict py_get_config() {
    SpacemitAudio::AudioConfig cfg = SpacemitAudio::GetConfig();
    py::dict result;
    result["sample_rate"] = cfg.sample_rate;
    result["channels"] = cfg.channels;
    result["chunk_size"] = cfg.chunk_size;
    result["capture_device"] = cfg.capture_device;
    result["player_device"] = cfg.player_device;
    return result;
}

// ============================================================================
// AudioCapture 绑定
// ============================================================================
class PyAudioCapture {
public:
    explicit PyAudioCapture(int device_index = -1)
        : capture_(device_index) {}

    void set_callback(py::function callback) {
        py_callback_ = callback;

        capture_.SetCallback([this](const uint8_t* data, size_t size){
            // 回调在音频线程中执行，需要使用gil锁
            py::gil_scoped_acquire acquire;
            py::bytes py_data(reinterpret_cast<const char*>(data), size);

            try {
                py_callback_(py_data);
            } catch (py::error_already_set& e) {
                // 异常处理
                e.restore();
                PyErr_Print();
            }
        });
    }

    bool start(int sample_rate = -1, int channels = -1, int chunk_size = -1) {
        py::gil_scoped_release release;  // 释放 GIL，避免阻塞
        return capture_.Start(sample_rate, channels, chunk_size);
    }

    void stop() {
        py::gil_scoped_release release;
        capture_.Stop();
    }

    void close() {
        py::gil_scoped_release release;
        capture_.Close();
    }

    bool is_running() const {
        return capture_.IsRunning();
    }

    static py::list list_devices() {
        auto devices = SpacemitAudio::AudioCapture::ListDevices();
        py::list result;
        for (const auto& [idx, name] : devices) {
            result.append(py::make_tuple(idx, name));
        }
        return result;
    }

private:
    SpacemitAudio::AudioCapture capture_;
    py::function py_callback_;
};

// ============================================================================
// AudioPlayer 绑定
// ============================================================================
class PyAudioPlayer {
public:
    explicit PyAudioPlayer(int device_index = -1)
        : player_(device_index) {}

    bool start(int sample_rate = -1, int channels = -1,
            int frames_per_buffer = -1) {
        py::gil_scoped_release release;
        return player_.Start(sample_rate, channels, frames_per_buffer);
    }

    bool write(py::bytes data) {
        std::string str = data;  // py::bytes -> std::string
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(str.data());

        py::gil_scoped_release release;
        return player_.Write(ptr, str.size());
    }

    bool play_file(const std::string& file_path) {
        py::gil_scoped_release release;
        return player_.PlayFile(file_path);
    }

    void stop() {
        py::gil_scoped_release release;
        player_.Stop();
    }

    void close() {
        py::gil_scoped_release release;
        player_.Close();
    }

    bool is_running() const {
        return player_.IsRunning();
    }

    static py::list list_devices() {
        auto devices = SpacemitAudio::AudioPlayer::ListDevices();
        py::list result;
        for (const auto& [idx, name] : devices) {
            result.append(py::make_tuple(idx, name));
        }
        return result;
    }

private:
    SpacemitAudio::AudioPlayer player_;
};

// ============================================================================
// 模块定义
// ============================================================================
PYBIND11_MODULE(_spacemit_audio, m) {
    m.doc() = "SpacemitAudio Python bindings";

    // Module-level functions
    m.def("init", &py_init,
            py::arg("sample_rate") = -1,
            py::arg("channels") = -1,
            py::arg("chunk_size") = -1,
            py::arg("capture_device") = -1,
            py::arg("player_device") = -1,
            "Initialize global audio configuration");

    m.def("get_config", &py_get_config,
            "Get current global configuration");

    // AudioCapture
    py::class_<PyAudioCapture>(m, "AudioCapture")
        .def(py::init<int>(), py::arg("device_index") = -1)
        .def("set_callback", &PyAudioCapture::set_callback, py::arg("callback"))
        .def("start", &PyAudioCapture::start,
                py::arg("sample_rate") = -1,
                py::arg("channels") = -1,
                py::arg("chunk_size") = -1)
        .def("stop", &PyAudioCapture::stop)
        .def("close", &PyAudioCapture::close)
        .def("is_running", &PyAudioCapture::is_running)
        .def_static("list_devices", &PyAudioCapture::list_devices)
        // Context Manager
        .def("__enter__", [](PyAudioCapture& self) -> PyAudioCapture& { return self; })
        .def("__exit__", [](PyAudioCapture& self, py::object, py::object, py::object) {
            self.close();
        });

    // AudioPlayer
    py::class_<PyAudioPlayer>(m, "AudioPlayer")
        .def(py::init<int>(), py::arg("device_index") = -1)
        .def("start", &PyAudioPlayer::start,
                py::arg("sample_rate") = -1,
                py::arg("channels") = -1,
                py::arg("frames_per_buffer") = -1)
        .def("write", &PyAudioPlayer::write, py::arg("data"))
        .def("play_file", &PyAudioPlayer::play_file, py::arg("file_path"))
        .def("stop", &PyAudioPlayer::stop)
        .def("close", &PyAudioPlayer::close)
        .def("is_running", &PyAudioPlayer::is_running)
        .def_static("list_devices", &PyAudioPlayer::list_devices)
        // Context Manager
        .def("__enter__", [](PyAudioPlayer& self) -> PyAudioPlayer& { return self; })
        .def("__exit__", [](PyAudioPlayer& self, py::object, py::object, py::object) {
            self.close();
        });
}
