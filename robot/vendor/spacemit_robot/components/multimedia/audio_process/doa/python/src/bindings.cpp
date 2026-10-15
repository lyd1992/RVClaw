/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * pybind11 bindings for SoundLocator
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "doa_service.h"

namespace py = pybind11;

using SpacemitAudio::MicrophonePosition;
using SpacemitAudio::MultiSoundLocator;
using SpacemitAudio::MultiSoundLocatorConfig;
using SpacemitAudio::MultiSoundLocatorResult;
using SpacemitAudio::SoundLocator;
using SpacemitAudio::SoundLocatorConfig;

namespace {

std::vector<MicrophonePosition> ParseMicrophones(py::iterable items) {
    std::vector<MicrophonePosition> microphones;
    for (py::handle item : items) {
        try {
            microphones.push_back(item.cast<MicrophonePosition>());
            continue;
        } catch (const py::cast_error&) {
        }

        py::sequence seq = py::reinterpret_borrow<py::sequence>(item);
        if (py::len(seq) < 2) {
            throw std::runtime_error(
                "microphones entries must be MicrophonePosition or (x, y[, z])");
        }
        MicrophonePosition mic;
        mic.x = seq[0].cast<float>();
        mic.y = seq[1].cast<float>();
        mic.z = py::len(seq) >= 3 ? seq[2].cast<float>() : 0.0f;
        microphones.push_back(mic);
    }
    return microphones;
}

}  // namespace

// Wrapper to accept numpy arrays
class PySoundLocator {
public:
    explicit PySoundLocator(const SoundLocatorConfig& config = {})
        : loc_(config) {}

    bool initialize() { return loc_.Initialize(); }
    void reset() { loc_.Reset(); }

    // Interleaved float32 numpy array [L, R, L, R, ...]
    bool process_float(
        py::array_t<float, py::array::c_style | py::array::forcecast> interleaved) {
        auto buf = interleaved.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected 1-D array for interleaved data");
        }
        size_t total = static_cast<size_t>(buf.shape[0]);
        if (total % 2 != 0) {
            throw std::runtime_error("Interleaved array length must be even");
        }
        const auto* ptr = static_cast<const float*>(buf.ptr);
        py::gil_scoped_release release;
        return loc_.Process(ptr, total / 2);
    }

    // Interleaved int16 numpy array
    bool process_int16(
        py::array_t<int16_t, py::array::c_style | py::array::forcecast> interleaved) {
        auto buf = interleaved.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Expected 1-D array for interleaved data");
        }
        size_t total = static_cast<size_t>(buf.shape[0]);
        if (total % 2 != 0) {
            throw std::runtime_error("Interleaved array length must be even");
        }
        const auto* ptr = static_cast<const int16_t*>(buf.ptr);
        py::gil_scoped_release release;
        return loc_.Process(ptr, total / 2);
    }

    // Two separate float32 numpy arrays
    bool process_separate(
        py::array_t<float, py::array::c_style | py::array::forcecast> ch0,
        py::array_t<float, py::array::c_style | py::array::forcecast> ch1) {
        auto b0 = ch0.request();
        auto b1 = ch1.request();
        if (b0.ndim != 1 || b1.ndim != 1) {
            throw std::runtime_error("Expected 1-D arrays");
        }
        if (b0.shape[0] != b1.shape[0]) {
            throw std::runtime_error("ch0 and ch1 must have same length");
        }
        size_t n = static_cast<size_t>(b0.shape[0]);
        const auto* p0 = static_cast<const float*>(b0.ptr);
        const auto* p1 = static_cast<const float*>(b1.ptr);
        py::gil_scoped_release release;
        return loc_.Process(p0, p1, n);
    }

    float get_tdoa() const { return loc_.GetTDOA(); }
    float get_doa() const { return loc_.GetDOA(); }
    float get_confidence() const { return loc_.GetConfidence(); }
    bool is_valid() const { return loc_.IsValid(); }
    float get_average_doa() const { return loc_.GetAverageDOA(); }
    int get_result_count() const { return loc_.GetResultCount(); }
    SoundLocatorConfig get_config() const { return loc_.GetConfig(); }
    int get_max_delay_samples() const { return loc_.GetMaxDelaySamples(); }

    PySoundLocator& enter() { return *this; }
    void exit(py::object, py::object, py::object) { reset(); }

private:
    SoundLocator loc_;
};

class PyMultiSoundLocator {
public:
    explicit PyMultiSoundLocator(const MultiSoundLocatorConfig& config = {})
        : loc_(config) {}

    bool initialize() { return loc_.Initialize(); }
    void reset() { loc_.Reset(); }

    bool process_float(
        py::array_t<float, py::array::c_style | py::array::forcecast> samples,
        int n_channels = 0) {
        auto buf = samples.request();
        const auto* ptr = static_cast<const float*>(buf.ptr);
        size_t n_frames = 0;
        int channels = n_channels;

        if (buf.ndim == 2) {
            n_frames = static_cast<size_t>(buf.shape[0]);
            const int array_channels = static_cast<int>(buf.shape[1]);
            if (channels == 0) {
                channels = array_channels;
            } else if (channels != array_channels) {
                throw std::runtime_error(
                    "n_channels does not match the second ndarray dimension");
            }
        } else if (buf.ndim == 1) {
            if (channels <= 0) {
                throw std::runtime_error(
                    "n_channels is required for 1-D interleaved arrays");
            }
            const size_t total = static_cast<size_t>(buf.shape[0]);
            if (total % static_cast<size_t>(channels) != 0) {
                throw std::runtime_error(
                    "interleaved array length must be divisible by n_channels");
            }
            n_frames = total / static_cast<size_t>(channels);
        } else {
            throw std::runtime_error(
                "expected a 1-D interleaved array or a 2-D (frames, channels) array");
        }

        py::gil_scoped_release release;
        return loc_.Process(ptr, n_frames, channels);
    }

    bool process_int16(
        py::array_t<int16_t, py::array::c_style | py::array::forcecast> samples,
        int n_channels = 0) {
        auto buf = samples.request();
        const auto* ptr = static_cast<const int16_t*>(buf.ptr);
        size_t n_frames = 0;
        int channels = n_channels;

        if (buf.ndim == 2) {
            n_frames = static_cast<size_t>(buf.shape[0]);
            const int array_channels = static_cast<int>(buf.shape[1]);
            if (channels == 0) {
                channels = array_channels;
            } else if (channels != array_channels) {
                throw std::runtime_error(
                    "n_channels does not match the second ndarray dimension");
            }
        } else if (buf.ndim == 1) {
            if (channels <= 0) {
                throw std::runtime_error(
                    "n_channels is required for 1-D interleaved arrays");
            }
            const size_t total = static_cast<size_t>(buf.shape[0]);
            if (total % static_cast<size_t>(channels) != 0) {
                throw std::runtime_error(
                    "interleaved array length must be divisible by n_channels");
            }
            n_frames = total / static_cast<size_t>(channels);
        } else {
            throw std::runtime_error(
                "expected a 1-D interleaved array or a 2-D (frames, channels) array");
        }

        py::gil_scoped_release release;
        return loc_.Process(ptr, n_frames, channels);
    }

    bool process_channels(py::sequence channels) {
        const int n_channels = static_cast<int>(py::len(channels));
        if (n_channels <= 0) {
            throw std::runtime_error("expected at least one channel array");
        }

        channel_arrays_.clear();
        channel_arrays_.reserve(n_channels);
        channel_ptrs_.clear();
        channel_ptrs_.reserve(n_channels);

        size_t n_frames = 0;
        for (int ch = 0; ch < n_channels; ++ch) {
            py::object item = channels[ch];
            auto array = py::array_t<
                float, py::array::c_style | py::array::forcecast>::ensure(
                    item);
            if (!array) {
                throw std::runtime_error("channel entries must be float arrays");
            }
            channel_arrays_.push_back(array);
            auto buf = channel_arrays_.back().request();
            if (buf.ndim != 1) {
                throw std::runtime_error("channel arrays must be 1-D");
            }
            if (ch == 0) {
                n_frames = static_cast<size_t>(buf.shape[0]);
            } else if (n_frames != static_cast<size_t>(buf.shape[0])) {
                throw std::runtime_error("all channel arrays must have same length");
            }
            channel_ptrs_.push_back(static_cast<const float*>(buf.ptr));
        }

        py::gil_scoped_release release;
        return loc_.Process(channel_ptrs_.data(), n_frames, n_channels);
    }

    MultiSoundLocatorResult get_result() const { return loc_.GetResult(); }
    float get_azimuth() const { return loc_.GetAzimuth(); }
    float get_confidence() const { return loc_.GetConfidence(); }
    bool is_valid() const { return loc_.IsValid(); }
    float get_average_azimuth() const { return loc_.GetAverageAzimuth(); }
    int get_result_count() const { return loc_.GetResultCount(); }
    float get_tdoa(int i, int j) const { return loc_.GetTDOA(i, j); }
    int get_max_delay_samples_pair(int i, int j) const {
        return loc_.GetMaxDelaySamplesPair(i, j);
    }
    MultiSoundLocatorConfig get_config() const { return loc_.GetConfig(); }

    // [A1] TDOA closure (N=3 only; 0 otherwise)
    float get_closure_residual() const { return loc_.GetClosureResidual(); }
    float get_closure_samples() const { return loc_.GetClosureSamples(); }
    // [A5] clamped quality, [0, 1]
    float get_quality() const { return loc_.GetQuality(); }
    // [A6] Mardia mean resultant length, [0, 1]
    float get_average_resultant_length() const {
        return loc_.GetAverageResultantLength();
    }
    // [A3] per-pair GCC peak confidences
    int get_pair_count() const { return loc_.GetPairCount(); }
    py::array_t<float> get_pair_confidences() const {
        const int n = loc_.GetPairCount();
        py::array_t<float> arr(static_cast<py::ssize_t>(n));
        if (n > 0) {
            auto r = arr.mutable_unchecked<1>();
            std::vector<float> buf(static_cast<size_t>(n));
            loc_.GetPairConfidences(buf.data(), n);
            for (int i = 0; i < n; ++i) r(i) = buf[static_cast<size_t>(i)];
        }
        return arr;
    }

    PyMultiSoundLocator& enter() { return *this; }
    void exit(py::object, py::object, py::object) { reset(); }

private:
    MultiSoundLocator loc_;
    std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>>
        channel_arrays_;
    std::vector<const float*> channel_ptrs_;
};

PYBIND11_MODULE(_spacemit_audio_process, m) {
    m.doc() = "Sound source localization (GCC-PHAT)";

    py::class_<SoundLocatorConfig>(m, "SoundLocatorConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &SoundLocatorConfig::sample_rate)
        .def_readwrite("mic_distance", &SoundLocatorConfig::mic_distance)
        .def_readwrite("sound_speed", &SoundLocatorConfig::sound_speed)
        .def_readwrite("fft_size", &SoundLocatorConfig::fft_size)
        .def_readwrite("frame_size", &SoundLocatorConfig::frame_size)
        .def_readwrite("avg_frames", &SoundLocatorConfig::avg_frames)
        .def_readwrite("confidence_threshold",
            &SoundLocatorConfig::confidence_threshold)
        .def_readwrite("upsample_factor",
            &SoundLocatorConfig::upsample_factor)
        .def_readwrite("use_fftw_measure",
            &SoundLocatorConfig::use_fftw_measure)
        .def("__repr__", [](const SoundLocatorConfig& c) {
            return "<SoundLocatorConfig sr=" + std::to_string(c.sample_rate)
                + " mic_dist=" + std::to_string(c.mic_distance)
                + " frame=" + std::to_string(c.frame_size)
                + " avg=" + std::to_string(c.avg_frames) + ">";
        });

    py::class_<PySoundLocator>(m, "SoundLocator")
        .def(py::init<const SoundLocatorConfig&>(),
            py::arg("config") = SoundLocatorConfig{})
        .def("initialize", &PySoundLocator::initialize)
        .def("reset", &PySoundLocator::reset)
        .def("process", &PySoundLocator::process_float,
            py::arg("interleaved"),
            "Process interleaved float32 stereo data")
        .def("process_int16", &PySoundLocator::process_int16,
            py::arg("interleaved"),
            "Process interleaved int16 stereo data")
        .def("process_separate", &PySoundLocator::process_separate,
            py::arg("ch0"), py::arg("ch1"),
            "Process two separate float32 channel arrays")
        .def_property_readonly("tdoa", &PySoundLocator::get_tdoa)
        .def_property_readonly("doa", &PySoundLocator::get_doa)
        .def_property_readonly("confidence", &PySoundLocator::get_confidence)
        .def_property_readonly("is_valid", &PySoundLocator::is_valid)
        .def_property_readonly("average_doa", &PySoundLocator::get_average_doa)
        .def_property_readonly("result_count", &PySoundLocator::get_result_count)
        .def_property_readonly("config", &PySoundLocator::get_config)
        .def_property_readonly("max_delay_samples",
            &PySoundLocator::get_max_delay_samples)
        .def("__enter__", &PySoundLocator::enter,
            py::return_value_policy::reference)
        .def("__exit__", &PySoundLocator::exit);

    py::class_<MicrophonePosition>(m, "MicrophonePosition")
        .def(py::init([](float x, float y, float z) {
            return MicrophonePosition{x, y, z};
        }), py::arg("x"), py::arg("y"), py::arg("z") = 0.0f)
        .def_readwrite("x", &MicrophonePosition::x)
        .def_readwrite("y", &MicrophonePosition::y)
        .def_readwrite("z", &MicrophonePosition::z)
        .def("__repr__", [](const MicrophonePosition& p) {
            return "<MicrophonePosition x=" + std::to_string(p.x)
                + " y=" + std::to_string(p.y)
                + " z=" + std::to_string(p.z) + ">";
        });

    py::class_<MultiSoundLocatorConfig>(m, "MultiSoundLocatorConfig")
        .def(py::init<>())
        .def_readwrite("sample_rate", &MultiSoundLocatorConfig::sample_rate)
        .def_readwrite("frame_size", &MultiSoundLocatorConfig::frame_size)
        .def_readwrite("avg_frames", &MultiSoundLocatorConfig::avg_frames)
        .def_readwrite("fft_size", &MultiSoundLocatorConfig::fft_size)
        .def_readwrite("upsample_factor", &MultiSoundLocatorConfig::upsample_factor)
        .def_readwrite("confidence_threshold",
            &MultiSoundLocatorConfig::confidence_threshold)
        .def_readwrite("sound_speed", &MultiSoundLocatorConfig::sound_speed)
        .def_readwrite("use_fftw_measure",
            &MultiSoundLocatorConfig::use_fftw_measure)
        .def_property("microphones",
            [](const MultiSoundLocatorConfig& c) { return c.microphones; },
            [](MultiSoundLocatorConfig& c, py::iterable items) {
                c.microphones = ParseMicrophones(items);
            })
        .def_readwrite("azimuth_offset_deg",
            &MultiSoundLocatorConfig::azimuth_offset_deg)
        .def_readwrite("max_frequency_hz",
            &MultiSoundLocatorConfig::max_frequency_hz)
        .def_readwrite("search_step_deg",
            &MultiSoundLocatorConfig::search_step_deg)
        .def_readwrite("margin_threshold",
            &MultiSoundLocatorConfig::margin_threshold)
        .def_readwrite("min_signal_rms",
            &MultiSoundLocatorConfig::min_signal_rms)
        .def_readwrite("max_avg_seconds",
            &MultiSoundLocatorConfig::max_avg_seconds)
        // [A5] quality threshold (default 0.0 = disabled; v1 backward-compat)
        .def_readwrite("quality_threshold",
            &MultiSoundLocatorConfig::quality_threshold)
        // [A2/P1.1] TDOA closure threshold (N=3 only). Effective threshold =
        // max(closure_threshold_samples, closure_threshold_fraction *
        // max_physical_TDOA_samples). P1.1 default samples=0.0, fraction=0.3
        // is array-scale invariant; set both <= 0 to disable the gate.
        .def_readwrite("closure_threshold_samples",
            &MultiSoundLocatorConfig::closure_threshold_samples)
        .def_readwrite("closure_threshold_fraction",
            &MultiSoundLocatorConfig::closure_threshold_fraction)
        .def("__repr__", [](const MultiSoundLocatorConfig& c) {
            return "<MultiSoundLocatorConfig sr=" + std::to_string(c.sample_rate)
                + " channels=" + std::to_string(c.microphones.size())
                + " frame=" + std::to_string(c.frame_size)
                + " avg=" + std::to_string(c.avg_frames) + ">";
        });

    py::class_<MultiSoundLocatorResult>(m, "MultiSoundLocatorResult")
        .def(py::init<>())
        .def_readwrite("azimuth_deg", &MultiSoundLocatorResult::azimuth_deg)
        .def_readwrite("confidence", &MultiSoundLocatorResult::confidence)
        .def_readwrite("peak_score", &MultiSoundLocatorResult::peak_score)
        .def_readwrite("score_margin", &MultiSoundLocatorResult::score_margin)
        // [A5] clamped consistency in [0, 1]
        .def_readwrite("quality", &MultiSoundLocatorResult::quality)
        // [A1] |τ_01+τ_12-τ_02| in seconds (N=3 only; 0 otherwise)
        .def_readwrite("closure_residual_sec",
            &MultiSoundLocatorResult::closure_residual_sec)
        .def_readwrite("valid_pairs", &MultiSoundLocatorResult::valid_pairs)
        .def_readwrite("valid", &MultiSoundLocatorResult::valid)
        .def("__repr__", [](const MultiSoundLocatorResult& r) {
            return "<MultiSoundLocatorResult azimuth="
                + std::to_string(r.azimuth_deg)
                + " confidence=" + std::to_string(r.confidence)
                + " valid=" + std::to_string(r.valid) + ">";
        });

    py::class_<PyMultiSoundLocator>(m, "MultiSoundLocator")
        .def(py::init<const MultiSoundLocatorConfig&>(),
            py::arg("config") = MultiSoundLocatorConfig{})
        .def_static("create_equilateral_triangle_config",
            &MultiSoundLocator::CreateEquilateralTriangleConfig,
            py::arg("side_length_m") = 0.063f)
        .def("initialize", &PyMultiSoundLocator::initialize)
        .def("reset", &PyMultiSoundLocator::reset)
        .def("process", &PyMultiSoundLocator::process_float,
            py::arg("samples"), py::arg("n_channels") = 0,
            "Process float32 data as (frames, channels) or 1-D interleaved")
        .def("process_int16", &PyMultiSoundLocator::process_int16,
            py::arg("samples"), py::arg("n_channels") = 0,
            "Process int16 data as (frames, channels) or 1-D interleaved")
        .def("process_channels", &PyMultiSoundLocator::process_channels,
            py::arg("channels"),
            "Process a sequence of separate float32 channel arrays")
        .def_property_readonly("result", &PyMultiSoundLocator::get_result)
        .def_property_readonly("azimuth", &PyMultiSoundLocator::get_azimuth)
        .def_property_readonly("confidence", &PyMultiSoundLocator::get_confidence)
        .def_property_readonly("is_valid", &PyMultiSoundLocator::is_valid)
        .def_property_readonly("average_azimuth",
            &PyMultiSoundLocator::get_average_azimuth)
        .def_property_readonly("result_count", &PyMultiSoundLocator::get_result_count)
        .def_property_readonly("config", &PyMultiSoundLocator::get_config)
        // [A1] TDOA closure observability
        .def_property_readonly("closure_residual",
            &PyMultiSoundLocator::get_closure_residual)
        .def_property_readonly("closure_samples",
            &PyMultiSoundLocator::get_closure_samples)
        // [A3] per-pair confidences
        .def_property_readonly("pair_count",
            &PyMultiSoundLocator::get_pair_count)
        .def_property_readonly("pair_confidences",
            &PyMultiSoundLocator::get_pair_confidences)
        // [A5] clamped consistency
        .def_property_readonly("quality", &PyMultiSoundLocator::get_quality)
        // [A6] Mardia mean resultant length
        .def_property_readonly("average_resultant_length",
            &PyMultiSoundLocator::get_average_resultant_length)
        .def("get_tdoa", &PyMultiSoundLocator::get_tdoa,
            py::arg("i"), py::arg("j"))
        .def("get_max_delay_samples_pair",
            &PyMultiSoundLocator::get_max_delay_samples_pair,
            py::arg("i"), py::arg("j"))
        .def("__enter__", &PyMultiSoundLocator::enter,
            py::return_value_policy::reference)
        .def("__exit__", &PyMultiSoundLocator::exit);
}
