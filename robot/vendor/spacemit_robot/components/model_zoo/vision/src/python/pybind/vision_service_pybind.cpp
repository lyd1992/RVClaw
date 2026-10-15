/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_service.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// Legacy flat result type for backward-compatible Python API (x1/y1/x2/y2/...).
struct PyFlatResult {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int label = -1;
    int track_id = -1;
    std::vector<vision::KeyPoint> keypoints;
    cv::Mat mask;
    std::vector<float> class_scores;
    std::string text;                          // OCR: recognized string
    std::vector<vision::KeyPoint> polygon;     // OCR: text-box quadrilateral corners
};

PyFlatResult ToFlatResult(const vision::Result& r) {
    PyFlatResult out{};
    out.score = vision::get_score(r);
    out.label = vision::get_label(r);
    out.track_id = vision::get_track_id(r);
    const vision::BoundingBox bbox = vision::get_bbox(r);
    out.x1 = bbox.x1;
    out.y1 = bbox.y1;
    out.x2 = bbox.x2;
    out.y2 = bbox.y2;

    if (const vision::Pose* p = std::get_if<vision::Pose>(&r)) {
        out.keypoints = p->keypoints;
    } else if (const vision::Segmentation* s = std::get_if<vision::Segmentation>(&r)) {
        if (s->mask && !s->mask->empty()) {
            out.mask = s->mask->clone();
        }
    } else if (const vision::Classification* c = std::get_if<vision::Classification>(&r)) {
        out.class_scores = c->class_scores;
    } else if (const vision::Action* a = std::get_if<vision::Action>(&r)) {
        out.class_scores = a->class_scores;
    } else if (const vision::Text* t = std::get_if<vision::Text>(&r)) {
        out.text = t->text;
        out.polygon = t->polygon;
    }
    return out;
}

std::vector<PyFlatResult> FlattenResults(const vision::ResultList& results) {
    std::vector<PyFlatResult> flat;
    flat.reserve(results.size());
    for (const vision::Result& r : results) {
        flat.push_back(ToFlatResult(r));
    }
    return flat;
}

VisionServiceInferParams MakeParams(
    float conf,
    float iou,
    int top_k = -1,
    float kp_threshold = -1.0f,
    float mask_threshold = -1.0f,
    int max_det = -1) {
    VisionServiceInferParams params{};
    params.conf_threshold = conf;
    params.iou_threshold = iou;
    params.top_k = top_k;
    params.kp_threshold = kp_threshold;
    params.mask_threshold = mask_threshold;
    params.max_det = max_det;
    return params;
}

cv::Mat NumpyToMatBgr(const py::array& arr) {
    py::buffer_info info = arr.request();
    if (info.ndim != 3) {
        throw std::invalid_argument("image must be a 3-D array HxWx3 (BGR uint8)");
    }
    if (info.shape[2] != 3) {
        throw std::invalid_argument("image must have 3 channels (BGR)");
    }
    if (info.itemsize != 1 || info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::invalid_argument("image dtype must be uint8");
    }
    const ssize_t row_stride = info.strides[0];
    const ssize_t col_stride = info.strides[1];
    if (col_stride != 3 || row_stride != info.shape[1] * 3) {
        throw std::invalid_argument("image must be C-contiguous HxWx3 (use numpy.ascontiguousarray)");
    }
    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    return cv::Mat(rows, cols, CV_8UC3, info.ptr);
}

py::array_t<uint8_t> MatToNumpyBGR(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array_t<uint8_t>();
    }
    if (mat.type() != CV_8UC3) {
        throw std::runtime_error("internal: expected BGR uint8 output");
    }
    const ssize_t h = mat.rows;
    const ssize_t w = mat.cols;
    const ssize_t c = 3;
    py::array_t<uint8_t> out({h, w, c});
    py::buffer_info dst = out.request();
    cv::Mat dst_wrapped(static_cast<int>(h), static_cast<int>(w), CV_8UC3, dst.ptr);
    mat.copyTo(dst_wrapped);
    return out;
}

py::array MatToNumpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array();
    }
    cv::Mat contiguous = mat.isContinuous() ? mat : mat.clone();
    const int channels = contiguous.channels();
    if (channels == 1) {
        py::array_t<uint8_t> out({contiguous.rows, contiguous.cols});
        py::buffer_info dst = out.request();
        cv::Mat dst_wrapped(contiguous.rows, contiguous.cols, CV_8UC1, dst.ptr);
        contiguous.copyTo(dst_wrapped);
        return out;
    }
    if (channels == 3) {
        return MatToNumpyBGR(contiguous);
    }
    throw std::runtime_error("unsupported cv::Mat channels in VisionServiceResult.mask");
}

cv::Mat NumpyToMat(const py::array& obj) {
    py::buffer_info info = obj.request();
    if (info.itemsize != 1 || info.format != py::format_descriptor<uint8_t>::format()) {
        throw std::invalid_argument("mask dtype must be uint8");
    }
    if (info.ndim == 2) {
        const int rows = static_cast<int>(info.shape[0]);
        const int cols = static_cast<int>(info.shape[1]);
        return cv::Mat(rows, cols, CV_8UC1, info.ptr).clone();
    }
    if (info.ndim == 3 && info.shape[2] == 3) {
        return NumpyToMatBgr(obj).clone();
    }
    throw std::invalid_argument("mask must be HxW or HxWx3 uint8 numpy array");
}

std::vector<float> ExtractEmbedding(const VisionServiceResponse& response) {
    if (response.results.empty()) {
        return {};
    }
    if (const vision::Embedding* e = std::get_if<vision::Embedding>(&response.results.front())) {
        return e->embedding;
    }
    throw std::runtime_error("infer_embedding: model did not return an Embedding result");
}

std::vector<float> ExtractClassScores(const VisionServiceResponse& response) {
    if (response.results.empty()) {
        return {};
    }
    if (const vision::Action* a = std::get_if<vision::Action>(&response.results.front())) {
        return a->class_scores;
    }
    if (const vision::Classification* c = std::get_if<vision::Classification>(&response.results.front())) {
        return c->class_scores;
    }
    throw std::runtime_error("infer_sequence: model did not return an Action/Classification result");
}

}  // namespace

PYBIND11_MODULE(_vision_service_cpp, m) {
    m.doc() = "Bindings for C++ VisionService (libvision).";

    py::enum_<VisionServiceStatus>(m, "VisionServiceStatus")
        .value("OK", VISION_SERVICE_OK)
        .value("INVALID_ARGUMENT", VISION_SERVICE_INVALID_ARGUMENT)
        .value("CREATE_FAILED", VISION_SERVICE_CREATE_FAILED)
        .value("INFER_FAILED", VISION_SERVICE_INFER_FAILED)
        .value("IO_FAILED", VISION_SERVICE_IO_FAILED)
        .export_values();

    py::class_<vision::KeyPoint>(m, "VisionServiceKeypoint")
        .def(py::init<>())
        .def_readwrite("x", &vision::KeyPoint::x)
        .def_readwrite("y", &vision::KeyPoint::y)
        .def_readwrite("visibility", &vision::KeyPoint::visibility);

    py::class_<PyFlatResult>(m, "VisionServiceResult")
        .def_readwrite("x1", &PyFlatResult::x1)
        .def_readwrite("y1", &PyFlatResult::y1)
        .def_readwrite("x2", &PyFlatResult::x2)
        .def_readwrite("y2", &PyFlatResult::y2)
        .def_readwrite("score", &PyFlatResult::score)
        .def_readwrite("label", &PyFlatResult::label)
        .def_readwrite("track_id", &PyFlatResult::track_id)
        .def_readwrite("keypoints", &PyFlatResult::keypoints)
        .def_readwrite("class_scores", &PyFlatResult::class_scores)
        .def_readwrite("text", &PyFlatResult::text)
        .def_readwrite("polygon", &PyFlatResult::polygon)
        .def_property(
            "mask",
            [](const PyFlatResult& r) -> py::object {
                if (r.mask.empty()) {
                    return py::none();
                }
                return MatToNumpy(r.mask);
            },
            [](PyFlatResult& r, const py::object& obj) {
                if (obj.is_none()) {
                    r.mask = cv::Mat();
                    return;
                }
                r.mask = NumpyToMat(obj.cast<py::array>());
            })
        .def("__repr__", [](const PyFlatResult& r) {
            return "<VisionServiceResult x1=" + std::to_string(r.x1) + " y1=" + std::to_string(r.y1) +
                " x2=" + std::to_string(r.x2) + " y2=" + std::to_string(r.y2) +
                " score=" + std::to_string(r.score) + " label=" + std::to_string(r.label) +
                " track_id=" + std::to_string(r.track_id) +
                " keypoints=" + std::to_string(r.keypoints.size()) +
                " class_scores=" + std::to_string(r.class_scores.size()) +
                " has_mask=" + (r.mask.empty() ? "false" : "true") + ">";
        });

    py::class_<VisionServiceInferParams>(m, "VisionServiceInferParams")
        .def(py::init<>())
        .def_readwrite("conf_threshold", &VisionServiceInferParams::conf_threshold)
        .def_readwrite("iou_threshold", &VisionServiceInferParams::iou_threshold)
        .def_readwrite("top_k", &VisionServiceInferParams::top_k)
        .def_readwrite("kp_threshold", &VisionServiceInferParams::kp_threshold)
        .def_readwrite("mask_threshold", &VisionServiceInferParams::mask_threshold)
        .def_readwrite("max_det", &VisionServiceInferParams::max_det);

    py::class_<VisionServiceResponse>(m, "VisionServiceResponse")
        .def(py::init<>())
        .def_readonly("ok", &VisionServiceResponse::ok)
        .def_readonly("error_message", &VisionServiceResponse::error_message)
        .def_property_readonly(
            "flat_results",
            [](const VisionServiceResponse& r) { return FlattenResults(r.results); })
        .def_property_readonly(
            "results",
            [](const VisionServiceResponse& r) { return FlattenResults(r.results); });

    py::class_<VisionServiceTiming>(m, "VisionServiceTiming")
        .def_readwrite("preprocess_ms", &VisionServiceTiming::preprocess_ms)
        .def_readwrite("model_infer_ms", &VisionServiceTiming::model_infer_ms)
        .def_readwrite("postprocess_ms", &VisionServiceTiming::postprocess_ms)
        .def_readwrite("detect_ms", &VisionServiceTiming::detect_ms)
        .def_readwrite("track_ms", &VisionServiceTiming::track_ms)
        .def_readwrite("infer_ms", &VisionServiceTiming::infer_ms)
        .def_readwrite("draw_ms", &VisionServiceTiming::draw_ms)
        .def_readwrite("embedding_ms", &VisionServiceTiming::embedding_ms)
        .def_readwrite("sequence_ms", &VisionServiceTiming::sequence_ms);

    py::class_<VisionServiceTimingOptions>(m, "VisionServiceTimingOptions")
        .def(py::init<>())
        .def_readwrite("enabled", &VisionServiceTimingOptions::enabled)
        .def_readwrite("print_to_stdout", &VisionServiceTimingOptions::print_to_stdout);

    py::class_<VisionService>(m, "VisionService")
        .def_static(
            "create",
            [](const std::string& config_path, const std::string& model_path_override, bool lazy_load) {
                auto svc = VisionService::Create(config_path, model_path_override, lazy_load);
                if (!svc) {
                    throw py::value_error(VisionService::LastCreateError());
                }
                return svc;
            },
            py::arg("config_path"),
            py::arg("model_path_override") = std::string(),
            py::arg("lazy_load") = false)
        .def_static("last_create_error", []() { return VisionService::LastCreateError(); })
        .def(
            "infer",
            [](VisionService& self, const std::string& path, const VisionServiceInferParams& params) {
                VisionServiceResponse response;
                const VisionServiceStatus st = self.Infer(path, &response, params);
                return py::make_tuple(st, response);
            },
            py::arg("image_path"),
            py::arg("params") = VisionServiceInferParams{})
        .def(
            "infer",
            [](VisionService& self, const py::array& arr, const VisionServiceInferParams& params) {
                cv::Mat mat = NumpyToMatBgr(arr);
                VisionServiceResponse response;
                const VisionServiceStatus st = self.Infer(mat, &response, params);
                return py::make_tuple(st, response);
            },
            py::arg("image_bgr_uint8"),
            py::arg("params") = VisionServiceInferParams{})
        .def(
            "infer_image",
            [](VisionService& self, const std::string& path, float conf, float iou) {
                VisionServiceResponse response;
                const VisionServiceStatus st =
                    self.Infer(path, &response, MakeParams(conf, iou));
                return py::make_tuple(st, FlattenResults(response.results), response);
            },
            py::arg("image_path"),
            py::arg("conf") = -1.0f,
            py::arg("iou") = -1.0f)
        .def(
            "infer_image",
            [](VisionService& self, const py::array& arr, float conf, float iou) {
                cv::Mat mat = NumpyToMatBgr(arr);
                VisionServiceResponse response;
                const VisionServiceStatus st =
                    self.Infer(mat, &response, MakeParams(conf, iou));
                return py::make_tuple(st, FlattenResults(response.results), response);
            },
            py::arg("image_bgr_uint8"),
            py::arg("conf") = -1.0f,
            py::arg("iou") = -1.0f)
        .def(
            // Open-vocabulary variant (e.g. YOLO-World): pass text prompts.
            // Empty prompts -> use the model's configured default vocabulary.
            "infer_image_prompts",
            [](VisionService& self, const py::array& arr,
                const std::vector<std::string>& prompts, float conf, float iou) {
                cv::Mat mat = NumpyToMatBgr(arr);
                VisionServiceRequest request{};
                request.image = mat;
                request.prompts = prompts;
                request.params = MakeParams(conf, iou);
                VisionServiceResponse response;
                const VisionServiceStatus st = self.Infer(request, &response);
                return py::make_tuple(st, FlattenResults(response.results), response);
            },
            py::arg("image_bgr_uint8"),
            py::arg("prompts"),
            py::arg("conf") = -1.0f,
            py::arg("iou") = -1.0f)
        .def(
            "infer_embedding",
            [](VisionService& self, const std::string& path) {
                VisionServiceResponse response;
                const VisionServiceStatus st = self.Infer(path, &response, VisionServiceInferParams{});
                return py::make_tuple(st, ExtractEmbedding(response), response);
            },
            py::arg("image_path"))
        .def(
            "infer_embedding",
            [](VisionService& self, const py::array& arr) {
                cv::Mat mat = NumpyToMatBgr(arr);
                VisionServiceResponse response;
                const VisionServiceStatus st = self.Infer(mat, &response, VisionServiceInferParams{});
                return py::make_tuple(st, ExtractEmbedding(response), response);
            },
            py::arg("image_bgr_uint8"))
        .def(
            "encode_text",
            [](VisionService& self, const std::string& text) {
                std::vector<float> embedding;
                const VisionServiceStatus st = self.EncodeText(text, &embedding);
                return py::make_tuple(st, embedding);
            },
            py::arg("text"))
        .def_static(
            "embedding_similarity",
            [](const std::vector<float>& a, const std::vector<float>& b) {
                return VisionService::EmbeddingSimilarity(a, b);
            },
            py::arg("embedding_a"),
            py::arg("embedding_b"))
        .def(
            "infer_sequence",
            [](VisionService& self, py::array_t<float> pts, int image_width, int image_height) {
                py::buffer_info info = pts.request();
                if (info.ndim != 1) {
                    throw std::invalid_argument("pts must be a 1-D float32 array");
                }
                if (info.itemsize != static_cast<ssize_t>(sizeof(float))) {
                    throw std::invalid_argument("pts dtype must be float32");
                }
                if (info.strides[0] != static_cast<ssize_t>(sizeof(float))) {
                    throw std::invalid_argument("pts must be C-contiguous");
                }
                if (info.shape[0] == 0) {
                    throw std::invalid_argument("pts must not be empty");
                }
                VisionServiceRequest request{};
                request.sequence_pts = static_cast<const float*>(info.ptr);
                request.sequence_count = static_cast<int>(info.shape[0]);
                request.sequence_width = image_width;
                request.sequence_height = image_height;
                VisionServiceResponse response;
                const VisionServiceStatus st = self.Infer(request, &response);
                return py::make_tuple(st, ExtractClassScores(response), response);
            },
            py::arg("pts"),
            py::arg("image_width"),
            py::arg("image_height"))
        .def("get_class_names", &VisionService::GetClassNames)
        .def(
            "get_sequence_class_names",
            &VisionService::GetClassNames,
            "Deprecated alias for get_class_names().")
        .def(
            "get_fall_down_class_index",
            [](VisionService& self) {
                const std::string value = self.GetConfigPathValue("fall_down_index");
                if (value.empty()) {
                    return -1;
                }
                try {
                    return std::stoi(value);
                } catch (...) {
                    return -1;
                }
            },
            "Read fall_down_index from config if present; otherwise -1.")
        .def(
            "draw",
            [](VisionService& self, const py::array& arr, const VisionServiceResponse& response) {
                cv::Mat in = NumpyToMatBgr(arr);
                cv::Mat out;
                const VisionServiceStatus st = self.Draw(in, response, &out);
                return py::make_tuple(st, MatToNumpyBGR(out));
            },
            py::arg("image_bgr_uint8"),
            py::arg("response"))
        .def("supports_draw", &VisionService::SupportsDraw)
        .def("release", &VisionService::Release)
        .def("set_timing_options", &VisionService::SetTimingOptions)
        .def("get_last_timing", &VisionService::GetLastTiming)
        .def("get_default_image", &VisionService::GetDefaultImage)
        .def("get_config_path_value", &VisionService::GetConfigPathValue, py::arg("config_key"))
        .def("last_error", [](const VisionService& self) { return self.LastError(); });
}
