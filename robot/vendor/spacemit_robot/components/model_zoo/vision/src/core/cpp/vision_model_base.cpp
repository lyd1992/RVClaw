/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_model_base.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <filesystem>  // NOLINT(build/c++17)

namespace vision_core {

namespace fs = std::filesystem;

static bool is_url(const std::string& path) {
    return path.rfind("http://", 0) == 0 || path.rfind("https://", 0) == 0;
}

static fs::path get_vision_root_dir() {
#ifdef CV_MODELZOO_VISION_ROOT
    return fs::path(CV_MODELZOO_VISION_ROOT);
#else
    return fs::current_path();
#endif
}
static std::string download_script_hint() {
    const fs::path hint =
        get_vision_root_dir() / "examples" / "<model>" / "scripts" / "download_models.sh";
    return hint.lexically_normal().string();
}

static std::string expand_tilde(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home || home[0] == '\0') return path;
    if (path.size() == 1 || path[1] == '/')
        return std::string(home) + path.substr(1);
    return path;
}

static std::string resolve_and_check_model_path(const std::string& raw_path) {
    if (raw_path.empty()) {
        throw std::runtime_error("Model path is empty");
    }
    if (is_url(raw_path)) {
        std::ostringstream oss;
        oss << "Model URL is not supported: " << raw_path
            << ". Please download the model first, e.g. run: " << download_script_hint();
        throw std::runtime_error(oss.str());
    }

    std::string expanded = expand_tilde(raw_path);
    fs::path p(expanded);
    p = p.lexically_normal();

    fs::path candidate;
    if (p.is_absolute()) {
        candidate = p;
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    } else {
        candidate = fs::current_path() / p;
        candidate = candidate.lexically_normal();
        if (fs::exists(candidate)) {
            return candidate.string();
        }

        candidate = get_vision_root_dir() / p;
        candidate = candidate.lexically_normal();
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }

    {
        std::ostringstream oss;
        oss << "Model file not found: " << p.string()
            << ". Please download the model first, e.g. run: " << download_script_hint();
        throw std::runtime_error(oss.str());
    }
}

Ort::Env& shared_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VisionModel");
    return env;
}

BaseModel::BaseModel(const std::string& model_path, bool lazy_load)
    : model_path_(model_path), model_loaded_(false), lazy_load_(lazy_load) {
#ifdef DEBUG
    owner_thread_ = std::thread::id{};  // 初始化为空，第一次调用时设置
#endif

    if (!lazy_load) {
        (void)0;  // Subclass may load in constructor
    }
}

BaseModel::~BaseModel() {
    release();
}

void BaseModel::warmup() {
    ensure_model_loaded();
}

void BaseModel::release() {
    session_.reset();
    input_shape_.clear();
    input_node_names_.clear();
    output_node_names_.clear();
    input_names_.clear();
    output_names_.clear();
    output_num_ = 0;
    model_loaded_ = false;
}

std::vector<ModelCapability> BaseModel::get_capabilities() const {
    return {};
}

bool BaseModel::supports_capability(ModelCapability capability) const {
    const std::vector<ModelCapability> capabilities = get_capabilities();
    for (const auto& cap : capabilities) {
        if (cap == capability) {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> BaseModel::get_input_shape() const {
    return input_shape_;
}

std::string BaseModel::get_model_info() const {
    std::string info = "Model Path: " + model_path_ + "\n";
    info += "Input Shape: ";
    for (size_t i = 0; i < input_shape_.size(); ++i) {
        info += std::to_string(input_shape_[i]);
        if (i < input_shape_.size() - 1) {
            info += ", ";
        }
    }
    return info;
}

RuntimeProfile BaseModel::get_runtime_profile() const {
    return runtime_profile_;
}

void BaseModel::reset_runtime_profile() {
    runtime_profile_ = RuntimeProfile{};
}

void BaseModel::set_runtime_preprocess_ms(double ms) {
    runtime_profile_.preprocess_ms = ms;
}

void BaseModel::set_runtime_model_infer_ms(double ms) {
    runtime_profile_.model_infer_ms = ms;
}

void BaseModel::set_runtime_postprocess_ms(double ms) {
    runtime_profile_.postprocess_ms = ms;
}

void BaseModel::set_runtime_detect_ms(double ms) {
    runtime_profile_.detect_ms = ms;
}

void BaseModel::set_runtime_track_ms(double ms) {
    runtime_profile_.track_ms = ms;
}

void BaseModel::set_runtime_total_ms(double ms) {
    runtime_profile_.total_ms = ms;
}

void BaseModel::ensure_model_loaded() {
    if (!model_loaded_) {
        load_model();
        model_loaded_ = true;
    }
}

void BaseModel::init_session(int num_threads, const std::string& provider) {
    // Idempotent: composite wrappers (e.g. ByteTrack) may call load_model() on an
    // eager-loaded sub-model; skip re-creating the ONNX session and re-acquiring EP cores.
    if (session_) {
        return;
    }

    model_path_ = resolve_and_check_model_path(model_path_);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (provider == "SpaceMITExecutionProvider") {
        Ort::Status status;
        // Keep det_10g behavior aligned with buffalo_l demo tuning.
        if (model_path_.find("det_10g") != std::string::npos) {
            std::unordered_map<std::string, std::string> provider_options;
            provider_options["SPACEMIT_EP_DISABLE_OP_TYPE_FILTER"] = "Transpose";
            status = Ort::SessionOptionsSpaceMITEnvInit(session_options, provider_options);
        } else {
            status = Ort::SessionOptionsSpaceMITEnvInit(session_options);
        }
        if (!status.IsOK()) {
            std::cerr << "SpaceMIT EP init failed: " << status.GetErrorMessage() << std::endl;
        }
    }

    session_ = std::make_unique<Ort::Session>(shared_ort_env(), model_path_.c_str(), session_options);
    if (provider == "SpaceMITExecutionProvider") {
        std::cout << "SpaceMIT EP initialized: " << model_path_ << std::endl;
    }

    size_t num_inputs = session_->GetInputCount();
    input_node_names_.resize(num_inputs);
    input_names_.resize(num_inputs, "");

    for (size_t i = 0; i < num_inputs; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator_);
        input_names_[i] = input_name.get();
        input_node_names_[i] = input_names_[i].c_str();
    }

    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = input_tensor_info.GetShape();

    size_t num_outputs = session_->GetOutputCount();
    output_node_names_.resize(num_outputs);
    output_names_.resize(num_outputs, "");
    output_num_ = num_outputs;

    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_[i] = output_name.get();
        output_node_names_[i] = output_names_[i].c_str();
    }

    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

std::vector<Ort::Value> BaseModel::run_session(const cv::Mat& input_blob) {
    check_thread_safety("run_session");
    ensure_model_loaded();

    if (input_blob.empty()) {
        throw std::runtime_error("Input blob is empty");
    }

    std::vector<int64_t> tensor_shape = input_shape_;

    if (!tensor_shape.empty() && tensor_shape[0] <= 0) {
        tensor_shape[0] = 1;
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_blob.ptr<float>()),
        input_blob.total(),
        tensor_shape.data(),
        tensor_shape.size());

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor,
        1,
        output_node_names_.data(),
        output_node_names_.size());
    const auto t1 = std::chrono::steady_clock::now();
    runtime_profile_.model_infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return outputs;
}

#ifdef DEBUG
void BaseModel::check_thread_safety(const char* method_name) const {
    std::thread::id current_thread = std::this_thread::get_id();

    if (owner_thread_ == std::thread::id{}) {
        // 第一次调用，记录线程 ID
        owner_thread_ = current_thread;
    } else if (owner_thread_ != current_thread) {
        // 检测到线程安全违规
        std::ostringstream oss;
        oss << "\n"
            << "========================================\n"
            << "THREAD SAFETY VIOLATION DETECTED!\n"
            << "========================================\n"
            << "Method: BaseModel::" << method_name << "()\n"
            << "Expected thread: " << owner_thread_ << "\n"
            << "Actual thread:   " << current_thread << "\n"
            << "\n"
            << "This model instance is being accessed from multiple threads.\n"
            << "ONNX Runtime Session::Run() is NOT thread-safe.\n"
            << "\n"
            << "Solutions:\n"
            << "1. Create separate model instances for each thread (RECOMMENDED)\n"
            << "2. Use a thread pool with one model per worker\n"
            << "3. Protect with mutex (NOT recommended)\n"
            << "\n"
            << "See docs/THREAD_SAFETY.md for examples.\n"
            << "========================================\n";

        std::cerr << oss.str() << std::endl;
        throw std::runtime_error("Thread safety violation in " + std::string(method_name));
    }
}
#endif

size_t BaseModel::expected_sequence_size() const {
    return 0;
}

std::vector<std::string> BaseModel::get_sequence_class_names() const {
    return {};
}

std::vector<std::string> BaseModel::get_dynamic_class_names() const {
    return {};
}

}  // namespace vision_core
