/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_MODEL_BASE_H
#define VISION_MODEL_BASE_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

#include "spacemit_ort_env.h"
#include "vision_infer_types.h"

#ifdef DEBUG
#include <thread>
#endif

namespace vision_core {

struct RuntimeProfile {
    double preprocess_ms = 0.0;
    double model_infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double detect_ms = 0.0;
    double track_ms = 0.0;
    double total_ms = 0.0;
};

enum class ModelCapability {
    kDraw,
};

Ort::Env& shared_ort_env();

/**
 * @brief Base class for all vision models
 *
 * @thread_safety IMPORTANT: This class is NOT thread-safe.
 *
 * Thread Safety Rules:
 * 1. load_model(), warmup(), release() must be called from a single thread
 * 2. Inference methods (detect, classify, etc.) CANNOT be called concurrently
 *    on the same instance from multiple threads
 * 3. ONNX Runtime Session::Run() is NOT thread-safe
 *
 * Correct Multi-threading Usage:
 * - Option 1: Create separate model instances for each thread (RECOMMENDED)
 * - Option 2: Use a thread pool with one model instance per worker thread
 * - Option 3: Protect with mutex (NOT recommended, serializes inference)
 *
 * See docs/THREAD_SAFETY.md for detailed examples.
 */
class BaseModel {
public:
    explicit BaseModel(const std::string& model_path, bool lazy_load = false);
    virtual ~BaseModel();

    /**
     * @brief Load model from disk
     * @thread_safety NOT thread-safe. Must be called before any inference.
     */
    virtual void load_model() = 0;

    /**
     * @brief Warm up the model with dummy input
     * @thread_safety NOT thread-safe. Call after load_model(), before inference.
     */
    virtual void warmup();

    /**
     * @brief Release model resources
     * @thread_safety NOT thread-safe. Do not call while inference is running.
     */
    virtual void release();

    /**
     * @brief Get model capabilities
     * @thread_safety Thread-safe (read-only after construction)
     */
    virtual std::vector<ModelCapability> get_capabilities() const;

    /**
     * @brief Check if model supports a capability
     * @thread_safety Thread-safe (read-only after construction)
     */
    bool supports_capability(ModelCapability capability) const;

    /**
     * @brief Get input shape
     * @thread_safety Thread-safe after load_model() completes
     */
    virtual std::vector<int64_t> get_input_shape() const;

    /**
     * @brief Get model information
     * @thread_safety Thread-safe (read-only)
     */
    virtual std::string get_model_info() const;

    /**
     * @brief Get runtime profiling data
     * @thread_safety NOT thread-safe. Only call from the same thread that ran inference.
     */
    RuntimeProfile get_runtime_profile() const;

    /**
     * @brief Reset runtime profiling data
     * @thread_safety NOT thread-safe.
     */
    void reset_runtime_profile();

    /**
     * @brief Unified inference entry (internal; not part of public vision_service.h API).
     * @thread_safety NOT thread-safe.
     */
    virtual InferResponse Run(const InferRequest& request) = 0;

    /**
     * @brief Declared inference intents for dispatch validation.
     * @thread_safety Thread-safe (read-only after construction).
     */
    virtual std::vector<InferIntent> supported_intents() const = 0;

    /**
     * @brief Expected number of floats in the raw sequence input buffer.
     * Sequence models override this; default returns 0 (not a sequence model).
     */
    virtual size_t expected_sequence_size() const;

    /**
     * @brief Sequence action class names (empty unless sequence-action model).
     */
    virtual std::vector<std::string> get_sequence_class_names() const;

    /**
     * @brief Model-provided class names that are only known at runtime, e.g.
     * open-vocabulary detectors (YOLO-World) whose labels come from the active
     * text prompts rather than a static label_file_path. Empty by default;
     * VisionService::GetClassNames() falls back to this when the config
     * declares no labels.
     */
    virtual std::vector<std::string> get_dynamic_class_names() const;

protected:
    std::string model_path_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<int64_t> input_shape_;
    bool model_loaded_;
    bool lazy_load_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    size_t output_num_ = 0;
    Ort::AllocatorWithDefaultOptions allocator_;

    void ensure_model_loaded();

    void init_session(int num_threads = 4, const std::string& provider = "SpaceMITExecutionProvider");

    /**
     * @brief Run ONNX Runtime inference
     * @thread_safety NOT thread-safe. ONNX Runtime Session::Run() is not thread-safe.
     */
    std::vector<Ort::Value> run_session(const cv::Mat& input_blob);

    Ort::MemoryInfo memory_info_{nullptr};
    RuntimeProfile runtime_profile_;

    void set_runtime_preprocess_ms(double ms);
    void set_runtime_model_infer_ms(double ms);
    void set_runtime_postprocess_ms(double ms);
    void set_runtime_detect_ms(double ms);
    void set_runtime_track_ms(double ms);
    void set_runtime_total_ms(double ms);

#ifdef DEBUG
    /**
     * @brief Check thread safety in debug mode
     * @note Only active in DEBUG builds. Zero overhead in release builds.
     */
    void check_thread_safety(const char* method_name) const;

private:
    mutable std::thread::id owner_thread_;
#else
    void check_thread_safety(const char* method_name) const {}
#endif
};

}  // namespace vision_core

#endif  // VISION_MODEL_BASE_H
