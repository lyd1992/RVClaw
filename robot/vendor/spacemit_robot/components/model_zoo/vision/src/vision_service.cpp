/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_service.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/imgcodecs.hpp>  // NOLINT(build/include_order)
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "common/cpp/datatype.h"
#include "common/cpp/drawing.h"
#include "common/cpp/embedding_utils.h"
#include "common/cpp/image_processing.h"
#include "core/cpp/vision_infer_types.h"
#include "core/cpp/vision_model_base.h"
#include "core/cpp/vision_model_factory.h"

struct VisionService::Impl {
    std::unique_ptr<vision_core::BaseModel> model;
    std::string config_path;
    std::vector<std::string> labels;
    std::string default_image_path;
    std::string last_config_path_value;
    VisionServiceTimingOptions timing_options;
    VisionServiceTiming last_timing;
    uint64_t timed_tracking_frame_count = 0;
    uint64_t timed_tracking_object_sum = 0;
};

thread_local std::string VisionService::g_last_error_;

namespace {

double ToMs(const std::chrono::steady_clock::duration& duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

void ResetAllTiming(VisionServiceTiming* timing) {
    if (timing == nullptr) {
        return;
    }
    timing->preprocess_ms = 0.0;
    timing->model_infer_ms = 0.0;
    timing->postprocess_ms = 0.0;
    timing->detect_ms = 0.0;
    timing->track_ms = 0.0;
    timing->infer_ms = 0.0;
    timing->draw_ms = 0.0;
    timing->embedding_ms = 0.0;
    timing->sequence_ms = 0.0;
}

void ResetDrawTiming(VisionServiceTiming* timing) {
    if (timing == nullptr) {
        return;
    }
    timing->draw_ms = 0.0;
}

void FillTimingFromRuntimeProfile(const vision_core::RuntimeProfile& profile,
                                    bool is_tracking,
                                    VisionServiceTiming* timing) {
    if (timing == nullptr) {
        return;
    }
    if (is_tracking) {
        timing->detect_ms = profile.detect_ms;
        timing->track_ms = profile.track_ms;
        timing->infer_ms = profile.total_ms;
    } else {
        timing->preprocess_ms = profile.preprocess_ms;
        timing->model_infer_ms = profile.model_infer_ms;
        timing->postprocess_ms = profile.postprocess_ms;
        timing->infer_ms = profile.total_ms;
    }
}

void MaybePrintImageTiming(const VisionServiceTimingOptions& options,
                            const VisionServiceTiming& timing,
                            bool is_tracking,
                            int tracked_count = -1,
                            double avg_tracked_count = 0.0) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }

    if (is_tracking) {
        std::cout << "[VisionService][Timing][Image][tracking] "
                    << "detect=" << timing.detect_ms << "ms, "
                    << "track=" << timing.track_ms << "ms, "
                    << "tracked=" << tracked_count << ", "
                    << "avg_tracked=" << avg_tracked_count << ", "
                    << "total=" << timing.infer_ms << "ms"
                    << std::endl;
    } else {
        std::cout << "[VisionService][Timing][Image][generic] "
                    << "preprocess=" << timing.preprocess_ms << "ms, "
                    << "model_infer=" << timing.model_infer_ms << "ms, "
                    << "postprocess=" << timing.postprocess_ms << "ms, "
                    << "total=" << timing.infer_ms << "ms"
                    << std::endl;
    }
}

void MaybePrintEmbeddingTiming(const VisionServiceTimingOptions& options,
                                const VisionServiceTiming& timing) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }
    std::cout << "[VisionService][Timing][Embedding] "
                << "preprocess=" << timing.preprocess_ms << "ms, "
                << "model_infer=" << timing.model_infer_ms << "ms, "
                << "postprocess=" << timing.postprocess_ms << "ms, "
                << "total=" << timing.embedding_ms << "ms"
                << std::endl;
}

void MaybePrintSequenceTiming(const VisionServiceTimingOptions& options,
                                const VisionServiceTiming& timing) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }
    std::cout << "[VisionService][Timing][Sequence] "
                << "preprocess=" << timing.preprocess_ms << "ms, "
                << "model_infer=" << timing.model_infer_ms << "ms, "
                << "postprocess=" << timing.postprocess_ms << "ms, "
                << "total=" << timing.sequence_ms << "ms"
                << std::endl;
}

void MaybePrintDrawTiming(const VisionServiceTimingOptions& options,
                            const VisionServiceTiming& timing) {
    if (!options.enabled || !options.print_to_stdout) {
        return;
    }
    std::cout << "[VisionService][Timing][Draw] "
                << "total=" << timing.draw_ms << "ms"
                << std::endl;
}

std::vector<std::string> loadLabelsForConfig(const YAML::Node& config, const std::string& config_file) {
    try {
        if (!config["label_file_path"]) {
            return {};
        }
        std::string label_file = config["label_file_path"].as<std::string>();
        label_file = vision_core::resolveResourcePath(label_file, config_file);
        return vision_common::load_labels(label_file);
    } catch (...) {
        return {};
    }
}

bool IntentDeclared(const std::vector<vision_core::InferIntent>& intents,
                    vision_core::InferIntent intent) {
    for (vision_core::InferIntent declared : intents) {
        if (declared == intent) {
            return true;
        }
    }
    return false;
}

bool ValidateIntentInputPair(const vision_core::InferRequest& request) {
    if (request.intent == vision_core::InferIntent::kInferSequence) {
        return std::holds_alternative<vision_core::SequenceInput>(request.input);
    }
    if (request.intent == vision_core::InferIntent::kEmbedText) {
        return std::holds_alternative<vision_core::TextInput>(request.input);
    }
    return std::holds_alternative<vision_core::ImageInput>(request.input);
}

std::optional<vision_core::InferIntent> PickImageIntent(
    const std::vector<vision_core::InferIntent>& intents) {
    if (intents.empty()) {
        return std::nullopt;
    }

    const vision_core::InferIntent priority[] = {
        vision_core::InferIntent::kTrack,
        vision_core::InferIntent::kEstimatePose,
        vision_core::InferIntent::kSegment,
        vision_core::InferIntent::kOcr,
        vision_core::InferIntent::kDetect,
        vision_core::InferIntent::kClassify,
    };
    for (vision_core::InferIntent preferred : priority) {
        if (IntentDeclared(intents, preferred)) {
            return preferred;
        }
    }

    return std::nullopt;
}

}  // namespace

VisionService::~VisionService() = default;

void VisionService::Release() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->model.reset();
    impl_->labels.clear();
    impl_->default_image_path.clear();
    impl_->last_config_path_value.clear();
    ResetAllTiming(&impl_->last_timing);
    impl_->timed_tracking_frame_count = 0;
    impl_->timed_tracking_object_sum = 0;
}

std::unique_ptr<VisionService> VisionService::Create(const std::string& config_path,
                                                    const std::string& model_path_override,
                                                    bool lazy_load) {
    g_last_error_.clear();
    if (config_path.empty()) {
        g_last_error_ = "config_path is empty";
        return nullptr;
    }

    std::unique_ptr<VisionService> service(new (std::nothrow) VisionService());
    if (!service) {
        g_last_error_ = "Failed to allocate VisionService";
        return nullptr;
    }
    service->impl_.reset(new (std::nothrow) Impl());
    if (!service->impl_) {
        g_last_error_ = "Failed to allocate VisionService::Impl";
        return nullptr;
    }

    try {
        std::filesystem::path config_file = std::filesystem::absolute(config_path);
        if (!std::filesystem::exists(config_file)) {
            g_last_error_ = "Config file not found: " + config_file.string();
            service->last_error_ = g_last_error_;
            return nullptr;
        }
        service->impl_->config_path = config_file.string();
        YAML::Node config = YAML::LoadFile(service->impl_->config_path);
        service->impl_->labels = loadLabelsForConfig(config, service->impl_->config_path);
        service->impl_->model = vision_core::createModelFromConfigPath(
            service->impl_->config_path, model_path_override, lazy_load, &config);
        service->last_error_.clear();
        g_last_error_.clear();
        return service;
    } catch (const std::exception& e) {
        g_last_error_ = e.what();
        service->last_error_ = e.what();
        return nullptr;
    } catch (...) {
        g_last_error_ = "Unknown error while creating model service";
        service->last_error_ = g_last_error_;
        return nullptr;
    }
}

const std::string& VisionService::LastCreateError() {
    return g_last_error_;
}

VisionServiceStatus VisionService::SetError(VisionServiceStatus code, const std::string& message) {
    g_last_error_ = message;
    last_error_ = message;
    return code;
}

const std::string& VisionService::LastError() const {
    if (!last_error_.empty()) {
        return last_error_;
    }
    return g_last_error_;
}

VisionServiceStatus VisionService::Infer(
    const VisionServiceRequest& request,
    VisionServiceResponse* response) {
    if (response == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "response must not be null");
    }
    response->results.clear();
    response->ok = false;
    response->error_message.clear();

    if (impl_ == nullptr || impl_->model == nullptr) {
        const std::string msg = "service/model must not be null";
        response->error_message = msg;
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, msg);
    }

    const bool is_sequence = (request.sequence_pts != nullptr);

    // Validate image input early (sequence input validated below).
    if (!is_sequence) {
        if (request.image.empty()) {
            response->error_message = "image must not be empty";
            return SetError(VISION_SERVICE_INVALID_ARGUMENT, response->error_message);
        }
        if (request.image.channels() != 3) {
            response->error_message = "image must be 3-channel BGR";
            return SetError(VISION_SERVICE_INVALID_ARGUMENT, response->error_message);
        }
    }

    try {
        const bool timing_enabled = impl_->timing_options.enabled;
        std::chrono::steady_clock::time_point t0;
        if (timing_enabled) {
            ResetAllTiming(&impl_->last_timing);
            t0 = std::chrono::steady_clock::now();
        }

        vision_core::BaseModel* model = impl_->model.get();
        const std::vector<vision_core::InferIntent> declared = model->supported_intents();

        // Decide the intent from input kind + declared capabilities.
        vision_core::InferIntent intent;
        if (is_sequence) {
            if (!IntentDeclared(declared, vision_core::InferIntent::kInferSequence)) {
                response->error_message = "current model does not support sequence inference";
                return SetError(VISION_SERVICE_INFER_FAILED, response->error_message);
            }
            intent = vision_core::InferIntent::kInferSequence;
        } else {
            const std::optional<vision_core::InferIntent> image_intent = PickImageIntent(declared);
            if (image_intent.has_value()) {
                intent = *image_intent;
            } else if (IntentDeclared(declared, vision_core::InferIntent::kEmbed)) {
                intent = vision_core::InferIntent::kEmbed;
            } else {
                response->error_message = "model does not support inference on image input";
                return SetError(VISION_SERVICE_INFER_FAILED, response->error_message);
            }
        }

        // Map public params -> internal params.
        vision_core::InferParams infer_params;
        infer_params.conf_threshold = request.params.conf_threshold;
        infer_params.iou_threshold = request.params.iou_threshold;
        infer_params.top_k = request.params.top_k;
        infer_params.kp_threshold = request.params.kp_threshold;
        infer_params.mask_threshold = request.params.mask_threshold;
        infer_params.max_det = request.params.max_det;
        infer_params.prompts = request.prompts;  // open-vocabulary text (YOLO-World)

        // Build the internal request input.
        vision_core::InferInput input;
        if (is_sequence) {
            const size_t seq_size = model->expected_sequence_size();
            if (seq_size == 0) {
                response->error_message = "model does not report expected sequence size";
                return SetError(VISION_SERVICE_INFER_FAILED, response->error_message);
            }
            // Guard against reading past the caller's buffer: if a count is
            // provided, it must cover what the model expects to consume.
            if (request.sequence_count > 0 &&
                static_cast<size_t>(request.sequence_count) < seq_size) {
                response->error_message =
                    "sequence_count (" + std::to_string(request.sequence_count) +
                    ") is smaller than the model's expected sequence size (" +
                    std::to_string(seq_size) + ")";
                return SetError(VISION_SERVICE_INVALID_ARGUMENT, response->error_message);
            }
            vision_core::SequenceInput seq;
            seq.image_width = request.sequence_width;
            seq.image_height = request.sequence_height;
            seq.pts.assign(request.sequence_pts, request.sequence_pts + seq_size);
            input = std::move(seq);
        } else {
            input = vision_core::ImageInput{request.image};
        }

        vision_core::InferRequest internal_request{std::move(input), intent, infer_params};
        if (!ValidateIntentInputPair(internal_request)) {
            response->error_message = "intent and input type mismatch";
            return SetError(VISION_SERVICE_INVALID_ARGUMENT, response->error_message);
        }

        vision_core::InferResponse internal_response = model->Run(internal_request);
        if (!internal_response.ok) {
            response->error_message = internal_response.error_message;
            return SetError(VISION_SERVICE_INFER_FAILED, internal_response.error_message);
        }

        // Internal ModelResult is an alias of vision::Result; move directly.
        response->results = std::move(internal_response.results);
        response->ok = true;

        // Timing.
        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto profile = model->get_runtime_profile();
            if (intent == vision_core::InferIntent::kEmbed ||
                intent == vision_core::InferIntent::kEmbedText) {
                impl_->last_timing.preprocess_ms = profile.preprocess_ms;
                impl_->last_timing.model_infer_ms = profile.model_infer_ms;
                impl_->last_timing.postprocess_ms = profile.postprocess_ms;
                impl_->last_timing.embedding_ms =
                    (profile.total_ms > 0.0) ? profile.total_ms : ToMs(t1 - t0);
                MaybePrintEmbeddingTiming(impl_->timing_options, impl_->last_timing);
            } else if (intent == vision_core::InferIntent::kInferSequence) {
                impl_->last_timing.preprocess_ms = profile.preprocess_ms;
                impl_->last_timing.model_infer_ms = profile.model_infer_ms;
                impl_->last_timing.postprocess_ms = profile.postprocess_ms;
                impl_->last_timing.sequence_ms =
                    (profile.total_ms > 0.0) ? profile.total_ms : ToMs(t1 - t0);
                MaybePrintSequenceTiming(impl_->timing_options, impl_->last_timing);
            } else {
                const bool is_tracking = (intent == vision_core::InferIntent::kTrack);
                FillTimingFromRuntimeProfile(profile, is_tracking, &impl_->last_timing);
                if (impl_->last_timing.infer_ms <= 0.0) {
                    impl_->last_timing.infer_ms = ToMs(t1 - t0);
                }
                if (is_tracking) {
                    int tracked_count = 0;
                    for (const auto& r : response->results) {
                        if (std::holds_alternative<vision_common::TrackingResult>(r)) {
                            const auto& tr = std::get<vision_common::TrackingResult>(r);
                            if (tr.track_id >= 0) {
                                ++tracked_count;
                            }
                        }
                    }
                    ++impl_->timed_tracking_frame_count;
                    impl_->timed_tracking_object_sum += static_cast<uint64_t>(tracked_count);
                    const double avg_tracked_count =
                        (impl_->timed_tracking_frame_count > 0)
                            ? (static_cast<double>(impl_->timed_tracking_object_sum) /
                                static_cast<double>(impl_->timed_tracking_frame_count))
                            : 0.0;
                    MaybePrintImageTiming(impl_->timing_options, impl_->last_timing, is_tracking,
                        tracked_count, avg_tracked_count);
                } else {
                    MaybePrintImageTiming(impl_->timing_options, impl_->last_timing, is_tracking);
                }
            }
        }

        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        response->error_message = e.what();
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        response->error_message = "Unknown error during inference";
        return SetError(VISION_SERVICE_INFER_FAILED, response->error_message);
    }
}

VisionServiceStatus VisionService::Infer(
    const std::string& image_path,
    VisionServiceResponse* response,
    const VisionServiceInferParams& params) {
    if (response == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "response must not be null");
    }
    response->results.clear();
    response->ok = false;
    if (image_path.empty()) {
        response->error_message = "image_path must not be empty";
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, response->error_message);
    }
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        response->error_message = std::string("Failed to read image: ") + image_path;
        return SetError(VISION_SERVICE_IO_FAILED, response->error_message);
    }
    VisionServiceRequest request;
    request.image = image;
    request.params = params;
    return Infer(request, response);
}

VisionServiceStatus VisionService::Infer(
    const cv::Mat& image,
    VisionServiceResponse* response,
    const VisionServiceInferParams& params) {
    VisionServiceRequest request;
    request.image = image;
    request.params = params;
    return Infer(request, response);
}

VisionServiceStatus VisionService::EncodeText(const std::string& text,
                                                std::vector<float>* out_embedding) {
    if (impl_ == nullptr || impl_->model == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model must not be null");
    }
    if (out_embedding == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "out_embedding must not be null");
    }
    if (text.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "text must not be empty");
    }

    try {
        vision_core::BaseModel* model = impl_->model.get();
        const std::vector<vision_core::InferIntent> declared = model->supported_intents();
        if (!IntentDeclared(declared, vision_core::InferIntent::kEmbedText)) {
            return SetError(VISION_SERVICE_INFER_FAILED,
                            "current model does not support text encoding");
        }

        const bool timing_enabled = impl_->timing_options.enabled;
        std::chrono::steady_clock::time_point t0;
        if (timing_enabled) {
            ResetAllTiming(&impl_->last_timing);
            t0 = std::chrono::steady_clock::now();
        }

        vision_core::TextInput text_input;
        text_input.text = text;
        vision_core::InferRequest internal_request{
            std::move(text_input), vision_core::InferIntent::kEmbedText, {}};
        if (!ValidateIntentInputPair(internal_request)) {
            return SetError(VISION_SERVICE_INVALID_ARGUMENT, "intent and input type mismatch");
        }

        vision_core::InferResponse internal_response = model->Run(internal_request);
        if (!internal_response.ok) {
            return SetError(VISION_SERVICE_INFER_FAILED, internal_response.error_message);
        }
        if (internal_response.results.empty()) {
            return SetError(VISION_SERVICE_INFER_FAILED, "model returned no embedding");
        }
        const vision_common::EmbeddingResult* emb =
            std::get_if<vision_common::EmbeddingResult>(&internal_response.results[0]);
        if (emb == nullptr) {
            return SetError(VISION_SERVICE_INFER_FAILED, "model did not return an Embedding result");
        }
        *out_embedding = emb->embedding;

        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            const auto profile = model->get_runtime_profile();
            impl_->last_timing.preprocess_ms = profile.preprocess_ms;
            impl_->last_timing.model_infer_ms = profile.model_infer_ms;
            impl_->last_timing.postprocess_ms = profile.postprocess_ms;
            impl_->last_timing.embedding_ms =
                (profile.total_ms > 0.0) ? profile.total_ms : ToMs(t1 - t0);
            MaybePrintEmbeddingTiming(impl_->timing_options, impl_->last_timing);
        }

        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    }
}

float VisionService::EmbeddingSimilarity(const std::vector<float>& embedding_a,
                                        const std::vector<float>& embedding_b) {
    if (embedding_a.empty() || embedding_b.empty() || embedding_a.size() != embedding_b.size()) {
        return 0.0f;
    }
    return vision_common::compute_similarity(embedding_a, embedding_b);
}

VisionServiceStatus VisionService::Draw(const cv::Mat& image,
                                        const VisionServiceResponse& response,
                                        cv::Mat* out_image) {
    if (impl_ == nullptr || impl_->model == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model must not be null");
    }
    if (out_image == nullptr) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "out_image must not be null");
    }
    if (image.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must not be empty");
    }
    if (image.channels() != 3) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "image must be 3-channel BGR");
    }
    if (!impl_->model->supports_capability(vision_core::ModelCapability::kDraw)) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "current model does not support draw");
    }
    if (response.results.empty()) {
        return SetError(VISION_SERVICE_INVALID_ARGUMENT, "No results to draw");
    }

    try {
        const bool timing_enabled = impl_->timing_options.enabled;
        if (timing_enabled) {
            ResetDrawTiming(&impl_->last_timing);
        }
        const auto t0 = timing_enabled ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        *out_image = image.clone();

        // Prefer static config labels; fall back to runtime model-provided
        // labels (e.g. YOLO-World prompts) so boxes show names, not "Class N".
        const std::vector<std::string>& draw_labels =
            !impl_->labels.empty() ? impl_->labels : impl_->model->get_dynamic_class_names();

        // draw_results handles ModelResult variants (alias of vision::Result).
        vision_common::draw_results(*out_image, response.results, draw_labels);

        if (timing_enabled) {
            const auto t1 = std::chrono::steady_clock::now();
            impl_->last_timing.draw_ms = ToMs(t1 - t0);
            MaybePrintDrawTiming(impl_->timing_options, impl_->last_timing);
        }

        last_error_.clear();
        return VISION_SERVICE_OK;
    } catch (const std::exception& e) {
        return SetError(VISION_SERVICE_INFER_FAILED, e.what());
    } catch (...) {
        return SetError(VISION_SERVICE_INFER_FAILED, "Unknown error during draw");
    }
}

bool VisionService::SupportsDraw() const {
    return impl_ != nullptr && impl_->model != nullptr &&
            impl_->model->supports_capability(vision_core::ModelCapability::kDraw);
}

void VisionService::SetTimingOptions(const VisionServiceTimingOptions& options) {
    if (impl_ == nullptr) {
        return;
    }
    const bool was_enabled = impl_->timing_options.enabled;
    impl_->timing_options = options;
    if (!options.enabled) {
        ResetAllTiming(&impl_->last_timing);
        impl_->timed_tracking_frame_count = 0;
        impl_->timed_tracking_object_sum = 0;
    } else if (!was_enabled) {
        ResetAllTiming(&impl_->last_timing);
        impl_->timed_tracking_frame_count = 0;
        impl_->timed_tracking_object_sum = 0;
    }
}

VisionServiceTiming VisionService::GetLastTiming() const {
    if (impl_ == nullptr) {
        return VisionServiceTiming{};
    }
    return impl_->last_timing;
}

std::string VisionService::GetDefaultImage() {
    if (impl_ == nullptr || impl_->model == nullptr) {
        SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model must not be null");
        return {};
    }

    if (!impl_->default_image_path.empty()) {
        return impl_->default_image_path;
    }

    try {
        if (impl_->config_path.empty() || !std::filesystem::exists(impl_->config_path)) {
            return {};
        }
        YAML::Node config = YAML::LoadFile(impl_->config_path);
        if (!config["test_image"]) {
            return {};
        }
        const std::string raw_path = config["test_image"].as<std::string>();
        impl_->default_image_path = vision_core::resolveResourcePath(raw_path, impl_->config_path);
        return impl_->default_image_path;
    } catch (...) {
        return {};
    }
}

std::string VisionService::GetConfigPathValue(const std::string& config_key) {
    if (impl_ == nullptr || impl_->model == nullptr || config_key.empty()) {
        SetError(VISION_SERVICE_INVALID_ARGUMENT, "service/model or config_key invalid");
        return {};
    }

    try {
        if (impl_->config_path.empty() || !std::filesystem::exists(impl_->config_path)) {
            return {};
        }
        YAML::Node config = YAML::LoadFile(impl_->config_path);
        if (!config[config_key]) {
            return {};
        }
        const std::string raw_path = config[config_key].as<std::string>();
        impl_->last_config_path_value = vision_core::resolveResourcePath(raw_path, impl_->config_path);
        return impl_->last_config_path_value;
    } catch (...) {
        return {};
    }
}

std::vector<std::string> VisionService::GetClassNames() const {
    if (impl_ == nullptr) {
        return {};
    }
    if (!impl_->labels.empty()) {
        return impl_->labels;
    }
    // Fall back to runtime model-provided labels (e.g. YOLO-World prompts) when
    // the config declares no static label_file_path.
    if (impl_->model != nullptr) {
        return impl_->model->get_dynamic_class_names();
    }
    return impl_->labels;
}
