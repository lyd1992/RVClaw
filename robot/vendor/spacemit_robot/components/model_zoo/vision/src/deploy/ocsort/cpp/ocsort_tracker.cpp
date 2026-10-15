/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ocsort_tracker.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> OCSortTracker::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for OCSortTracker");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for OCSortTracker");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    float det_thresh = vision_core::yaml_utils::getFloat(default_params, "det_thresh", 0.3f);
    int max_age = vision_core::yaml_utils::getInt(default_params, "max_age", 30);
    int min_hits = vision_core::yaml_utils::getInt(default_params, "min_hits", 3);
    int delta_t = vision_core::yaml_utils::getInt(default_params, "delta_t", 3);
    float inertia = vision_core::yaml_utils::getFloat(default_params, "inertia", 0.2f);
    bool use_byte = vision_core::yaml_utils::getBool(default_params, "use_byte", false);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<OCSortTracker>(
        model_path,
        conf_threshold,
        iou_threshold,
        det_thresh,
        max_age,
        min_hits,
        delta_t,
        inertia,
        use_byte,
        num_threads,
        lazy_load,
        provider);
}

OCSortTracker::OCSortTracker(const std::string& model_path,
                            float conf_threshold,
                            float iou_threshold,
                            float det_thresh,
                            int max_age,
                            int min_hits,
                            int delta_t,
                            float inertia,
                            bool use_byte,
                            int num_threads,
                            bool lazy_load,
                            const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        det_thresh_(det_thresh),
        max_age_(max_age),
        min_hits_(min_hits),
        delta_t_(delta_t),
        inertia_(inertia),
        use_byte_(use_byte),
        num_threads_(num_threads),
        provider_(provider) {
    // Initialize detector
    detector_ = std::make_unique<YOLOv8Detector>(
        model_path, conf_threshold_, iou_threshold_, num_threads_, lazy_load, provider_);

    // Initialize tracker (will be fully initialized after detector loads)
    if (!lazy_load) {
        load_model();
    }
}

void OCSortTracker::load_model() {
    if (model_loaded_) {
        return;
    }
    // Load detection model (idempotent if detector was eager-loaded in ctor).
    detector_->load_model();

    // Initialize OC-SORT tracker after detector is loaded
    tracker_ = std::make_unique<ocsort::OCSort>(
        det_thresh_, max_age_, min_hits_, iou_threshold_, delta_t_,
        "iou", inertia_, use_byte_);

    // Copy input shape from detector
    input_shape_ = detector_->get_input_shape();
    model_loaded_ = true;
}

cv::Mat OCSortTracker::preprocess(const cv::Mat& image) {
    ensure_model_loaded();
    return detector_->preprocess(image);
}

vision_common::TrackingResultList OCSortTracker::track(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Run detection
    const auto t_det0 = std::chrono::steady_clock::now();
    vision_common::DetectionResultList detections = detector_->detect(image, conf_threshold, iou_threshold);
    const auto t_det1 = std::chrono::steady_clock::now();
    set_runtime_detect_ms(std::chrono::duration<double, std::milli>(t_det1 - t_det0).count());

    // Convert to Eigen matrix format for OC-SORT tracker
    const auto t_track0 = std::chrono::steady_clock::now();
    Eigen::MatrixXf dets = convert_results_to_dets(detections);

    // Update tracker
    std::vector<Eigen::RowVectorXf> tracks = tracker_->update(dets);

    // Convert back to TrackingResult format with track_id and preserve label information
    vision_common::TrackingResultList results = convert_tracks_to_results(tracks, detections);
    const auto t_track1 = std::chrono::steady_clock::now();
    set_runtime_track_ms(std::chrono::duration<double, std::milli>(t_track1 - t_track0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> OCSortTracker::supported_intents() const {
    return {vision_core::InferIntent::kTrack};
}

vision_core::InferResponse OCSortTracker::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kTrack);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "OCSortTracker expects ImageInput";
        return response;
    }

    vision_common::TrackingResultList task_results = track(image_input->image, request.params.conf_threshold, request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> OCSortTracker::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

Eigen::MatrixXf OCSortTracker::convert_results_to_dets(
    const vision_common::DetectionResultList& results) {
    if (results.empty()) {
        return Eigen::MatrixXf(0, 6);
    }

    // Each row: [x1, y1, x2, y2, score, label]
    Eigen::MatrixXf dets(results.size(), 6);

    for (size_t i = 0; i < results.size(); ++i) {
        dets(i, 0) = results[i].bbox.x1;
        dets(i, 1) = results[i].bbox.y1;
        dets(i, 2) = results[i].bbox.x2;
        dets(i, 3) = results[i].bbox.y2;
        dets(i, 4) = results[i].score;
        dets(i, 5) = static_cast<float>(results[i].label);
    }

    return dets;
}

float OCSortTracker::calculate_iou(float x1_1, float y1_1, float x2_1, float y2_1,
                                    float x1_2, float y1_2, float x2_2,
                                    float y2_2) {
    float inter_x1 = std::max(x1_1, x1_2);
    float inter_y1 = std::max(y1_1, y1_2);
    float inter_x2 = std::min(x2_1, x2_2);
    float inter_y2 = std::min(y2_1, y2_2);

    if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) {
        return 0.0f;
    }

    float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    float union_area = area1 + area2 - inter_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

vision_common::TrackingResultList OCSortTracker::convert_tracks_to_results(
    const std::vector<Eigen::RowVectorXf>& tracks,
    const vision_common::DetectionResultList& detections) {

    vision_common::TrackingResultList results;
    results.reserve(tracks.size());

    for (const auto& track : tracks) {
        vision_common::TrackingResult result;

        // OC-SORT output format: [x1, y1, x2, y2, track_id, ...]
        // Note: The exact format depends on OC-SORT implementation
        if (track.size() >= 5) {
            result.bbox = vision_common::BoundingBox{
                static_cast<float>(track(0)),
                static_cast<float>(track(1)),
                static_cast<float>(track(2)),
                static_cast<float>(track(3))
            };
            result.track_id = static_cast<int>(track(4));

            // Default score (OC-SORT may not preserve score)
            result.score = 1.0f;
            result.state = vision_common::TrackingResult::State::Confirmed;

            // Match with detections to find label using IoU
            float max_iou = 0.0f;
            int best_match_idx = -1;

            for (size_t i = 0; i < detections.size(); ++i) {
                float iou = vision_common::iou(result.bbox, detections[i].bbox);
                if (iou > max_iou && iou > 0.5f) {
                    max_iou = iou;
                    best_match_idx = static_cast<int>(i);
                }
            }

            // Use label from matched detection, or -1 if no good match
            if (best_match_idx >= 0) {
                result.label = detections[best_match_idx].label;
                result.score = detections[best_match_idx].score;
            } else {
                result.label = -1;
            }

            results.push_back(result);
        }
    }

    return results;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<OCSortTracker> registrar("OCSortTracker");

}  // namespace vision_deploy
