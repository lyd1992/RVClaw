/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "byte_track_tracker.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> ByteTrackTracker::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for ByteTrackTracker");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for ByteTrackTracker");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int frame_rate = vision_core::yaml_utils::getInt(default_params, "frame_rate", 30);
    int track_buffer = vision_core::yaml_utils::getInt(default_params, "track_buffer", 30);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<ByteTrackTracker>(
        model_path,
        conf_threshold,
        iou_threshold,
        frame_rate,
        track_buffer,
        num_threads,
        lazy_load,
        provider);
}

ByteTrackTracker::ByteTrackTracker(const std::string& model_path,
                                    float conf_threshold,
                                    float iou_threshold,
                                    int frame_rate,
                                    int track_buffer,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        frame_rate_(frame_rate),
        track_buffer_(track_buffer),
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

void ByteTrackTracker::load_model() {
    if (model_loaded_) {
        return;
    }
    // Load detection model (idempotent if detector was eager-loaded in ctor).
    detector_->load_model();

    // Initialize tracker after detector is loaded
    tracker_ = std::make_unique<BYTETracker>(frame_rate_, track_buffer_);

    // Copy input shape from detector
    input_shape_ = detector_->get_input_shape();
    model_loaded_ = true;
}

cv::Mat ByteTrackTracker::preprocess(const cv::Mat& image) {
    ensure_model_loaded();
    return detector_->preprocess(image);
}

vision_common::TrackingResultList ByteTrackTracker::track(
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

    // Convert to Object format for tracker
    const auto t_track0 = std::chrono::steady_clock::now();
    std::vector<Object> objects = convert_results_to_objects(detections);

    // Update tracker
    std::vector<STrack> stracks = tracker_->update(objects);

    // Convert back to TrackingResult format with track_id and preserve label information
    vision_common::TrackingResultList results = convert_stracks_to_results(stracks, detections);
    const auto t_track1 = std::chrono::steady_clock::now();
    set_runtime_track_ms(std::chrono::duration<double, std::milli>(t_track1 - t_track0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> ByteTrackTracker::supported_intents() const {
    return {vision_core::InferIntent::kTrack};
}

vision_core::InferResponse ByteTrackTracker::Run(const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kTrack);
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "ByteTrackTracker expects ImageInput";
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

std::vector<vision_core::ModelCapability> ByteTrackTracker::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

std::vector<Object> ByteTrackTracker::convert_results_to_objects(
    const vision_common::DetectionResultList& results) {
    std::vector<Object> objects;
    objects.reserve(results.size());

    for (const auto& result : results) {
        Object obj;
        obj.rect.x = result.bbox.x1;
        obj.rect.y = result.bbox.y1;
        obj.rect.width = vision_common::width(result.bbox);
        obj.rect.height = vision_common::height(result.bbox);
        obj.label = result.label;
        obj.prob = result.score;
        objects.push_back(obj);
    }

    return objects;
}

vision_common::TrackingResultList ByteTrackTracker::convert_stracks_to_results(
    const std::vector<STrack>& stracks,
    const vision_common::DetectionResultList& detections) {
    vision_common::TrackingResultList results;
    results.reserve(stracks.size());

    for (const auto& strack : stracks) {
        vision_common::TrackingResult result;

        // Convert tlwh to xyxy
        float track_x1 = strack.tlwh[0];
        float track_y1 = strack.tlwh[1];
        float track_x2 = strack.tlwh[0] + strack.tlwh[2];
        float track_y2 = strack.tlwh[1] + strack.tlwh[3];

        result.bbox = vision_common::BoundingBox{track_x1, track_y1, track_x2, track_y2};
        result.score = strack.score;
        result.track_id = strack.track_id;
        result.state = vision_common::TrackingResult::State::Confirmed;

        // Match with detections to find label
        // Find the detection with highest IoU
        float max_iou = 0.0f;
        int best_match_idx = -1;

        for (size_t i = 0; i < detections.size(); i++) {
            float iou = vision_common::iou(result.bbox, detections[i].bbox);

            if (iou > max_iou && iou > 0.5f) {  // Threshold to ensure good match
                max_iou = iou;
                best_match_idx = static_cast<int>(i);
            }
        }

        // Use label from matched detection, or -1 if no good match
        if (best_match_idx >= 0) {
            result.label = detections[best_match_idx].label;
        } else {
            result.label = -1;
        }

        results.push_back(result);
    }

    return results;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<ByteTrackTracker> registrar("ByteTrackTracker");

}  // namespace vision_deploy

