/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BYTE_TRACK_TRACKER_H
#define BYTE_TRACK_TRACKER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"
#include "include/BYTETracker.h"
#include "yolov8_detector.h"

namespace YAML {
class Node;
}

namespace cv {
class Mat;
}

namespace vision_deploy {

/**
 * @brief ByteTrack Tracker with integrated detection
 *
 * Combines YOLOv8 detection and ByteTrack tracking for video analysis.
 */
class ByteTrackTracker : public vision_core::BaseModel, public vision_core::ITrackingModel {
public:
    ByteTrackTracker(const std::string& model_path,
                    float conf_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int frame_rate = 30,
                    int track_buffer = 30,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~ByteTrackTracker() = default;

    /**
     * @brief Load detection model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for detection
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /**
     * @brief Update tracker with new detections
     * @param image          Input image
     * @param conf_threshold Detector confidence override; <= 0 keeps config default
     * @param iou_threshold  Detector NMS IoU override; <= 0 keeps config default
     * @return Vector of tracking results with track_id
     */
    vision_common::TrackingResultList track(const cv::Mat& image,
                                            float conf_threshold = -1.0f,
                                            float iou_threshold = -1.0f) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

private:
    float conf_threshold_;
    float iou_threshold_;
    int frame_rate_;
    int track_buffer_;
    int num_threads_;
    std::string provider_;

    // Detection model
    std::unique_ptr<YOLOv8Detector> detector_;

    // Tracking model
    std::unique_ptr<BYTETracker> tracker_;

    /**
     * @brief Convert DetectionResult to Object format for tracker
     */
    std::vector<Object> convert_results_to_objects(const vision_common::DetectionResultList& results);

    /**
     * @brief Convert STrack to TrackingResult format, preserving label from detections
     */
    vision_common::TrackingResultList convert_stracks_to_results(
        const std::vector<STrack>& stracks,
        const vision_common::DetectionResultList& detections);
};

}  // namespace vision_deploy

#endif  // BYTE_TRACK_TRACKER_H
