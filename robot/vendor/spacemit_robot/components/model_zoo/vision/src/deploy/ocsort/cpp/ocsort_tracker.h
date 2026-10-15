/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OCSORT_TRACKER_H
#define OCSORT_TRACKER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"
#include "include/OCSort.hpp"
#include "yolov8_detector.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief OC-SORT Tracker with integrated detection
 * 
 * Combines YOLOv8 detection and OC-SORT tracking for video analysis.
 * OC-SORT is an observation-centric SORT algorithm that improves tracking
 * by using observation-centric re-update and momentum.
 */
class OCSortTracker : public vision_core::BaseModel, public vision_core::ITrackingModel {
public:
    OCSortTracker(const std::string& model_path,
                    float conf_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    float det_thresh = 0.3f,
                    int max_age = 30,
                    int min_hits = 3,
                    int delta_t = 3,
                    float inertia = 0.2f,
                    bool use_byte = false,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~OCSortTracker() = default;

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
    float det_thresh_;
    int max_age_;
    int min_hits_;
    int delta_t_;
    float inertia_;
    bool use_byte_;
    int num_threads_;
    std::string provider_;

    // Detection model
    std::unique_ptr<YOLOv8Detector> detector_;

    // OC-SORT tracker
    std::unique_ptr<ocsort::OCSort> tracker_;

    /**
     * @brief Convert Result to Eigen matrix format for OC-SORT tracker
     * Each row: [x1, y1, x2, y2, score, label]
     */
    Eigen::MatrixXf convert_results_to_dets(const vision_common::DetectionResultList& results);

    /**
     * @brief Convert OC-SORT output to TrackingResult format
     * @param tracks OC-SORT output, each row: [x1, y1, x2, y2, track_id, ...]
     * @param detections Original detections for label matching
     */
    vision_common::TrackingResultList convert_tracks_to_results(
        const std::vector<Eigen::RowVectorXf>& tracks,
        const vision_common::DetectionResultList& detections);

    /**
     * @brief Calculate IoU between two bounding boxes
     */
    static float calculate_iou(float x1_1, float y1_1, float x2_1, float y2_1,
                                float x1_2, float y1_2, float x2_2,
                                float y2_2);
};

}  // namespace vision_deploy

#endif  // OCSORT_TRACKER_H
