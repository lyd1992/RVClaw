/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DRAWING_H
#define DRAWING_H

#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "datatype.h"

namespace vision_common {

/**
 * @brief Draw detection results on image
 * @param image Input/output image
 * @param detections Vector of detections (x1, y1, x2, y2, class_id, score)
 * @param labels Optional class labels (if empty, only show class_id)
 * @param box_color Box color (BGR)
 * @param line_thickness Line thickness
 */
void draw_detections(
    cv::Mat& image,
    const std::vector<vision_common::DetectionResult>& detections,
    const std::vector<std::string>& labels = {},
    const cv::Scalar& box_color = cv::Scalar(0, 255, 0),
    int line_thickness = 2
);

/**
 * @brief Draw pose estimation results with keypoints and skeleton
 * @param image Input/output image
 * @param results Vector of PoseResult structures containing bounding boxes and keypoints
 * @param point_confidence_threshold Minimum visibility threshold for keypoints (default: 0.2)
 * @param box_color Box color (BGR, default: green)
 * @param kp_color Keypoint color (BGR, default: blue)
 * @param line_thickness Line thickness for skeleton (default: 2)
 * @param kp_radius Radius of keypoint circles (default: 5)
 */
void draw_keypoints(
    cv::Mat& image,
    const std::vector<PoseResult>& results,
    float point_confidence_threshold = 0.2f,
    const cv::Scalar& box_color = cv::Scalar(0, 255, 0),
    const cv::Scalar& kp_color = cv::Scalar(255, 0, 0),
    int line_thickness = 2,
    int kp_radius = 5
);

/**
 * @brief Draw segmentation results with masks
 * @param image Input/output image
 * @param results Vector of SegmentationResult structures containing bounding boxes and masks
 * @param labels Optional class labels (if empty, only show class_id)
 * @param alpha Transparency for mask overlay (default: 0.5)
 * @param box_color Box color (BGR, default: green)
 * @param line_thickness Line thickness for bounding box (default: 2)
 */
void draw_segmentation(
    cv::Mat& image,
    const std::vector<SegmentationResult>& results,
    const std::vector<std::string>& labels = {},
    float alpha = 0.5f,
    const cv::Scalar& box_color = cv::Scalar(0, 255, 0),
    int line_thickness = 2
);

/**
 * @brief Draw tracking results with track IDs
 * @param image Input/output image
 * @param results Vector of TrackingResult structures containing bounding boxes and track IDs
 * @param labels Optional class labels (if empty, only show class_id)
 * @param line_thickness Line thickness for bounding box (default: 2)
 */
void draw_tracking_results(
    cv::Mat& image,
    const std::vector<TrackingResult>& results,
    const std::vector<std::string>& labels = {},
    int line_thickness = 2
);

/**
 * @brief Draw OCR text results: quadrilateral polygon + recognized string.
 * @note cv::putText renders ASCII only; non-ASCII text (e.g. Chinese) appears
 *       as '?' on the image, though the string in the result is intact.
 * @param image Input/output image
 * @param results Vector of TextResult (polygon + text + score)
 * @param color Polygon/text color
 * @param line_thickness Line thickness
 */
void draw_text(
    cv::Mat& image,
    const std::vector<TextResult>& results,
    const cv::Scalar& color = cv::Scalar(0, 255, 0),
    int line_thickness = 2
);

/**
 * @brief Draw ModelResult variant (auto-detects type and calls appropriate function)
 * @param image Input/output image
 * @param results Vector of ModelResult variants
 * @param labels Optional class labels
 * @param line_thickness Line thickness
 */
void draw_results(
    cv::Mat& image,
    const std::vector<ModelResult>& results,
    const std::vector<std::string>& labels = {},
    int line_thickness = 2
);

}  // namespace vision_common

#endif  // DRAWING_H
