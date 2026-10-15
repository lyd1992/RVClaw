/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NMS_H
#define NMS_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "datatype.h"

namespace vision_common {

/**
 * @brief Calculate IoU between two boxes
 * @param box1 First box
 * @param box2 Second box
 * @return IoU value
 */
float calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2);

/**
 * @brief Calculate IoU between two bounding boxes
 * @param bbox1 First bounding box
 * @param bbox2 Second bounding box
 * @return IoU value
 */
float calculate_iou(const BoundingBox& bbox1, const BoundingBox& bbox2);

/**
 * @brief Non-maximum suppression
 * @param boxes Vector of boxes
 * @param scores Vector of scores
 * @param iou_threshold IoU threshold
 * @return Indices of boxes to keep
 */
std::vector<int> nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float iou_threshold
);

/**
 * @brief Multi-class non-maximum suppression for DetectionResult
 * @param objects Vector of DetectionResult objects with different classes
 * @param iou_threshold IoU threshold for NMS
 * @return Filtered results after per-class NMS
 */
std::vector<DetectionResult> multi_class_nms(
    const std::vector<DetectionResult>& objects,
    float iou_threshold
);

/**
 * @brief Multi-class non-maximum suppression for PoseResult
 * @param objects Vector of PoseResult objects with different classes
 * @param iou_threshold IoU threshold for NMS
 * @return Filtered results after per-class NMS
 */
std::vector<PoseResult> multi_class_nms(
    const std::vector<PoseResult>& objects,
    float iou_threshold
);

/**
 * @brief Multi-class non-maximum suppression for SegmentationResult
 * @param objects Vector of SegmentationResult objects with different classes
 * @param iou_threshold IoU threshold for NMS
 * @return Filtered results after per-class NMS
 */
std::vector<SegmentationResult> multi_class_nms(
    const std::vector<SegmentationResult>& objects,
    float iou_threshold
);

}  // namespace vision_common

#endif  // NMS_H

