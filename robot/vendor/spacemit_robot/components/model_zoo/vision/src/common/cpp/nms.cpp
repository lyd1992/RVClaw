/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nms.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace vision_common {

float calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float x1_inter = std::max(box1.x, box2.x);
    float y1_inter = std::max(box1.y, box2.y);
    float x2_inter = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2_inter = std::min(box1.y + box1.height, box2.y + box2.height);

    // Keep IoU behavior aligned with Python NMS implementation (inclusive coordinates).
    float width_inter = std::max(0.0f, x2_inter - x1_inter + 1.0f);
    float height_inter = std::max(0.0f, y2_inter - y1_inter + 1.0f);
    float area_inter = width_inter * height_inter;

    float area1 = (box1.width + 1.0f) * (box1.height + 1.0f);
    float area2 = (box2.width + 1.0f) * (box2.height + 1.0f);
    float area_union = area1 + area2 - area_inter;

    if (area_union == 0) {
        return 0.0f;
    }
    return area_inter / area_union;
}

float calculate_iou(const BoundingBox& bbox1, const BoundingBox& bbox2) {
    return iou(bbox1, bbox2);
}

std::vector<int> nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float iou_threshold) {
    if (boxes.empty()) {
        return std::vector<int>();
    }

    // Create indices and sort by score
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&scores](int i, int j) { return scores[i] > scores[j]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);

        if (indices.size() == 1) {
            break;
        }

        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); ++i) {
            float iou = calculate_iou(boxes[current], boxes[indices[i]]);
            if (iou < iou_threshold) {
                new_indices.push_back(indices[i]);
            }
        }
        indices = new_indices;
    }

    return keep;
}

// Template helper for multi-class NMS
template<typename T>
std::vector<T> multi_class_nms_impl(
    const std::vector<T>& objects,
    float iou_threshold) {
    if (objects.empty()) {
        return std::vector<T>();
    }

    // Group by class - find unique labels
    std::vector<int> unique_labels;
    for (const auto& result : objects) {
        if (std::find(unique_labels.begin(), unique_labels.end(), result.label) == unique_labels.end()) {
            unique_labels.push_back(result.label);
        }
    }

    std::vector<T> final_results;
    for (int label : unique_labels) {
        // Collect results for this class
        std::vector<T> results_class;
        for (const auto& result : objects) {
            if (result.label == label) {
                results_class.push_back(result);
            }
        }

        // Sort by score descending
        std::sort(results_class.begin(), results_class.end(),
            [](const T& a, const T& b) { return a.score > b.score; });

        // Apply NMS
        std::vector<bool> suppressed(results_class.size(), false);
        for (size_t i = 0; i < results_class.size(); ++i) {
            if (suppressed[i]) continue;

            final_results.push_back(results_class[i]);

            // Suppress overlapping boxes
            for (size_t j = i + 1; j < results_class.size(); ++j) {
                if (suppressed[j]) continue;

                float iou = calculate_iou(results_class[i].bbox, results_class[j].bbox);
                if (iou >= iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return final_results;
}

std::vector<DetectionResult> multi_class_nms(
    const std::vector<DetectionResult>& objects,
    float iou_threshold) {
    return multi_class_nms_impl(objects, iou_threshold);
}

std::vector<PoseResult> multi_class_nms(
    const std::vector<PoseResult>& objects,
    float iou_threshold) {
    return multi_class_nms_impl(objects, iou_threshold);
}

std::vector<SegmentationResult> multi_class_nms(
    const std::vector<SegmentationResult>& objects,
    float iou_threshold) {
    return multi_class_nms_impl(objects, iou_threshold);
}

}  // namespace vision_common

