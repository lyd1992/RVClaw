/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DATATYPE_H
#define DATATYPE_H

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "vision_service.h"  // public result types (namespace vision)
#include "common/cpp/vision_geometry.h"  // internal helpers over those types

/**
 * @file datatype.h
 * @brief Internal aliases + helpers over the PUBLIC result types.
 *
 * The canonical result structures are defined once in include/vision_service.h
 * (namespace vision). Internal code historically uses the vision_common::*Result
 * names, so this header maps those names onto the public types and provides the
 * variant visitor helpers used across deploy/common. There is no separate
 * internal definition and no conversion layer.
 */

namespace vision_common {

// ----------------------------------------------------------------------------
// Aliases onto the public types (single source of truth: vision::)
// ----------------------------------------------------------------------------
using BoundingBox = vision::BoundingBox;
using KeyPoint = vision::KeyPoint;

using DetectionResult = vision::Detection;
using ClassificationResult = vision::Classification;
using PoseResult = vision::Pose;
using SegmentationResult = vision::Segmentation;
using EmbeddingResult = vision::Embedding;
using TrackingResult = vision::Tracking;
using ActionResult = vision::Action;
using TextResult = vision::Text;

using ModelResult = vision::Result;

using DetectionResultList = std::vector<DetectionResult>;
using ClassificationResultList = std::vector<ClassificationResult>;
using PoseResultList = std::vector<PoseResult>;
using SegmentationResultList = std::vector<SegmentationResult>;
using TrackingResultList = std::vector<TrackingResult>;
using TextResultList = std::vector<TextResult>;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/**
 * @brief Build a top-k classification result list from per-class scores.
 * @param scores Per-class logits or probabilities (modified in-place when apply_softmax).
 * @param k Number of top predictions to return (default 5).
 * @param apply_softmax Whether to apply softmax before ranking.
 */
inline ClassificationResultList build_classification_top_k(
    std::vector<float> scores, int k = 5, bool apply_softmax = true) {
    ClassificationResultList results;
    if (scores.empty()) {
        return results;
    }

    if (apply_softmax) {
        const float max_score = *std::max_element(scores.begin(), scores.end());
        float exp_sum = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            exp_sum += score;
        }
        for (float& score : scores) {
            score /= exp_sum;
        }
    }

    ClassificationResult summary;
    summary.class_scores = scores;
    const auto ranked = top_k(summary, k);

    results.reserve(ranked.size());
    for (size_t i = 0; i < ranked.size(); ++i) {
        ClassificationResult item;
        item.label = ranked[i].first;
        item.score = ranked[i].second;
        if (i == 0) {
            item.class_scores = scores;
        }
        results.push_back(item);
    }
    return results;
}

// ----------------------------------------------------------------------------
// Type-checking helpers
// ----------------------------------------------------------------------------

inline bool is_detection(const ModelResult& r) {
    return std::holds_alternative<DetectionResult>(r);
}
inline bool is_classification(const ModelResult& r) {
    return std::holds_alternative<ClassificationResult>(r);
}
inline bool is_pose(const ModelResult& r) {
    return std::holds_alternative<PoseResult>(r);
}
inline bool is_segmentation(const ModelResult& r) {
    return std::holds_alternative<SegmentationResult>(r);
}
inline bool is_embedding(const ModelResult& r) {
    return std::holds_alternative<EmbeddingResult>(r);
}
inline bool is_tracking(const ModelResult& r) {
    return std::holds_alternative<TrackingResult>(r);
}
inline bool is_action(const ModelResult& r) {
    return std::holds_alternative<ActionResult>(r);
}

// ----------------------------------------------------------------------------
// Type-safe accessors (returns nullptr if wrong type)
// ----------------------------------------------------------------------------

inline const DetectionResult* as_detection(const ModelResult& r) {
    return std::get_if<DetectionResult>(&r);
}
inline const ClassificationResult* as_classification(const ModelResult& r) {
    return std::get_if<ClassificationResult>(&r);
}
inline const PoseResult* as_pose(const ModelResult& r) {
    return std::get_if<PoseResult>(&r);
}
inline const SegmentationResult* as_segmentation(const ModelResult& r) {
    return std::get_if<SegmentationResult>(&r);
}
inline const EmbeddingResult* as_embedding(const ModelResult& r) {
    return std::get_if<EmbeddingResult>(&r);
}
inline const TrackingResult* as_tracking(const ModelResult& r) {
    return std::get_if<TrackingResult>(&r);
}
inline const ActionResult* as_action(const ModelResult& r) {
    return std::get_if<ActionResult>(&r);
}

// ----------------------------------------------------------------------------
// Common property accessors (work across most result types)
// ----------------------------------------------------------------------------

inline float get_score(const ModelResult& r) {
    return std::visit([](const auto& result) -> float { return result.score; }, r);
}

inline BoundingBox get_bbox(const ModelResult& r) {
    return std::visit([](const auto& result) -> BoundingBox {
        using T = std::decay_t<decltype(result)>;
        if constexpr (
            std::is_same_v<T, DetectionResult> ||
            std::is_same_v<T, PoseResult> ||
            std::is_same_v<T, SegmentationResult> ||
            std::is_same_v<T, TrackingResult>) {
            return result.bbox;
        } else {
            return BoundingBox{};
        }
    }, r);
}

inline int get_label(const ModelResult& r) {
    return std::visit([](const auto& result) -> int {
        using T = std::decay_t<decltype(result)>;
        if constexpr (
            std::is_same_v<T, DetectionResult> ||
            std::is_same_v<T, ClassificationResult> ||
            std::is_same_v<T, PoseResult> ||
            std::is_same_v<T, SegmentationResult> ||
            std::is_same_v<T, TrackingResult> ||
            std::is_same_v<T, ActionResult>) {
            return result.label;
        } else {
            return -1;
        }
    }, r);
}

template<typename Visitor>
auto visit_result(const ModelResult& r, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), r);
}

}  // namespace vision_common

#endif  // DATATYPE_H
