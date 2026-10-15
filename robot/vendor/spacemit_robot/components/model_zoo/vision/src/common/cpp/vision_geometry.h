/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_GEOMETRY_H
#define VISION_GEOMETRY_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "vision_service.h"  // public POD result types (namespace vision)

/**
 * @file vision_geometry.h
 * @brief Internal helpers (geometry, ranking, similarity) over the public
 *        POD result types. The public header carries only data; all logic on
 *        those types lives here, in src/, and is never part of the ABI.
 */

namespace vision_common {

// ---------------------------------------------------------------------------
// BoundingBox helpers
// ---------------------------------------------------------------------------

inline float width(const vision::BoundingBox& b) { return b.x2 - b.x1; }
inline float height(const vision::BoundingBox& b) { return b.y2 - b.y1; }
inline float area(const vision::BoundingBox& b) { return width(b) * height(b); }
inline float center_x(const vision::BoundingBox& b) { return (b.x1 + b.x2) / 2.0f; }
inline float center_y(const vision::BoundingBox& b) { return (b.y1 + b.y2) / 2.0f; }
inline bool is_valid(const vision::BoundingBox& b) { return b.x2 > b.x1 && b.y2 > b.y1; }

// IoU (pixel coordinates with +1, consistent with the Python implementation).
inline float iou(const vision::BoundingBox& a, const vision::BoundingBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1 + 1.0f);
    float ih = std::max(0.0f, iy2 - iy1 + 1.0f);
    float inter = iw * ih;
    float a1 = (a.x2 - a.x1 + 1.0f) * (a.y2 - a.y1 + 1.0f);
    float a2 = (b.x2 - b.x1 + 1.0f) * (b.y2 - b.y1 + 1.0f);
    float uni = a1 + a2 - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

// ---------------------------------------------------------------------------
// KeyPoint helpers
// ---------------------------------------------------------------------------

inline bool is_visible(const vision::KeyPoint& kp) { return kp.visibility > 0.5f; }

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

// Top-k (index, score) pairs from a classification's per-class scores.
inline std::vector<std::pair<int, float>> top_k(const vision::Classification& c, int k) {
    std::vector<std::pair<int, float>> scored;
    scored.reserve(c.class_scores.size());
    for (size_t i = 0; i < c.class_scores.size(); ++i) {
        scored.emplace_back(static_cast<int>(i), c.class_scores[i]);
    }
    const int kk = std::min(k, static_cast<int>(scored.size()));
    std::partial_sort(
        scored.begin(), scored.begin() + kk, scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    scored.resize(kk);
    return scored;
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

inline float similarity(const vision::Embedding& a, const vision::Embedding& b) {
    if (a.embedding.size() != b.embedding.size() || a.embedding.empty()) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.embedding.size(); ++i) {
        dot += a.embedding[i] * b.embedding[i];
        na += a.embedding[i] * a.embedding[i];
        nb += b.embedding[i] * b.embedding[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0 ? dot / denom : 0.0f;
}

inline float distance(const vision::Embedding& a, const vision::Embedding& b) {
    if (a.embedding.size() != b.embedding.size() || a.embedding.empty()) {
        return std::numeric_limits<float>::max();
    }
    float sum = 0.0f;
    for (size_t i = 0; i < a.embedding.size(); ++i) {
        float d = a.embedding[i] - b.embedding[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

}  // namespace vision_common

#endif  // VISION_GEOMETRY_H
