/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "embedding_utils.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace vision_common {

std::vector<float> normalize_embedding(const std::vector<float>& embedding) {
    // Compute L2 norm
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;

    // Normalize
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = embedding[i] / norm;
    }

    return normalized;
}

float compute_similarity(const std::vector<float>& embedding1,
                        const std::vector<float>& embedding2) {
    if (embedding1.size() != embedding2.size()) {
        throw std::runtime_error("Embedding dimensions do not match");
    }

    // Compute cosine similarity (dot product of normalized embeddings)
    float similarity = 0.0f;
    for (size_t i = 0; i < embedding1.size(); ++i) {
        similarity += embedding1[i] * embedding2[i];
    }

    return similarity;
}

}  // namespace vision_common

