/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Internal inference request/response types (not installed to include/).
 */

#ifndef VISION_INFER_TYPES_H
#define VISION_INFER_TYPES_H

#include <string>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

#include "common/cpp/datatype.h"

namespace vision_core {

struct ImageInput {
    cv::Mat image;
};

struct SequenceInput {
    std::vector<float> pts;
    int image_width = 0;
    int image_height = 0;
};

struct TextInput {
    std::string text;
};

using InferInput = std::variant<ImageInput, SequenceInput, TextInput>;

enum class InferIntent {
    kDetect,
    kClassify,
    kEstimatePose,
    kSegment,
    kTrack,
    kEmbed,
    kEmbedText,
    kInferSequence,
    kOcr,
};

// Unified inference-time parameters (field <= 0 means "use model default").
struct InferParams {
    float conf_threshold = -1.f;
    float iou_threshold = -1.f;

    int top_k = -1;
    float kp_threshold = -1.f;
    float mask_threshold = -1.f;
    int max_det = -1;

    // Open-vocabulary text prompts (e.g. YOLO-World). Empty means "use the
    // model's configured default vocabulary". Ignored by models that don't
    // consume text.
    std::vector<std::string> prompts;
};

struct InferRequest {
    InferInput input;
    InferIntent intent;
    InferParams params;
};

struct InferResponse {
    std::vector<vision_common::ModelResult> results;
    bool ok = true;
    std::string error_message;
};

}  // namespace vision_core

#endif  // VISION_INFER_TYPES_H
