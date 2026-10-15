/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "align_face.h"

#include <cmath>

namespace vision_common {

const cv::Point2f kArcFaceSrc5[5] = {
    cv::Point2f(38.2946f, 51.6963f),
    cv::Point2f(73.5318f, 51.5014f),
    cv::Point2f(56.0252f, 71.7366f),
    cv::Point2f(41.5493f, 92.3655f),
    cv::Point2f(70.7299f, 92.2041f),
};

namespace {

// Similarity transform (scale + rotation + translation), aligned with buffalo_l reference.
bool estimate_similarity_affine(const cv::Point2f src[5],
                                const cv::Point2f dst[5],
                                float matrix[6]) {
    float src_mean_x = 0.0f;
    float src_mean_y = 0.0f;
    float dst_mean_x = 0.0f;
    float dst_mean_y = 0.0f;
    for (int i = 0; i < 5; ++i) {
        src_mean_x += src[i].x;
        src_mean_y += src[i].y;
        dst_mean_x += dst[i].x;
        dst_mean_y += dst[i].y;
    }
    src_mean_x /= 5.0f;
    src_mean_y /= 5.0f;
    dst_mean_x /= 5.0f;
    dst_mean_y /= 5.0f;

    float src_var = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
    for (int i = 0; i < 5; ++i) {
        const float xs = src[i].x - src_mean_x;
        const float ys = src[i].y - src_mean_y;
        const float xd = dst[i].x - dst_mean_x;
        const float yd = dst[i].y - dst_mean_y;
        src_var += xs * xs + ys * ys;
        a += xd * xs + yd * ys;
        b += yd * xs - xd * ys;
    }
    if (src_var <= 1e-6f) {
        return false;
    }
    a /= src_var;
    b /= src_var;

    matrix[0] = a;
    matrix[1] = -b;
    matrix[2] = dst_mean_x - a * src_mean_x + b * src_mean_y;
    matrix[3] = b;
    matrix[4] = a;
    matrix[5] = dst_mean_y - b * src_mean_x - a * src_mean_y;
    return true;
}

}  // namespace

cv::Mat align_face_5pt(const cv::Mat& image,
                        const cv::Point2f landmarks[5],
                        int output_size) {
    if (image.empty() || output_size <= 0) {
        return {};
    }

    const float scale = static_cast<float>(output_size) / 112.0f;
    cv::Point2f dst[5];
    for (int i = 0; i < 5; ++i) {
        dst[i] = cv::Point2f(kArcFaceSrc5[i].x * scale, kArcFaceSrc5[i].y * scale);
    }

    float M[6];
    if (!estimate_similarity_affine(landmarks, dst, M)) {
        return {};
    }

    cv::Mat affine(2, 3, CV_32F, M);
    cv::Mat aligned;
    cv::warpAffine(image, aligned, affine, cv::Size(output_size, output_size),
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return aligned;
}

}  // namespace vision_common
