/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALIGN_FACE_H
#define ALIGN_FACE_H

#include <opencv2/opencv.hpp>

namespace vision_common {

/** InsightFace / ArcFace 5-point template for 112x112 aligned faces. */
extern const cv::Point2f kArcFaceSrc5[5];

/**
 * @brief Align a face using 5 landmarks to a square output (default 112x112).
 * @param image Source image (BGR).
 * @param landmarks Five facial keypoints in image coordinates.
 * @param output_size Output width/height (default 112).
 * @return Aligned face crop, or empty Mat on failure.
 */
cv::Mat align_face_5pt(const cv::Mat& image,
                        const cv::Point2f landmarks[5],
                        int output_size = 112);

}  // namespace vision_common

#endif  // ALIGN_FACE_H
