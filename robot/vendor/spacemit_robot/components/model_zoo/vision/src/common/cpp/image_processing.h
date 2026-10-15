/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace vision_common {

/**
 * @brief Letterbox preprocessing for YOLO models
 * @param image Input image in BGR format
 * @param target_size Target size (height, width)
 * @param pad_color Padding color (BGR)
 * @return Resized and padded image
 */
cv::Mat letterbox(
    const cv::Mat& image,
    const std::pair<int, int>& target_size,
    const cv::Scalar& pad_color = cv::Scalar(114, 114, 114)
);

/**
 * @brief Preprocess image for classification models
 * @param image Input image in BGR format
 * @param input_shape Target input shape (height, width)
 * @param mean Mean values for normalization (default: ImageNet mean)
 * @param std Std values for normalization (default: ImageNet std)
 * @param resize_size Resize to this size before cropping (if (0,0), skip resize)
 * @param center_crop Whether to center crop
 * @param interpolation OpenCV interpolation flag for resize (default INTER_LINEAR)
 * @return Preprocessed tensor ready for inference
 */
cv::Mat preprocess_classification(
    const cv::Mat& image,
    const std::pair<int, int>& input_shape,
    const cv::Scalar& mean = cv::Scalar(0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f),
    const cv::Scalar& std = cv::Scalar(0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f),
    const cv::Size& resize_size = cv::Size(0, 0),
    bool center_crop = true,
    int interpolation = cv::INTER_LINEAR
);

/**
 * @brief Load labels from file (one label per line)
 * @param label_file Path to label file
 * @return Vector of label strings
 */
std::vector<std::string> load_labels(const std::string& label_file);

/**
 * @brief Resolve relative path for resource files (e.g. when running from build dir).
 * Tries path as-is, then "../" + path. Absolute paths unchanged.
 * @param path Path from config (e.g. "assets/labels/imagenet.txt")
 * @return Path that exists, or original path
 */
std::string resolve_path_for_resource(const std::string& path);

/**
 * @brief Load labels from ImageNet-format file (first token per line is WordNet ID, rest is name).
 * @param label_file Path to label file (e.g. imagenet.txt)
 * @return Vector of display names (one per line, WordNet ID stripped)
 */
std::vector<std::string> load_labels_imagenet(const std::string& label_file);

}  // namespace vision_common

#endif  // IMAGE_PROCESSING_H

