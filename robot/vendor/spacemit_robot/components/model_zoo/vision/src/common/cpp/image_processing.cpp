/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_processing.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <filesystem>  // NOLINT(build/c++17)

namespace fs = std::filesystem;

namespace vision_common {

cv::Mat letterbox(
    const cv::Mat& image,
    const std::pair<int, int>& target_size,
    const cv::Scalar& pad_color) {
    int input_h = target_size.first;   // height
    int input_w = target_size.second;  // width

    int orig_h = image.rows;
    int orig_w = image.cols;

    // Calculate scale ratio (similar to Python: min(input_h/orig_h, input_w/orig_w))
    float r = std::min(static_cast<float>(input_h) / orig_h,
                        static_cast<float>(input_w) / orig_w);

    // Compute new dimensions
    int new_w = static_cast<int>(std::round(orig_w * r));
    int new_h = static_cast<int>(std::round(orig_h * r));

    // Compute padding (similar to Python: dw, dh = (input_w - new_w)/2, (input_h - new_h)/2)
    float dw = (input_w - new_w) / 2.0f;
    float dh = (input_h - new_h) / 2.0f;

    // Resize image if needed
    cv::Mat resized;
    if (orig_w != new_w || orig_h != new_h) {
        cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    // Add padding (similar to Python: top, bottom = round(dh-0.1), round(dh+0.1))
    int top = static_cast<int>(std::round(dh - 0.1));
    int bottom = static_cast<int>(std::round(dh + 0.1));
    int left = static_cast<int>(std::round(dw - 0.1));
    int right = static_cast<int>(std::round(dw + 0.1));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                        cv::BORDER_CONSTANT, pad_color);

    return padded;
}

cv::Mat preprocess_classification(
    const cv::Mat& image,
    const std::pair<int, int>& input_shape,
    const cv::Scalar& mean,
    const cv::Scalar& std,
    const cv::Size& resize_size,
    bool center_crop,
    int interpolation) {
    cv::Mat img = image;

    // Resize on uint8 first — much faster than resizing float32 data
    if (resize_size.width > 0 && resize_size.height > 0) {
        cv::resize(img, img, resize_size, 0, 0, interpolation);
    }

    // Center crop on uint8
    if (center_crop && (resize_size.width > 0 && resize_size.height > 0)) {
        int y0 = (img.rows - input_shape.first) / 2;
        int x0 = (img.cols - input_shape.second) / 2;
        img = img(cv::Rect(x0, y0, input_shape.second, input_shape.first)).clone();
    } else {
        cv::resize(img, img, cv::Size(input_shape.second, input_shape.first),
                    0, 0, interpolation);
    }

    // blobFromImage: BGR->RGB (swapRB), float conversion, 1/255 scale,
    // and HWC->CHW in one optimized call
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0,
                                            cv::Size(), cv::Scalar(),
                                            true, false, CV_32F);

    // Per-channel mean/std normalization directly on contiguous CHW memory
    const int channel_size = input_shape.first * input_shape.second;
    float* blob_data = blob.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        float* ch = blob_data + c * channel_size;
        const float m = static_cast<float>(mean[c]) / 255.0f;
        const float inv_s = 255.0f / static_cast<float>(std[c]);
        for (int i = 0; i < channel_size; ++i) {
            ch[i] = (ch[i] - m) * inv_s;
        }
    }

    return blob;
}

std::vector<std::string> load_labels(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);

    if (!file.is_open()) {
        return labels;  // Return empty vector if file cannot be opened
    }

    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace from line
        if (!line.empty()) {
            // Remove trailing whitespace
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (!line.empty()) {
                labels.push_back(line);
            }
        }
    }

    file.close();
    return labels;
}

std::string resolve_path_for_resource(const std::string& path) {
    if (path.empty() || (path.size() >= 1 && path[0] == '/') ||
        (path.size() >= 2 && path[1] == ':')) {
        return path;
    }
    if (fs::exists(path)) return path;
    std::string with_parent = "../" + path;
    if (fs::exists(with_parent)) return with_parent;
    return path;
}

std::vector<std::string> load_labels_imagenet(const std::string& label_file) {
    std::vector<std::string> labels;
    std::ifstream file(label_file);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty()) continue;
            // ImageNet format: "n01440764 tench, Tinca tinca" -> "tench, Tinca tinca"
            size_t pos = line.find(' ');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                labels.push_back(line.substr(pos + 1));
            } else {
                labels.push_back(line);
            }
        }
    }
    return labels;
}

}  // namespace vision_common

