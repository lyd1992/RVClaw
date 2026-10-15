/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo_utils.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace vision_common {

std::tuple<float, float, float, float> dfl_decode(
    const float* boxes, int anchor_idx, int anchors,
    int grid_w, float scale_w, float scale_h,
    float scale2orign, int pad_w, int pad_h) {
    constexpr int kDflLen = 16;
    float xywh[4] = {0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        float exp_sum = 0.0f;
        size_t offset = i * kDflLen * anchors + anchor_idx;
        float exp_dfl[kDflLen];

        for (int dfl_idx = 0; dfl_idx < kDflLen; dfl_idx++) {
            exp_dfl[dfl_idx] = exp(boxes[offset]);
            exp_sum += exp_dfl[dfl_idx];
            offset += anchors;
        }

        offset = i * kDflLen * anchors + anchor_idx;  // Reset offset for weighted sum
        for (int dfl_idx = 0; dfl_idx < kDflLen; dfl_idx++) {
            xywh[i] += (exp_dfl[dfl_idx] / exp_sum) * dfl_idx;
            offset += anchors;
        }
    }

    int h_idx = anchor_idx / grid_w;
    int w_idx = anchor_idx % grid_w;

    float x1 = ((w_idx - xywh[0] + 0.5f) * scale_w - pad_w) / scale2orign;
    float y1 = ((h_idx - xywh[1] + 0.5f) * scale_h - pad_h) / scale2orign;
    float x2 = ((w_idx + xywh[2] + 0.5f) * scale_w - pad_w) / scale2orign;
    float y2 = ((h_idx + xywh[3] + 0.5f) * scale_h - pad_h) / scale2orign;

    return std::make_tuple(x1, y1, x2, y2);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void xywh2xyxy(const float* xywh, float* xyxy) {
    float x = xywh[0];
    float y = xywh[1];
    float w = xywh[2];
    float h = xywh[3];

    xyxy[0] = x - w / 2.0f;  // x1
    xyxy[1] = y - h / 2.0f;  // y1
    xyxy[2] = x + w / 2.0f;  // x2
    xyxy[3] = y + h / 2.0f;  // y2
}

void scale_coords(const cv::Size& img1_shape, float* coords, const cv::Size& img0_shape) {
    // Python: scale_coords(img1_shape, coords, img0_shape)
    // img1_shape: (height, width) - preprocessed image shape
    // img0_shape: (height, width) - original image shape
    // cv::Size is (width, height), so we need to convert

    // Python: gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    // img1_shape[0] = height, img1_shape[1] = width
    float gain = std::min(static_cast<float>(img1_shape.height) / img0_shape.height,
                        static_cast<float>(img1_shape.width) / img0_shape.width);

    // Python: pad =(img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    // pad[0] = pad_w, pad[1] = pad_h
    float pad_w = (img1_shape.width - img0_shape.width * gain) / 2.0f;
    float pad_h = (img1_shape.height - img0_shape.height * gain) / 2.0f;

    // Python: coords[[0, 2]] -= pad[0]  # x padding
    // Python: coords[[1, 3]] -= pad[1]  # y padding
    // Python: coords[:4] /= gain
    coords[0] = (coords[0] - pad_w) / gain;  // x1
    coords[1] = (coords[1] - pad_h) / gain;  // y1
    coords[2] = (coords[2] - pad_w) / gain;  // x2
    coords[3] = (coords[3] - pad_h) / gain;  // y2

    // Clip to image bounds (same as Python clip_boxes)
    coords[0] = std::max(0.0f, std::min(coords[0], static_cast<float>(img0_shape.width)));
    coords[1] = std::max(0.0f, std::min(coords[1], static_cast<float>(img0_shape.height)));
    coords[2] = std::max(0.0f, std::min(coords[2], static_cast<float>(img0_shape.width)));
    coords[3] = std::max(0.0f, std::min(coords[3], static_cast<float>(img0_shape.height)));
}

std::vector<std::vector<float>> flatten_yolo_tensor(
    const float* tensor,
    int batch, int channels, int height, int width) {
    std::vector<std::vector<float>> flattened;
    flattened.reserve(batch * height * width);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                std::vector<float> row;
                row.reserve(channels);
                size_t base_idx = b * channels * height * width +
                                    h * width * channels + w * channels;
                for (int c = 0; c < channels; ++c) {
                    row.push_back(tensor[base_idx + c]);
                }
                flattened.push_back(row);
            }
        }
    }

    return flattened;
}

std::vector<std::vector<float>> process_box_dfl(
    const float* position,
    const std::pair<int, int>& input_shape,
    int reg_max) {
    // position shape: (1, 4*reg_max, grid_h, grid_w)
    int grid_h = input_shape.first;
    int grid_w = input_shape.second;
    int channels = 4 * reg_max;

    // Create grid
    std::vector<std::vector<float>> grid;
    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            grid.push_back({static_cast<float>(w), static_cast<float>(h)});
        }
    }

    // Process DFL for each position
    std::vector<std::vector<float>> boxes;
    boxes.reserve(grid_h * grid_w);

    for (int h = 0; h < grid_h; ++h) {
        for (int w = 0; w < grid_w; ++w) {
            float box_xy[4] = {0, 0, 0, 0};

            // Process each coordinate (x, y, w, h)
            for (int coord = 0; coord < 4; ++coord) {
                float exp_sum = 0.0f;
                std::vector<float> exp_vals(reg_max);

                // Compute softmax
                size_t base_idx = coord * reg_max * grid_h * grid_w +
                                    h * grid_w * reg_max + w * reg_max;

                float max_val = position[base_idx];
                for (int i = 1; i < reg_max; ++i) {
                    if (position[base_idx + i] > max_val) {
                        max_val = position[base_idx + i];
                    }
                }

                for (int i = 0; i < reg_max; ++i) {
                    exp_vals[i] = std::exp(position[base_idx + i] - max_val);
                    exp_sum += exp_vals[i];
                }

                // Compute expected value
                for (int i = 0; i < reg_max; ++i) {
                    box_xy[coord] += (exp_vals[i] / exp_sum) * static_cast<float>(i);
                }
            }

            // Compute stride
            float stride_h = static_cast<float>(input_shape.first) / grid_h;
            float stride_w = static_cast<float>(input_shape.second) / grid_w;

            // Convert to xyxy format
            float x = (static_cast<float>(w) + 0.5f - box_xy[0]) * stride_w;
            float y = (static_cast<float>(h) + 0.5f - box_xy[1]) * stride_h;
            float x2 = (static_cast<float>(w) + 0.5f + box_xy[2]) * stride_w;
            float y2 = (static_cast<float>(h) + 0.5f + box_xy[3]) * stride_h;

            boxes.push_back({x, y, x2, y2});
        }
    }

    return boxes;
}

std::vector<std::vector<float>> scale_boxes_letterbox(
    const std::vector<std::vector<float>>& boxes,
    float ratio,
    const std::pair<float, float>& pad,
    const std::pair<int, int>& orig_shape) {
    std::vector<std::vector<float>> scaled_boxes;
    scaled_boxes.reserve(boxes.size());

    float dw = pad.first;
    float dh = pad.second;
    int orig_w = orig_shape.second;
    int orig_h = orig_shape.first;

    for (const auto& box : boxes) {
        if (box.size() < 4) continue;

        float x1 = (box[0] - dw) / ratio;
        float y1 = (box[1] - dh) / ratio;
        float x2 = (box[2] - dw) / ratio;
        float y2 = (box[3] - dh) / ratio;

        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        scaled_boxes.push_back({x1, y1, x2, y2});
    }

    return scaled_boxes;
}

}  // namespace vision_common

