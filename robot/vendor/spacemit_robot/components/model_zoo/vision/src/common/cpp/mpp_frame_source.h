/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MPP_FRAME_SOURCE_H
#define MPP_FRAME_SOURCE_H

#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace vision_mpp {

struct MppFrameSourceConfig {
    bool        use_mpp    = false;
    bool        use_vi     = false;
    bool        cpu_color  = false;
    int         camera_id  = 0;
    std::string v4l2_dev;
    int         width      = 640;
    int         height     = 480;
    int         fps        = 30;
    int         timeout_ms = 1000;
    std::string format     = "MJPEG";
    int vi_chn        = 0;
    int sensor_width  = 3864;
    int sensor_height = 2192;
    int mipi_lanes    = 4;
    int mbps          = 800;
};

class MppFrameSource {
public:
    explicit MppFrameSource(const MppFrameSourceConfig& cfg);
    ~MppFrameSource();

    bool open();
    bool read(cv::Mat* out_bgr);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision_mpp

#endif  // MPP_FRAME_SOURCE_H
