/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MPP_EXAMPLE_HELPERS_H
#define MPP_EXAMPLE_HELPERS_H

#include <cstdlib>
#include <string>

#include "mpp_frame_source.h"

namespace vision_mpp {

inline bool ParseMppArgs(int argc, char** argv, int camera_id, MppFrameSourceConfig* cfg) {
    if (cfg == nullptr) return false;
    cfg->camera_id = camera_id;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--use-mpp") {
            cfg->use_mpp = true;
        } else if (arg == "--mpp-vi") {
            cfg->use_vi = true;
        } else if (arg == "--mpp-format" && i + 1 < argc) {
            cfg->format = argv[++i];
        } else if (arg == "--mpp-width" && i + 1 < argc) {
            cfg->width = std::atoi(argv[++i]);
        } else if (arg == "--mpp-height" && i + 1 < argc) {
            cfg->height = std::atoi(argv[++i]);
        } else if (arg == "--mpp-fps" && i + 1 < argc) {
            cfg->fps = std::atoi(argv[++i]);
        } else if (arg == "--mpp-timeout" && i + 1 < argc) {
            cfg->timeout_ms = std::atoi(argv[++i]);
        } else if (arg == "--v4l2-dev" && i + 1 < argc) {
            cfg->v4l2_dev = argv[++i];
        } else if (arg == "--mpp-chn" && i + 1 < argc) {
            cfg->vi_chn = std::atoi(argv[++i]);
        } else if (arg == "--mpp-cpu-color") {
            cfg->cpu_color = true;
        }
    }
    return cfg->use_mpp;
}

inline const char* MppUsage() {
    return "  --use-mpp             Use MPP camera backend instead of cv::VideoCapture\n"
            "  --mpp-vi              Use MPP VI/ISP path (default UVC)\n"
            "  --mpp-format <fmt>    UVC pixel format: MJPEG (default) | YUYV | NV12\n"
            "  --mpp-width <w>       MPP capture width (default 640)\n"
            "  --mpp-height <h>      MPP capture height (default 480)\n"
            "  --mpp-fps <f>         MPP capture fps (default 30)\n"
            "  --mpp-timeout <ms>    MPP frame timeout ms (default 1000)\n"
            "  --v4l2-dev <path>     Override /dev/video<camera_id>\n"
            "  --mpp-chn <id>        VI channel id (with --mpp-vi, default 0)\n"
            "  --mpp-cpu-color       Force CPU cvtColor for NV12->BGR (skip V2D)\n";
}

}  // namespace vision_mpp

#endif  // MPP_EXAMPLE_HELPERS_H
