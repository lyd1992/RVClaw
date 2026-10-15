// Copyright 2026 RVClaw contributors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/calib3d.hpp>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "helmet_compliance.h"
#include "vision_service.h"

namespace {

std::atomic<bool> running{true};
constexpr int kMediaOutputFd = 3;

void stop_handler(int) { running.store(false); }

struct Options {
    std::string config;
    std::string model;
    std::string device;
    std::string input_video;
    std::string ppe_config;
    std::string ppe_model;
    int width = 960;
    int height = 540;
    int fps = 30;
    int ppe_fps = 5;
    int meter_fps = 5;
    int jpeg_quality = 78;
    bool hardhat_only = false;
    bool ppe_only = false;
    bool head_roi_ppe = false;
    bool meter_only = false;
    std::string meter_type = "pointer";
    double meter_start_angle = 225.0;
    double meter_end_angle = 315.0;
    double meter_min = 0.0;
    double meter_max = 1.0;
    std::string meter_direction = "counterclockwise";
    double meter_roi_x = 0.0;
    double meter_roi_y = 0.0;
    double meter_roi_width = 1.0;
    double meter_roi_height = 1.0;
    double display_roi_x = 0.10;
    double display_roi_y = 0.15;
    double display_roi_width = 0.80;
    double display_roi_height = 0.45;
    int meter_decimal_places = 0;
    int meter_digit_count = 0;
    double meter_min_confidence = 0.58;
    int meter_consistency_count = 3;
    double meter_normal_min = 0.0;
    double meter_normal_max = 0.0;
    bool meter_has_normal_min = false;
    bool meter_has_normal_max = false;
    double meter_max_change = 0.0;
    bool meter_has_max_change = false;
    std::string meter_reference_image;
    std::string meter_trial_dir;
    double meter_marker_a_x = 0.0;
    double meter_marker_a_y = 0.0;
    double meter_marker_b_x = 0.0;
    double meter_marker_b_y = 0.0;
    bool meter_has_markers = false;
};

cv::Mat align_meter_frame(const cv::Mat& frame, const Options& options, bool* aligned_ok = nullptr) {
    if (aligned_ok) *aligned_ok = false;
    if (frame.empty() || options.meter_reference_image.empty() || !options.meter_has_markers) {
        return frame;
    }
    static std::string loaded_path;
    static cv::Mat reference_gray;
    if (loaded_path != options.meter_reference_image) {
        loaded_path = options.meter_reference_image;
        const cv::Mat reference = cv::imread(loaded_path, cv::IMREAD_GRAYSCALE);
        reference_gray = reference;
    }
    if (reference_gray.empty()) return frame;
    cv::Mat current_gray;
    if (frame.channels() == 1) {
        current_gray = frame;
    } else {
        cv::cvtColor(frame, current_gray, cv::COLOR_BGR2GRAY);
    }
    const std::array<cv::Point2f, 2> targets = {
        cv::Point2f(
            static_cast<float>(options.meter_marker_a_x * current_gray.cols),
            static_cast<float>(options.meter_marker_a_y * current_gray.rows)),
        cv::Point2f(
            static_cast<float>(options.meter_marker_b_x * current_gray.cols),
            static_cast<float>(options.meter_marker_b_y * current_gray.rows)),
    };
    const std::array<cv::Point2f, 2> reference_points = {
        cv::Point2f(
            static_cast<float>(options.meter_marker_a_x * reference_gray.cols),
            static_cast<float>(options.meter_marker_a_y * reference_gray.rows)),
        cv::Point2f(
            static_cast<float>(options.meter_marker_b_x * reference_gray.cols),
            static_cast<float>(options.meter_marker_b_y * reference_gray.rows)),
    };
    std::vector<cv::Point2f> source_points;
    std::vector<cv::Point2f> destination_points;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const cv::Point2f center = reference_points[index];
        const int half = std::max(8, std::min(reference_gray.cols, reference_gray.rows) / 24);
        const cv::Rect patch_box(
            std::max(0, static_cast<int>(center.x) - half),
            std::max(0, static_cast<int>(center.y) - half),
            std::min(reference_gray.cols, static_cast<int>(center.x) + half) -
                std::max(0, static_cast<int>(center.x) - half),
            std::min(reference_gray.rows, static_cast<int>(center.y) + half) -
                std::max(0, static_cast<int>(center.y) - half));
        if (patch_box.width < 8 || patch_box.height < 8 ||
            patch_box.width >= current_gray.cols || patch_box.height >= current_gray.rows) {
            return frame;
        }
        cv::Mat marker_template;
        const int scaled_width = std::max(
            8, static_cast<int>(std::round(patch_box.width *
                static_cast<double>(current_gray.cols) / reference_gray.cols)));
        const int scaled_height = std::max(
            8, static_cast<int>(std::round(patch_box.height *
                static_cast<double>(current_gray.rows) / reference_gray.rows)));
        cv::resize(reference_gray(patch_box), marker_template,
                   cv::Size(scaled_width, scaled_height), 0.0, 0.0,
                   cv::INTER_LINEAR);
        if (marker_template.cols >= current_gray.cols || marker_template.rows >= current_gray.rows) {
            return frame;
        }
        cv::Mat matches;
        cv::matchTemplate(current_gray, marker_template, matches,
                          cv::TM_CCOEFF_NORMED);
        double score = 0.0;
        cv::Point location;
        cv::minMaxLoc(matches, nullptr, &score, nullptr, &location);
        if (score < 0.45) return frame;
        source_points.emplace_back(
            static_cast<float>(location.x + marker_template.cols * 0.5),
            static_cast<float>(location.y + marker_template.rows * 0.5));
        destination_points.push_back(targets[index]);
    }
    if (source_points.size() != 2 ||
        cv::norm(source_points[0] - source_points[1]) < 12.0) return frame;
    cv::Mat inlier_mask;
    const cv::Mat transform = cv::estimateAffinePartial2D(
        source_points, destination_points, inlier_mask, cv::RANSAC, 5.0);
    if (transform.empty()) return frame;
    cv::Mat aligned;
    cv::warpAffine(frame, aligned, transform, frame.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);
    if (aligned_ok) *aligned_ok = true;
    return aligned;
}

struct CaptureState {
    std::mutex mutex;
    std::condition_variable ready;
    cv::Mat latest_frame;
    double captured_at = 0.0;
    std::uint64_t sequence = 0;
    bool failed = false;
    bool eof = false;
    std::string error;
};

class HardwareMjpegDecoder {
public:
    HardwareMjpegDecoder(std::string device, int width, int height, int fps)
        : device_(std::move(device)), width_(width), height_(height), fps_(fps) {}

    ~HardwareMjpegDecoder() { stop(); }

    bool start() {
        int output_pipe[2];
        if (pipe(output_pipe) != 0) {
            std::cerr << "Unable to create JPEG decoder pipe: "
                      << std::strerror(errno) << std::endl;
            return false;
        }

        child_pid_ = fork();
        if (child_pid_ < 0) {
            std::cerr << "Unable to start JPEG decoder: "
                      << std::strerror(errno) << std::endl;
            close(output_pipe[0]);
            close(output_pipe[1]);
            return false;
        }

        if (child_pid_ == 0) {
            close(output_pipe[0]);
            if (output_pipe[1] != kMediaOutputFd) {
                if (dup2(output_pipe[1], kMediaOutputFd) < 0) _exit(126);
                close(output_pipe[1]);
            }
            // stcodec/MPP writes diagnostic messages to stdout. Keep those
            // bytes away from the packed BGR frame stream on fd 3.
            if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) _exit(126);

            const std::string video_size =
                std::to_string(width_) + "x" + std::to_string(height_);
            const std::string framerate = std::to_string(fps_);
            execlp(
                "ffmpeg",
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-video_size",
                video_size.c_str(),
                "-framerate",
                framerate.c_str(),
                "-c:v",
                "mjpeg_stcodec",
                "-i",
                device_.c_str(),
                "-an",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:3",
                static_cast<char*>(nullptr));
            _exit(127);
        }

        close(output_pipe[1]);
        output_fd_ = output_pipe[0];
        std::cerr << "C70 hardware JPEG decoder started" << std::endl;
        return true;
    }

    bool read_frame(cv::Mat* frame) {
        frame->create(height_, width_, CV_8UC3);
        auto* data = frame->ptr<unsigned char>();
        std::size_t remaining = frame->total() * frame->elemSize();
        while (remaining > 0 && running.load()) {
            const ssize_t count = read(output_fd_, data, remaining);
            if (count > 0) {
                data += count;
                remaining -= static_cast<std::size_t>(count);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            return false;
        }
        return remaining == 0;
    }

    void stop() {
        if (output_fd_ >= 0) {
            close(output_fd_);
            output_fd_ = -1;
        }
        if (child_pid_ > 0) {
            kill(child_pid_, SIGTERM);
            int status = 0;
            waitpid(child_pid_, &status, 0);
            child_pid_ = -1;
        }
    }

private:
    std::string device_;
    int width_;
    int height_;
    int fps_;
    int output_fd_ = -1;
    pid_t child_pid_ = -1;
};

class HardwareMjpegEncoder {
public:
    HardwareMjpegEncoder(int width, int height, int fps)
        : width_(width), height_(height), fps_(fps) {}

    ~HardwareMjpegEncoder() { stop(); }

    bool start() {
        int input_pipe[2];
        int output_pipe[2];
        if (pipe(input_pipe) != 0) {
            std::cerr << "Unable to create JPEG encoder pipes: "
                      << std::strerror(errno) << std::endl;
            return false;
        }
        if (pipe(output_pipe) != 0) {
            std::cerr << "Unable to create JPEG encoder pipes: "
                      << std::strerror(errno) << std::endl;
            close(input_pipe[0]);
            close(input_pipe[1]);
            return false;
        }

        child_pid_ = fork();
        if (child_pid_ < 0) {
            std::cerr << "Unable to start JPEG encoder: "
                      << std::strerror(errno) << std::endl;
            close(input_pipe[0]);
            close(input_pipe[1]);
            close(output_pipe[0]);
            close(output_pipe[1]);
            return false;
        }

        if (child_pid_ == 0) {
            dup2(input_pipe[0], STDIN_FILENO);
            close(input_pipe[0]);
            close(input_pipe[1]);
            close(output_pipe[0]);
            if (output_pipe[1] != kMediaOutputFd) {
                if (dup2(output_pipe[1], kMediaOutputFd) < 0) _exit(126);
                close(output_pipe[1]);
            }
            // Keep stcodec/MPP stdout diagnostics out of the MJPEG byte stream.
            if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) _exit(126);

            const std::string video_size =
                std::to_string(width_) + "x" + std::to_string(height_);
            const std::string framerate = std::to_string(fps_);
            execlp(
                "ffmpeg",
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                video_size.c_str(),
                "-framerate",
                framerate.c_str(),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "mjpeg_stcodec",
                "-f",
                "mjpeg",
                "pipe:3",
                static_cast<char*>(nullptr));
            _exit(127);
        }

        close(input_pipe[0]);
        close(output_pipe[1]);
        input_fd_ = input_pipe[1];
        output_fd_ = output_pipe[0];
        output_thread_ = std::thread(&HardwareMjpegEncoder::forward_output, this);
        std::cerr << "C70 hardware JPEG encoder started" << std::endl;
        return true;
    }

    bool write_frame(const cv::Mat& frame) {
        const cv::Mat continuous = frame.isContinuous() ? frame : frame.clone();
        const auto* data = continuous.ptr<unsigned char>();
        std::size_t remaining = continuous.total() * continuous.elemSize();
        while (remaining > 0 && running.load()) {
            const ssize_t written = write(input_fd_, data, remaining);
            if (written > 0) {
                data += written;
                remaining -= static_cast<std::size_t>(written);
                continue;
            }
            if (written < 0 && errno == EINTR) continue;
            std::cerr << "Hardware JPEG encoder input failed: "
                      << std::strerror(errno) << std::endl;
            return false;
        }
        return remaining == 0;
    }

    void stop() {
        if (input_fd_ >= 0) {
            close(input_fd_);
            input_fd_ = -1;
        }
        if (child_pid_ > 0) {
            int status = 0;
            waitpid(child_pid_, &status, 0);
            child_pid_ = -1;
        }
        if (output_thread_.joinable()) output_thread_.join();
    }

private:
    void forward_output() {
        std::vector<unsigned char> buffer(65536);
        bool client_connected = true;
        while (true) {
            const ssize_t count = read(output_fd_, buffer.data(), buffer.size());
            if (count == 0) break;
            if (count < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (!client_connected) continue;

            const unsigned char* data = buffer.data();
            ssize_t remaining = count;
            while (remaining > 0) {
                const ssize_t written = write(STDOUT_FILENO, data, remaining);
                if (written > 0) {
                    data += written;
                    remaining -= written;
                    continue;
                }
                if (written < 0 && errno == EINTR) continue;
                client_connected = false;
                running.store(false);
                break;
            }
        }
        close(output_fd_);
        output_fd_ = -1;
    }

    int width_;
    int height_;
    int fps_;
    int input_fd_ = -1;
    int output_fd_ = -1;
    pid_t child_pid_ = -1;
    std::thread output_thread_;
};

bool parse_options(int argc, char** argv, Options* options) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--hardhat-only") {
            options->hardhat_only = true;
            continue;
        }
        if (arg == "--ppe-only") {
            options->ppe_only = true;
            continue;
        }
        if (arg == "--head-roi-ppe") {
            options->head_roi_ppe = true;
            continue;
        }
        if (arg == "--meter-only") {
            options->meter_only = true;
            continue;
        }
        if (index + 1 >= argc) {
            std::cerr << "Missing value for " << arg << std::endl;
            return false;
        }
        const std::string value = argv[++index];
        if (arg == "--config") options->config = value;
        else if (arg == "--model") options->model = value;
        else if (arg == "--device") options->device = value;
        else if (arg == "--input-video") options->input_video = value;
        else if (arg == "--ppe-config") options->ppe_config = value;
        else if (arg == "--ppe-model") options->ppe_model = value;
        else if (arg == "--width") options->width = std::stoi(value);
        else if (arg == "--height") options->height = std::stoi(value);
        else if (arg == "--fps") options->fps = std::stoi(value);
        else if (arg == "--ppe-fps") options->ppe_fps = std::stoi(value);
        else if (arg == "--meter-fps") options->meter_fps = std::stoi(value);
        // Accepted for compatibility with older launch commands. SafetyVision
        // upstream logic does not use a minimum person-size filter.
        else if (arg == "--min-person-height") (void)value;
        else if (arg == "--jpeg-quality") options->jpeg_quality = std::stoi(value);
        else if (arg == "--meter-type") options->meter_type = value;
        else if (arg == "--meter-start-angle") options->meter_start_angle = std::stod(value);
        else if (arg == "--meter-end-angle") options->meter_end_angle = std::stod(value);
        else if (arg == "--meter-min") options->meter_min = std::stod(value);
        else if (arg == "--meter-max") options->meter_max = std::stod(value);
        else if (arg == "--meter-direction") options->meter_direction = value;
        else if (arg == "--meter-roi-x") options->meter_roi_x = std::stod(value);
        else if (arg == "--meter-roi-y") options->meter_roi_y = std::stod(value);
        else if (arg == "--meter-roi-width") options->meter_roi_width = std::stod(value);
        else if (arg == "--meter-roi-height") options->meter_roi_height = std::stod(value);
        else if (arg == "--display-roi-x") options->display_roi_x = std::stod(value);
        else if (arg == "--display-roi-y") options->display_roi_y = std::stod(value);
        else if (arg == "--display-roi-width") options->display_roi_width = std::stod(value);
        else if (arg == "--display-roi-height") options->display_roi_height = std::stod(value);
        else if (arg == "--meter-decimal-places") options->meter_decimal_places = std::stoi(value);
        else if (arg == "--meter-digit-count") options->meter_digit_count = std::stoi(value);
        else if (arg == "--meter-min-confidence") options->meter_min_confidence = std::stod(value);
        else if (arg == "--meter-consistency-count") options->meter_consistency_count = std::stoi(value);
        else if (arg == "--meter-normal-min") { options->meter_normal_min = std::stod(value); options->meter_has_normal_min = true; }
        else if (arg == "--meter-normal-max") { options->meter_normal_max = std::stod(value); options->meter_has_normal_max = true; }
        else if (arg == "--meter-max-change") { options->meter_max_change = std::stod(value); options->meter_has_max_change = true; }
        else if (arg == "--meter-reference-image") options->meter_reference_image = value;
        else if (arg == "--meter-trial-dir") options->meter_trial_dir = value;
        else if (arg == "--meter-marker-a-x") options->meter_marker_a_x = std::stod(value);
        else if (arg == "--meter-marker-a-y") options->meter_marker_a_y = std::stod(value);
        else if (arg == "--meter-marker-b-x") options->meter_marker_b_x = std::stod(value);
        else if (arg == "--meter-marker-b-y") { options->meter_marker_b_y = std::stod(value); options->meter_has_markers = true; }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return false;
        }
    }
    const bool ppe_pair_is_valid = options->ppe_config.empty() == options->ppe_model.empty();
    const bool detector_mode_is_valid =
        !(options->hardhat_only && options->ppe_only) &&
        !(options->head_roi_ppe && (options->hardhat_only || options->ppe_only)) &&
        !(options->ppe_only && !options->ppe_config.empty()) &&
        (!options->head_roi_ppe || !options->ppe_config.empty());
    const bool source_is_valid =
        options->device.empty() != options->input_video.empty();
    const bool model_pair_is_valid =
        (options->meter_only && options->meter_type == "pointer") ||
        (!options->config.empty() && !options->model.empty());
    return model_pair_is_valid && source_is_valid &&
           options->fps > 0 && options->ppe_fps > 0 && options->meter_fps > 0 &&
           ppe_pair_is_valid &&
           detector_mode_is_valid && (!options->meter_only ||
                                      options->meter_type == "pointer" ||
                                      options->meter_type == "digital") &&
           options->meter_roi_x >= 0.0 && options->meter_roi_y >= 0.0 &&
           options->meter_roi_width > 0.0 && options->meter_roi_height > 0.0 &&
           options->meter_roi_x + options->meter_roi_width <= 1.0 &&
           options->meter_roi_y + options->meter_roi_height <= 1.0 &&
           options->display_roi_x >= 0.0 && options->display_roi_y >= 0.0 &&
           options->display_roi_width > 0.0 && options->display_roi_height > 0.0 &&
           options->display_roi_x + options->display_roi_width <= 1.0 &&
           options->display_roi_y + options->display_roi_height <= 1.0 &&
           options->meter_decimal_places >= 0 && options->meter_decimal_places <= 6 &&
           options->meter_digit_count >= 0 && options->meter_digit_count <= 16 &&
           options->meter_min_confidence >= 0.0 && options->meter_min_confidence <= 1.0 &&
           options->meter_consistency_count >= 1 && options->meter_consistency_count <= 10 &&
           (!options->meter_has_normal_min || !options->meter_has_normal_max ||
            options->meter_normal_min <= options->meter_normal_max) &&
           (!options->meter_has_max_change || options->meter_max_change >= 0.0);
}

void disable_dynamic_framerate(const std::string& device) {
    const int fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        std::cerr << "Warning: unable to open C70 controls: "
                  << std::strerror(errno) << std::endl;
        return;
    }

    v4l2_control control{};
    control.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
    control.value = 0;
    if (ioctl(fd, VIDIOC_S_CTRL, &control) < 0) {
        std::cerr << "Warning: unable to disable C70 dynamic framerate: "
                  << std::strerror(errno) << std::endl;
    } else {
        std::cerr << "C70 dynamic framerate disabled" << std::endl;
    }
    close(fd);
}

struct MeterReading {
    std::string status = "checking";
    std::string raw_text;
    std::string method;
    double value = 0.0;
    double angle_deg = 0.0;
    float confidence = 0.0f;
    bool valid = false;
};

bool apply_digital_format(MeterReading* reading, const Options& options) {
    if (reading == nullptr || !reading->valid) return false;
    std::string digits;
    for (const char character : reading->raw_text) {
        if (character >= '0' && character <= '9') digits.push_back(character);
    }
    if (options.meter_digit_count > 0 &&
        static_cast<int>(digits.size()) != options.meter_digit_count) {
        reading->valid = false;
        reading->status = "unreadable";
        return false;
    }
    if (options.meter_decimal_places > 0) {
        if (static_cast<int>(digits.size()) <= options.meter_decimal_places) {
            reading->valid = false;
            reading->status = "unreadable";
            return false;
        }
        const std::size_t decimal_at = digits.size() - options.meter_decimal_places;
        reading->raw_text = digits.substr(0, decimal_at) + "." + digits.substr(decimal_at);
        try {
            reading->value = std::stod(reading->raw_text);
        } catch (const std::exception&) {
            reading->valid = false;
            reading->status = "unreadable";
            return false;
        }
    }
    if (options.meter_has_normal_min && reading->value < options.meter_normal_min) {
        reading->status = "out_of_range";
    }
    if (options.meter_has_normal_max && reading->value > options.meter_normal_max) {
        reading->status = "out_of_range";
    }
    if (reading->confidence < options.meter_min_confidence) {
        reading->valid = false;
        reading->status = "unreadable";
        return false;
    }
    return true;
}

struct MeterTextItem {
    std::string text;
    float score = 0.0f;
    cv::Rect2f bbox;
    bool in_display = false;
};

struct DigitalMeterState {
    std::mutex mutex;
    MeterReading reading;
    std::size_t text_count = 0;
    std::vector<MeterTextItem> text_items;
    std::string raw_text;
    double inference_ms = 0.0;
    std::uint64_t inference_count = 0;
    bool failed = false;
    std::string error;
};

MeterReading read_pointer_meter(const cv::Mat& frame, const Options& options) {
    MeterReading output;
    if (frame.empty()) return output;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1.2,
                     std::max(20, frame.rows / 4), 100, 32,
                     std::max(12, frame.rows / 8), std::max(20, frame.rows / 2));
    if (circles.empty()) return output;
    const cv::Vec3f circle = *std::max_element(
        circles.begin(), circles.end(), [](const cv::Vec3f& lhs, const cv::Vec3f& rhs) {
            return lhs[2] < rhs[2];
        });
    const cv::Point2f center(circle[0], circle[1]);
    const float radius = circle[2];
    cv::Mat edges;
    cv::Canny(gray, edges, 60, 160);
    cv::Mat mask = cv::Mat::zeros(edges.size(), edges.type());
    cv::circle(mask, center, std::max(2, static_cast<int>(radius * .92f)), 255, cv::FILLED);
    edges &= mask;
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, std::max(12, static_cast<int>(radius * .22f)),
                    std::max(10.0, radius * .35), 8);
    cv::Point2f best_tip;
    double best_score = -1.0;
    for (const auto& line : lines) {
        const cv::Point2f a(line[0], line[1]);
        const cv::Point2f b(line[2], line[3]);
        const double da = cv::norm(a - center);
        const double db = cv::norm(b - center);
        const cv::Point2f near = da < db ? a : b;
        const cv::Point2f tip = da < db ? b : a;
        if (cv::norm(near - center) > radius * .30) continue;
        const double score = cv::norm(tip - near) / std::max(1.0f, radius);
        if (score > best_score) { best_score = score; best_tip = tip; }
    }
    if (best_score < 0.0) return output;
    const double angle = std::fmod(std::atan2(center.y - best_tip.y, best_tip.x - center.x) * 180.0 / CV_PI + 360.0, 360.0);
    const bool clockwise = options.meter_direction == "clockwise";
    const double span = clockwise
        ? std::fmod(options.meter_end_angle - options.meter_start_angle + 360.0, 360.0)
        : std::fmod(options.meter_start_angle - options.meter_end_angle + 360.0, 360.0);
    const double offset = clockwise
        ? std::fmod(angle - options.meter_start_angle + 360.0, 360.0)
        : std::fmod(options.meter_start_angle - angle + 360.0, 360.0);
    const double ratio = std::clamp(offset / std::max(span, 1e-6), 0.0, 1.0);
    output.status = "stable";
    output.method = "pointer_geometry";
    output.value = options.meter_min + ratio * (options.meter_max - options.meter_min);
    output.angle_deg = angle;
    output.confidence = static_cast<float>(std::min(0.99, 0.45 + best_score / 4.0));
    output.valid = true;
    return output;
}

MeterReading stabilize_meter_reading(
    const MeterReading& current,
    bool digital,
    std::deque<MeterReading>* history,
    int required_count = 3,
    double max_change = 0.0,
    bool has_max_change = false) {
    constexpr std::size_t kWindow = 5;
    const std::size_t kRequired = static_cast<std::size_t>(
        std::clamp(required_count, 1, static_cast<int>(kWindow)));
    history->push_back(current);
    while (history->size() > kWindow) history->pop_front();
    std::vector<MeterReading> valid;
    for (const auto& item : *history) if (item.valid) valid.push_back(item);
    if (valid.size() < kRequired) {
        MeterReading pending = current;
        pending.valid = false;
        pending.status = history->size() == kWindow && digital ? "unreadable" : "checking";
        return pending;
    }
    std::sort(valid.begin(), valid.end(), [](const MeterReading& lhs, const MeterReading& rhs) {
        return lhs.value < rhs.value;
    });
    const double tolerance = digital ? 1e-3 : 0.02;
    if (valid.back().value - valid.front().value > tolerance) {
        MeterReading pending = current;
        pending.valid = false;
        pending.status = "checking";
        return pending;
    }
    if (has_max_change && valid.back().value - valid.front().value > max_change) {
        MeterReading pending = current;
        pending.valid = false;
        pending.status = "checking";
        return pending;
    }
    MeterReading stable = valid[valid.size() / 2];
    stable.status = std::any_of(
        valid.begin(), valid.end(), [](const MeterReading& item) {
            return item.status == "out_of_range";
        }) ? "out_of_range" : "stable";
    stable.valid = true;
    return stable;
}

MeterReading read_digital_meter(
    const VisionServiceResponse& response,
    const cv::Size& image_size,
    const cv::Rect& meter_roi,
    const cv::Size& frame_size,
    const Options& options,
    std::vector<MeterTextItem>* text_items) {
    MeterReading output;
    static const std::regex numeric_pattern(R"([-+]?\d+(?:\.\d+)?)");
    float best_score = -1.0f;
    text_items->clear();
    for (const auto& item : response.results) {
        const auto* text = std::get_if<vision::Text>(&item);
        if (!text || text->polygon.empty()) continue;
        float min_x = static_cast<float>(image_size.width);
        float min_y = static_cast<float>(image_size.height);
        float max_x = 0.0f;
        float max_y = 0.0f;
        for (const auto& point : text->polygon) {
            min_x = std::min(min_x, point.x);
            min_y = std::min(min_y, point.y);
            max_x = std::max(max_x, point.x);
            max_y = std::max(max_y, point.y);
        }
        const float normalized_x = min_x / std::max(1, image_size.width);
        const float normalized_y = min_y / std::max(1, image_size.height);
        const float normalized_width = (max_x - min_x) / std::max(1, image_size.width);
        const float normalized_height = (max_y - min_y) / std::max(1, image_size.height);
        const float center_x = normalized_x + normalized_width * 0.5f;
        const float center_y = normalized_y + normalized_height * 0.5f;
        const bool in_display =
            center_x >= options.display_roi_x &&
            center_x <= options.display_roi_x + options.display_roi_width &&
            center_y >= options.display_roi_y &&
            center_y <= options.display_roi_y + options.display_roi_height;
        text_items->push_back({
            text->text,
            text->score,
            cv::Rect2f(
                (meter_roi.x + normalized_x * meter_roi.width) /
                    std::max(1, frame_size.width),
                (meter_roi.y + normalized_y * meter_roi.height) /
                    std::max(1, frame_size.height),
                normalized_width * meter_roi.width / std::max(1, frame_size.width),
                normalized_height * meter_roi.height / std::max(1, frame_size.height)),
            in_display,
        });
        if (!in_display || text->score < best_score) continue;
        std::smatch match;
        std::string normalized = text->text;
        for (char& character : normalized) {
            if (character == 'O' || character == 'o') character = '0';
            else if (character == 'I' || character == 'l') character = '1';
            else if (character == ',') character = '.';
        }
        if (!std::regex_search(normalized, match, numeric_pattern)) continue;
        try {
            output.value = std::stod(match.str());
        } catch (const std::exception&) {
            continue;
        }
        best_score = text->score;
        output.status = "stable";
        output.confidence = text->score;
        output.raw_text = match.str();
        output.method = "ppocr";
        output.valid = true;
    }
    if (!output.valid) output.status = "unreadable";
    if (output.valid) apply_digital_format(&output, options);
    return output;
}

struct SevenSegmentDigit {
    cv::Rect box;
    char value = '?';
    float confidence = 0.0f;
};

std::pair<char, float> decode_seven_segment_digit(const cv::Mat& mask, const cv::Rect& box) {
    const cv::Rect clipped = box & cv::Rect(0, 0, mask.cols, mask.rows);
    if (clipped.width < 5 || clipped.height < 8) return {'?', 0.0f};
    const cv::Mat glyph = mask(clipped);
    if (static_cast<double>(clipped.width) / clipped.height < 0.42) {
        return {'1', 0.82f};
    }
    static const std::array<std::array<float, 7>, 10> segments = {{
        {{1, 1, 1, 1, 1, 1, 0}}, // 0
        {{0, 1, 1, 0, 0, 0, 0}}, // 1
        {{1, 1, 0, 1, 1, 0, 1}}, // 2
        {{1, 1, 1, 1, 0, 0, 1}}, // 3
        {{0, 1, 1, 0, 0, 1, 1}}, // 4
        {{1, 0, 1, 1, 0, 1, 1}}, // 5
        {{1, 0, 1, 1, 1, 1, 1}}, // 6
        {{1, 1, 1, 0, 0, 0, 0}}, // 7
        {{1, 1, 1, 1, 1, 1, 1}}, // 8
        {{1, 1, 1, 1, 0, 1, 1}}, // 9
    }};
    const std::array<cv::Rect, 7> regions = {{
        cv::Rect(static_cast<int>(clipped.width * .25), static_cast<int>(clipped.height * .03),
                 std::max(1, static_cast<int>(clipped.width * .50)),
                 std::max(1, static_cast<int>(clipped.height * .14))),
        cv::Rect(static_cast<int>(clipped.width * .78), static_cast<int>(clipped.height * .16),
                 std::max(1, static_cast<int>(clipped.width * .18)),
                 std::max(1, static_cast<int>(clipped.height * .28))),
        cv::Rect(static_cast<int>(clipped.width * .78), static_cast<int>(clipped.height * .60),
                 std::max(1, static_cast<int>(clipped.width * .18)),
                 std::max(1, static_cast<int>(clipped.height * .28))),
        cv::Rect(static_cast<int>(clipped.width * .25), static_cast<int>(clipped.height * .83),
                 std::max(1, static_cast<int>(clipped.width * .50)),
                 std::max(1, static_cast<int>(clipped.height * .14))),
        cv::Rect(static_cast<int>(clipped.width * .04), static_cast<int>(clipped.height * .60),
                 std::max(1, static_cast<int>(clipped.width * .18)),
                 std::max(1, static_cast<int>(clipped.height * .28))),
        cv::Rect(static_cast<int>(clipped.width * .04), static_cast<int>(clipped.height * .16),
                 std::max(1, static_cast<int>(clipped.width * .18)),
                 std::max(1, static_cast<int>(clipped.height * .28))),
        cv::Rect(static_cast<int>(clipped.width * .25), static_cast<int>(clipped.height * .43),
                 std::max(1, static_cast<int>(clipped.width * .50)),
                 std::max(1, static_cast<int>(clipped.height * .14))),
    }};
    std::array<float, 7> occupancy{};
    for (std::size_t index = 0; index < regions.size(); ++index) {
        const cv::Rect region = regions[index] & cv::Rect(0, 0, glyph.cols, glyph.rows);
        occupancy[index] = region.area() > 0
            ? static_cast<float>(cv::countNonZero(glyph(region))) / region.area()
            : 0.0f;
    }
    int best_digit = -1;
    float best_score = -1.0f;
    for (int digit = 0; digit < 10; ++digit) {
        float score = 0.0f;
        for (std::size_t index = 0; index < occupancy.size(); ++index) {
            const float observed = occupancy[index] >= .55f ? 1.0f : 0.0f;
            score += observed == segments[digit][index] ? 1.0f : 0.0f;
        }
        score /= static_cast<float>(occupancy.size());
        if (score > best_score) {
            best_score = score;
            best_digit = digit;
        }
    }
    return best_digit >= 0 ? std::make_pair(static_cast<char>('0' + best_digit), best_score)
                           : std::make_pair('?', 0.0f);
}

MeterReading read_seven_segment_meter(
    const cv::Mat& frame, const cv::Rect& roi, const Options& options) {
    MeterReading output;
    if (frame.empty() || roi.width < 20 || roi.height < 20) return output;
    cv::Mat enlarged;
    cv::resize(frame(roi), enlarged, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);
    cv::Mat hsv;
    cv::cvtColor(enlarged, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(0, 70, 55), cv::Scalar(18, 255, 255), mask);
    cv::Mat red_high;
    cv::inRange(hsv, cv::Scalar(165, 70, 55), cv::Scalar(179, 255, 255), red_high);
    mask |= red_high;
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    const int min_height = std::max(12, static_cast<int>(enlarged.rows * .04));
    const int max_height = static_cast<int>(enlarged.rows * .18);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<SevenSegmentDigit> candidates;
    for (const auto& contour : contours) {
        const cv::Rect box = cv::boundingRect(contour);
        if (box.height < min_height || box.height > max_height || box.width < 5) continue;
        if (box.area() < static_cast<int>(box.width * box.height * .10)) continue;
        std::vector<cv::Rect> parts{box};
        if (box.width > static_cast<int>(box.height * 1.05)) {
            const cv::Mat local = mask(box);
            const int center = local.cols / 2;
            int split = center;
            int best_column = INT_MAX;
            const int left = std::max(1, static_cast<int>(local.cols * .35));
            const int right = std::min(local.cols - 2, static_cast<int>(local.cols * .65));
            for (int column = left; column <= right; ++column) {
                const int score = cv::countNonZero(local.col(column));
                if (score < best_column) {
                    best_column = score;
                    split = column;
                }
            }
            if (split > 4 && box.width - split > 4) {
                parts = {
                    cv::Rect(box.x, box.y, split, box.height),
                    cv::Rect(box.x + split, box.y, box.width - split, box.height),
                };
            }
        }
        for (const auto& part : parts) {
            const auto decoded = decode_seven_segment_digit(mask, part);
            if (decoded.first != '?' && decoded.second >= .52f) {
                candidates.push_back({part, decoded.first, decoded.second});
            }
        }
    }
    if (candidates.empty()) return output;
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        const int lhs_center = lhs.box.y + lhs.box.height / 2;
        const int rhs_center = rhs.box.y + rhs.box.height / 2;
        return lhs_center == rhs_center ? lhs.box.x < rhs.box.x : lhs_center < rhs_center;
    });
    std::vector<std::vector<SevenSegmentDigit>> rows;
    for (const auto& candidate : candidates) {
        const int center = candidate.box.y + candidate.box.height / 2;
        bool placed = false;
        for (auto& row : rows) {
            const int row_center = row.front().box.y + row.front().box.height / 2;
            if (std::abs(center - row_center) <=
                std::max(18, static_cast<int>(candidate.box.height * .55))) {
                row.push_back(candidate);
                placed = true;
                break;
            }
        }
        if (!placed) rows.push_back({candidate});
    }
    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.front().box.y < rhs.front().box.y;
    });
    std::vector<std::string> readings;
    float confidence_sum = 0.0f;
    int confidence_count = 0;
    for (auto& row : rows) {
        if (row.size() < 2) continue;
        std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.box.x < rhs.box.x;
        });
        const int row_height = row.front().box.height;
        std::string current;
        for (std::size_t index = 0; index < row.size(); ++index) {
            if (index > 0) {
                const int gap = row[index].box.x - (row[index - 1].box.x + row[index - 1].box.width);
                if (gap > static_cast<int>(row_height * .45) && !current.empty()) {
                    if (current.size() >= 4) readings.push_back(current);
                    current.clear();
                }
            }
            current.push_back(row[index].value);
            confidence_sum += row[index].confidence;
            ++confidence_count;
        }
        if (current.size() >= 4) readings.push_back(current);
    }
    if (readings.empty()) return output;
    output.raw_text = readings.front();
    output.method = "seven_segment";
    try {
        output.value = std::stod(readings.front());
    } catch (const std::exception&) {
        return MeterReading();
    }
    output.status = "stable";
    output.confidence = confidence_count > 0 ? confidence_sum / confidence_count : 0.0f;
    output.valid = true;
    apply_digital_format(&output, options);
    return output;
}

cv::Mat prepare_digital_meter_roi(const cv::Mat& frame, const cv::Rect& roi) {
    const cv::Mat source = frame(roi);
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
    cv::Mat lab;
    cv::cvtColor(resized, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    cv::Mat enhanced;
    cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    return enhanced;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        if (character == '\n') {
            escaped += "\\n";
        } else if (character == '\r') {
            escaped += "\\r";
        } else if (character == '\t') {
            escaped += "\\t";
        } else {
            escaped.push_back(character);
        }
    }
    return escaped;
}

void save_meter_trial(const Options& options, const std::string& trial_id,
                      double captured_at, std::uint64_t sequence,
                      const cv::Mat& frame, const cv::Mat& crop,
                      const MeterReading& reading, bool alignment_ok,
                      double inference_ms) {
    const std::string prefix = options.meter_trial_dir + "/" + trial_id;
    if (!cv::imwrite(prefix + "-scene.jpg", frame) ||
        !cv::imwrite(prefix + "-crop.png", crop)) {
        throw std::runtime_error("Unable to save meter trial images");
    }
    const bool valid = alignment_ok && reading.valid;
    std::string reason;
    if (!alignment_ok) reason = "alignment_failed";
    else if (!reading.valid) {
        reason = reading.raw_text.empty() ? "no_digits" :
            (reading.confidence < options.meter_min_confidence ? "low_confidence" : "invalid_format");
    }
    const std::string status = !alignment_ok ? "alignment_failed" :
        (!valid ? "unreadable" : (reading.status == "out_of_range" ? "out_of_range" : "recognized"));
    std::ofstream output(prefix + ".json.tmp");
    output << std::setprecision(15)
           << "{\"trial_id\":\"" << trial_id << "\",\"captured_at\":" << captured_at
           << ",\"frame_sequence\":" << sequence
           << ",\"inference_ms\":" << inference_ms
           << ",\"status\":\"" << status << "\",\"reason\":\"" << reason
           << "\",\"value\":" << (valid ? std::to_string(reading.value) : "null")
           << ",\"raw_text\":\"" << json_escape(reading.raw_text)
           << "\",\"confidence\":" << reading.confidence
           << ",\"alignment_ok\":" << (alignment_ok ? "true" : "false")
           << ",\"single_frame\":true}";
    output.close();
    if (!output || std::rename((prefix + ".json.tmp").c_str(), (prefix + ".json").c_str()) != 0) {
        throw std::runtime_error("Unable to publish meter trial result");
    }
}

std::string meter_text_items_json(const std::vector<MeterTextItem>& items) {
    std::ostringstream output;
    output << "[";
    for (std::size_t index = 0; index < items.size(); ++index) {
        if (index > 0) output << ",";
        const auto& item = items[index];
        output << "{\"text\":\"" << json_escape(item.text)
               << "\",\"confidence\":" << std::fixed << std::setprecision(4)
               << item.score
               << ",\"bbox\":{\"x\":" << item.bbox.x
               << ",\"y\":" << item.bbox.y
               << ",\"width\":" << item.bbox.width
               << ",\"height\":" << item.bbox.height
               << "},\"in_display\":" << (item.in_display ? "true" : "false") << "}";
    }
    output << "]";
    return output.str();
}

void draw_meter_text_items(
    cv::Mat* image,
    const std::vector<MeterTextItem>& items) {
    for (const auto& item : items) {
        const cv::Rect box(
            static_cast<int>(std::round(item.bbox.x * image->cols)),
            static_cast<int>(std::round(item.bbox.y * image->rows)),
            std::max(1, static_cast<int>(std::round(item.bbox.width * image->cols))),
            std::max(1, static_cast<int>(std::round(item.bbox.height * image->rows))));
        const cv::Scalar color = item.in_display
            ? cv::Scalar(54, 205, 80)
            : cv::Scalar(235, 180, 40);
        cv::rectangle(*image, box, color, 2, cv::LINE_AA);
    }
}

// OpenCV's built-in Hershey fonts cannot render UTF-8 text. Keep the full
// OCR string in the API, but restrict the video overlay to readable ASCII.
std::string overlay_text(const std::string& value) {
    std::string output;
    bool omitted_non_ascii = false;
    for (const unsigned char character : value) {
        if (character >= 0x20 && character <= 0x7e) {
            output.push_back(static_cast<char>(character));
            omitted_non_ascii = false;
        } else if (!omitted_non_ascii) {
            if (!output.empty() && output.back() != ' ') output.push_back(' ');
            output += "[unicode]";
            omitted_non_ascii = true;
        }
    }
    return output;
}

cv::Ptr<cv::freetype::FreeType2> overlay_font() {
    static cv::Ptr<cv::freetype::FreeType2> font = [] {
        const char* configured_path = std::getenv("RVCLAW_OVERLAY_FONT");
        const std::string font_path = configured_path != nullptr
            ? configured_path
            : "/usr/share/fonts/opentype/noto/NotoSansSC-Regular.ttf";
        try {
            auto value = cv::freetype::createFreeType2();
            value->loadFontData(font_path, 0);
            return value;
        } catch (const cv::Exception& error) {
            std::cerr << "Unable to load UTF-8 overlay font " << font_path
                      << ": " << error.what() << std::endl;
            return cv::Ptr<cv::freetype::FreeType2>();
        }
    }();
    return font;
}

void draw_overlay_text(
    cv::Mat* image,
    const std::string& text,
    const cv::Point& origin,
    int font_height,
    const cv::Scalar& color) {
    const auto font = overlay_font();
    if (font) {
        font->putText(*image, text, origin, font_height, color, -1, cv::LINE_AA, false);
    } else {
        cv::putText(*image, overlay_text(text), origin, cv::FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv::LINE_AA);
    }
}

std::string meter_overlay_text(const std::string& raw_text) {
    const std::size_t ocr_marker = raw_text.find(" | OCR: ");
    return ocr_marker == std::string::npos ? raw_text : raw_text.substr(0, ocr_marker);
}

void draw_meter_result(cv::Mat* image, const MeterReading& reading) {
    if (!reading.valid) {
        cv::putText(*image, "METER CHECKING", cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX,
                    0.65, cv::Scalar(40, 210, 235), 2, cv::LINE_AA);
        if (!reading.raw_text.empty()) {
            draw_overlay_text(image, meter_overlay_text(reading.raw_text), cv::Point(12, 62),
                              26, cv::Scalar(40, 210, 235));
        }
        return;
    }
    std::ostringstream label;
    label << "METER " << std::fixed << std::setprecision(2) << reading.value
          << " angle=" << std::setprecision(1) << reading.angle_deg;
    cv::putText(*image, label.str(), cv::Point(12, 32), cv::FONT_HERSHEY_SIMPLEX,
                0.62, cv::Scalar(54, 205, 80), 2, cv::LINE_AA);
    if (!reading.raw_text.empty()) {
        draw_overlay_text(image, meter_overlay_text(reading.raw_text), cv::Point(12, 62),
                          26, cv::Scalar(54, 205, 80));
    }
}

std::size_t general_person_count(const VisionServiceResponse& response) {
    std::size_t count = 0;
    for (const auto& item : response.results) {
        if (vision::get_label(item) == 0) ++count;
    }
    return count;
}

std::vector<rvclaw::PersonTarget> tracked_people(
    const VisionServiceResponse& response) {
    std::vector<rvclaw::PersonTarget> people;
    for (const auto& item : response.results) {
        if (vision::get_label(item) != 0) continue;
        const int track_id = vision::get_track_id(item);
        if (track_id < 0) continue;
        const auto bbox = vision::get_bbox(item);
        people.push_back({
            {bbox.x1, bbox.y1, bbox.x2, bbox.y2},
            track_id,
            vision::get_score(item),
        });
    }
    return people;
}

std::vector<rvclaw::PersonTarget> safetyvision_people(
    const VisionServiceResponse& response) {
    std::vector<rvclaw::PersonTarget> people;
    for (const auto& item : response.results) {
        if (vision::get_label(item) != 11) continue;
        const auto bbox = vision::get_bbox(item);
        people.push_back({{bbox.x1, bbox.y1, bbox.x2, bbox.y2}});
    }
    return people;
}

std::vector<rvclaw::PpeDetection> ppe_detections(
    const VisionServiceResponse& response) {
    std::vector<rvclaw::PpeDetection> detections;
    for (const auto& item : response.results) {
        const int label = vision::get_label(item);
        if (label != 3 && label != 7) continue;
        const auto bbox = vision::get_bbox(item);
        detections.push_back({
            {bbox.x1, bbox.y1, bbox.x2, bbox.y2},
            label == 3 ? rvclaw::HelmetState::kHelmet
                       : rvclaw::HelmetState::kNoHelmet,
            vision::get_score(item),
        });
    }
    return detections;
}

std::vector<rvclaw::PpeResult> hardhat_results(
    const VisionServiceResponse& response) {
    std::vector<rvclaw::PpeResult> results;
    results.reserve(response.results.size());
    for (const auto& item : response.results) {
        const int label = vision::get_label(item);
        if (label != 0) continue;
        const auto bbox = vision::get_bbox(item);
        results.push_back({
            {bbox.x1, bbox.y1, bbox.x2, bbox.y2},
            rvclaw::HelmetState::kHelmet,
            vision::get_score(item),
        });
    }
    return results;
}

struct HeadTile {
    std::size_t person_index = 0;
    cv::Rect tile;
    cv::Rect content;
};

cv::Rect clipped_rect(const rvclaw::Box& box, const cv::Size& frame_size) {
    const int x1 = std::clamp(
        static_cast<int>(std::floor(box.x1)), 0, frame_size.width - 1);
    const int y1 = std::clamp(
        static_cast<int>(std::floor(box.y1)), 0, frame_size.height - 1);
    const int x2 = std::clamp(
        static_cast<int>(std::ceil(box.x2)), x1 + 1, frame_size.width);
    const int y2 = std::clamp(
        static_cast<int>(std::ceil(box.y2)), y1 + 1, frame_size.height);
    return {x1, y1, x2 - x1, y2 - y1};
}

cv::Mat build_head_mosaic(
    const cv::Mat& frame,
    const std::vector<rvclaw::PersonTarget>& people,
    std::vector<HeadTile>* mappings) {
    constexpr int kMosaicSize = 640;
    constexpr int kTileSize = kMosaicSize / 2;
    constexpr std::size_t kMaxPeople = 4;
    cv::Mat mosaic(kMosaicSize, kMosaicSize, CV_8UC3, cv::Scalar(114, 114, 114));
    mappings->clear();

    std::vector<std::size_t> order(people.size());
    for (std::size_t index = 0; index < people.size(); ++index) order[index] = index;
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return rvclaw::area(people[lhs].bbox) > rvclaw::area(people[rhs].bbox);
    });

    const std::size_t count = std::min(kMaxPeople, order.size());
    for (std::size_t slot = 0; slot < count; ++slot) {
        const std::size_t person_index = order[slot];
        if (!rvclaw::head_zone_visible(
                people[person_index].bbox,
                static_cast<float>(frame.cols),
                static_cast<float>(frame.rows))) {
            continue;
        }
        const auto zone = rvclaw::head_zone(
            people[person_index].bbox,
            static_cast<float>(frame.cols),
            static_cast<float>(frame.rows));
        const cv::Rect source = clipped_rect(zone, frame.size());
        if (source.empty()) continue;

        const std::size_t tile_slot = mappings->size();
        const cv::Rect tile(
            static_cast<int>(tile_slot % 2) * kTileSize,
            static_cast<int>(tile_slot / 2) * kTileSize,
            kTileSize,
            kTileSize);
        const double scale = std::min(
            static_cast<double>(kTileSize - 16) / source.width,
            static_cast<double>(kTileSize - 16) / source.height);
        const cv::Size resized_size(
            std::max(1, static_cast<int>(std::round(source.width * scale))),
            std::max(1, static_cast<int>(std::round(source.height * scale))));
        const cv::Rect content(
            tile.x + (tile.width - resized_size.width) / 2,
            tile.y + (tile.height - resized_size.height) / 2,
            resized_size.width,
            resized_size.height);
        cv::resize(frame(source), mosaic(content), resized_size, 0.0, 0.0, cv::INTER_LINEAR);
        mappings->push_back({person_index, tile, content});
    }
    return mosaic;
}

std::vector<float> hardhat_scores_by_person(
    const VisionServiceResponse& response,
    const std::vector<HeadTile>& mappings,
    const std::vector<rvclaw::PersonTarget>& people,
    const cv::Size& frame_size) {
    const std::size_t person_count = people.size();
    std::vector<float> scores(person_count, -1.0f);
    for (const auto& person : people) {
        if (!rvclaw::head_zone_visible(
                person.bbox,
                static_cast<float>(frame_size.width),
                static_cast<float>(frame_size.height))) {
            scores[&person - people.data()] = -2.0f;
        }
    }
    for (const auto& mapping : mappings) scores[mapping.person_index] = 0.0f;
    for (const auto& item : response.results) {
        if (vision::get_label(item) != 0) continue;
        const auto bbox = vision::get_bbox(item);
        const cv::Point center(
            static_cast<int>(std::round((bbox.x1 + bbox.x2) * 0.5f)),
            static_cast<int>(std::round((bbox.y1 + bbox.y2) * 0.5f)));
        for (const auto& mapping : mappings) {
            if (!mapping.content.contains(center)) continue;
            scores[mapping.person_index] = std::max(
                scores[mapping.person_index], vision::get_score(item));
            break;
        }
    }
    return scores;
}

cv::Scalar state_color(rvclaw::HelmetState state) {
    if (state == rvclaw::HelmetState::kHelmet) return cv::Scalar(54, 205, 80);
    if (state == rvclaw::HelmetState::kNoHelmet) return cv::Scalar(50, 50, 235);
    return cv::Scalar(40, 210, 235);
}

void draw_ppe_result(cv::Mat* image, const rvclaw::PpeResult& result) {
    const cv::Scalar color = state_color(result.state);
    const cv::Rect bounds(0, 0, image->cols, image->rows);
    const cv::Point top_left(
        std::max(0, static_cast<int>(std::round(result.bbox.x1))),
        std::max(0, static_cast<int>(std::round(result.bbox.y1))));
    const cv::Point bottom_right(
        std::min(image->cols - 1, static_cast<int>(std::round(result.bbox.x2))),
        std::min(image->rows - 1, static_cast<int>(std::round(result.bbox.y2))));
    cv::rectangle(*image, top_left, bottom_right, color, 3, cv::LINE_AA);

    std::ostringstream label;
    if (result.track_id >= 0) {
        label << "P-" << std::setw(2) << std::setfill('0') << result.track_id << " ";
    }
    label << rvclaw::helmet_state_text(result.state);
    if (result.state == rvclaw::HelmetState::kHelmet && result.score > 0.0f) {
        label << " " << std::fixed << std::setprecision(2) << result.score;
    }
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.58, 2, &baseline);
    const int text_x = top_left.x;
    const int text_y = std::max(text_size.height + 6, top_left.y - 6);
    const cv::Rect background =
        cv::Rect(text_x, text_y - text_size.height - 6, text_size.width + 10,
                 text_size.height + baseline + 8) & bounds;
    cv::rectangle(*image, background, color, cv::FILLED);
    cv::putText(
        *image,
        label.str(),
        cv::Point(text_x + 5, text_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.58,
        cv::Scalar(20, 20, 20),
        2,
        cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, &options)) {
        std::cerr << "Usage: yolov8_mjpeg --config YAML --model ONNX "
                     "(--device V4L2 | --input-video FILE) "
                     "[--hardhat-only | --ppe-only] "
                     "[--ppe-config YAML --ppe-model ONNX --ppe-fps 5 "
                     "--head-roi-ppe] "
                     "[--meter-only --meter-type pointer|digital --meter-fps 5] "
                     "[--width 960 --height 540 --fps 30 --jpeg-quality 78]"
                  << std::endl;
        return 2;
    }

    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    std::signal(SIGPIPE, stop_handler);

    std::unique_ptr<VisionService> service;
    if (!options.meter_only || options.meter_type == "digital") {
        service = VisionService::Create(options.config, options.model, false);
    }
    if ((!options.meter_only || options.meter_type == "digital") && !service) {
        std::cerr << "YOLOv8 model initialization failed: "
                  << VisionService::LastCreateError() << std::endl;
        return 3;
    }

    std::unique_ptr<VisionService> ppe_service;
    if (!options.meter_only && !options.ppe_config.empty()) {
        ppe_service = VisionService::Create(
            options.ppe_config, options.ppe_model, false);
        if (!ppe_service) {
            std::cerr << "PPE model initialization failed; general detection will continue: "
                      << VisionService::LastCreateError() << std::endl;
        } else {
            std::cerr << "RVCLAW_PPE {\"enabled\":true,\"target_fps\":"
                      << options.ppe_fps << ",\"mode\":\""
                      << (options.head_roi_ppe ? "tracked-head-roi-mosaic"
                                               : "serial-full-core")
                      << "\"}"
                      << std::endl;
        }
    }

    std::unique_ptr<HardwareMjpegDecoder> decoder;
    cv::VideoCapture video_capture;
    if (options.input_video.empty()) {
        disable_dynamic_framerate(options.device);
        decoder = std::make_unique<HardwareMjpegDecoder>(
            options.device, options.width, options.height, options.fps);
        if (!decoder->start()) {
            std::cerr << "Unable to open C70 camera: " << options.device << std::endl;
            return 4;
        }
        std::cerr << "C70_CAPTURE {\"width\":" << options.width
                  << ",\"height\":" << options.height
                  << ",\"fps\":" << options.fps
                  << ",\"decoder\":\"mjpeg_stcodec\"}" << std::endl;
    } else {
        if (!video_capture.open(options.input_video)) {
            std::cerr << "Unable to open input video: "
                      << options.input_video << std::endl;
            return 4;
        }
        std::cerr << "VIDEO_FILE_CAPTURE {\"path\":\""
                  << options.input_video << "\",\"width\":" << options.width
                  << ",\"height\":" << options.height
                  << ",\"fps\":" << options.fps << "}" << std::endl;
    }
    HardwareMjpegEncoder encoder(options.width, options.height, options.fps);
    if (!encoder.start()) {
        if (decoder) decoder->stop();
        return 5;
    }

    std::uint64_t inference_count = 0;
    double ppe_inference_ms = 0.0;
    std::uint64_t ppe_inference_count = 0;
    int exit_code = 0;
    CaptureState capture_state;
    DigitalMeterState digital_meter_state;
    std::vector<rvclaw::PpeResult> latest_ppe_results;
    rvclaw::HelmetStateTracker helmet_tracker;
    std::deque<MeterReading> meter_history;

    // Keep V4L2 acquisition running independently from inference. The old
    // single loop waited for the next 30 fps camera frame and then performed
    // inference/drawing/JPEG encoding, which added the two stage durations and
    // reduced the annotated stream to about 15 fps. This producer always keeps
    // the newest frame so inference can overlap the next camera capture.
    std::thread capture_thread([&]() {
        auto next_frame = std::chrono::steady_clock::now();
        const auto frame_period = std::chrono::microseconds(
            1000000 / options.fps);
        while (running.load()) {
            cv::Mat frame;
            const bool read_ok = options.input_video.empty()
                ? decoder->read_frame(&frame)
                : video_capture.read(frame);
            if (!read_ok || frame.empty()) {
                std::lock_guard<std::mutex> lock(capture_state.mutex);
                if (options.input_video.empty()) {
                    capture_state.failed = true;
                    capture_state.error = "Unable to read a C70 frame";
                } else {
                    capture_state.eof = true;
                }
                capture_state.ready.notify_all();
                return;
            }
            if (!options.input_video.empty()) {
                if (frame.cols != options.width || frame.rows != options.height) {
                    cv::resize(
                        frame, frame, cv::Size(options.width, options.height),
                        0.0, 0.0, cv::INTER_LINEAR);
                }
                next_frame += frame_period;
            }
            {
                std::lock_guard<std::mutex> lock(capture_state.mutex);
                frame.copyTo(capture_state.latest_frame);
                capture_state.captured_at = std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                ++capture_state.sequence;
            }
            capture_state.ready.notify_all();
            if (!options.input_video.empty()) {
                std::this_thread::sleep_until(next_frame);
            }
        }
    });

    const bool async_digital_meter =
        options.meter_only && options.meter_type == "digital";
    std::thread digital_meter_thread;
    if (async_digital_meter) {
        digital_meter_thread = std::thread([&]() {
            std::uint64_t consumed_sequence = 0;
            std::string completed_trial;
            std::deque<MeterReading> history;
            auto next_inference = std::chrono::steady_clock::now();
            const auto inference_period = std::chrono::microseconds(
                1000000 / options.meter_fps);

            while (running.load()) {
                const auto now = std::chrono::steady_clock::now();
                if (now < next_inference) {
                    std::unique_lock<std::mutex> lock(capture_state.mutex);
                    capture_state.ready.wait_until(
                        lock,
                        next_inference,
                        [&]() {
                            return capture_state.failed || capture_state.eof ||
                                   !running.load();
                        });
                }

                std::string trial_id;
                double requested_at = 0.0;
                if (!options.meter_trial_dir.empty()) {
                    std::ifstream request(options.meter_trial_dir + "/request");
                    request >> trial_id >> requested_at;
                    if (!request || trial_id == completed_trial ||
                        !std::regex_match(trial_id, std::regex("[a-f0-9]{32}"))) trial_id.clear();
                }
                cv::Mat frame;
                double captured_at = 0.0;
                {
                    std::unique_lock<std::mutex> lock(capture_state.mutex);
                    capture_state.ready.wait_for(
                        lock,
                        std::chrono::milliseconds(250),
                        [&]() {
                            return capture_state.sequence != consumed_sequence ||
                                   capture_state.failed || capture_state.eof ||
                                   !running.load();
                        });
                    if (!running.load() || capture_state.failed) return;
                    if (capture_state.sequence == consumed_sequence) {
                        if (capture_state.eof) return;
                        continue;
                    }
                    capture_state.latest_frame.copyTo(frame);
                    captured_at = capture_state.captured_at;
                    consumed_sequence = capture_state.sequence;
                }
                if (!trial_id.empty() && captured_at < requested_at) continue;

                const auto started = std::chrono::steady_clock::now();
                bool alignment_ok = false;
                const cv::Mat aligned_frame = align_meter_frame(frame, options, &alignment_ok);
                const cv::Mat& analysis_frame = aligned_frame.empty() ? frame : aligned_frame;
                const cv::Rect meter_roi(
                    static_cast<int>(std::round(options.meter_roi_x * analysis_frame.cols)),
                    static_cast<int>(std::round(options.meter_roi_y * analysis_frame.rows)),
                    std::max(1, static_cast<int>(std::round(
                        options.meter_roi_width * analysis_frame.cols))),
                    std::max(1, static_cast<int>(std::round(
                        options.meter_roi_height * analysis_frame.rows))));
                const cv::Rect safe_meter_roi =
                    meter_roi & cv::Rect(0, 0, analysis_frame.cols, analysis_frame.rows);
                const cv::Mat ocr_frame = prepare_digital_meter_roi(analysis_frame, safe_meter_roi);
                VisionServiceResponse response;
                if (service->Infer(ocr_frame, &response) !=
                    VISION_SERVICE_OK) {
                    std::lock_guard<std::mutex> lock(digital_meter_state.mutex);
                    digital_meter_state.failed = true;
                    digital_meter_state.error =
                        "PP-OCRv5 inference failed: " + service->LastError();
                    capture_state.ready.notify_all();
                    return;
                }
                std::vector<MeterTextItem> text_items;
                MeterReading ocr_reading = read_digital_meter(
                    response, ocr_frame.size(), safe_meter_roi, frame.size(), options, &text_items);
                const cv::Rect display_roi(
                    safe_meter_roi.x + static_cast<int>(std::round(
                        options.display_roi_x * safe_meter_roi.width)),
                    safe_meter_roi.y + static_cast<int>(std::round(
                        options.display_roi_y * safe_meter_roi.height)),
                    std::max(1, static_cast<int>(std::round(
                        options.display_roi_width * safe_meter_roi.width))),
                    std::max(1, static_cast<int>(std::round(
                        options.display_roi_height * safe_meter_roi.height))));
                MeterReading seven_segment_reading =
                    read_seven_segment_meter(analysis_frame, display_roi, options);
                MeterReading current_reading = ocr_reading;
                if (seven_segment_reading.valid) {
                    current_reading = seven_segment_reading;
                }
                if (!trial_id.empty()) {
                    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    try {
                        save_meter_trial(options, trial_id, captured_at, consumed_sequence,
                                         frame, ocr_frame, current_reading, alignment_ok, elapsed_ms);
                    } catch (const std::exception& error) {
                        std::cerr << "Meter trial failed: " << error.what() << std::endl;
                    }
                    completed_trial = trial_id;
                }
                MeterReading reading = stabilize_meter_reading(
                    current_reading, true, &history,
                    options.meter_consistency_count,
                    options.meter_max_change,
                    options.meter_has_max_change);
                const auto finished = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lock(digital_meter_state.mutex);
                    digital_meter_state.reading = reading;
                    digital_meter_state.text_count = response.results.size();
                    digital_meter_state.text_items = std::move(text_items);
                    digital_meter_state.raw_text = reading.raw_text;
                    digital_meter_state.inference_ms =
                        std::chrono::duration<double, std::milli>(
                            finished - started).count();
                    ++digital_meter_state.inference_count;
                }
                next_inference = started + inference_period;
            }
        });
    }

    std::uint64_t consumed_sequence = 0;

    while (running.load()) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(capture_state.mutex);
            capture_state.ready.wait_for(
                lock,
                std::chrono::milliseconds(250),
                [&]() {
                    return capture_state.sequence != consumed_sequence ||
                           capture_state.failed || capture_state.eof ||
                           !running.load();
                });
            if (!running.load()) break;
            if (capture_state.failed) {
                std::cerr << capture_state.error << std::endl;
                exit_code = 5;
                break;
            }
            if (capture_state.sequence == consumed_sequence) {
                if (capture_state.eof) break;
                continue;
            }
            capture_state.latest_frame.copyTo(frame);
            consumed_sequence = capture_state.sequence;
        }

        const auto started = std::chrono::steady_clock::now();
        VisionServiceResponse response;
        MeterReading meter_reading;
        std::vector<MeterTextItem> meter_text_items;
        std::size_t detection_count = 0;
        const cv::Rect meter_roi(
            static_cast<int>(std::round(options.meter_roi_x * frame.cols)),
            static_cast<int>(std::round(options.meter_roi_y * frame.rows)),
            std::max(1, static_cast<int>(std::round(options.meter_roi_width * frame.cols))),
            std::max(1, static_cast<int>(std::round(options.meter_roi_height * frame.rows))));
        const cv::Rect safe_meter_roi = meter_roi & cv::Rect(0, 0, frame.cols, frame.rows);
        const cv::Mat meter_frame = options.meter_only ? frame(safe_meter_roi) : frame;
        if (options.meter_only && options.meter_type == "pointer") {
            meter_reading = read_pointer_meter(meter_frame, options);
        } else if (async_digital_meter) {
            std::lock_guard<std::mutex> lock(digital_meter_state.mutex);
            if (digital_meter_state.failed) {
                std::cerr << digital_meter_state.error << std::endl;
                exit_code = 5;
                break;
            }
            meter_reading = digital_meter_state.reading;
            detection_count = digital_meter_state.text_count;
            meter_text_items = digital_meter_state.text_items;
        } else if (service->Infer(frame, &response) != VISION_SERVICE_OK) {
            std::cerr << "YOLOv8 inference failed: " << service->LastError() << std::endl;
            exit_code = 5;
            break;
        }
        if (options.meter_only && !async_digital_meter) {
            meter_reading = stabilize_meter_reading(
                meter_reading, options.meter_type == "digital", &meter_history,
                options.meter_consistency_count,
                options.meter_max_change,
                options.meter_has_max_change);
        }

        cv::Mat annotated;
        std::size_t person_count = 0;
        std::vector<rvclaw::PersonTarget> people;
        if (options.meter_only) {
            annotated = frame;
            if (async_digital_meter) {
                draw_meter_text_items(&annotated, meter_text_items);
            }
            draw_meter_result(&annotated, meter_reading);
        } else if (options.ppe_only) {
            annotated = frame;
            const auto people = safetyvision_people(response);
            person_count = people.size();
            latest_ppe_results = rvclaw::resolve_safetyvision(
                people, ppe_detections(response));
        } else if (options.hardhat_only) {
            annotated = frame;
            latest_ppe_results = hardhat_results(response);
        } else if (options.head_roi_ppe) {
            annotated = frame;
            people = tracked_people(response);
            person_count = people.size();
            helmet_tracker.update_people(people);
            latest_ppe_results = helmet_tracker.results(people);
        } else if (!response.results.empty() &&
            service->Draw(frame, response, &annotated) != VISION_SERVICE_OK) {
            annotated = frame;
        } else if (response.results.empty()) {
            annotated = frame;
        }

        const auto finished = std::chrono::steady_clock::now();
        double inference_ms =
            std::chrono::duration<double, std::milli>(finished - started).count();
        if (async_digital_meter) {
            std::lock_guard<std::mutex> lock(digital_meter_state.mutex);
            inference_ms = digital_meter_state.inference_ms;
            inference_count = digital_meter_state.inference_count;
        } else {
            ++inference_count;
            detection_count = response.results.size();
        }
        if (options.ppe_only) {
            ppe_inference_ms = inference_ms;
            ppe_inference_count = inference_count;
        }

        if (!options.meter_only && !options.ppe_only && !options.hardhat_only && !options.head_roi_ppe) {
            person_count = general_person_count(response);
        }
        if (ppe_service && person_count > 0) {
            const int ppe_interval = std::max(1, options.fps / options.ppe_fps);
            const bool ppe_due =
                inference_count % static_cast<std::uint64_t>(ppe_interval) == 0 ||
                (options.head_roi_ppe && helmet_tracker.needs_confirmation(people));
            if (ppe_due) {
                // Both EP sessions stay loaded, but inference is serialized so
                // ByteTrack and the hard-hat detector never contend for K3
                // compute threads. All tracked head ROIs share one PPE pass.
                const auto ppe_started = std::chrono::steady_clock::now();
                VisionServiceResponse ppe_response;
                std::vector<HeadTile> head_tiles;
                const cv::Mat ppe_input = options.head_roi_ppe
                    ? build_head_mosaic(frame, people, &head_tiles)
                    : frame;
                const auto ppe_status = ppe_service->Infer(ppe_input, &ppe_response);
                const auto ppe_finished = std::chrono::steady_clock::now();
                ppe_inference_ms = std::chrono::duration<double, std::milli>(
                                       ppe_finished - ppe_started)
                                       .count();
                ++ppe_inference_count;
                if (ppe_status == VISION_SERVICE_OK) {
                    if (options.head_roi_ppe) {
                        helmet_tracker.observe_hardhats(
                            people,
                            hardhat_scores_by_person(
                                ppe_response, head_tiles, people, frame.size()));
                        latest_ppe_results = helmet_tracker.results(people);
                    } else {
                        latest_ppe_results = rvclaw::resolve_safetyvision(
                            safetyvision_people(ppe_response),
                            ppe_detections(ppe_response));
                    }
                } else {
                    std::cerr << "PPE inference failed: "
                              << ppe_service->LastError() << std::endl;
                }
            }
        } else if (!options.ppe_only && !options.hardhat_only && person_count == 0) {
            latest_ppe_results.clear();
        }

        int helmet_count = 0;
        int no_helmet_count = 0;
        int checking_count = 0;
        for (const auto& result : latest_ppe_results) {
            draw_ppe_result(&annotated, result);
            if (result.state == rvclaw::HelmetState::kHelmet) {
                ++helmet_count;
            } else if (result.state == rvclaw::HelmetState::kNoHelmet) {
                ++no_helmet_count;
            } else {
                ++checking_count;
            }
        }

        if (options.meter_only) {
            const std::string meter_pipeline = async_digital_meter
                ? "Astra Digital Meter / PP-OCRv5"
                : "Astra Pointer Meter / OpenCV";
            cv::putText(annotated, meter_pipeline, cv::Point(12, annotated.rows - 16),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(42, 226, 232), 2, cv::LINE_AA);
        } else cv::putText(
            annotated,
            options.head_roi_ppe
                ? "YOLOv8 ByteTrack + Head ROI Hard-Hat / SpaceMIT EP"
                : (options.ppe_only
                ? "SafetyVision PPE / SpaceMIT EP"
                : (options.hardhat_only
                ? "RVClaw Hard-Hat YOLOv8n / SpaceMIT EP"
                : (ppe_service ? "YOLOv8n + PPE / SpaceMIT EP"
                               : "YOLOv8n / SpaceMIT EP"))),
            cv::Point(12, annotated.rows - 16),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(42, 226, 232),
            2,
            cv::LINE_AA);

        if (!encoder.write_frame(annotated)) {
            exit_code = 5;
            break;
        }

        std::cerr << "RVCLAW_METRICS {\"detections\":" << detection_count
                  << ",\"persons\":" << person_count
                  << ",\"helmet\":" << helmet_count
                  << ",\"no_helmet\":" << no_helmet_count
                  << ",\"checking\":" << checking_count
                  << ",\"inference_ms\":" << std::fixed << std::setprecision(1)
                  << inference_ms << ",\"inferences\":" << inference_count
                  << ",\"ppe_ready\":"
                  << ((ppe_service || options.hardhat_only || options.ppe_only)
                          ? "true" : "false")
                  << ",\"ppe_inference_ms\":" << ppe_inference_ms
                  << ",\"ppe_inferences\":" << ppe_inference_count
                  << ",\"meter_type\":\"" << (options.meter_only ? options.meter_type : "") << "\""
                  << ",\"meter_raw_text\":\""
                  << (options.meter_only ? json_escape(meter_reading.raw_text) : "") << "\""
                  << ",\"meter_value_method\":\""
                  << (options.meter_only ? json_escape(meter_reading.method) : "") << "\""
                  << ",\"meter_status\":\"" << (options.meter_only ? meter_reading.status : "") << "\""
                  << ",\"meter_value\":" << (options.meter_only && meter_reading.valid ? std::to_string(meter_reading.value) : "null")
                  << ",\"meter_angle_deg\":" << (options.meter_only && meter_reading.valid ? std::to_string(meter_reading.angle_deg) : "null")
                  << ",\"meter_confidence\":" << (options.meter_only ? std::to_string(meter_reading.confidence) : "null")
                  << ",\"meter_text_items\":"
                  << (options.meter_only && async_digital_meter
                          ? meter_text_items_json(meter_text_items) : "[]")
                  << "}"
                  << std::endl;
    }

    running.store(false);
    capture_state.ready.notify_all();
    if (digital_meter_thread.joinable()) digital_meter_thread.join();
    if (capture_thread.joinable()) capture_thread.join();
    if (decoder) decoder->stop();
    video_capture.release();
    encoder.stop();
    return exit_code;
}
