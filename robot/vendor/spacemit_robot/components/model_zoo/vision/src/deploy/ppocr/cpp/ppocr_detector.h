/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PPOCR_DETECTOR_H
#define PPOCR_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief PP-OCRv5 two-stage OCR: text detection (DBNet) + recognition (CRNN/CTC).
 *
 * Holds two ONNX sessions: the DB text detector (BaseModel::session_) and a
 * self-managed recognition session. Infer(image) runs the full pipeline:
 * detect quadrilateral text boxes -> perspective-crop each -> recognize the
 * string via CTC greedy decode against a character dictionary. Returns a list
 * of vision::Text (polygon + text + score).
 */
class PPOCRDetector : public vision_core::BaseModel {
public:
    PPOCRDetector(
        const std::string& model_path,
        const std::string& rec_model_path,
        const std::string& dict_path,
        int det_limit_side_len = 960,
        float det_db_thresh = 0.3f,
        float det_db_box_thresh = 0.6f,
        float det_db_unclip_ratio = 2.0f,
        int rec_img_h = 48,
        int rec_img_w_max = 320,
        int num_threads = 4,
        bool lazy_load = false,
        const std::string& provider = "SpaceMITExecutionProvider",
        bool rec_only = false);

    virtual ~PPOCRDetector() = default;

    void load_model() override;

    // Full OCR pipeline: detect + recognize.
    vision_common::TextResultList detect_text(const cv::Mat& image);

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    struct TextBox {
        std::vector<cv::Point> points;  // 4 corners ordered TL, TR, BR, BL
        float score = 0.0f;
    };

    // --- detection (DBNet) ---
    std::vector<float> det_preprocess(const cv::Mat& image, int* net_h, int* net_w);
    std::vector<TextBox> db_postprocess(const cv::Mat& prob_map, int ori_h, int ori_w,
                                        int net_h, int net_w);
    std::vector<cv::Point> unclip(const std::vector<cv::Point>& poly);
    float box_score(const cv::Mat& prob_map, const std::vector<cv::Point>& box);
    static void sort_box_points(std::vector<cv::Point>& pts);

    // --- recognition (CRNN + CTC) ---
    static cv::Mat crop_text_box(const cv::Mat& image, const std::vector<cv::Point>& box);
    std::string rec_run(const cv::Mat& crop, float* out_score);

    void load_dict(const std::string& dict_path);
    void validate_dict_size();

    std::string rec_model_path_;
    std::string dict_path_;
    int det_limit_side_len_;
    float det_db_thresh_;
    float det_db_box_thresh_;
    float det_db_unclip_ratio_;
    int rec_img_h_;
    int rec_img_w_max_;
    int num_threads_;
    std::string provider_;
    // Fixed digital-display regions already isolate one reading, so DBNet text
    // detection is unnecessary and can miss joined LCD or seven-segment glyphs.
    bool rec_only_;

    std::vector<std::string> dict_;  // index -> character; index 0 is CTC blank

    // Self-managed recognition session (det uses BaseModel::session_).
    std::unique_ptr<Ort::Session> rec_session_;
    std::vector<const char*> rec_input_names_;
    std::vector<const char*> rec_output_names_;
    std::vector<std::string> rec_input_names_str_;
    std::vector<std::string> rec_output_names_str_;
};

}  // namespace vision_deploy

#endif  // PPOCR_DETECTOR_H
