/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV5_FACE_DETECTOR_H
#define YOLOV5_FACE_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief Fixed-size prediction structure to avoid dynamic vector allocation
 * Stores: [x, y, w, h, obj_conf, cls_conf] in xywh format
 */
struct Prediction {
    float x;
    float y;
    float w;
    float h;
    float obj_conf;
    float cls_conf;

    // Convenience constructor
    Prediction(float x_, float y_, float w_, float h_, float obj_conf_, float cls_conf_)
        : x(x_), y(y_), w(w_), h(h_), obj_conf(obj_conf_), cls_conf(cls_conf_) {}
};

/**
 * @brief YOLOv5-Face Detector
 * 
 * Detects faces in images using YOLOv5-Face model.
 */
class YOLOv5FaceDetector : public vision_core::BaseModel, public vision_core::IDetectionModel {
public:
    YOLOv5FaceDetector(const std::string& model_path,
                        float conf_threshold = 0.25f,
                        float iou_threshold = 0.45f,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YOLOv5FaceDetector() = default;

    /**
     * @brief Load YOLOv5-Face ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for YOLOv5-Face inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    vision_common::DetectionResultList detect(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) override;


    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    vision_common::DetectionResultList postprocess(
        std::vector<Ort::Value>& outputs,
        const cv::Size& orig_size,
        float conf_threshold,
        float iou_threshold);

private:
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    std::string provider_;

    vision_common::DetectionResultList non_max_suppression_face(
        const std::vector<Prediction>& predictions,
        float conf_threshold,
        float iou_threshold);
    std::vector<float> make_grid(int nx, int ny);
};

}  // namespace vision_deploy

#endif  // YOLOV5_FACE_DETECTOR_H
