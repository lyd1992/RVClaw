/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_TASK_INTERFACES_H
#define VISION_TASK_INTERFACES_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common/cpp/datatype.h"

namespace vision_core {

/**
 * @brief Detection model interface
 *
 * Models that detect objects in images (bounding boxes + classes).
 * Examples: YOLOv5, YOLOv8, YOLOv11, YOLOv5-face, YOLOv5-gesture
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kDetect")]] IDetectionModel {
public:
    virtual ~IDetectionModel() = default;

    /**
     * @brief Detect objects in an image
     * @param image          Input image in BGR format
     * @param conf_threshold Confidence threshold; <= 0 keeps the model's configured default
     * @param iou_threshold  NMS IoU threshold; <= 0 keeps the model's configured default
     * @return List of detection results
     */
    virtual vision_common::DetectionResultList detect(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) = 0;
};

/**
 * @brief Classification model interface
 *
 * Models that classify entire images into categories.
 * Examples: ResNet, Emotion recognition
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kClassify")]] IClassificationModel {
public:
    virtual ~IClassificationModel() = default;

    /**
     * @brief Classify an image
     * @param image Input image in BGR format
     * @return Classification result (single result, but returned as list for consistency)
     */
    virtual vision_common::ClassificationResultList classify(const cv::Mat& image) = 0;
};

/**
 * @brief Segmentation model interface
 *
 * Models that perform instance or semantic segmentation.
 * Examples: YOLOv8-seg, PP-LiteSeg
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kSegment")]] ISegmentationModel {
public:
    virtual ~ISegmentationModel() = default;

    /**
     * @brief Segment objects in an image
     * @param image          Input image in BGR format
     * @param conf_threshold Confidence threshold; <= 0 keeps the model's configured default
     * @param iou_threshold  NMS IoU threshold; <= 0 keeps the model's configured default
     * @return List of segmentation results (bbox + mask)
     */
    virtual vision_common::SegmentationResultList segment(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) = 0;
};

/**
 * @brief Pose estimation model interface
 *
 * Models that detect human poses (keypoints).
 * Examples: YOLOv8-pose
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kEstimatePose")]] IPoseModel {
public:
    virtual ~IPoseModel() = default;

    /**
     * @brief Estimate poses in an image
     * @param image          Input image in BGR format
     * @param conf_threshold Confidence threshold; <= 0 keeps the model's configured default
     * @param iou_threshold  NMS IoU threshold; <= 0 keeps the model's configured default
     * @return List of pose results (bbox + keypoints)
     */
    virtual vision_common::PoseResultList estimate_pose(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) = 0;
};

/**
 * @brief Tracking model interface
 *
 * Models that track objects across video frames.
 * Examples: ByteTrack, OC-SORT
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kTrack")]] ITrackingModel {
public:
    virtual ~ITrackingModel() = default;

    /**
     * @brief Update tracker with new frame
     * @param image          Input image in BGR format
     * @param conf_threshold Detector confidence override; <= 0 keeps the model's configured default
     * @param iou_threshold  Detector NMS IoU override; <= 0 keeps the model's configured default
     * @return List of tracking results (bbox + track_id)
     */
    virtual vision_common::TrackingResultList track(
        const cv::Mat& image,
        float conf_threshold = -1.0f,
        float iou_threshold = -1.0f) = 0;
};

/**
 * @brief Embedding model interface
 *
 * Models that extract feature embeddings from images.
 * Examples: ArcFace (face recognition)
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kEmbed")]] IEmbeddingModel {
public:
    virtual ~IEmbeddingModel() = default;

    /**
     * @brief Extract embedding from an image
     * @param image Input image in BGR format (usually a cropped face/object)
     * @return Embedding result with feature vector
     */
    virtual vision_common::EmbeddingResult infer_embedding(const cv::Mat& image) = 0;

    /**
     * @brief Calculate similarity between two embeddings
     * @param embedding_a First embedding
     * @param embedding_b Second embedding
     * @return Similarity score (cosine similarity, 0-1)
     */
    static float calculate_similarity(const vision_common::EmbeddingResult& embedding_a,
        const vision_common::EmbeddingResult& embedding_b) {
        return vision_common::similarity(embedding_a, embedding_b);
    }
};

/**
 * @brief Sequence action recognition model interface
 *
 * Models that recognize actions from temporal sequences (e.g., skeleton sequences).
 * Examples: STGCN (fall detection)
 */
class [[deprecated("Use BaseModel::Run with InferIntent::kInferSequence")]] ISequenceActionModel {
public:
    virtual ~ISequenceActionModel() = default;

    /**
     * @brief Recognize action from skeleton sequence
     * @param pts Pointer to skeleton data (30 frames × 17 keypoints × 2 coords)
     * @param image_width Original image width (for normalization)
     * @param image_height Original image height (for normalization)
     * @return Action recognition result with class probabilities
     */
    virtual vision_common::ActionResult infer_sequence(const float* pts,
        int image_width,
        int image_height) = 0;

    /**
     * @brief Get action class names
     * @return List of action class names (e.g., ["standing", "walking", "falling"])
     */
    virtual std::vector<std::string> get_class_names() const { return {}; }
};

}  // namespace vision_core

#endif  // VISION_TASK_INTERFACES_H
