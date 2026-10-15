/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OCSORT_HPP
#define OCSORT_HPP

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Association.hpp"
#include "KalmanBoxTracker.hpp"
#include "lapjv.hpp"

namespace ocsort {

    class OCSort {
    public:
        OCSort(float det_thresh_, int max_age_ = 30, int min_hits_ = 3, float iou_threshold_ = 0.3,
                int delta_t_ = 3, std::string asso_func_ = "iou", float inertia_ = 0.2,
                bool use_byte_ = false);
        std::vector<Eigen::RowVectorXf> update(Eigen::MatrixXf dets);

    public:
        float det_thresh;
        int max_age;
        int min_hits;
        float iou_threshold;
        int delta_t;
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> asso_func;
        float inertia;
        bool use_byte;
        std::vector<KalmanBoxTracker> trackers;
        int frame_count;
    };

}  // namespace ocsort

#endif  // OCSORT_HPP
