/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef STRACK_H
#define STRACK_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "kalmanFilter.h"

using cv::Scalar;
using std::vector;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
    STrack(vector<float> tlwh_, float score);
    ~STrack();

    static vector<float> tlbr_to_tlwh(vector<float> &tlbr);
    static void multi_predict(vector<STrack*> &stracks,
        byte_kalman::KalmanFilter &kalman_filter);
    void static_tlwh();
    void static_tlbr();
    vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
    vector<float> to_xyah();
    void mark_lost();
    void mark_removed();
    int next_id();
    int end_frame();

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
    void re_activate(STrack &new_track, int frame_id, bool new_id = false);
    void update(STrack &new_track, int frame_id);

public:
    bool is_activated;
    int track_id;
    int state;

    vector<float> _tlwh;
    vector<float> tlwh;
    vector<float> tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};

#endif  // STRACK_H
