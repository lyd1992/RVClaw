/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef KALMANFILTER_HPP
#define KALMANFILTER_HPP

#include <any>
#include <map>
#include <vector>

#include <Eigen/Dense>  // NOLINT(build/include_order) -- Eigen may pull in C headers
namespace ocsort {
    class KalmanFilterNew {
    public:
        KalmanFilterNew();
        KalmanFilterNew(int dim_x_, int dim_z_);
        void predict();
        void update(Eigen::VectorXf* z_);
        void freeze();
        void unfreeze();
        KalmanFilterNew& operator=(const KalmanFilterNew&) = default;

    public:
        int dim_z = 4;
        int dim_x = 7;
        int dim_u = 0;
        /// state: This is the Kalman state variable [7,1].
        Eigen::VectorXf x;
        // P: Covariance matrix. Initially declared as an identity matrix. Data type is float. [7,7].
        Eigen::MatrixXf P;
        // Q: Process noise covariance matrix. [7,7].
        Eigen::MatrixXf Q;
        // B: Control matrix. Not used in target tracking. [n,n].
        Eigen::MatrixXf B;
        // F: Prediction matrix / state transition matrix. [7,7].
        Eigen::Matrix<float, 7, 7> F;
        // H: Observation model / matrix. [4,7].
        Eigen::Matrix<float, 4, 7> H;
        // R: Observation noise covariance matrix. [4,4].
        Eigen::Matrix<float, 4, 4> R;
        // _alpha_sq: Fading memory control, controlling the update weight. Float.
        float _alpha_sq = 1.0;
        // M: Measurement matrix, state x to measurement z. [7,4]. Opposite of H.
        Eigen::MatrixXf M;
        // z: Measurement vector. [4,1].
        Eigen::VectorXf z;
        /* The following variables are intermediate variables used in calculations */
        // K: Kalman gain. [7,4].
        Eigen::MatrixXf K;
        // y: Measurement residual. [4,1].
        Eigen::MatrixXf y;
        // S: Measurement residual covariance.
        Eigen::MatrixXf S;
        // SI: Transpose of measurement residual covariance (simplified for subsequent calculations).
        Eigen::MatrixXf SI;
        // Identity matrix of size [dim_x,dim_x], used for convenient calculations. This cannot be changed.
        const Eigen::MatrixXf I = Eigen::MatrixXf::Identity(dim_x, dim_x);
        // There will always be a copy of x, P after predict() is called.
        // Eigen assign requires compatible dimensions; init ensures rows/cols match.
        Eigen::VectorXf x_prior;
        Eigen::MatrixXf P_prior;
        // there will always be a copy of x,P after update() is called
        Eigen::VectorXf x_post;
        Eigen::MatrixXf P_post;
        // keeps all observations. When there is a 'z', it is directly pushed back.
        std::vector<Eigen::VectorXf*> history_obs;
        // The following is newly added by ocsort.
        // Tracking state: target still matching this trajectory; default false.
        bool observed = false;
        std::vector<Eigen::VectorXf*> new_history;  // Used to create a virtual trajectory.

        /* todo: Let's change the way we store variables, as C++ does not have Python's self.dict.
        Using map<string,any> incurs high memory overhead, and there are errors when assigning values to Eigen data.
        Someone in the group suggested using metadata to achieve this, but I don't know how to do it.
        Therefore, here we will use a structure to save variables.
        */
        struct Data {
            Eigen::VectorXf x;
            Eigen::MatrixXf P;
            Eigen::MatrixXf Q;
            Eigen::MatrixXf B;
            Eigen::MatrixXf F;
            Eigen::MatrixXf H;
            Eigen::MatrixXf R;
            float _alpha_sq = 1.;
            Eigen::MatrixXf M;
            Eigen::VectorXf z;
            Eigen::MatrixXf K;
            Eigen::MatrixXf y;
            Eigen::MatrixXf S;
            Eigen::MatrixXf SI;
            Eigen::VectorXf x_prior;
            Eigen::MatrixXf P_prior;
            Eigen::VectorXf x_post;
            Eigen::MatrixXf P_post;
            std::vector<Eigen::VectorXf*> history_obs;
            bool observed = false;
            // The following is to determine whether the data has been saved due to freezing.
            bool IsInitialized = false;
        };
        struct Data attr_saved;
    };

}  // namespace ocsort

#endif  // KALMANFILTER_HPP
