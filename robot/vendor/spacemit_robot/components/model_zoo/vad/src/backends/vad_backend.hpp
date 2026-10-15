/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * VAD Backend Interface
 *
 * 后端抽象接口定义，支持多种 VAD 实现。
 */

#ifndef VAD_BACKEND_HPP
#define VAD_BACKEND_HPP

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "../vad_types.hpp"
#include "../vad_callback.hpp"

namespace vad {

// =============================================================================
// VAD Backend Interface
// =============================================================================

/**
 * @brief VAD 后端抽象接口
 *
 * 所有 VAD 后端实现都必须继承此接口。
 */
class IVadBackend {
public:
    virtual ~IVadBackend() = default;

    // -------------------------------------------------------------------------
    // 生命周期管理
    // -------------------------------------------------------------------------

    /// @brief 初始化后端
    /// @param config 配置
    /// @return 错误信息
    virtual ErrorInfo initialize(const VadConfig& config) = 0;

    /// @brief 释放资源
    virtual void shutdown() = 0;

    /// @brief 检查是否已初始化
    virtual bool isInitialized() const = 0;

    // -------------------------------------------------------------------------
    // 后端信息
    // -------------------------------------------------------------------------

    /// @brief 获取后端类型
    virtual BackendType getType() const = 0;

    /// @brief 获取后端名称
    virtual std::string getName() const = 0;

    /// @brief 获取后端版本
    virtual std::string getVersion() const { return "1.0.0"; }

    /// @brief 检查是否支持流式模式
    virtual bool supportsStreaming() const { return true; }

    /// @brief 获取支持的采样率列表
    virtual std::vector<int> getSupportedSampleRates() const {
        return {8000, 16000};
    }

    /// @brief 获取推荐的帧大小
    virtual int getRecommendedFrameSize() const { return 512; }

    // -------------------------------------------------------------------------
    // 检测接口
    // -------------------------------------------------------------------------

    /// @brief 检测单帧音频
    /// @param audio 音频数据
    /// @param result 输出检测结果
    /// @return 错误信息
    virtual ErrorInfo detect(const AudioChunk& audio, DetectionResult& result) = 0;

    /// @brief 重置内部状态
    virtual void reset() = 0;

    // -------------------------------------------------------------------------
    // 流式检测
    // -------------------------------------------------------------------------

    /// @brief 开始流式检测
    virtual ErrorInfo startStream() {
        if (streaming_.load()) {
            return ErrorInfo::error(ErrorCode::ALREADY_STARTED, "Stream already started");
        }
        streaming_.store(true);
        return ErrorInfo::ok();
    }

    /// @brief 发送音频数据
    virtual ErrorInfo feedAudio(const AudioChunk& audio) {
        if (!streaming_.load()) {
            return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
        }
        DetectionResult result;
        auto err = detect(audio, result);
        if (err.isOk() && callback_) {
            callback_->onResult(result);

            // 检测状态变化
            if (result.state == VadState::SPEECH_START) {
                callback_->onSpeechStart(result.timestamp_ms);
            } else if (result.state == VadState::SPEECH_END) {
                callback_->onSpeechEnd(result.timestamp_ms, result.speech_duration_ms);
            }
        }
        return err;
    }

    /// @brief 停止流式检测
    virtual ErrorInfo stopStream() {
        if (!streaming_.load()) {
            return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
        }
        streaming_.store(false);
        return ErrorInfo::ok();
    }

    /// @brief 检查流式检测是否活跃
    virtual bool isStreamActive() const { return streaming_.load(); }

    // -------------------------------------------------------------------------
    // 回调设置
    // -------------------------------------------------------------------------

    /// @brief 设置回调
    virtual void setCallback(IVadCallback* callback) {
        callback_ = callback;
    }

    /// @brief 获取回调
    IVadCallback* getCallback() const { return callback_; }

    // -------------------------------------------------------------------------
    // 动态配置
    // -------------------------------------------------------------------------

    /// @brief 设置阈值
    virtual ErrorInfo setThresholds(float trigger, float stop) {
        (void)trigger;
        (void)stop;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Not supported by this backend");
    }

    /// @brief 获取当前配置
    virtual VadConfig getConfig() const { return config_; }

protected:
    IVadCallback* callback_ = nullptr;
    VadConfig config_;
    std::atomic<bool> streaming_{false};

    // 辅助方法: 触发回调
    void notifyStart() {
        if (callback_) callback_->onStart();
    }

    void notifyComplete() {
        if (callback_) callback_->onComplete();
    }

    void notifyClose() {
        if (callback_) callback_->onClose();
    }

    void notifyResult(const DetectionResult& result) {
        if (callback_) callback_->onResult(result);
    }

    void notifySpeechStart(int64_t timestamp_ms) {
        if (callback_) callback_->onSpeechStart(timestamp_ms);
    }

    void notifySpeechEnd(int64_t timestamp_ms, int duration_ms) {
        if (callback_) callback_->onSpeechEnd(timestamp_ms, duration_ms);
    }

    void notifyError(const ErrorInfo& error) {
        if (callback_) callback_->onError(error);
    }
};

// =============================================================================
// Backend Factory
// =============================================================================

/**
 * @brief VAD 后端工厂
 *
 * 负责创建和管理 VAD 后端实例。
 */
class VadBackendFactory {
public:
    /// @brief 创建 VAD 后端实例
    /// @param type 后端类型
    /// @return 后端实例，失败返回 nullptr
    static std::unique_ptr<IVadBackend> create(BackendType type);

    /// @brief 检查后端类型是否可用
    /// @param type 后端类型
    /// @return true 表示可用
    static bool isAvailable(BackendType type);

    /// @brief 获取所有可用的后端类型
    /// @return 可用后端列表
    static std::vector<BackendType> getAvailableBackends();

    /// @brief 获取后端默认采样率
    /// @param type 后端类型
    /// @return 默认采样率
    static int getDefaultSampleRate(BackendType type);

    /// @brief 获取后端推荐帧大小
    /// @param type 后端类型
    /// @return 推荐帧大小
    static int getRecommendedFrameSize(BackendType type);
};

}  // namespace vad

#endif  // VAD_BACKEND_HPP
