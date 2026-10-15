/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * VAD Callback Interface
 *
 * 内部回调接口定义。
 */

#ifndef VAD_CALLBACK_HPP
#define VAD_CALLBACK_HPP

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "vad_types.hpp"

namespace vad {

// =============================================================================
// VAD Callback Interface
// =============================================================================

class IVadCallback {
public:
    virtual ~IVadCallback() = default;

    /// @brief 检测会话开始
    virtual void onStart() {}

    /// @brief 检测会话完成
    virtual void onComplete() {}

    /// @brief 会话关闭
    virtual void onClose() {}

    /// @brief 收到检测结果
    /// @param result 检测结果
    virtual void onResult(const DetectionResult& result) = 0;

    /// @brief 检测到语音开始
    /// @param timestamp_ms 语音开始的时间戳（毫秒）
    virtual void onSpeechStart(int64_t timestamp_ms) {}

    /// @brief 检测到语音结束
    /// @param timestamp_ms 语音结束的时间戳（毫秒）
    /// @param duration_ms 语音段持续时间（毫秒）
    virtual void onSpeechEnd(int64_t timestamp_ms, int duration_ms) {}

    /// @brief 发生错误
    /// @param error 错误信息
    virtual void onError(const ErrorInfo& error) = 0;
};

// =============================================================================
// Lambda Callback Types
// =============================================================================

using OnStartCallback = std::function<void()>;
using OnCompleteCallback = std::function<void()>;
using OnCloseCallback = std::function<void()>;
using OnResultCallback = std::function<void(const DetectionResult&)>;
using OnSpeechStartCallback = std::function<void(int64_t)>;
using OnSpeechEndCallback = std::function<void(int64_t, int)>;
using OnErrorCallback = std::function<void(const ErrorInfo&)>;

// =============================================================================
// Lambda Callback Adapter
// =============================================================================

class LambdaCallback : public IVadCallback {
public:
    class Builder {
    public:
        Builder& onStart(OnStartCallback cb) {
            on_start_ = std::move(cb);
            return *this;
        }

        Builder& onComplete(OnCompleteCallback cb) {
            on_complete_ = std::move(cb);
            return *this;
        }

        Builder& onClose(OnCloseCallback cb) {
            on_close_ = std::move(cb);
            return *this;
        }

        Builder& onResult(OnResultCallback cb) {
            on_result_ = std::move(cb);
            return *this;
        }

        Builder& onSpeechStart(OnSpeechStartCallback cb) {
            on_speech_start_ = std::move(cb);
            return *this;
        }

        Builder& onSpeechEnd(OnSpeechEndCallback cb) {
            on_speech_end_ = std::move(cb);
            return *this;
        }

        Builder& onError(OnErrorCallback cb) {
            on_error_ = std::move(cb);
            return *this;
        }

        std::unique_ptr<LambdaCallback> build() {
            auto cb = std::make_unique<LambdaCallback>();
            cb->on_start_ = std::move(on_start_);
            cb->on_complete_ = std::move(on_complete_);
            cb->on_close_ = std::move(on_close_);
            cb->on_result_ = std::move(on_result_);
            cb->on_speech_start_ = std::move(on_speech_start_);
            cb->on_speech_end_ = std::move(on_speech_end_);
            cb->on_error_ = std::move(on_error_);
            return cb;
        }

    private:
        OnStartCallback on_start_;
        OnCompleteCallback on_complete_;
        OnCloseCallback on_close_;
        OnResultCallback on_result_;
        OnSpeechStartCallback on_speech_start_;
        OnSpeechEndCallback on_speech_end_;
        OnErrorCallback on_error_;
    };

    static Builder create() { return Builder(); }

    void onStart() override {
        if (on_start_) on_start_();
    }

    void onComplete() override {
        if (on_complete_) on_complete_();
    }

    void onClose() override {
        if (on_close_) on_close_();
    }

    void onResult(const DetectionResult& result) override {
        if (on_result_) on_result_(result);
    }

    void onSpeechStart(int64_t ts) override {
        if (on_speech_start_) on_speech_start_(ts);
    }

    void onSpeechEnd(int64_t ts, int dur) override {
        if (on_speech_end_) on_speech_end_(ts, dur);
    }

    void onError(const ErrorInfo& error) override {
        if (on_error_) on_error_(error);
    }

private:
    friend class Builder;

    OnStartCallback on_start_;
    OnCompleteCallback on_complete_;
    OnCloseCallback on_close_;
    OnResultCallback on_result_;
    OnSpeechStartCallback on_speech_start_;
    OnSpeechEndCallback on_speech_end_;
    OnErrorCallback on_error_;
};

// =============================================================================
// Simple Callback (for testing)
// =============================================================================

class SimpleCallback : public IVadCallback {
public:
    void onResult(const DetectionResult& result) override {
        last_result_ = result;
        has_result_ = true;
    }

    void onError(const ErrorInfo& error) override {
        last_error_ = error;
        has_error_ = true;
    }

    bool hasResult() const { return has_result_; }
    bool hasError() const { return has_error_; }

    const DetectionResult& getLastResult() const { return last_result_; }
    const ErrorInfo& getLastError() const { return last_error_; }

    void reset() {
        has_result_ = false;
        has_error_ = false;
    }

private:
    DetectionResult last_result_;
    ErrorInfo last_error_;
    bool has_result_ = false;
    bool has_error_ = false;
};

}  // namespace vad

#endif  // VAD_CALLBACK_HPP
