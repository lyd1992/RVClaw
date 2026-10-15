/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_CALLBACK_HPP
#define ASR_CALLBACK_HPP

#include <functional>
#include <memory>
#include <utility>

#include "asr_types.hpp"

namespace asr {

// =============================================================================
// ASR Callback Interface (回调接口)
// =============================================================================
//
// 用户可以选择两种方式实现回调:
// 1. 继承此类并重写虚函数
// 2. 使用 CallbackBuilder 设置 lambda 函数
//

class IASRCallback {
public:
    virtual ~IASRCallback() = default;

    // -------------------------------------------------------------------------
    // 连接生命周期回调 (主要用于流式模式)
    // -------------------------------------------------------------------------

    /// @brief 识别会话已开始
    /// @note 流式模式: start() 成功后调用
    /// @note 离线模式: recognize() 开始处理时调用
    virtual void onStart() {}

    /// @brief 识别会话已结束
    /// @note 所有结果已返回, 会话正常结束
    virtual void onComplete() {}

    /// @brief 会话已关闭
    /// @note 无论成功或失败, 最后都会调用此方法
    virtual void onClose() {}

    // -------------------------------------------------------------------------
    // 结果回调
    // -------------------------------------------------------------------------

    /// @brief 收到识别结果
    /// @param result 识别结果 (可能是中间结果或最终结果)
    /// @note 流式模式: 可能多次调用, 检查 SentenceResult::is_final 判断是否为最终结果
    /// @note 离线模式: 通常只调用一次, 返回完整结果
    virtual void onResult(const RecognitionResult& result) = 0;

    /// @brief 收到句子级别结果 (便捷回调)
    /// @param sentence 单个句子结果
    /// @param is_final 是否为最终结果
    /// @note 默认实现为空, 如需句子级别粒度可重写此方法
    virtual void onSentence(const SentenceResult& sentence, bool is_final) {
        (void)sentence;
        (void)is_final;
    }

    // -------------------------------------------------------------------------
    // 错误回调
    // -------------------------------------------------------------------------

    /// @brief 发生错误
    /// @param error 错误信息
    virtual void onError(const ErrorInfo& error) = 0;

    // -------------------------------------------------------------------------
    // VAD 相关回调 (可选)
    // -------------------------------------------------------------------------

    /// @brief 检测到语音开始
    /// @param timestamp_ms 语音开始的时间戳(毫秒)
    virtual void onSpeechStart(int64_t timestamp_ms) {
        (void)timestamp_ms;
    }

    /// @brief 检测到语音结束
    /// @param timestamp_ms 语音结束的时间戳(毫秒)
    virtual void onSpeechEnd(int64_t timestamp_ms) {
        (void)timestamp_ms;
    }
};

// =============================================================================
// Callback using std::function (函数式回调)
// =============================================================================

using OnStartCallback = std::function<void()>;
using OnCompleteCallback = std::function<void()>;
using OnCloseCallback = std::function<void()>;
using OnResultCallback = std::function<void(const RecognitionResult&)>;
using OnSentenceCallback = std::function<void(const SentenceResult&, bool is_final)>;
using OnErrorCallback = std::function<void(const ErrorInfo&)>;
using OnSpeechStartCallback = std::function<void(int64_t timestamp_ms)>;
using OnSpeechEndCallback = std::function<void(int64_t timestamp_ms)>;

// =============================================================================
// Lambda Callback Adapter (Lambda适配器)
// =============================================================================
//
// 使用示例:
//
//   auto callback = LambdaCallback::create()
//       .onResult([](const RecognitionResult& r) {
//           std::cout << "Result: " << r.getText() << std::endl;
//       })
//       .onError([](const ErrorInfo& e) {
//           std::cerr << "Error: " << e.message << std::endl;
//       })
//       .build();
//
//   engine.setCallback(std::move(callback));
//

class LambdaCallback : public IASRCallback {
public:
    class Builder {
    public:
        Builder& onStart(OnStartCallback cb) { on_start_ = std::move(cb); return *this; }
        Builder& onComplete(OnCompleteCallback cb) { on_complete_ = std::move(cb); return *this; }
        Builder& onClose(OnCloseCallback cb) { on_close_ = std::move(cb); return *this; }
        Builder& onResult(OnResultCallback cb) { on_result_ = std::move(cb); return *this; }
        Builder& onSentence(OnSentenceCallback cb) { on_sentence_ = std::move(cb); return *this; }
        Builder& onError(OnErrorCallback cb) { on_error_ = std::move(cb); return *this; }
        Builder& onSpeechStart(OnSpeechStartCallback cb) { on_speech_start_ = std::move(cb); return *this; }
        Builder& onSpeechEnd(OnSpeechEndCallback cb) { on_speech_end_ = std::move(cb); return *this; }

        std::unique_ptr<LambdaCallback> build() {
            auto cb = std::make_unique<LambdaCallback>();
            cb->on_start_ = std::move(on_start_);
            cb->on_complete_ = std::move(on_complete_);
            cb->on_close_ = std::move(on_close_);
            cb->on_result_ = std::move(on_result_);
            cb->on_sentence_ = std::move(on_sentence_);
            cb->on_error_ = std::move(on_error_);
            cb->on_speech_start_ = std::move(on_speech_start_);
            cb->on_speech_end_ = std::move(on_speech_end_);
            return cb;
        }

    private:
        OnStartCallback on_start_;
        OnCompleteCallback on_complete_;
        OnCloseCallback on_close_;
        OnResultCallback on_result_;
        OnSentenceCallback on_sentence_;
        OnErrorCallback on_error_;
        OnSpeechStartCallback on_speech_start_;
        OnSpeechEndCallback on_speech_end_;
    };

    static Builder create() { return Builder(); }

    void onStart() override { if (on_start_) on_start_(); }
    void onComplete() override { if (on_complete_) on_complete_(); }
    void onClose() override { if (on_close_) on_close_(); }

    void onResult(const RecognitionResult& result) override {
        if (on_result_) on_result_(result);
    }

    void onSentence(const SentenceResult& sentence, bool is_final) override {
        if (on_sentence_) on_sentence_(sentence, is_final);
    }

    void onError(const ErrorInfo& error) override {
        if (on_error_) on_error_(error);
    }

    void onSpeechStart(int64_t timestamp_ms) override {
        if (on_speech_start_) on_speech_start_(timestamp_ms);
    }

    void onSpeechEnd(int64_t timestamp_ms) override {
        if (on_speech_end_) on_speech_end_(timestamp_ms);
    }

private:
    OnStartCallback on_start_;
    OnCompleteCallback on_complete_;
    OnCloseCallback on_close_;
    OnResultCallback on_result_;
    OnSentenceCallback on_sentence_;
    OnErrorCallback on_error_;
    OnSpeechStartCallback on_speech_start_;
    OnSpeechEndCallback on_speech_end_;
};

// =============================================================================
// Simple Callback (简单同步回调 - 用于快速测试)
// =============================================================================

class SimpleCallback : public IASRCallback {
public:
    void onResult(const RecognitionResult& result) override {
        last_result_ = result;
        has_result_ = true;
    }

    void onError(const ErrorInfo& error) override {
        last_error_ = error;
        has_error_ = true;
    }

    bool hasResult() const { return has_result_; }
    bool hasError() const { return has_error_; }

    const RecognitionResult& getResult() const { return last_result_; }
    const ErrorInfo& getError() const { return last_error_; }

    void reset() {
        has_result_ = false;
        has_error_ = false;
        last_result_ = {};
        last_error_ = {};
    }

private:
    bool has_result_ = false;
    bool has_error_ = false;
    RecognitionResult last_result_;
    ErrorInfo last_error_;
};

}  // namespace asr

#endif  // ASR_CALLBACK_HPP
