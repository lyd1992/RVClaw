/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_BACKEND_HPP
#define ASR_BACKEND_HPP

#include <memory>
#include <vector>
#include <atomic>
#include <string>

#include "../asr_types.hpp"
#include "../asr_config.hpp"
#include "../asr_callback.hpp"

namespace asr {

// =============================================================================
// ASR Backend Interface (ASR后端抽象接口)
// =============================================================================
//
// 所有ASR后端必须实现此接口。
// 这是策略模式的核心, 允许运行时切换不同的ASR实现。
//
// 实现新后端的步骤:
// 1. 继承 IASRBackend
// 2. 实现所有纯虚函数
// 3. 在 ASRBackendFactory 中注册
//

class IASRBackend {
public:
    virtual ~IASRBackend() = default;

    // -------------------------------------------------------------------------
    // 生命周期管理
    // -------------------------------------------------------------------------

    /// @brief 初始化后端
    /// @param config 配置参数
    /// @return 错误信息, OK表示成功
    virtual ErrorInfo initialize(const ASRConfig& config) = 0;

    /// @brief 释放资源
    virtual void shutdown() = 0;

    /// @brief 检查是否已初始化
    virtual bool isInitialized() const = 0;

    // -------------------------------------------------------------------------
    // 后端信息
    // -------------------------------------------------------------------------

    /// @brief 获取后端类型
    virtual BackendType getType() const = 0;

    /// @brief 获取后端名称 (用于日志)
    virtual std::string getName() const = 0;

    /// @brief 获取后端版本
    virtual std::string getVersion() const = 0;

    /// @brief 检查是否支持流式模式
    virtual bool supportsStreaming() const = 0;

    /// @brief 获取支持的音频格式列表
    virtual std::vector<AudioFormat> getSupportedFormats() const = 0;

    /// @brief 获取支持的采样率列表
    virtual std::vector<int> getSupportedSampleRates() const = 0;

    // -------------------------------------------------------------------------
    // 离线识别 (批处理模式)
    // -------------------------------------------------------------------------

    /// @brief 同步识别完整音频
    /// @param audio 音频数据块
    /// @param result [out] 识别结果
    /// @return 错误信息
    virtual ErrorInfo recognize(const AudioChunk& audio, RecognitionResult& result) = 0;

    /// @brief 同步识别音频文件
    /// @param file_path 音频文件路径
    /// @param result [out] 识别结果
    /// @return 错误信息
    virtual ErrorInfo recognizeFile(const std::string& file_path, RecognitionResult& result) {
        (void)file_path;
        (void)result;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "File recognition not implemented");
    }

    // -------------------------------------------------------------------------
    // 流式识别
    // -------------------------------------------------------------------------

    /// @brief 开始流式识别会话
    /// @return 错误信息
    virtual ErrorInfo startStream() {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Streaming not supported");
    }

    /// @brief 发送音频数据块
    /// @param audio 音频数据块
    /// @return 错误信息
    virtual ErrorInfo feedAudio(const AudioChunk& audio) {
        (void)audio;
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
    }

    /// @brief 结束流式识别会话
    /// @return 错误信息
    virtual ErrorInfo stopStream() {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
    }

    /// @brief 刷新缓冲区并立即识别 (不关闭会话)
    /// @return 错误信息
    /// @note 用于用户 VAD 检测到句子结束时手动触发识别
    virtual ErrorInfo flushStream() {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
    }

    /// @brief 检查流式会话是否活跃
    virtual bool isStreamActive() const { return false; }

    // -------------------------------------------------------------------------
    // 回调设置
    // -------------------------------------------------------------------------

    /// @brief 设置回调处理器
    /// @param callback 回调接口指针 (生命周期由调用者管理)
    virtual void setCallback(IASRCallback* callback) {
        callback_ = callback;
    }

    /// @brief 获取当前回调处理器
    IASRCallback* getCallback() const { return callback_; }

    // -------------------------------------------------------------------------
    // 配置热更新 (可选)
    // -------------------------------------------------------------------------

    /// @brief 更新热词
    virtual ErrorInfo updateHotwords(const std::vector<std::string>& hotwords) {
        (void)hotwords;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Hotword update not supported");
    }

    /// @brief 更新语言设置
    virtual ErrorInfo setLanguage(Language language) {
        (void)language;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Language update not supported");
    }

protected:
    IASRCallback* callback_ = nullptr;

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

    void notifyResult(const RecognitionResult& result) {
        if (callback_) callback_->onResult(result);
    }

    void notifySentence(const SentenceResult& sentence, bool is_final) {
        if (callback_) callback_->onSentence(sentence, is_final);
    }

    void notifyError(const ErrorInfo& error) {
        if (callback_) callback_->onError(error);
    }

    void notifySpeechStart(int64_t timestamp_ms) {
        if (callback_) callback_->onSpeechStart(timestamp_ms);
    }

    void notifySpeechEnd(int64_t timestamp_ms) {
        if (callback_) callback_->onSpeechEnd(timestamp_ms);
    }
};

// =============================================================================
// Backend Factory (后端工厂)
// =============================================================================

class ASRBackendFactory {
public:
    /// @brief 创建ASR后端实例
    /// @param type 后端类型
    /// @return 后端实例, 失败返回nullptr
    static std::unique_ptr<IASRBackend> create(BackendType type);

    /// @brief 检查后端类型是否可用
    /// @param type 后端类型
    /// @return 是否可用
    static bool isAvailable(BackendType type);

    /// @brief 获取所有可用的后端类型
    static std::vector<BackendType> getAvailableBackends();
};

}  // namespace asr

#endif  // ASR_BACKEND_HPP
