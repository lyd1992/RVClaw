/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ASR_ENGINE_HPP
#define ASR_ENGINE_HPP

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>

#include "asr_types.hpp"
#include "asr_config.hpp"
#include "asr_callback.hpp"
#include "backends/asr_backend.hpp"

namespace asr {

// =============================================================================
// ASR Engine (ASR引擎 - 用户主接口)
// =============================================================================
//
// 这是用户直接使用的主类, 封装了后端选择、生命周期管理和API调用。
// 设计参考阿里云FunASR SDK的接口风格。
//
// 使用示例 1 - 离线识别:
//
//   asr::ASREngine engine;
//   auto config = asr::ASRConfig::sensevoice("~/.cache/models/asr/sensevoice");
//   engine.initialize(config);
//
//   auto result = engine.recognize(audio_data, audio_size);
//   std::cout << result.getText() << std::endl;
//
// 使用示例 2 - 流式识别 (回调模式):
//
//   asr::ASREngine engine;
//   auto config = asr::ASRConfig::sensevoice("~/.cache/models/asr/sensevoice")
//                    .withStreaming(100);  // 100ms chunks
//
//   auto callback = asr::LambdaCallback::create()
//       .onResult([](const asr::RecognitionResult& r) {
//           std::cout << r.getText() << std::endl;
//       })
//       .onError([](const asr::ErrorInfo& e) {
//           std::cerr << e.message << std::endl;
//       })
//       .build();
//
//   engine.setCallback(std::move(callback));
//   engine.initialize(config);
//
//   engine.start();
//   while (has_audio) {
//       engine.sendAudio(chunk);
//   }
//   engine.stop();
//

class ASREngine {
public:
    ASREngine();
    ~ASREngine();

    // 禁止拷贝
    ASREngine(const ASREngine&) = delete;
    ASREngine& operator=(const ASREngine&) = delete;

    // 允许移动
    ASREngine(ASREngine&&) noexcept;
    ASREngine& operator=(ASREngine&&) noexcept;

    // -------------------------------------------------------------------------
    // 初始化与释放
    // -------------------------------------------------------------------------

    /// @brief 初始化ASR引擎
    /// @param config 配置参数
    /// @return 错误信息, OK表示成功
    ErrorInfo initialize(const ASRConfig& config);

    /// @brief 释放资源
    void shutdown();

    /// @brief 检查是否已初始化
    bool isInitialized() const;

    /// @brief 获取当前配置
    const ASRConfig& getConfig() const { return config_; }

    // -------------------------------------------------------------------------
    // 回调设置
    // -------------------------------------------------------------------------

    /// @brief 设置回调 (使用智能指针, 引擎管理生命周期)
    void setCallback(std::unique_ptr<IASRCallback> callback);

    /// @brief 设置回调 (原始指针, 调用者管理生命周期)
    void setCallback(IASRCallback* callback);

    // -------------------------------------------------------------------------
    // 离线识别 API (批处理模式)
    // -------------------------------------------------------------------------

    /// @brief 识别PCM S16音频
    /// @param audio 音频数据 (16-bit signed)
    /// @param samples 样本数
    /// @return 识别结果
    RecognitionResult recognize(const int16_t* audio, size_t samples);

    /// @brief 识别PCM Float音频
    /// @param audio 音频数据 (float, [-1.0, 1.0])
    /// @param samples 样本数
    /// @return 识别结果
    RecognitionResult recognize(const float* audio, size_t samples);

    /// @brief 识别音频数据块
    /// @param chunk 音频块
    /// @return 识别结果
    RecognitionResult recognize(const AudioChunk& chunk);

    /// @brief 识别音频文件
    /// @param file_path 文件路径
    /// @return 识别结果
    RecognitionResult recognizeFile(const std::string& file_path);

    // -------------------------------------------------------------------------
    // 流式识别 API
    // -------------------------------------------------------------------------

    /// @brief 开始流式识别会话
    /// @return 错误信息
    /// @note 结果通过回调返回
    ErrorInfo start();

    /// @brief 发送音频数据 (PCM S16)
    /// @param audio 音频数据
    /// @param samples 样本数
    /// @return 错误信息
    ErrorInfo sendAudio(const int16_t* audio, size_t samples);

    /// @brief 发送音频数据 (PCM Float)
    /// @param audio 音频数据
    /// @param samples 样本数
    /// @return 错误信息
    ErrorInfo sendAudio(const float* audio, size_t samples);

    /// @brief 发送音频数据块
    /// @param chunk 音频块
    /// @return 错误信息
    ErrorInfo sendAudio(const AudioChunk& chunk);

    /// @brief 结束流式识别会话
    /// @return 错误信息
    ErrorInfo stop();

    /// @brief 刷新缓冲区并立即识别 (不关闭会话)
    /// @return 错误信息
    /// @note 用于用户 VAD 检测到句子结束时手动触发识别
    ///       识别结果通过回调返回，会话保持活跃可继续发送音频
    ErrorInfo flush();

    /// @brief 检查流式会话是否活跃
    bool isStreaming() const;

    // -------------------------------------------------------------------------
    // 状态与调试
    // -------------------------------------------------------------------------

    /// @brief 获取最后一次请求ID
    std::string getLastRequestId() const { return last_request_id_; }

    /// @brief 获取首包延迟 (毫秒)
    int64_t getFirstPacketLatency() const { return first_packet_latency_ms_; }

    /// @brief 获取最后结果延迟 (毫秒)
    int64_t getLastPacketLatency() const { return last_packet_latency_ms_; }

    /// @brief 获取最后的错误信息
    ErrorInfo getLastError() const { return last_error_; }

    /// @brief 获取后端类型
    BackendType getBackendType() const;

    /// @brief 获取后端名称
    std::string getBackendName() const;

    // -------------------------------------------------------------------------
    // 动态配置更新
    // -------------------------------------------------------------------------

    /// @brief 更新热词
    ErrorInfo updateHotwords(const std::vector<std::string>& hotwords);

    /// @brief 设置语言
    ErrorInfo setLanguage(Language language);

    // -------------------------------------------------------------------------
    // 静态工具方法
    // -------------------------------------------------------------------------

    /// @brief 获取支持的后端列表
    static std::vector<BackendType> getAvailableBackends();

    /// @brief 检查后端是否可用
    static bool isBackendAvailable(BackendType type);

    /// @brief 获取库版本
    static std::string getVersion();

private:
    ASRConfig config_;
    std::unique_ptr<IASRBackend> backend_;
    std::unique_ptr<IASRCallback> owned_callback_;  // 引擎管理的回调
    IASRCallback* callback_ = nullptr;  // 当前使用的回调

    std::atomic<bool> initialized_{false};
    std::atomic<bool> streaming_{false};
    mutable std::mutex mutex_;

    // 性能追踪
    std::string last_request_id_;
    int64_t first_packet_latency_ms_ = 0;
    int64_t last_packet_latency_ms_ = 0;
    ErrorInfo last_error_;

    // 内部辅助方法
    ErrorInfo createBackend();
    void updateLatencyMetrics(const RecognitionResult& result);
    std::string generateRequestId();
};

// =============================================================================
// 便捷函数 (快速使用)
// =============================================================================

namespace quick {

/// @brief 快速识别音频 (使用默认SenseVoice)
/// @param audio 音频数据
/// @param samples 样本数
/// @param model_dir 模型目录 (默认 ~/.cache/models/asr/sensevoice)
/// @return 识别文本
std::string recognize(const float* audio, size_t samples,
        const std::string& model_dir = "");

/// @brief 快速识别音频文件
/// @param file_path 文件路径
/// @param model_dir 模型目录
/// @return 识别文本
std::string recognizeFile(const std::string& file_path,
        const std::string& model_dir = "");

}  // namespace quick

}  // namespace asr

#endif  // ASR_ENGINE_HPP
