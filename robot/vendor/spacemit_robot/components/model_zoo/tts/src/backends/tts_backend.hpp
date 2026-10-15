/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TTS_BACKEND_HPP
#define TTS_BACKEND_HPP

#include <cstdio>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tts_config.hpp"
#include "tts_types.hpp"

namespace tts {

// =============================================================================
// TTS Backend Interface (TTS后端抽象接口)
// =============================================================================
//
// 所有TTS后端必须实现此接口。
// 这是策略模式的核心, 允许运行时切换不同的TTS实现。
//
// 实现新后端的步骤:
// 1. 继承 ITtsBackend
// 2. 实现所有纯虚函数
// 3. 在 TtsBackendFactory 中注册
//
// 已实现的后端:
// - MatchaZHBackend:   中文 (matcha-icefall-zh-baker)
// - MatchaENBackend:   英文 (matcha-icefall-en_US-ljspeech)
// - MatchaZHENBackend: 中英混合 (matcha-icefall-zh-en)
//

class ITtsBackend {
public:
    virtual ~ITtsBackend() = default;

    // -------------------------------------------------------------------------
    // 生命周期管理
    // -------------------------------------------------------------------------

    /// @brief 初始化后端
    /// @param config 配置参数
    /// @return 错误信息, OK表示成功
    virtual ErrorInfo initialize(const TtsConfig& config) = 0;

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

    /// @brief 获取支持的说话人数量
    virtual int getNumSpeakers() const = 0;

    /// @brief 获取默认采样率
    virtual int getSampleRate() const = 0;

    /// @brief 获取支持的音频格式列表
    virtual std::vector<AudioFormat> getSupportedFormats() const {
        return {AudioFormat::PCM_F32LE, AudioFormat::PCM_S16LE, AudioFormat::WAV};
    }

    // -------------------------------------------------------------------------
    // 离线合成 (批处理模式)
    // -------------------------------------------------------------------------

    /// @brief 同步合成文本
    /// @param text 要合成的文本
    /// @param result [out] 合成结果 (音频数据)
    /// @return 错误信息
    virtual ErrorInfo synthesize(const std::string& text, SynthesisResult& result) = 0;

    /// @brief 同步合成文本并保存到文件
    /// @param text 要合成的文本
    /// @param file_path 输出文件路径
    /// @return 错误信息
    virtual ErrorInfo synthesizeToFile(const std::string& text, const std::string& file_path) {
        SynthesisResult result;
        auto err = synthesize(text, result);
        if (!err.isOk()) {
            return err;
        }
        // 子类可以覆盖此方法以提供更高效的实现
        return saveToFile(result.audio, file_path);
    }

    // -------------------------------------------------------------------------
    // 流式合成 (可选)
    // -------------------------------------------------------------------------

    /// @brief 开始流式合成会话
    /// @return 错误信息
    virtual ErrorInfo startStream() {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Streaming not supported");
    }

    /// @brief 发送文本进行合成
    /// @param text 要合成的文本
    /// @return 错误信息
    /// @note 可多次调用，每次发送一段文本
    virtual ErrorInfo feedText(const std::string& text) {
        (void)text;
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
    }

    /// @brief 刷新缓冲区并立即合成 (不关闭会话)
    /// @return 错误信息
    /// @note 用于强制输出当前已缓冲的文本
    virtual ErrorInfo flushStream() {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
    }

    /// @brief 结束流式合成会话
    /// @return 错误信息
    virtual ErrorInfo stopStream() {
        return ErrorInfo::error(ErrorCode::NOT_STARTED, "Stream not started");
    }

    /// @brief 检查流式会话是否活跃
    virtual bool isStreamActive() const { return false; }

    // -------------------------------------------------------------------------
    // 回调设置
    // -------------------------------------------------------------------------

    /// @brief 设置回调处理器
    /// @param callback 回调接口指针 (生命周期由调用者管理)
    virtual void setCallback(ITtsCallback* callback) {
        callback_ = callback;
    }

    /// @brief 获取当前回调处理器
    ITtsCallback* getCallback() const { return callback_; }

    // -------------------------------------------------------------------------
    // 动态配置 (可选)
    // -------------------------------------------------------------------------

    /// @brief 设置语速
    /// @param speed 语速倍率 (>1.0快, <1.0慢)
    /// @return 错误信息
    virtual ErrorInfo setSpeed(float speed) {
        (void)speed;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Speed update not supported");
    }

    /// @brief 设置说话人
    /// @param speaker_id 说话人ID
    /// @return 错误信息
    virtual ErrorInfo setSpeaker(int speaker_id) {
        (void)speaker_id;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Speaker update not supported");
    }

    /// @brief 设置音量
    /// @param volume 音量 [0, 100]
    /// @return 错误信息
    virtual ErrorInfo setVolume(int volume) {
        (void)volume;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Volume update not supported");
    }

    /// @brief 设置音调
    /// @param pitch 音调倍率
    /// @return 错误信息
    virtual ErrorInfo setPitch(float pitch) {
        (void)pitch;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Pitch update not supported");
    }

    struct LexiconEntry {
        std::string word;
        std::string phoneme;
        std::string locale = "zh";
    };

    /// @brief 更新自定义发音词典
    /// @param entries word->phoneme->locale 映射列表
    /// @return 错误信息
    virtual ErrorInfo updateLexicon(
        const std::vector<LexiconEntry>& entries) {
        (void)entries;
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR, "Lexicon update not supported");
    }

protected:
    ITtsCallback* callback_ = nullptr;

    // -------------------------------------------------------------------------
    // 辅助方法: 触发回调
    // -------------------------------------------------------------------------

    void notifyOpen() {
        if (callback_) callback_->onOpen();
    }

    void notifyComplete() {
        if (callback_) callback_->onComplete();
    }

    void notifyClose() {
        if (callback_) callback_->onClose();
    }

    void notifyAudioChunk(const AudioChunk& chunk) {
        if (callback_) callback_->onAudioChunk(chunk);
    }

    void notifyError(const ErrorInfo& error) {
        if (callback_) callback_->onError(error);
    }

    // -------------------------------------------------------------------------
    // 辅助方法: 文件操作
    // -------------------------------------------------------------------------

    /// @brief 保存音频到WAV文件
    /// @param audio 音频数据
    /// @param file_path 文件路径
    /// @return 错误信息
    ErrorInfo saveToFile(const AudioChunk& audio, const std::string& file_path) {
        if (audio.isEmpty()) {
            return ErrorInfo::error(ErrorCode::INVALID_CONFIG, "Empty audio data");
        }

        // WAV 文件头
        struct WavHeader {
            char riff[4] = {'R', 'I', 'F', 'F'};
            uint32_t file_size;
            char wave[4] = {'W', 'A', 'V', 'E'};
            char fmt[4] = {'f', 'm', 't', ' '};
            uint32_t fmt_size = 16;
            uint16_t audio_format = 1;  // PCM
            uint16_t num_channels = 1;
            uint32_t sample_rate;
            uint32_t byte_rate;
            uint16_t block_align;
            uint16_t bits_per_sample = 16;
            char data[4] = {'d', 'a', 't', 'a'};
            uint32_t data_size;
        };

        auto int16_data = audio.toInt16();
        uint32_t data_size = static_cast<uint32_t>(int16_data.size() * sizeof(int16_t));

        WavHeader header;
        header.sample_rate = static_cast<uint32_t>(audio.sample_rate);
        header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
        header.block_align = header.num_channels * (header.bits_per_sample / 8);
        header.data_size = data_size;
        header.file_size = sizeof(WavHeader) - 8 + data_size;

        FILE* file = fopen(file_path.c_str(), "wb");
        if (!file) {
            return ErrorInfo::error(ErrorCode::FILE_WRITE_ERROR,
                "Failed to open file: " + file_path);
        }

        fwrite(&header, sizeof(WavHeader), 1, file);
        fwrite(int16_data.data(), sizeof(int16_t), int16_data.size(), file);
        fclose(file);

        return ErrorInfo::ok();
    }
};

// =============================================================================
// Backend Factory (后端工厂)
// =============================================================================

class TtsBackendFactory {
public:
    /// @brief 创建TTS后端实例
    /// @param type 后端类型
    /// @return 后端实例, 失败返回nullptr
    static std::unique_ptr<ITtsBackend> create(BackendType type);

    /// @brief 检查后端类型是否可用
    /// @param type 后端类型
    /// @return 是否可用
    static bool isAvailable(BackendType type);

    /// @brief 获取所有可用的后端类型
    static std::vector<BackendType> getAvailableBackends();

    /// @brief 获取后端的默认采样率
    /// @param type 后端类型
    /// @return 采样率 (Hz)
    static int getDefaultSampleRate(BackendType type) {
        return tts::getDefaultSampleRate(type);
    }

    /// @brief 获取后端名称
    /// @param type 后端类型
    /// @return 后端名称字符串
    static const char* getBackendName(BackendType type) {
        return backendTypeToString(type);
    }
};

}  // namespace tts

#endif  // TTS_BACKEND_HPP
