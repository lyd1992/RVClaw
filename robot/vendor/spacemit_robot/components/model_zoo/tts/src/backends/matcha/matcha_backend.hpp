/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef MATCHA_BACKEND_HPP
#define MATCHA_BACKEND_HPP

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/matcha/matcha_tts_config.hpp"
#include "backends/tts_backend.hpp"

namespace tts {

// =============================================================================
// Matcha Backend Base Class
// =============================================================================
//
// 所有 Matcha-TTS 后端的公共基类。
// 实现 ONNX 模型加载、声学模型推理、声码器处理等公共功能。
// 派生类只需实现 textToTokenIds() 方法。
//
// 继承结构:
//   ITtsBackend
//       └── MatchaBackend (此类 - 公共基类)
//               ├── MatchaZhBackend    (中文)
//               ├── MatchaEnBackend    (英文)
//               └── MatchaZhEnBackend  (中英混合)
//

class MatchaBackend : public ITtsBackend {
public:
    /// @brief 构造函数
    /// @param type 后端类型 (MATCHA_ZH, MATCHA_EN, MATCHA_ZH_EN)
    explicit MatchaBackend(BackendType type);

    ~MatchaBackend() override;

    // -------------------------------------------------------------------------
    // ITtsBackend 接口实现
    // -------------------------------------------------------------------------

    ErrorInfo initialize(const TtsConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override;

    BackendType getType() const override;
    std::string getName() const override;
    std::string getVersion() const override;
    bool supportsStreaming() const override;
    int getNumSpeakers() const override;
    int getSampleRate() const override;

    ErrorInfo synthesize(const std::string& text, SynthesisResult& result) override;
    ErrorInfo synthesizeToFile(const std::string& text, const std::string& file_path) override;

    ErrorInfo setSpeed(float speed) override;
    ErrorInfo setSpeaker(int speaker_id) override;

protected:
    // -------------------------------------------------------------------------
    // 派生类必须实现的方法
    // -------------------------------------------------------------------------

    /// @brief 将文本转换为 token IDs
    /// @param text 输入文本
    /// @return token ID 列表
    virtual std::vector<int64_t> textToTokenIds(const std::string& text) = 0;

    /// @brief 获取模型子目录名
    /// @return 模型目录名 (如 "matcha-icefall-zh-baker")
    virtual std::string getModelSubdir() const = 0;

    /// @brief 是否使用 blank tokens
    /// @return true 如果需要在 tokens 之间插入 blank
    virtual bool usesBlankTokens() const = 0;

    /// @brief 派生类特有的初始化
    /// @param config 配置
    /// @return 错误信息
    virtual ErrorInfo initializeLanguageSpecific(const TtsConfig& config) = 0;

    /// @brief 派生类特有的清理
    virtual void shutdownLanguageSpecific() {}

    // -------------------------------------------------------------------------
    // 受保护的辅助方法 (供派生类使用)
    // -------------------------------------------------------------------------

    /// @brief 获取 token 到 ID 的映射表
    const std::unordered_map<std::string, int64_t>& getTokenToIdMap() const { return token_to_id_; }

    /// @brief 获取配置
    const TtsConfig& getConfig() const { return config_; }

    /// @brief 获取旧配置 (TTSConfig)
    const TTSConfig& getInternalConfig() const { return internal_config_; }

    /// @brief 获取模型目录
    std::string getModelDir() const;

    /// @brief 添加 blank tokens
    /// @param tokens 原始 tokens
    /// @return 带 blank 的 tokens
    std::vector<int64_t> addBlankTokens(const std::vector<int64_t>& tokens);

    /// @brief 将 token IDs 合成为音频样本
    std::vector<float> synthesizeTokenIdsToAudio(
        const std::vector<int64_t>& token_ids,
        int* output_sample_rate);

    /// @brief 检查 espeak-ng 是否可用
    bool checkEspeakNgAvailable();

    /// @brief 使用 espeak-ng 将英文转 IPA
    std::string processEnglishTextToPhonemes(const std::string& text);

private:
    // -------------------------------------------------------------------------
    // ONNX 推理
    // -------------------------------------------------------------------------

    /// @brief 运行声学模型
    std::vector<float> runAcousticModel(
        const std::vector<int64_t>& tokens,
        int speaker_id,
        float speed,
        bool use_cpu_fallback = false);

    /// @brief 运行声码器
    std::vector<float> runVocoder(const std::vector<float>& mel, int mel_dim);

    /// @brief 创建内部配置
    void createInternalConfig();

    /// @brief 提取模型元数据
    void extractModelMetadata();

    /// @brief 预热模型
    void warmUpModels();

    /// @brief Finish ONNX Runtime profiling and print emitted file paths.
    void endOrtProfiling();

protected:
    // 后端类型
    BackendType type_;

    // 配置
    TtsConfig config_;
    TTSConfig internal_config_;

    // Token 映射
    std::unordered_map<std::string, int64_t> token_to_id_;

    // 模型参数
    int mel_dim_ = 80;
    int num_speakers_ = 1;
    int64_t pad_id_ = 0;
    int sample_rate_ = 22050;

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> acoustic_model_;
    std::unique_ptr<Ort::Session> acoustic_cpu_fallback_model_;
    std::unique_ptr<Ort::Session> vocoder_model_;
    bool trace_enabled_ = false;
    bool ort_profiling_enabled_ = false;
    std::string ort_profile_prefix_;

    // 状态
    bool initialized_ = false;

    // 当前参数
    float current_speed_ = 1.0f;
    int current_speaker_ = 0;

    // 线程安全
    mutable std::mutex inference_mutex_;

    // ISTFT 参数 (从 vocoder 元数据读取)
    int32_t istft_n_fft_ = 1024;
    int32_t istft_hop_length_ = 256;
    int32_t istft_win_length_ = 1024;
};

// =============================================================================
// 派生类声明 (forward declarations)
// =============================================================================

class MatchaZhBackend;
class MatchaEnBackend;
class MatchaZhEnBackend;

}  // namespace tts

#endif  // MATCHA_BACKEND_HPP
