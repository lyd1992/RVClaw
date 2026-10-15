/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpacemitAudioSDK 流式识别示例 (定时 Flush)
 *
 * 使用 SpacemitAudio::AudioCapture 采集麦克风，每 3 秒自动 Flush 触发识别。
 * 简化演示版本，无 VAD。
 *
 * 音频流程: 16kHz 2ch → 混音 → 16kHz mono → ASR
 *
 * Usage:
 *   ./asr_stream_demo [选项]
 *
 * Examples:
 *   ./asr_stream_demo                # 默认设备，总时长 30 秒
 *   ./asr_stream_demo -l             # 列出设备
 *   ./asr_stream_demo -i 0 -t 60    # 设备 0，总时长 60 秒
 *   ./asr_stream_demo -i 0 -c 1     # 强制单通道采集
 *   ./asr_stream_demo -f 5          # 每 5 秒 flush 一次
 */

#include <csignal>
#include <cmath>
#include <cstring>

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "asr_service.h"
#include "audio_base.hpp"
#include "audio_resampler.hpp"

// 音频参数
constexpr int INPUT_SAMPLE_RATE = 16000;
constexpr int DEFAULT_INPUT_CHANNELS = 2;
constexpr int OUTPUT_SAMPLE_RATE = 16000;
constexpr int OUTPUT_CHANNELS = 1;

// 全局停止标志
static std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    std::cout << "\n收到信号 " << signum << "，停止中..." << std::endl;
    g_running = false;
}

// =============================================================================
// StreamCallback - 自定义回调类（多态）
// =============================================================================

class StreamCallback : public SpacemiT::AsrEngineCallback {
public:
    StreamCallback() = default;

    void OnOpen() override {
        std::cout << "    [回调] 识别开始" << std::endl;
        start_time_ = std::chrono::steady_clock::now();
    }

    void OnEvent(std::shared_ptr<SpacemiT::RecognitionResult> result) override {
        if (!result) return;

        std::string text = result->GetText();
        bool is_final = result->IsSentenceEnd();

        if (!text.empty()) {
            if (is_final) {
                std::cout << "    [回调] 最终结果: " << text << std::endl;
                const std::string emotion = result->GetEmotion();
                if (!emotion.empty()) {
                    std::cout << "    [回调] 情绪: " << emotion << std::endl;
                }

                // 显示详细信息（仅最终结果）
                auto sentences = result->GetSentences();
                if (sentences.size() > 1) {
                    std::cout << "    [回调] 句子数: " << sentences.size() << std::endl;
                }

                // 性能信息
                if (result->GetAudioDuration() > 0) {
                    std::cout << "    [回调] 音频: " << result->GetAudioDuration()
                        << "ms, 处理: " << result->GetProcessingTime()
                        << "ms, RTF: " << std::fixed << std::setprecision(3)
                        << result->GetRTF() << std::endl;
                }
            } else {
                std::cout << "    [回调] 中间结果: " << text << std::endl;
            }
        }

        // 保存最后的结果
        {
            std::lock_guard<std::mutex> lock(mutex_);
            last_text_ = text;
            is_final_ = is_final;
            last_result_ = result;
        }
    }

    void OnComplete() override {
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time_).count();
        std::cout << "    [回调] 识别完成 (耗时: " << duration_ms << " ms)" << std::endl;
    }

    void OnError(std::shared_ptr<SpacemiT::RecognitionResult> result) override {
        if (result) {
            std::cerr << "    [回调] 错误: " << result->GetText() << std::endl;
        } else {
            std::cerr << "    [回调] 未知错误" << std::endl;
        }
    }

    void OnClose() override {
        std::cout << "    [回调] 连接关闭" << std::endl;
    }

    // 获取最后的识别结果
    std::string getLastText() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_text_;
    }

    bool isFinal() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_final_;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        last_text_.clear();
        is_final_ = false;
        last_result_.reset();
    }

    // 获取最后的完整结果
    std::shared_ptr<SpacemiT::RecognitionResult> getLastResult() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_result_;
    }

private:
    mutable std::mutex mutex_;
    std::string last_text_;
    bool is_final_ = false;
    std::shared_ptr<SpacemiT::RecognitionResult> last_result_;
    std::chrono::steady_clock::time_point start_time_;
};

// =============================================================================
// Utility Functions
// =============================================================================

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [选项]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input <N>    输入设备索引 (-1 为默认)" << std::endl;
    std::cout << "  -c, --channels <N> 输入通道数 (默认 2, K3 麦克风使用 2)" << std::endl;
    std::cout << "  -t, --time <N>     录音时长秒数 (默认 30)" << std::endl;
    std::cout << "  -f, --flush <N>    Flush 间隔秒数 (默认 3)" << std::endl;
    std::cout << "  -l, --list         列出可用音频设备" << std::endl;
    std::cout << "  -p, --provider <EP> EP: cpu | spacemit (默认 spacemit)" << std::endl;
    std::cout << "  --enable-emotion   启用 SenseVoice 情绪识别 (默认关闭)" << std::endl;
    std::cout << "  -h, --help         显示帮助" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << "              # 默认设备，30 秒，每 3 秒 flush" << std::endl;
    std::cout << "  " << program << " -l           # 列出设备" << std::endl;
    std::cout << "  " << program << " -i 0 -t 60   # 设备 0，60 秒" << std::endl;
    std::cout << "  " << program << " -i 0 -c 1    # 强制单通道采集" << std::endl;
    std::cout << "  " << program << " -f 5         # 每 5 秒 flush 一次" << std::endl;
}

void listDevices() {
    std::cout << "可用音频输入设备:" << std::endl;
    std::cout << "==================" << std::endl;

    auto devices = SpacemitAudio::AudioCapture::ListDevices();

    if (devices.empty()) {
        std::cout << "  未找到设备!" << std::endl;
    } else {
        for (const auto& [index, name] : devices) {
            std::cout << "  [" << index << "] " << name << std::endl;
        }
    }
    std::cout << std::endl;
}

// 线程安全的音频缓冲区（带重采样）
class AudioBuffer {
public:
    AudioBuffer()
        : resampler_(makeResamplerConfig(DEFAULT_INPUT_CHANNELS))
        , channels_(DEFAULT_INPUT_CHANNELS) {
        resampler_.initialize();
    }

    void setChannels(int channels) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (channels != channels_) {
            channels_ = channels;
            resampler_ = Resampler(makeResamplerConfig(channels));
            resampler_.initialize();
        }
    }

    // 接收音频数据，重采样后存储为 16kHz mono
    void append(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        const int16_t* samples = reinterpret_cast<const int16_t*>(data);
        size_t num_samples = size / sizeof(int16_t);
        size_t num_frames = num_samples / channels_;

        // 转换为 float 并混合为单声道
        std::vector<float> mono_float(num_frames);
        if (channels_ == 1) {
            for (size_t i = 0; i < num_frames; ++i) {
                mono_float[i] = samples[i] / 32768.0f;
            }
        } else {
            for (size_t i = 0; i < num_frames; ++i) {
                float sum = 0.0f;
                for (int ch = 0; ch < channels_; ++ch) {
                    sum += samples[i * channels_ + ch] / 32768.0f;
                }
                mono_float[i] = sum / channels_;
            }
        }

        // 重采样 (如果需要)
        const std::vector<float>* output_data = &mono_float;
        std::vector<float> resampled;
        if (INPUT_SAMPLE_RATE != OUTPUT_SAMPLE_RATE) {
            resampled = resampler_.process(mono_float);
            output_data = &resampled;
        }

        // 转换回 int16_t 并存储
        size_t old_size = buffer_.size();
        buffer_.resize(old_size + output_data->size() * sizeof(int16_t));
        int16_t* out_ptr = reinterpret_cast<int16_t*>(buffer_.data() + old_size);
        for (size_t i = 0; i < output_data->size(); ++i) {
            float sample = (*output_data)[i];
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
            out_ptr[i] = static_cast<int16_t>(sample * 32767.0f);
        }
    }

    std::vector<uint8_t> getAndClear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<uint8_t> result = std::move(buffer_);
        buffer_.clear();
        return result;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

    // 返回 16kHz mono 样本数
    size_t sampleCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size() / sizeof(int16_t);
    }

    // 返回秒数 (基于 16kHz)
    float durationSeconds() const {
        return sampleCount() / static_cast<float>(OUTPUT_SAMPLE_RATE);
    }

private:
    static Resampler::Config makeResamplerConfig(int channels) {
        Resampler::Config config;
        config.input_sample_rate = INPUT_SAMPLE_RATE;
        config.output_sample_rate = OUTPUT_SAMPLE_RATE;
        config.channels = 1;  // 已经混合为单声道
        config.method = ResampleMethod::LINEAR_DOWNSAMPLE;
        (void)channels;  // 参数仅用于未来扩展
        return config;
    }

    mutable std::mutex mutex_;
    std::vector<uint8_t> buffer_;  // 存储 16kHz mono PCM16
    Resampler resampler_;
    int channels_ = DEFAULT_INPUT_CHANNELS;
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // 解析参数
    int device_index = -1;
    int input_channels = DEFAULT_INPUT_CHANNELS;
    int total_seconds = 30;
    int flush_interval = 3;  // 每 3 秒 flush 一次
    std::string provider = "spacemit";
    bool enable_emotion = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-l" || arg == "--list") {
            listDevices();
            return 0;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "-f" || arg == "--flush") && i + 1 < argc) {
            flush_interval = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            device_index = std::stoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--channels") && i + 1 < argc) {
            input_channels = std::stoi(argv[++i]);
            if (input_channels < 1) input_channels = DEFAULT_INPUT_CHANNELS;
        } else if ((arg == "-t" || arg == "--time") && i + 1 < argc) {
            total_seconds = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--provider") && i + 1 < argc) {
            provider = argv[++i];
        } else if (arg == "--enable-emotion") {
            enable_emotion = true;
        }
    }

    // 设置信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "========================================" << std::endl;
    std::cout << "  SpacemitAudioSDK 流式识别 (定时 Flush)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    std::cout << "设备索引: " << (device_index == -1 ? "默认" : std::to_string(device_index)) << std::endl;
    std::cout << "输入通道: " << input_channels << std::endl;
    std::cout << "总时长: " << total_seconds << " 秒" << std::endl;
    std::cout << "Flush 间隔: " << flush_interval << " 秒" << std::endl;
    std::cout << "Provider: " << provider << std::endl;
    std::cout << "情绪识别: " << (enable_emotion ? "启用" : "禁用") << std::endl;
    std::cout << std::endl;

    // 列出设备
    listDevices();

    // 使用 AsrConfig 创建引擎
    std::cout << ">>> 初始化 ASR 引擎 (使用 AsrConfig)..." << std::endl;

    SpacemiT::AsrConfig config = SpacemiT::AsrConfig::Preset("sensevoice");  // 使用默认模型路径
    config.language = "zh";
    config.punctuation = true;
    config.provider = provider;
    config.enable_emotion = enable_emotion;

    auto asrEngine = std::make_shared<SpacemiT::AsrEngine>(config);

    if (!asrEngine->IsInitialized()) {
        std::cerr << "ASR 引擎初始化失败!" << std::endl;
        return 1;
    }

    // 显示当前配置
    auto cfg = asrEngine->GetConfig();
    std::cout << "引擎: " << cfg.engine << ", 语言: " << cfg.language
        << ", 标点: " << (cfg.punctuation ? "是" : "否")
        << ", 情绪识别: " << (cfg.enable_emotion ? "是" : "否") << std::endl;
    std::cout << std::endl;

    std::cout << ">>> Warmup..." << std::endl;
    {
        std::vector<float> silence(8000, 0.0f);
        auto t0 = std::chrono::steady_clock::now();
        asrEngine->Recognize(silence, 16000);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Warmup done: " << std::fixed << std::setprecision(0)
            << ms << " ms" << std::endl;
    }
    std::cout << std::endl;

    // 创建回调实例（多态）
    auto callback = std::make_shared<StreamCallback>();
    asrEngine->SetCallback(callback);
    std::cout << ">>> 已设置流式回调 (AsrEngineCallback 多态)" << std::endl;
    std::cout << std::endl;

    // 音频缓冲区
    AudioBuffer audio_buffer;
    audio_buffer.setChannels(input_channels);

    // 创建音频采集器
    SpacemitAudio::AudioCapture capture(device_index);

    // 设置回调 - 仅收集音频
    capture.SetCallback([&](const uint8_t* data, size_t size) {
        audio_buffer.append(data, size);
    });

    // 启动采集
    std::cout << ">>> 启动音频采集..." << std::endl;
    if (!capture.Start(INPUT_SAMPLE_RATE, input_channels, 4096)) {
        std::cerr << "音频采集启动失败!" << std::endl;
        std::cerr << "尝试运行 -l 查看可用设备" << std::endl;
        return 1;
    }

    std::cout << "音频采集已启动 (" << INPUT_SAMPLE_RATE << "Hz, "
        << input_channels << "ch → " << OUTPUT_SAMPLE_RATE << "Hz mono)" << std::endl;
    std::cout << "每 " << flush_interval << " 秒自动 Flush 触发识别" << std::endl;
    std::cout << "按 Ctrl+C 停止" << std::endl;
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "         实时识别结果" << std::endl;
    std::cout << "========================================" << std::endl;

    // 启动流式会话 (只启动一次)
    asrEngine->Start();

    // 主循环：每 flush_interval 秒 Flush 一次
    auto start_time = std::chrono::steady_clock::now();
    auto last_flush_time = start_time;
    int sentence_count = 0;

    while (g_running) {
        auto now = std::chrono::steady_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        auto since_last_flush = std::chrono::duration_cast<std::chrono::seconds>(now - last_flush_time).count();

        // 检查是否到达总时长
        if (total_elapsed >= total_seconds) {
            break;
        }

        // 检查是否需要 Flush (定时)
        if (since_last_flush >= flush_interval) {
            // 获取当前缓冲区的音频数据
            auto pcm_data = audio_buffer.getAndClear();

            if (!pcm_data.empty()) {
                // 发送音频
                asrEngine->SendAudioFrame(pcm_data);

                // Flush 触发识别
                sentence_count++;
                std::cout << "\n[句子 " << sentence_count << "] 定时 Flush ("
                    << flush_interval << "s)..." << std::endl;
                asrEngine->Flush();

                last_flush_time = now;
            }
        } else {
            // 定期发送累积的音频 (不触发识别)
            auto pcm_data = audio_buffer.getAndClear();
            if (!pcm_data.empty()) {
                asrEngine->SendAudioFrame(pcm_data);
            }
        }

        // 显示状态
        std::cout << "\r[" << total_elapsed << "/" << total_seconds << "s] "
            << "句子: " << sentence_count
            << ", 下次 Flush: " << (flush_interval - since_last_flush) << "s"
            << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << std::endl;

    // 处理剩余的音频数据
    auto remaining_data = audio_buffer.getAndClear();
    if (!remaining_data.empty()) {
        asrEngine->SendAudioFrame(remaining_data);
        sentence_count++;
        std::cout << "\n[句子 " << sentence_count << " - 剩余] Flush..." << std::endl;
        asrEngine->Flush();
    }

    // 停止流式会话
    asrEngine->Stop();

    // 停止采集
    std::cout << "\n>>> 停止音频采集..." << std::endl;
    capture.Stop();
    capture.Close();

    // 显示统计
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "共识别 " << sentence_count << " 句" << std::endl;
    std::cout << "========================================" << std::endl;

    // 显示识别结果汇总
    if (!callback->getLastText().empty()) {
        std::cout << std::endl;
        std::cout << ">>> 最后识别结果: " << callback->getLastText() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Done." << std::endl;

    return 0;
}
