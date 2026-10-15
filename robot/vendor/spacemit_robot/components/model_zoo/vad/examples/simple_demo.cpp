/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * VAD Simple Demo
 *
 * 演示如何使用 SpaceVadSDK 进行语音活动检测。
 */

#include <cmath>
#include <cstdlib>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "vad_service.h"

// 生成测试音频（正弦波模拟语音，静音模拟无语音）
std::vector<float> generateTestAudio(int sample_rate, float duration_sec, bool is_speech) {
    size_t num_samples = static_cast<size_t>(sample_rate * duration_sec);
    std::vector<float> audio(num_samples);

    if (is_speech) {
        // 生成正弦波模拟语音
        float frequency = 440.0f;  // 440Hz
        for (size_t i = 0; i < num_samples; ++i) {
            audio[i] = 0.5f * std::sin(2.0f * M_PI * frequency * i / sample_rate);
        }
    } else {
        // 生成静音（很小的噪声）
        for (size_t i = 0; i < num_samples; ++i) {
            unsigned int seed = static_cast<unsigned int>(i);
            audio[i] = 0.001f * (static_cast<float>(rand_r(&seed)) / RAND_MAX - 0.5f);
        }
    }

    return audio;
}

int main() {
    std::cout << "=== VAD Simple Demo ===" << std::endl;

    // 创建 VAD 引擎（使用默认 Silero 配置）
    auto config = SpacemiT::VadConfig::Preset("silero")
        .withTriggerThreshold(0.5f)
        .withStopThreshold(0.35f);

    auto engine = std::make_shared<SpacemiT::VadEngine>(config);

    if (!engine->IsInitialized()) {
        std::cerr << "Failed to initialize VAD engine" << std::endl;
        std::cerr << "Make sure Silero VAD model is available at "
                << "~/.cache/models/vad/silero/silero_vad.onnx" << std::endl;
        return 1;
    }

    std::cout << "VAD Engine initialized: " << engine->GetEngineName() << std::endl;

    // 测试参数
    int sample_rate = 16000;
    int window_size = 512;  // 32ms at 16kHz
    float window_duration = static_cast<float>(window_size) / sample_rate;
    double frame_duration_ms = static_cast<double>(window_size) * 1000.0 / sample_rate;

    // 生成测试序列：静音 -> 语音 -> 静音
    std::vector<std::pair<std::string, bool>> test_sequence = {
        {"Silence", false},
        {"Silence", false},
        {"Speech",  true},
        {"Speech",  true},
        {"Speech",  true},
        {"Silence", false},
        {"Silence", false},
    };

    std::cout << "\nProcessing test audio frames:" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    std::cout << "Frame duration: " << std::fixed << std::setprecision(2)
        << frame_duration_ms << " ms" << std::endl;

    double total_processing_ms = 0.0;
    double total_audio_ms = 0.0;
    size_t success_count = 0;

    for (size_t i = 0; i < test_sequence.size(); ++i) {
        auto& [label, is_speech] = test_sequence[i];

        // 生成音频帧
        auto audio = generateTestAudio(sample_rate, window_duration, is_speech);

        // 检测
        auto start_time = std::chrono::steady_clock::now();
        auto result = engine->Detect(audio, sample_rate);
        auto end_time = std::chrono::steady_clock::now();

        double processing_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double audio_ms = static_cast<double>(audio.size()) * 1000.0 / sample_rate;
        double rtf = audio_ms > 0.0 ? processing_ms / audio_ms : 0.0;

        if (result && result->IsSuccess()) {
            total_processing_ms += processing_ms;
            total_audio_ms += audio_ms;
            success_count++;

            std::cout << "Frame " << i << " (" << label << "): "
                << "prob=" << result->GetProbability()
                << ", state=" << static_cast<int>(result->GetState())
                << ", processing=" << std::fixed << std::setprecision(3)
                << processing_ms << " ms"
                << ", RTF=" << std::setprecision(4) << rtf;

            if (result->IsSpeechStart()) {
                std::cout << " [SPEECH START]";
            } else if (result->IsSpeechEnd()) {
                std::cout << " [SPEECH END]";
            }

            std::cout << std::endl;
        } else {
            std::cerr << "Detection failed" << std::endl;
        }
    }

    if (success_count > 0 && total_audio_ms > 0.0) {
        double avg_processing_ms = total_processing_ms / success_count;
        double avg_rtf = total_processing_ms / total_audio_ms;

        std::cout << "\nPerformance summary:" << std::endl;
        std::cout << "  Frames: " << success_count << std::endl;
        std::cout << "  Total audio: " << std::fixed << std::setprecision(2)
            << total_audio_ms << " ms" << std::endl;
        std::cout << "  Total processing: " << std::setprecision(3)
            << total_processing_ms << " ms" << std::endl;
        std::cout << "  Avg processing/frame: " << avg_processing_ms << " ms" << std::endl;
        std::cout << "  Avg RTF: " << std::setprecision(4) << avg_rtf << std::endl;
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
