/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpacemitAudioSDK 静态文件识别示例
 *
 * Usage:
 *   ./asr_file_demo <audio1.wav> [audio2.wav ...] [--model-dir DIR] [--rounds N] [--provider EP] [--enable-emotion]
 *
 * Examples:
 *   ./asr_file_demo ~/test.wav
 *   ./asr_file_demo a.wav b.wav c.wav
 *   ./asr_file_demo a.wav b.wav --model-dir ~/.cache/models/asr/sensevoice
 */

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "asr_service.h"

struct FileResult {
    int round;
    std::string file;
    double audio_ms;
    double process_ms;
    double rtf;
    std::string text;
    std::string emotion;
};

class ErrorCaptureCallback : public SpacemiT::AsrEngineCallback {
public:
    void Clear() {
        last_error_.clear();
    }

    const std::string& LastError() const {
        return last_error_;
    }

    void OnError(std::shared_ptr<SpacemiT::RecognitionResult> result) override {
        if (result && !result->GetText().empty()) {
            last_error_ = result->GetText();
        } else {
            last_error_ = "unknown ASR error";
        }
    }

private:
    std::string last_error_;
};

void printUsage(const char* program) {
    std::cout << "Usage: " << program
        << " <audio1.wav> [audio2.wav ...] [OPTIONS]"
        << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  audio files   One or more WAV audio files" << std::endl;
    std::cout << "  --engine      Engine: sensevoice | zipformer | qwen3-asr (default: sensevoice)" << std::endl;
    std::cout << "  --model-dir   Path to SenseVoice model directory" << std::endl;
    std::cout << "                Default: ~/.cache/models/asr/sensevoice" << std::endl;
    std::cout << "  --rounds N    Run N rounds of recognition (default: 1)" << std::endl;
    std::cout << "  --provider    EP: cpu | spacemit (default: spacemit)" << std::endl;
    std::cout << "  --hotwords    Comma-separated hotwords (e.g. \"SpacemiT,进迭时空\")" << std::endl;
    std::cout << "  --hotword-boost  Hotword boost weight (default: 2.0)" << std::endl;
    std::cout << "  --enable-emotion  Enable SenseVoice emotion recognition (default: off)" << std::endl;
    std::cout << "  --endpoint    Qwen3-ASR llama-server URL" << std::endl;
    std::cout << "                Default: http://127.0.0.1:8063/v1/chat/completions" << std::endl;
    std::cout << "  --model       Qwen3-ASR model tag (default: qwen3-asr)" << std::endl;
    std::cout << "  --timeout     Qwen3-ASR timeout in seconds (default: 60)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << " ~/test.wav" << std::endl;
    std::cout << "  " << program << " a.wav --hotwords \"SpacemiT,进迭时空\" --hotword-boost 3.0" << std::endl;
    std::cout << "  " << program
                << " a.wav b.wav --engine qwen3-asr"
                << " --endpoint http://10.0.90.72:8063/v1/chat/completions"
                << std::endl;
    std::cout << "  " << program << " a.wav b.wav --model-dir /path/to/models" << std::endl;
}

std::string expandHome(const std::string& path) {
    if (!path.empty() && path[0] == '~') {
        const char* home = getenv("HOME");
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    // Parse args
    std::vector<std::string> audio_files;
    std::string engine_name = "sensevoice";
    std::string model_dir;
    bool model_dir_set = false;
    std::string provider = "spacemit";
    std::string hotwords_str;
    float hotword_boost = 2.0f;
    std::string endpoint = "http://127.0.0.1:8063/v1/chat/completions";
    std::string model_tag = "qwen3-asr";
    int timeout = 60;
    int rounds = 1;
    bool enable_emotion = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            engine_name = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
            model_dir_set = true;
        } else if (arg == "--rounds" && i + 1 < argc) {
            rounds = std::atoi(argv[++i]);
            if (rounds < 1) rounds = 1;
        } else if (arg == "--provider" && i + 1 < argc) {
            provider = argv[++i];
        } else if (arg == "--hotwords" && i + 1 < argc) {
            hotwords_str = argv[++i];
        } else if (arg == "--hotword-boost" && i + 1 < argc) {
            hotword_boost = std::stof(argv[++i]);
        } else if (arg == "--enable-emotion") {
            enable_emotion = true;
        } else if (arg == "--endpoint" && i + 1 < argc) {
            endpoint = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_tag = argv[++i];
        } else if (arg == "--timeout" && i + 1 < argc) {
            timeout = std::atoi(argv[++i]);
        } else {
            audio_files.push_back(expandHome(argv[i]));
        }
    }

    if (audio_files.empty()) {
        std::cerr << "Error: no audio files specified" << std::endl;
        return 1;
    }

    // Initialize engine once
    std::cout << "========================================" << std::endl;
    std::cout << "    SpacemitAudioSDK 文件识别测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << ">>> 创建 ASR 引擎 (" << engine_name << ")..." << std::endl;
    SpacemiT::AsrConfig config = SpacemiT::AsrConfig::Preset(engine_name);
    config.language = "auto";
    config.punctuation = true;
    config.enable_emotion = enable_emotion;

    if (engine_name == "qwen3-asr") {
        config.endpoint = endpoint;
        config.model = model_tag;
        config.timeout = timeout;
    } else {
        if (model_dir_set) {
            config.model_dir = expandHome(model_dir);
        }
        config.provider = provider;
    }

    // 解析热词
    if (!hotwords_str.empty()) {
        std::vector<std::string> words;
        std::string word;
        for (char c : hotwords_str) {
            if (c == ',') {
                if (!word.empty()) {
                    words.push_back(word);
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        if (!word.empty()) words.push_back(word);
        config.hotwords = words;
        config.hotword_boost = hotword_boost;
    }

    auto engine = std::make_shared<SpacemiT::AsrEngine>(config);
    if (!engine->IsInitialized()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return 1;
    }

    auto error_callback = std::make_shared<ErrorCaptureCallback>();
    engine->SetCallback(error_callback);

    auto cfg = engine->GetConfig();
    std::cout << "引擎类型: " << cfg.engine << std::endl;
    std::cout << "语言: " << cfg.language << std::endl;
    std::cout << "标点: " << (cfg.punctuation ? "启用" : "禁用") << std::endl;
    std::cout << "情绪识别: " << (cfg.enable_emotion ? "启用" : "禁用") << std::endl;
    std::cout << "采样率: " << cfg.sample_rate << " Hz" << std::endl;
    if (engine_name == "qwen3-asr") {
        std::cout << "Endpoint: " << cfg.endpoint << std::endl;
        std::cout << "Model: " << cfg.model << std::endl;
        std::cout << "Timeout: " << cfg.timeout << "s" << std::endl;
    } else {
        std::cout << "Provider: " << provider << std::endl;
    }
    if (!cfg.hotwords.empty()) {
        std::cout << "热词: ";
        for (size_t j = 0; j < cfg.hotwords.size(); ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << cfg.hotwords[j];
        }
        std::cout << " (boost=" << cfg.hotword_boost << ")" << std::endl;
    }
    std::cout << "文件数: " << audio_files.size() << std::endl;
    std::cout << "轮次: " << rounds << std::endl;
    std::cout << std::endl;

    // --- Warmup: 跑一次哑推理，加热 EP JIT 缓存 ---
    if (engine_name != "qwen3-asr") {
        std::cout << ">>> Warmup (excluded from benchmark)..." << std::endl;
        {
            std::vector<float> silence(8000, 0.0f);  // 0.5s 静音 @16kHz
            auto t0 = std::chrono::steady_clock::now();
            engine->Recognize(silence, 16000);
            auto t1 = std::chrono::steady_clock::now();
            double warmup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "Warmup done: " << std::fixed << std::setprecision(0)
                << warmup_ms << " ms" << std::endl;
        }
        std::cout << std::endl;
    }

    // Recognize each file, multiple rounds
    std::vector<FileResult> results;
    bool had_failure = false;

    for (int round = 0; round < rounds; ++round) {
        if (rounds > 1) {
            std::cout << "======== 第 " << (round + 1) << "/" << rounds << " 轮 ========" << std::endl;
        }

        for (size_t i = 0; i < audio_files.size(); ++i) {
            const auto& file = audio_files[i];
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "[" << (i + 1) << "/" << audio_files.size() << "] " << file << std::endl;

            error_callback->Clear();
            auto result = engine->Call(file);

            if (result && !result->IsEmpty()) {
                FileResult fr;
                fr.round = round + 1;
                fr.file = file;
                fr.text = result->GetText();
                fr.emotion = result->GetEmotion();
                fr.audio_ms = result->GetAudioDuration();
                fr.process_ms = result->GetProcessingTime();
                fr.rtf = result->GetRTF();
                results.push_back(fr);

                std::cout << "文本: " << fr.text << std::endl;
                if (!fr.emotion.empty()) {
                    std::cout << "情绪: " << fr.emotion << std::endl;
                }
                std::cout << "音频: " << std::fixed << std::setprecision(0) << fr.audio_ms << " ms"
                    << "  处理: " << fr.process_ms << " ms"
                    << "  RTF: " << std::setprecision(3) << fr.rtf << std::endl;
            } else if (!error_callback->LastError().empty()) {
                had_failure = true;
                std::cerr << "识别失败: " << error_callback->LastError() << std::endl;
            } else {
                had_failure = true;
                std::cerr << "识别失败或未检测到语音" << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Summary table
    if (results.size() > 1) {
        std::cout << "========================================" << std::endl;
        std::cout << "              汇总" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::left << std::setw(40) << "文件"
            << std::right << std::setw(10) << "音频(ms)"
            << std::setw(10) << "处理(ms)"
            << std::setw(8) << "RTF" << std::endl;
        std::cout << std::string(68, '-') << std::endl;

        double total_audio = 0, total_process = 0;
        for (const auto& r : results) {
            // Extract filename from path
            std::string name = r.file;
            size_t pos = name.rfind('/');
            if (pos != std::string::npos) name = name.substr(pos + 1);

            std::cout << std::left << std::setw(40) << name
                << std::right << std::fixed
                << std::setw(10) << std::setprecision(0) << r.audio_ms
                << std::setw(10) << r.process_ms
                << std::setw(8) << std::setprecision(3) << r.rtf << std::endl;
            total_audio += r.audio_ms;
            total_process += r.process_ms;
        }

        std::cout << std::string(68, '-') << std::endl;
        std::cout << std::left << std::setw(40) << "Total"
            << std::right << std::fixed
            << std::setw(10) << std::setprecision(0) << total_audio
            << std::setw(10) << total_process
            << std::setw(8) << std::setprecision(3) << (total_process / total_audio) << std::endl;
    }

    engine.reset();
    std::cout << std::endl << "Done." << std::endl;
    _Exit(had_failure ? 1 : 0);
}
