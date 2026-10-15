/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstdint>
#include <cstring>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio_base.hpp"
#include "audio_resampler.hpp"
#include "tts_service.h"

// =============================================================================
// 引擎选择 (复用 tts_file_demo 的格式: matcha:zh-en, kokoro:xiaoxiao)
// =============================================================================

struct EngineSelection {
    SpacemiT::BackendType backend;
    std::string voice;  // Only used by Kokoro
};

// Kokoro known voices: {full_name, short_name}
static const std::vector<std::pair<std::string, std::string>> kKokoroVoices = {
    // Chinese female
    {"zf_xiaobei",  "xiaobei"},
    {"zf_xiaoni",   "xiaoni"},
    {"zf_xiaoxiao", "xiaoxiao"},
    {"zf_xiaoyi",   "xiaoyi"},
    // Chinese male
    {"zm_yunxi",    "yunxi"},
    {"zm_yunyang",  "yunyang"},
    {"zm_yunjian",  "yunjian"},
    {"zm_yunfan",   "yunfan"},
    // American English female
    {"af_heart",    "heart"},
    {"af_alloy",    "alloy"},
    {"af_aoede",    "aoede"},
    {"af_bella",    "bella"},
    {"af_jessica",  "jessica"},
    {"af_kore",     "kore"},
    {"af_nicole",   "nicole"},
    {"af_nova",     "nova"},
    {"af_river",    "river"},
    {"af_sarah",    "sarah"},
    {"af_sky",      "sky"},
    // American English male
    {"am_adam",     "adam"},
    {"am_echo",     "echo"},
    {"am_eric",     "eric"},
    {"am_fenrir",   "fenrir"},
    {"am_liam",     "liam"},
    {"am_michael",  "michael"},
    {"am_onyx",     "onyx"},
    {"am_puck",     "puck"},
    // British English female
    {"bf_alice",    "alice"},
    {"bf_emma",     "emma"},
    {"bf_isabella", "isabella"},
    {"bf_lily",     "lily"},
    // British English male
    {"bm_daniel",   "daniel"},
    {"bm_fable",    "fable"},
    {"bm_george",   "george"},
    {"bm_lewis",    "lewis"},
};

std::string resolveVoiceName(const std::string& input) {
    if (input.empty()) return input;

    if (input.find('_') != std::string::npos) {
        return input;
    }

    std::vector<std::string> matches;
    for (const auto& [full, shortname] : kKokoroVoices) {
        if (shortname == input) {
            matches.push_back(full);
        }
    }

    if (matches.size() == 1) {
        std::cout << "音色: " << input << " -> " << matches[0] << std::endl;
        return matches[0];
    }

    if (matches.size() > 1) {
        std::cerr << "错误: 音色名 '" << input << "' 有多个匹配:\n";
        for (const auto& m : matches) {
            std::cerr << "  " << m << "\n";
        }
        std::cerr << "请使用完整名称，如 -l kokoro:" << matches[0] << "\n";
        exit(1);
    }

    std::cerr << "警告: 未知音色 '" << input << "'，将直接使用该名称\n";
    return input;
}

EngineSelection parseEngine(const std::string& spec) {
    EngineSelection sel;
    sel.backend = SpacemiT::BackendType::MATCHA_ZH_EN;

    auto colon = spec.find(':');
    std::string engine = (colon != std::string::npos) ? spec.substr(0, colon) : spec;
    std::string variant = (colon != std::string::npos) ? spec.substr(colon + 1) : "";

    if (engine == "matcha") {
        if (variant.empty() || variant == "zh-en" || variant == "zhen") {
            sel.backend = SpacemiT::BackendType::MATCHA_ZH_EN;
        } else if (variant == "zh") {
            sel.backend = SpacemiT::BackendType::MATCHA_ZH;
        } else if (variant == "en") {
            sel.backend = SpacemiT::BackendType::MATCHA_EN;
        } else {
            std::cerr << "错误: 未知 Matcha 变体 '" << variant << "'\n"
                << "可用变体: zh, en, zh-en\n";
            exit(1);
        }
        return sel;
    }

    if (engine == "kokoro") {
        sel.backend = SpacemiT::BackendType::KOKORO;
        sel.voice = resolveVoiceName(variant);
        return sel;
    }

    // Hint for old format users
    if (engine == "zh" || engine == "en" || engine == "zh-en" || engine == "zhen") {
        std::cerr << "错误: 旧格式 '-l " << spec << "' 已不再支持\n"
            << "请使用新格式: -l matcha:" << spec << "\n";
        exit(1);
    }

    std::cerr << "错误: 未知引擎 '" << engine << "'\n"
        << "可用引擎: matcha, kokoro\n"
        << "用法: -l matcha:zh 或 -l kokoro:zf_xiaobei\n";
    exit(1);
}

void printVoiceList() {
    std::cout << "Kokoro 可用音色列表:\n"
        << "\n"
        << "中文女声 (zf_):\n"
        << "  zf_xiaobei      小北 (默认)\n"
        << "  zf_xiaoni       小妮\n"
        << "  zf_xiaoxiao     小小\n"
        << "  zf_xiaoyi       小一\n"
        << "\n"
        << "中文男声 (zm_):\n"
        << "  zm_yunxi        云希\n"
        << "  zm_yunyang      云阳\n"
        << "  zm_yunjian      云健\n"
        << "  zm_yunfan       云帆\n"
        << "\n"
        << "美式英语女声 (af_):\n"
        << "  af_heart        Heart\n"
        << "  af_alloy        Alloy\n"
        << "  af_aoede        Aoede\n"
        << "  af_bella        Bella\n"
        << "  af_jessica      Jessica\n"
        << "  af_kore         Kore\n"
        << "  af_nicole       Nicole\n"
        << "  af_nova         Nova\n"
        << "  af_river        River\n"
        << "  af_sarah        Sarah\n"
        << "  af_sky          Sky\n"
        << "\n"
        << "美式英语男声 (am_):\n"
        << "  am_adam         Adam\n"
        << "  am_echo         Echo\n"
        << "  am_eric         Eric\n"
        << "  am_fenrir       Fenrir\n"
        << "  am_liam         Liam\n"
        << "  am_michael      Michael\n"
        << "  am_onyx         Onyx\n"
        << "  am_puck         Puck\n"
        << "\n"
        << "英式英语女声 (bf_):\n"
        << "  bf_alice        Alice\n"
        << "  bf_emma         Emma\n"
        << "  bf_isabella     Isabella\n"
        << "  bf_lily         Lily\n"
        << "\n"
        << "英式英语男声 (bm_):\n"
        << "  bm_daniel       Daniel\n"
        << "  bm_fable        Fable\n"
        << "  bm_george       George\n"
        << "  bm_lewis        Lewis\n"
        << "\n"
        << "用法: -l kokoro:<voice>  支持短名 (xiaobei) 和全名 (zf_xiaobei)\n"
        << std::endl;
}

// =============================================================================
// 音频处理工具
// =============================================================================

/**
 * @brief 将音频上采样到指定采样率
 * @param samples 输入音频样本 (int16)
 * @param src_rate 源采样率
 * @param dst_rate 目标采样率
 * @return 重采样后的音频
 */
std::vector<int16_t> resampleAudio(const std::vector<int16_t>& samples, int src_rate, int dst_rate) {
    if (samples.empty() || src_rate == dst_rate) {
        return samples;
    }

    // 转换为 float
    std::vector<float> input_float(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        input_float[i] = static_cast<float>(samples[i]) / 32768.0f;
    }

    // 创建重采样器
    Resampler::Config config;
    config.input_sample_rate = src_rate;
    config.output_sample_rate = dst_rate;
    config.channels = 1;
    config.method = ResampleMethod::LINEAR_UPSAMPLE;

    Resampler resampler(config);
    if (!resampler.initialize()) {
        std::cerr << "[重采样] 初始化失败" << std::endl;
        return samples;
    }

    // 重采样
    std::vector<float> output_float = resampler.process(input_float);

    // 转回 int16
    std::vector<int16_t> output(output_float.size());
    for (size_t i = 0; i < output_float.size(); ++i) {
        float s = std::clamp(output_float[i], -1.0f, 1.0f);
        output[i] = static_cast<int16_t>(s * 32767.0f);
    }

    return output;
}

/**
 * @brief 将单声道音频转换为双声道
 * @param samples 单声道音频样本
 * @return 双声道音频样本 (L, R, L, R, ...)
 */
std::vector<int16_t> monoToStereo(const std::vector<int16_t>& samples) {
    std::vector<int16_t> stereo(samples.size() * 2);
    for (size_t i = 0; i < samples.size(); ++i) {
        stereo[i * 2] = samples[i];      // Left
        stereo[i * 2 + 1] = samples[i];  // Right
    }
    return stereo;
}

// =============================================================================
// 音频队列 - 线程安全的音频块队列
// =============================================================================

struct AudioChunk {
    std::vector<int16_t> samples;
    int sample_rate;
    int sentence_index;
    bool is_end_marker;  // 是否是结束标记
};

class AudioQueue {
public:
    void push(AudioChunk chunk) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(chunk));
        cv_.notify_one();
    }

    bool pop(AudioChunk& chunk, int timeout_ms = 5000) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                [this] { return !queue_.empty() || stopped_; })) {
            return false;  // 超时
        }
        if (stopped_ && queue_.empty()) {
            return false;
        }
        chunk = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<AudioChunk> queue_;
    bool stopped_ = false;
};

// =============================================================================
// 文本切分器 - 按标点符号切分
// =============================================================================

class TextSplitter {
public:
    // 按标点符号切分文本
    std::vector<std::string> split(const std::string& text) {
        std::vector<std::string> sentences;
        std::string buffer;

        for (size_t i = 0; i < text.size(); ) {
            // 获取一个 UTF-8 字符
            unsigned char c = text[i];
            int char_len = 1;
            if ((c & 0x80) == 0) {
                char_len = 1;  // ASCII
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;  // 中文
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }

            std::string ch = text.substr(i, char_len);
            buffer += ch;
            i += char_len;

            // 遇到句末标点，输出完整句子
            if (isSentenceEnd(ch) && !buffer.empty()) {
                sentences.push_back(buffer);
                buffer.clear();
            }
        }

        // 输出剩余文本
        if (!buffer.empty()) {
            sentences.push_back(buffer);
        }

        return sentences;
    }

private:
    bool isSentenceEnd(const std::string& ch) {
        // 中文句末标点
        static const std::vector<std::string> cn_puncts = {
            "。", "！", "？", "；"
        };
        for (const auto& p : cn_puncts) {
            if (ch == p) return true;
        }
        // 英文句末标点
        static const std::string en_puncts = ".!?;";
        if (ch.size() == 1 && en_puncts.find(ch[0]) != std::string::npos) {
            return true;
        }
        return false;
    }
};

// =============================================================================
// 合成线程 - 模拟 LLM 输出并逐句合成
// =============================================================================

void synthesisThread(SpacemiT::TtsEngine& engine,
        const std::vector<std::string>& sentences,
        AudioQueue& queue,
        int delay_ms) {
    std::cout << "[合成] 开始合成, 共 " << sentences.size() << " 句" << std::endl;

    int sentence_index = 0;
    for (const auto& sentence : sentences) {
        sentence_index++;

        // 模拟 LLM 输出延迟 (按字符数计算)
        int char_count = 0;
        for (size_t i = 0; i < sentence.size(); ) {
            unsigned char c = sentence[i];
            int char_len = 1;
            if ((c & 0x80) == 0) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            i += char_len;
            char_count++;
        }

        std::cout << "[LLM] 生成第 " << sentence_index << " 句 (" << char_count << " 字): "
                << sentence << std::endl;

        // 模拟 LLM 生成延迟
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms * char_count));

        // 合成
        auto start = std::chrono::steady_clock::now();
        auto result = engine.Call(sentence);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (result && result->IsSuccess()) {
            auto audio_int16 = result->GetAudioInt16();
            int sample_rate = result->GetSampleRate();
            int duration_ms = result->GetDurationMs();
            float rtf = result->GetRTF();

            std::cout << "[合成] 第 " << sentence_index << " 句完成: "
                    << audio_int16.size() << " 样本, "
                    << duration_ms << " ms, "
                    << "耗时 " << elapsed << " ms, "
                    << "RTF=" << rtf << std::endl;

            // 放入队列
            AudioChunk chunk;
            chunk.samples = std::move(audio_int16);
            chunk.sample_rate = sample_rate;
            chunk.sentence_index = sentence_index;
            chunk.is_end_marker = false;
            queue.push(std::move(chunk));
        } else {
            std::cerr << "[合成] 第 " << sentence_index << " 句失败: "
                    << (result ? result->GetMessage() : "unknown error") << std::endl;
        }
    }

    // 发送结束标记
    AudioChunk end_chunk;
    end_chunk.is_end_marker = true;
    end_chunk.sentence_index = -1;
    queue.push(std::move(end_chunk));

    std::cout << "[合成] 全部完成" << std::endl;
}

// =============================================================================
// 播放线程
// =============================================================================

void playbackThread(AudioQueue& queue, bool enable_play, int output_rate, int channels, int device_index) {
    std::unique_ptr<SpacemitAudio::AudioPlayer> player;

    if (enable_play) {
        player = std::make_unique<SpacemitAudio::AudioPlayer>(device_index);
        if (!player->Start(output_rate, channels)) {
            std::cerr << "[播放] 启动播放器失败" << std::endl;
            enable_play = false;
        } else {
            std::cout << "[播放] 播放器已启动, 采样率: " << output_rate << " Hz, 声道: " << channels << std::endl;
        }
    }

    int played_count = 0;
    while (true) {
        AudioChunk chunk;
        if (!queue.pop(chunk, 10000)) {
            std::cout << "[播放] 队列超时，退出" << std::endl;
            break;
        }

        // 检查结束标记
        if (chunk.is_end_marker) {
            std::cout << "[播放] 收到结束标记" << std::endl;
            break;
        }

        if (chunk.samples.empty()) {
            continue;
        }

        played_count++;
        int duration_ms = chunk.samples.size() * 1000 / chunk.sample_rate;
        std::cout << "[播放] 播放第 " << chunk.sentence_index << " 句: "
                << chunk.samples.size() << " 样本 @ " << chunk.sample_rate << " Hz, "
                << duration_ms << " ms" << std::endl;

        if (enable_play && player) {
            // 重采样到目标采样率
            auto resampled = resampleAudio(chunk.samples, chunk.sample_rate, output_rate);
            std::cout << "[播放] 重采样: " << chunk.sample_rate << " -> " << output_rate
                    << " Hz (" << resampled.size() << " 样本)" << std::endl;

            // 声道转换
            std::vector<int16_t> final_audio;
            if (channels == 2) {
                final_audio = monoToStereo(resampled);
                std::cout << "[播放] 转换为双声道 (" << final_audio.size() << " 样本)" << std::endl;
            } else {
                final_audio = std::move(resampled);
            }

            // 转换为字节数组
            std::vector<uint8_t> bytes(final_audio.size() * 2);
            memcpy(bytes.data(), final_audio.data(), bytes.size());
            player->Write(bytes);
            // Write 是阻塞的，不需要额外等待
        } else {
            // 不播放时模拟播放延迟
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    if (player) {
        player->Stop();
        player->Close();
    }

    std::cout << "[播放] 播放完成, 共 " << played_count << " 句" << std::endl;
}

// =============================================================================
// 主函数
// =============================================================================

void printUsage(const char* program) {
    std::cout << "用法: " << program << " [选项]\n"
        << "\n"
        << "选项:\n"
        << "  -p <text>         自定义文本\n"
        << "  -e <engine>       引擎选择 (格式: 引擎:变体)\n"
        << "                    matcha:zh / matcha:en / matcha:zh-en (默认)\n"
        << "                    kokoro / kokoro:<voice>\n"
        << "  -o <device>       输出设备索引 (-1 为默认)\n"
        << "  -l, --list        列出可用音频输出设备\n"
        << "  --provider <provider>           推理后端: auto/cpu/spacemit (默认: auto)\n"
        << "  --output-rate <N> 输出采样率 (默认: 48000)\n"
        << "  --channels <N>    输出声道数: 1=单声道, 2=双声道 (默认: 1)\n"
        << "  --no-play         不播放音频\n"
        << "  --delay <ms>      模拟 LLM 输出延迟 (默认: 5 ms/字符)\n"
        << "  --list-voices     列出 Kokoro 可用音色\n"
        << "  -h                显示帮助\n"
        << "\n"
        << "示例:\n"
        << "  " << program << " -l                              # 列出输出设备\n"
        << "  " << program << " -p \"你好\" -e matcha:zh           # 中文合成\n"
        << "  " << program << " -p \"你好\" -e kokoro:xiaoxiao     # Kokoro 音色\n"
        << "  " << program << " -o 0 -p \"你好\"                   # 指定输出设备 0\n"
        << std::endl;
}

void listOutputDevices() {
    std::cout << "可用音频输出设备:" << std::endl;
    std::cout << "==================" << std::endl;

    auto devices = SpacemitAudio::AudioPlayer::ListDevices();

    if (devices.empty()) {
        std::cout << "  未找到设备!" << std::endl;
    } else {
        for (const auto& [index, name] : devices) {
            std::cout << "  [" << index << "] " << name << std::endl;
        }
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // 默认的中英混合长文本
    std::string text =
        "大家好，今天我们来讨论一下人工智能的发展。"
        "AI技术在recent years取得了remarkable progress。"
        "特别是在Natural Language Processing领域，"
        "像ChatGPT这样的large language models已经能够进行流畅的对话。"
        "这些技术正在改变我们的生活方式。"
        "比如说，语音助手可以帮助我们控制smart home devices。"
        "Machine Learning可以帮助医生诊断疾病。"
        "未来，AI将会更加intelligent，更加helpful。"
        "让我们一起期待这个exciting的未来吧！";

    std::string engine_spec = "matcha:zh-en";
    bool enable_play = true;
    int delay_ms = 5;  // 默认 5ms/字符，可用 --delay 0 禁用
    int output_rate = 48000;  // 输出采样率
    int channels = 1;  // 输出声道数
    int output_device = -1;  // 输出设备索引
    std::string provider = "auto";

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--list") == 0) {
            listOutputDevices();
            return 0;
        } else if (strcmp(argv[i], "--list-voices") == 0) {
            printVoiceList();
            return 0;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            engine_spec = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_device = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
            provider = argv[++i];
        } else if (strcmp(argv[i], "--no-play") == 0) {
            enable_play = false;
        } else if (strcmp(argv[i], "--delay") == 0 && i + 1 < argc) {
            delay_ms = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--output-rate") == 0 && i + 1 < argc) {
            output_rate = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--channels") == 0 && i + 1 < argc) {
            channels = std::stoi(argv[++i]);
            if (channels < 1 || channels > 2) {
                std::cerr << "错误: 声道数必须是 1 或 2" << std::endl;
                return 1;
            }
        }
    }

    auto selection = parseEngine(engine_spec);

    int sample_rate;
    switch (selection.backend) {
        case SpacemiT::BackendType::MATCHA_ZH:
        case SpacemiT::BackendType::MATCHA_EN:
            sample_rate = 22050;
            break;
        case SpacemiT::BackendType::MATCHA_ZH_EN:
            sample_rate = 16000;
            break;
        case SpacemiT::BackendType::KOKORO:
            sample_rate = 24000;
            break;
        default:
            sample_rate = 22050;
    }

    std::cout << "============================================" << std::endl;
    std::cout << "        流式 TTS 演示程序" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "引擎: " << engine_spec << std::endl;
    std::cout << "模型采样率: " << sample_rate << " Hz" << std::endl;
    std::cout << "输出采样率: " << output_rate << " Hz" << std::endl;
    std::cout << "输出声道: " << channels << std::endl;
    std::cout << "输出设备: " << (output_device == -1 ? "默认" : std::to_string(output_device)) << std::endl;
    std::cout << "播放: " << (enable_play ? "是" : "否") << std::endl;
    std::cout << "LLM 延迟: " << delay_ms << " ms/字符" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl;

    // 创建 TTS 引擎
    std::cout << "初始化 TTS 引擎..." << std::endl;

    SpacemiT::TtsConfig config;
    config.backend = selection.backend;
    config.sample_rate = sample_rate;
    config.provider = provider;

    if (selection.backend == SpacemiT::BackendType::KOKORO && !selection.voice.empty()) {
        config.voice = selection.voice;
    }

    SpacemiT::TtsEngine engine(config);
    if (!engine.IsInitialized()) {
        std::cerr << "TTS 引擎初始化失败!" << std::endl;
        return 1;
    }

    std::cout << "引擎: " << engine.GetEngineName() << std::endl;
    std::cout << "采样率: " << engine.GetSampleRate() << " Hz" << std::endl;
    std::cout << std::endl;

    // 切分文本
    TextSplitter splitter;
    auto sentences = splitter.split(text);

    std::cout << "========== 开始流式合成 ==========" << std::endl;
    std::cout << "输入文本: " << text << std::endl;
    std::cout << "切分为 " << sentences.size() << " 句" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;

    // 创建音频队列
    AudioQueue audio_queue;

    // 启动播放线程
    std::thread player_thread(playbackThread, std::ref(audio_queue),
            enable_play, output_rate, channels, output_device);

    // 启动合成线程
    std::thread synthesis_thread(synthesisThread, std::ref(engine),
            std::ref(sentences), std::ref(audio_queue), delay_ms);

    // 等待线程结束
    synthesis_thread.join();
    player_thread.join();

    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "        演示完成" << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
