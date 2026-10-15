/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include <cstring>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tts_service.h"

// Engine selection result from parsing "-l" argument
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

// Resolve a voice name: accept both full ("zf_xiaobei") and short ("xiaobei")
std::string resolveVoiceName(const std::string& input) {
    if (input.empty()) return input;

    // Already a full name (contains '_') — check if it's known, pass through either way
    if (input.find('_') != std::string::npos) {
        return input;
    }

    // Short name lookup
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

    // No match — might be a valid voice not in our list, pass through
    std::cerr << "警告: 未知音色 '" << input << "'，将直接使用该名称\n"
        << "使用 --list-voices 查看可用音色列表\n";
    return input;
}

void printUsage(const char* program) {
    std::cout << "用法: " << program << " [选项]\n"
        << "\n"
        << "选项:\n"
        << "  -p <text>      直接合成指定文本\n"
        << "  -l <engine>    引擎选择 (格式: 引擎:变体)\n"
        << "  -o <file>      输出文件 (默认: output.wav)\n"
        << "  -s <speed>     语速倍率 (默认: 1.0)\n"
        << "  --provider <provider>           推理后端: auto/cpu/spacemit (默认: auto)\n"
        << "  --lexicon <entries>  自定义发音 (格式: word:phoneme[:locale], 多条逗号分隔)\n"
        << "                 locale=zh (默认): phoneme 为带声调拼音, 空格分隔, 如 \"wei4 ni3\"\n"
        << "                 locale=en (matcha:en 或 matcha:zh-en): phoneme 为英文单词/短语, 走 espeak 生成 IPA\n"
        << "  --list-voices  列出 Kokoro 可用音色\n"
        << "  -h             显示帮助\n"
        << "\n"
        << "引擎格式:\n"
        << "  matcha         Matcha 中英混合 (= matcha:zh-en)\n"
        << "  matcha:zh      Matcha 中文 (22050Hz)\n"
        << "  matcha:en      Matcha 英文 (22050Hz)\n"
        << "  matcha:zh-en   Matcha 中英混合 (16000Hz)\n"
        << "  kokoro         Kokoro 默认音色 (24000Hz)\n"
        << "  kokoro:<voice> Kokoro 指定音色 (支持短名和全名)\n"
        << "                 短名: kokoro:xiaobei  全名: kokoro:zf_xiaobei\n"
        << "\n"
        << "交互模式:\n"
        << "  不带 -p 参数时进入交互模式，输入文本后按 Enter 合成\n"
        << "  输入 'q' 或 'quit' 退出\n"
        << "\n"
        << "示例:\n"
        << "  " << program << "                                  # 交互模式\n"
        << "  " << program << " -p \"你好世界\" -l matcha:zh       # 中文合成\n"
        << "  " << program << " -p \"Hello\" -l matcha:en         # 英文合成\n"
        << "  " << program << " -p \"今天学Python\" -l matcha:zh-en  # 中英混合\n"
        << "  " << program << " -p \"你好\" -l kokoro              # Kokoro 默认音色\n"
        << "  " << program << " -p \"你好\" -l kokoro:yunxi        # Kokoro 短名\n"
        << "  " << program << " -p \"你好\" -l kokoro:zm_yunxi     # Kokoro 全名\n"
        << "  # 中文热词 (matcha:zh 或 matcha:zh-en): 纠正多音字, phoneme 为带声调拼音\n"
        << "  " << program << " -p \"你好,我是语音合成模型,很高兴为你服务\" -l matcha:zh \\\n"
        << "      --lexicon \"为你:wei4 ni3\"    # 把 '为你' 从 wei2 ni3 纠正为 wei4 ni3\n"
        << "  # 英文热词 (matcha:en 或 matcha:zh-en): locale=en, phoneme 为英文单词, 让 espeak 按英文读\n"
        << "  " << program << " -p \"你好,我是 SpaceMIT 的语音合成模型\" -l matcha:zh-en \\\n"
        << "      --lexicon \"SpaceMIT:space meet:en\"\n"
        << "  # 混合: 一次传多条, 用 locale 分别指定\n"
        << "  " << program << " -p \"你好,我是 SpaceMIT 的语音合成模型,很高兴为你服务\" -l matcha:zh-en \\\n"
        << "      --lexicon \"为你:wei4 ni3:zh,SpaceMIT:space meet:en\"\n"
        << std::endl;
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

EngineSelection parseEngine(const std::string& spec) {
    EngineSelection sel;
    sel.backend = SpacemiT::BackendType::MATCHA_ZH_EN;

    // Split on ':'
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

bool synthesize(SpacemiT::TtsEngine& engine, const std::string& text,
                const std::string& output_file) {
    std::cout << "合成中: \"" << text << "\"" << std::endl;

    auto result = engine.Call(text);

    if (!result || !result->IsSuccess()) {
        std::cerr << "合成失败";
        if (result) {
            std::cerr << ": " << result->GetMessage();
        }
        std::cerr << std::endl;
        return false;
    }

    // 显示信息
    std::cout << "采样率: " << result->GetSampleRate() << " Hz" << std::endl;
    std::cout << "时长: " << result->GetDurationMs() << " ms" << std::endl;
    std::cout << "处理时间: " << result->GetProcessingTimeMs() << " ms" << std::endl;
    std::cout << "RTF: " << result->GetRTF() << std::endl;

    // 保存文件
    if (result->SaveToFile(output_file)) {
        std::cout << "已保存: " << output_file << std::endl;
        return true;
    } else {
        std::cerr << "保存失败: " << output_file << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::string text;
    std::string engine_spec = "matcha:zh-en";
    std::string output_file = "output.wav";
    float speed = 1.0f;
    std::string lexicon_str;
    std::string provider = "auto";
    bool interactive = true;

    // 解析参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--list-voices") == 0) {
            printVoiceList();
            return 0;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            text = argv[++i];
            interactive = false;
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            engine_spec = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            speed = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
            provider = argv[++i];
        } else if (strcmp(argv[i], "--lexicon") == 0 && i + 1 < argc) {
            lexicon_str = argv[++i];
        }
    }

    // 解析引擎选择
    auto selection = parseEngine(engine_spec);

    std::cout << "初始化 TTS 引擎 (" << engine_spec << ")..." << std::endl;

    // 创建配置
    SpacemiT::TtsConfig config;
    config.backend = selection.backend;
    config.speech_rate = speed;
    config.provider = provider;

    // Kokoro: set voice
    if (selection.backend == SpacemiT::BackendType::KOKORO && !selection.voice.empty()) {
        config.voice = selection.voice;
    }

    // 根据后端设置采样率
    switch (selection.backend) {
        case SpacemiT::BackendType::MATCHA_ZH:
        case SpacemiT::BackendType::MATCHA_EN:
            config.sample_rate = 22050;
            break;
        case SpacemiT::BackendType::MATCHA_ZH_EN:
            config.sample_rate = 16000;
            break;
        case SpacemiT::BackendType::KOKORO:
            config.sample_rate = 24000;
            break;
        default:
            config.sample_rate = 22050;
    }

    // 创建引擎
    SpacemiT::TtsEngine engine(config);

    if (!engine.IsInitialized()) {
        std::cerr << "引擎初始化失败!" << std::endl;
        return 1;
    }

    std::cout << "引擎: " << engine.GetEngineName() << std::endl;
    std::cout << "采样率: " << engine.GetSampleRate() << " Hz" << std::endl;
    std::cout << "说话人数: " << engine.GetNumSpeakers() << std::endl;

    // 解析并应用自定义发音词典 (格式: word:phoneme[:locale], 多条逗号分隔)
    if (!lexicon_str.empty()) {
        std::vector<SpacemiT::PronunciationEntry> entries;
        std::string item;
        for (size_t i = 0; i <= lexicon_str.size(); ++i) {
            if (i == lexicon_str.size() || lexicon_str[i] == ',') {
                auto first_colon = item.find(':');
                if (first_colon != std::string::npos) {
                    SpacemiT::PronunciationEntry e;
                    e.word = item.substr(0, first_colon);
                    std::string rest = item.substr(first_colon + 1);
                    auto second_colon = rest.rfind(':');
                    if (second_colon != std::string::npos) {
                        e.phoneme = rest.substr(0, second_colon);
                        e.locale = rest.substr(second_colon + 1);
                    } else {
                        e.phoneme = rest;
                    }
                    entries.push_back(std::move(e));
                }
                item.clear();
            } else {
                item += lexicon_str[i];
            }
        }
        if (!entries.empty()) {
            engine.UpdateLexicon(entries);
            std::cout << "自定义发音: ";
            for (size_t j = 0; j < entries.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << entries[j].word << " -> " << entries[j].phoneme
                        << " (" << entries[j].locale << ")";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    if (interactive) {
        // 交互模式
        std::cout << "进入交互模式，输入文本后按 Enter 合成 (输入 q 退出)" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        std::string line;
        int count = 0;

        while (true) {
            std::cout << "> ";
            std::getline(std::cin, line);

            if (line.empty()) {
                continue;
            }

            if (line == "q" || line == "quit" || line == "exit") {
                std::cout << "再见!" << std::endl;
                break;
            }

            // 生成输出文件名
            std::string out = output_file;
            if (count > 0) {
                size_t dot = out.rfind('.');
                if (dot != std::string::npos) {
                    out = out.substr(0, dot) + "_" + std::to_string(count) + out.substr(dot);
                } else {
                    out = out + "_" + std::to_string(count);
                }
            }

            synthesize(engine, line, out);
            std::cout << std::endl;
            count++;
        }
    } else {
        // 直接模式
        if (text.empty()) {
            std::cerr << "错误: 请使用 -p 指定文本" << std::endl;
            printUsage(argv[0]);
            return 1;
        }

        if (!synthesize(engine, text, output_file)) {
            return 1;
        }
    }

    return 0;
}
