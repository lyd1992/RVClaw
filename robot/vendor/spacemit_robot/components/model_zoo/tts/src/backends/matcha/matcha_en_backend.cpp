/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/matcha/matcha_en_backend.hpp"

#include <cctype>
#include <chrono>
#include <cstdint>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "text/phoneme_utils.hpp"
#include "text/text_normalizer.hpp"
#include "text/text_utils.hpp"

namespace tts {

namespace {

constexpr double kPi = 3.14159265358979323846;

bool IsPausePunctuation(char ch) {
    return ch == ',' || ch == '.' || ch == '?' || ch == '!' ||
        ch == ';' || ch == ':';
}

int PauseMsForPunctuation(char ch) {
    if (ch == ',' || ch == ';' || ch == ':') {
        return 180;
    }
    return 360;
}

struct EnglishSegment {
    std::string text;
    int pause_ms = 0;
};

std::vector<EnglishSegment> SplitByPausePunctuation(const std::string& text) {
    std::vector<EnglishSegment> segments;
    std::string current;

    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        if (!IsPausePunctuation(ch)) {
            current += ch;
            continue;
        }

        int pause_ms = PauseMsForPunctuation(ch);
        while (i + 1 < text.size() && IsPausePunctuation(text[i + 1])) {
            ++i;
            pause_ms = std::max(pause_ms, PauseMsForPunctuation(text[i]));
        }
        if (!current.empty()) {
            segments.push_back({current, pause_ms});
        }
        current.clear();
    }

    if (!current.empty()) {
        segments.push_back({current, 0});
    }

    return segments;
}

}  // namespace

// =============================================================================
// 构造与析构
// =============================================================================

MatchaEnBackend::MatchaEnBackend()
    : MatchaBackend(BackendType::MATCHA_EN)
    , espeak_initialized_(false) {
}

MatchaEnBackend::~MatchaEnBackend() {
    shutdownLanguageSpecific();
}

// =============================================================================
// MatchaBackend 纯虚方法实现
// =============================================================================

std::string MatchaEnBackend::getModelSubdir() const {
    return "matcha-icefall-en_US-ljspeech";
}

bool MatchaEnBackend::usesBlankTokens() const {
    return true;
}

ErrorInfo MatchaEnBackend::initializeLanguageSpecific(const TtsConfig& config) {
    // 检查 espeak-ng 是否可用
    if (!checkEspeakNgAvailable()) {
        return ErrorInfo::error(ErrorCode::INTERNAL_ERROR,
            "espeak-ng is required for English TTS but not available. "
            "Please install: brew install espeak-ng (macOS) or apt-get install espeak-ng (Linux)");
    }

    espeak_initialized_ = true;
    std::cout << "Info: espeak-ng found and available for English TTS." << std::endl;

    return ErrorInfo::ok();
}

void MatchaEnBackend::shutdownLanguageSpecific() {
    espeak_initialized_ = false;
    lexicon_.clear();
}

ErrorInfo MatchaEnBackend::updateLexicon(
    const std::vector<LexiconEntry>& entries) {
    for (const auto& e : entries) {
        if (!e.word.empty() && !e.phoneme.empty()) {
            lexicon_[e.word] = e.phoneme;
        }
    }
    return ErrorInfo::ok();
}

// =============================================================================
// 初始化 espeak-ng
// =============================================================================

void MatchaEnBackend::initializeEspeak() {
    // espeak-ng 通过命令行调用，不需要特殊初始化
    espeak_initialized_ = checkEspeakNgAvailable();
}

ErrorInfo MatchaEnBackend::synthesize(
    const std::string& input_text,
    SynthesisResult& result) {
    if (!isInitialized()) {
        return ErrorInfo::error(ErrorCode::NOT_INITIALIZED, "Backend not initialized");
    }

    if (input_text.empty()) {
        return ErrorInfo::error(ErrorCode::INVALID_TEXT, "Empty text");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        std::string normalized_text = text::normalizeText(input_text, text::Language::EN);
        std::vector<EnglishSegment> segments = SplitByPausePunctuation(normalized_text);
        std::vector<float> audio_samples;
        int output_sample_rate = getSampleRate();

        for (const auto& segment : segments) {
            std::vector<int64_t> token_ids = processEnglishWithLexicon(segment.text);
            int segment_sample_rate = output_sample_rate;
            std::vector<float> segment_audio =
                synthesizeTokenIdsToAudio(token_ids, &segment_sample_rate);
            output_sample_rate = segment_sample_rate;
            smoothSegmentBoundary(&segment_audio, output_sample_rate);
            audio_samples.insert(
                audio_samples.end(), segment_audio.begin(), segment_audio.end());

            if (segment.pause_ms > 0 && output_sample_rate > 0) {
                size_t silence_samples =
                    static_cast<size_t>(output_sample_rate) * segment.pause_ms / 1000;
                audio_samples.insert(audio_samples.end(), silence_samples, 0.0f);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.audio = AudioChunk::fromFloat(audio_samples, output_sample_rate, true);
        result.audio_duration_ms = result.audio.getDurationMs();
        result.processing_time_ms = duration.count();
        result.calculateRTF();
        result.success = true;

        SentenceInfo sentence;
        sentence.text = input_text;
        sentence.begin_time_ms = 0;
        sentence.end_time_ms = result.audio_duration_ms;
        sentence.is_final = true;
        result.sentences.push_back(sentence);

        if (callback_) {
            notifyAudioChunk(result.audio);
        }

        return ErrorInfo::ok();
    } catch (const std::exception& e) {
        return ErrorInfo::error(ErrorCode::SYNTHESIS_FAILED,
            std::string("Synthesis failed: ") + e.what());
    }
}

// =============================================================================
// 文本转 Token IDs (英文)
// =============================================================================

std::vector<int64_t> MatchaEnBackend::textToTokenIds(const std::string& text) {
    std::vector<int64_t> token_ids;
    const auto& token_to_id = getTokenToIdMap();

    // 检查是否包含中文字符 - 如果包含则跳过
    if (text::containsChinese(text)) {
        return token_ids;  // 静默跳过中文
    }

    // 添加开始 token (^) - sherpa-onnx 风格
    auto start_it = token_to_id.find("^");
    if (start_it != token_to_id.end()) {
        token_ids.push_back(start_it->second);
    }

    auto body_ids = processEnglishWithLexicon(text);
    token_ids.insert(token_ids.end(), body_ids.begin(), body_ids.end());

    // 添加结束 token ($) - sherpa-onnx 风格
    auto end_it = token_to_id.find("$");
    if (end_it != token_to_id.end()) {
        token_ids.push_back(end_it->second);
    }

    return token_ids;
}

std::vector<int64_t> MatchaEnBackend::processEnglishWithLexicon(
    const std::string& text) {
    if (text.empty()) {
        return {};
    }

    if (!lexicon_.empty()) {
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        const std::string* replacement = nullptr;

        for (const auto& [word, phoneme] : lexicon_) {
            size_t search_pos = 0;
            while (true) {
                size_t pos = text.find(word, search_pos);
                if (pos == std::string::npos) {
                    break;
                }

                if (isLexiconWordMatch(text, pos, word.length()) &&
                    (best_pos == std::string::npos || pos < best_pos ||
                    (pos == best_pos && word.length() > best_len))) {
                    best_pos = pos;
                    best_len = word.length();
                    replacement = &phoneme;
                }

                search_pos = pos + 1;
            }
        }

        if (replacement != nullptr) {
            std::vector<int64_t> result;
            auto before = processEnglishWithLexicon(text.substr(0, best_pos));
            result.insert(result.end(), before.begin(), before.end());

            auto lex_ids = processPlainEnglishToIds(*replacement);
            result.insert(result.end(), lex_ids.begin(), lex_ids.end());

            auto after = processEnglishWithLexicon(text.substr(best_pos + best_len));
            result.insert(result.end(), after.begin(), after.end());
            return result;
        }
    }

    return processPlainEnglishToIds(text);
}

bool MatchaEnBackend::isLexiconWordMatch(
    const std::string& text,
    size_t pos,
    size_t len) const {
    auto is_word_char = [](unsigned char ch) {
        return std::isalnum(ch) || ch == '_';
    };

    bool left_ok = pos == 0 ||
        !is_word_char(static_cast<unsigned char>(text[pos - 1]));
    size_t end = pos + len;
    bool right_ok = end >= text.size() ||
        !is_word_char(static_cast<unsigned char>(text[end]));
    return left_ok && right_ok;
}

std::vector<int64_t> MatchaEnBackend::processPlainEnglishToIds(
    const std::string& text) {
    std::vector<int64_t> token_ids;
    std::string segment;

    auto flush_segment = [&]() {
        if (segment.empty()) {
            return;
        }

        // 使用 espeak-ng 获取 IPA 音素
        std::string phonemes = processEnglishTextToPhonemes(segment);
        if (phonemes.empty()) {
            std::cerr << "Error: espeak-ng failed to process text" << std::endl;
            segment.clear();
            return;
        }

        // 转换为 Matcha English token 表兼容格式
        std::string gruut_phonemes = text::convertToMatchaEnUs(phonemes);
        appendTokenString(gruut_phonemes, &token_ids);
        segment.clear();
    };

    for (char ch : text) {
        if (IsPausePunctuation(ch)) {
            flush_segment();
            appendTokenString(std::string(1, ch), &token_ids);
        } else {
            segment += ch;
        }
    }
    flush_segment();

    return token_ids;
}

void MatchaEnBackend::appendTokenString(
    const std::string& token_text,
    std::vector<int64_t>* token_ids) const {
    const auto& token_to_id = getTokenToIdMap();

    // 处理音素字符
    std::vector<std::string> phoneme_chars = text::splitUtf8(token_text);
    bool last_was_space = false;

    for (const auto& phoneme_char : phoneme_chars) {
        if (phoneme_char.empty()) continue;

        // 过滤问题字符
        if (phoneme_char == "\u200D" ||  // Zero-width joiner
            phoneme_char == "\u200C" ||  // Zero-width non-joiner
            phoneme_char == "\uFEFF" ||  // Byte order mark
            phoneme_char == "\u00A0" ||  // Non-breaking space
            (phoneme_char.size() == 1 &&
            std::iscntrl(static_cast<unsigned char>(phoneme_char[0])))) {
            continue;
        }

        // 处理空格 - 限制连续空格
        if (phoneme_char == " ") {
            if (last_was_space) {
                continue;
            }
            last_was_space = true;
        } else {
            last_was_space = false;
        }

        auto token_it = token_to_id.find(phoneme_char);
        if (token_it != token_to_id.end()) {
            token_ids->push_back(token_it->second);
        } else if (phoneme_char != " ") {
            // 只记录非空格的未知 token
            std::cerr << "Warning: Unknown phoneme token: '" << phoneme_char << "'" << std::endl;
        }
    }
}

void MatchaEnBackend::smoothSegmentBoundary(
    std::vector<float>* audio,
    int sample_rate) const {
    if (audio == nullptr || audio->empty() || sample_rate <= 0) {
        return;
    }

    size_t fade_samples = static_cast<size_t>(sample_rate) * 20 / 1000;
    fade_samples = std::min(fade_samples, audio->size() / 4);
    if (fade_samples == 0) {
        return;
    }

    for (size_t i = 0; i < fade_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(fade_samples);
        float fade = 0.5f * (1.0f - std::cos(static_cast<float>(kPi) * t));
        (*audio)[i] *= fade;

        size_t tail_idx = audio->size() - 1 - i;
        (*audio)[tail_idx] *= fade;
    }
}

}  // namespace tts
