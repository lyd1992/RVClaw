/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/kokoro/kokoro_phonemizer.hpp"

#include <cpp-pinyin/G2pglobal.h>
#include <cpp-pinyin/Pinyin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backends/matcha/tts_model_downloader.hpp"
#include "text/phoneme_utils.hpp"
#include "text/text_normalizer.hpp"
#include "text/text_utils.hpp"

namespace fs = std::filesystem;

namespace {
struct PcloseDeleter {
    void operator()(FILE* p) const { if (p) pclose(p); }
};
}  // namespace

namespace tts {

// =============================================================================
// Kokoro IPA Vocabulary (114 entries, from official Kokoro-82M config.json)
// =============================================================================

void KokoroPhonemizer::initVocab() {
    // Official Kokoro v1.0 vocabulary from hexgrad/Kokoro-82M config.json
    // 114 entries, max ID = 177. Per-Unicode-character tokenization.
    const std::vector<std::pair<std::string, int64_t>> vocab_entries = {
        // Punctuation
        {";", 1}, {":", 2}, {",", 3}, {".", 4}, {"!", 5}, {"?", 6},
        {"\xe2\x80\x94", 9},    // em dash (U+2014)
        {"\xe2\x80\xa6", 10},   // ellipsis (U+2026)
        {"\"", 11},
        {"(", 12}, {")", 13},
        {"\xe2\x80\x9c", 14},   // left double quotation (U+201C)
        {"\xe2\x80\x9d", 15},   // right double quotation (U+201D)
        {" ", 16},
        {"\xcc\x83", 17},       // combining tilde (U+0303)

        // Affricate single-glyph characters
        {"\xca\xa3", 18},       // U+02A3
        {"\xca\xa5", 19},       // U+02A5
        {"\xca\xa6", 20},       // U+02A6
        {"\xca\xa8", 21},       // U+02A8

        // Modifier letters
        {"\xe1\xb5\x9d", 22},   // U+1D5D
        {"\xea\xad\xa7", 23},   // U+AB67

        // Uppercase letters (sparse)
        {"A", 24}, {"I", 25},
        {"O", 31}, {"Q", 33}, {"S", 35}, {"T", 36},
        {"W", 39}, {"Y", 41},

        // Modifier letter
        {"\xe1\xb5\x8a", 42},   // U+1D4A

        // Lowercase letters a-z
        {"a", 43}, {"b", 44}, {"c", 45}, {"d", 46}, {"e", 47}, {"f", 48},
        {"h", 50}, {"i", 51}, {"j", 52}, {"k", 53}, {"l", 54}, {"m", 55},
        {"n", 56}, {"o", 57}, {"p", 58}, {"q", 59}, {"r", 60}, {"s", 61},
        {"t", 62}, {"u", 63}, {"v", 64}, {"w", 65}, {"x", 66}, {"y", 67},
        {"z", 68},

        // IPA vowels
        {"\xc9\x91", 69},       // U+0251
        {"\xc9\x90", 70},       // U+0250
        {"\xc9\x92", 71},       // U+0252
        {"\xc3\xa6", 72},       // U+00E6
        {"\xce\xb2", 75},       // U+03B2
        {"\xc9\x94", 76},       // U+0254
        {"\xc9\x95", 77},       // U+0255
        {"\xc3\xa7", 78},       // U+00E7
        {"\xc9\x96", 80},       // U+0256
        {"\xc3\xb0", 81},       // U+00F0
        {"\xca\xa4", 82},       // U+02A4
        {"\xc9\x99", 83},       // U+0259
        {"\xc9\x9a", 85},       // U+025A
        {"\xc9\x9b", 86},       // U+025B
        {"\xc9\x9c", 87},       // U+025C
        {"\xc9\x9f", 90},       // U+025F
        {"\xc9\xa1", 92},       // U+0261
        {"\xc9\xa5", 99},       // U+0265
        {"\xc9\xa8", 101},      // U+0268
        {"\xc9\xaa", 102},      // U+026A
        {"\xca\x9d", 103},      // U+029D

        // IPA consonants and nasals
        {"\xc9\xaf", 110},      // U+026F
        {"\xc9\xb0", 111},      // U+0270
        {"\xc5\x8b", 112},      // U+014B
        {"\xc9\xb3", 113},      // U+0273
        {"\xc9\xb2", 114},      // U+0272
        {"\xc9\xb4", 115},      // U+0274
        {"\xc3\xb8", 116},      // U+00F8
        {"\xc9\xb8", 118},      // U+0278
        {"\xce\xb8", 119},      // U+03B8
        {"\xc5\x93", 120},      // U+0153
        {"\xc9\xb9", 123},      // U+0279
        {"\xc9\xbe", 125},      // U+027E
        {"\xc9\xbb", 126},      // U+027B
        {"\xca\x81", 128},      // U+0281
        {"\xc9\xbd", 129},      // U+027D
        {"\xca\x82", 130},      // U+0282
        {"\xca\x83", 131},      // U+0283
        {"\xca\x88", 132},      // U+0288
        {"\xca\xa7", 133},      // U+02A7
        {"\xca\x8a", 135},      // U+028A
        {"\xca\x8b", 136},      // U+028B
        {"\xca\x8c", 138},      // U+028C
        {"\xc9\xa3", 139},      // U+0263
        {"\xc9\xa4", 140},      // U+0264
        {"\xcf\x87", 142},      // U+03C7
        {"\xca\x8e", 143},      // U+028E
        {"\xca\x92", 147},      // U+0292
        {"\xca\x94", 148},      // U+0294

        // Prosodic markers
        {"\xcb\x88", 156},      // U+02C8 primary stress
        {"\xcb\x8c", 157},      // U+02CC secondary stress
        {"\xcb\x90", 158},      // U+02D0 long

        // Aspiration and palatalization
        {"\xca\xb0", 162},      // U+02B0
        {"\xca\xb2", 164},      // U+02B2

        // Arrow tone markers
        {"\xe2\x86\x93", 169},  // U+2193
        {"\xe2\x86\x92", 171},  // U+2192
        {"\xe2\x86\x97", 172},  // U+2197
        {"\xe2\x86\x98", 173},  // U+2198

        // Latin small letter iota
        {"\xe1\xb5\xbb", 177},  // U+1D7B
    };

    for (const auto& [token, id] : vocab_entries) {
        vocab_[token] = id;
    }
}

// =============================================================================
// Pinyin -> IPA Static Mapping Tables
// =============================================================================
// Based on Misaki G2P's Chinese pipeline: jieba -> pypinyin -> pinyin-to-ipa

const std::unordered_map<std::string, std::string> KokoroPhonemizer::initial_to_ipa_ = {
    {"b", "p"},    {"p", "p\xca\xb0"},   {"m", "m"},    {"f", "f"},
    {"d", "t"},    {"t", "t\xca\xb0"},    {"n", "n"},    {"l", "l"},
    {"g", "k"},    {"k", "k\xca\xb0"},    {"h", "x"},
    {"j", "t\xc9\x95"},  {"q", "t\xc9\x95\xca\xb0"},  {"x", "\xc9\x95"},
    {"zh", "\xca\x88\xca\x82"},   {"ch", "\xca\x88\xca\x82\xca\xb0"},
    {"sh", "\xca\x82"},           {"r", "\xc9\xbb"},
    {"z", "ts"},   {"c", "ts\xca\xb0"},   {"s", "s"},
    {"y", "j"},    {"w", "w"},
};

const std::unordered_map<std::string, std::string> KokoroPhonemizer::final_to_ipa_ = {
    {"a",    "a"},
    {"ai",   "ai"},
    {"an",   "an"},
    {"ang",  "a\xc5\x8b"},
    {"ao",   "au"},
    {"e",    "\xc9\xa4"},
    {"ei",   "ei"},
    {"en",   "\xc9\x99n"},
    {"eng",  "\xc9\x99\xc5\x8b"},
    {"er",   "\xc9\x99\xc9\xbb"},
    {"i",    "i"},
    {"ia",   "ja"},
    {"ian",  "j\xc9\x9bn"},
    {"iang", "ja\xc5\x8b"},
    {"iao",  "jau"},
    {"ie",   "je"},
    {"in",   "in"},
    {"ing",  "i\xc5\x8b"},
    {"iong", "j\xca\x8a\xc5\x8b"},
    {"iu",   "jou"},
    {"o",    "o"},
    {"ong",  "\xca\x8a\xc5\x8b"},
    {"ou",   "ou"},
    {"u",    "u"},
    {"ua",   "wa"},
    {"uai",  "wai"},
    {"uan",  "wan"},
    {"uang", "wa\xc5\x8b"},
    {"ue",   "\xc9\xa5" "e"},
    {"ui",   "wei"},
    {"un",   "w\xc9\x99n"},
    {"uo",   "wo"},
    {"v",    "y"},
    {"ve",   "\xc9\xa5" "e"},
    {"van",  "\xc9\xa5\xc9\x9bn"},
    {"vn",   "yn"},
};

// Ordered initials for longest-match parsing (two-char initials first)
const std::vector<std::string> KokoroPhonemizer::initials_ordered_ = {
    "zh", "ch", "sh",
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
};

// =============================================================================
// Constructor / Destructor
// =============================================================================

KokoroPhonemizer::KokoroPhonemizer() {
    initVocab();
}

KokoroPhonemizer::~KokoroPhonemizer() = default;

// =============================================================================
// Public Methods
// =============================================================================

void KokoroPhonemizer::initPinyin() {
    // Reuse TTSModelDownloader to ensure cpp-pinyin is available
    TTSModelDownloader downloader;
    if (!downloader.ensureCppPinyin()) {
        throw std::runtime_error("Failed to download cpp-pinyin dictionary.");
    }

    std::string pinyin_dict_dir = downloader.getCppPinyinPath();
    std::cout << "[KokoroPhonemizer] Using cpp-pinyin dictionary at: " << pinyin_dict_dir << std::endl;

    Pinyin::setDictionaryPath(fs::path(pinyin_dict_dir));
    pinyin_converter_ = std::make_unique<Pinyin::Pinyin>();

    std::cout << "[KokoroPhonemizer] cpp-pinyin initialized successfully." << std::endl;

    // Check espeak-ng availability for English support
    espeak_available_ = isEspeakAvailable();
    if (espeak_available_) {
        std::cout << "[KokoroPhonemizer] espeak-ng detected, English support enabled." << std::endl;
    } else {
        std::cout << "[KokoroPhonemizer] espeak-ng not found, English text will be skipped." << std::endl;
    }
}

bool KokoroPhonemizer::isEspeakAvailable() {
    std::string command = "espeak-ng --version 2>/dev/null";
    std::unique_ptr<FILE, PcloseDeleter> pipe(popen(command.c_str(), "r"));
    if (!pipe) return false;

    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        result += buffer;
    }

    int exit_status = pclose(pipe.release());
    return exit_status == 0 && !result.empty();
}

std::vector<int64_t> KokoroPhonemizer::textToTokenIds(const std::string& text) const {
    if (text.empty()) return {};

    if (!pinyin_converter_) {
        std::cerr << "[KokoroPhonemizer] cpp-pinyin not initialized, call initPinyin() first" << std::endl;
        return {};
    }

    // Step 1: Text normalization
    std::string normalized = text::normalizeText(text, text::Language::ZH);

    // Step 2: Split into UTF-8 characters and segment by language
    auto chars = text::splitUtf8(normalized);
    std::string combined_ipa;

    size_t i = 0;
    while (i < chars.size()) {
        const std::string& ch = chars[i];

        // --- Chinese segment ---
        if (text::isChineseChar(ch)) {
            std::string chinese_segment;
            while (i < chars.size() && text::isChineseChar(chars[i])) {
                chinese_segment += chars[i];
                i++;
            }

            // Convert Chinese segment via cpp-pinyin
            Pinyin::PinyinResVector pinyin_result = pinyin_converter_->hanziToPinyin(
                chinese_segment,
                Pinyin::ManTone::Style::TONE3,
                Pinyin::Error::Default,
                false,  // candidates
                false,  // v_to_u
                true);  // neutral_tone_with_five

            for (const auto& res : pinyin_result) {
                if (!res.error) {
                    std::string ipa = pinyinToIPA(res.pinyin);
                    if (!ipa.empty()) {
                        combined_ipa += ipa;
                    }
                }
            }
            continue;
        }

        // --- English segment ---
        if (text::isEnglishLetter(ch)) {
            std::string english_segment;
            while (i < chars.size() &&
                (text::isEnglishLetter(chars[i]) || chars[i] == " " ||
                chars[i] == "'" || chars[i] == "-")) {
                english_segment += chars[i];
                i++;
            }

            // Trim trailing spaces
            while (!english_segment.empty() && english_segment.back() == ' ') {
                english_segment.pop_back();
            }

            if (!english_segment.empty()) {
                std::string ipa = englishToIPA(english_segment);
                if (!ipa.empty()) {
                    combined_ipa += ipa;
                }
            }
            continue;
        }

        // --- Punctuation / other characters ---
        // 注: 数字在 step 1 (line 287) 已由 FST 转为中文字符, 这里不再需要数字分支。
        std::string mapped = text::mapChinesePunctToAscii(ch);
        if (mapped.empty()) mapped = ch;

        if (vocab_.count(mapped)) {
            combined_ipa += mapped;
        }
        i++;
    }

    if (combined_ipa.empty()) {
        std::cerr << "[KokoroPhonemizer] No IPA output for text: " << text << std::endl;
        return {};
    }

    // Step 3: Convert IPA to token IDs
    std::vector<int64_t> ids = ipaToTokenIds(combined_ipa);

    // Step 4: Add start/end padding
    std::vector<int64_t> padded;
    padded.reserve(ids.size() + 2);
    padded.push_back(PAD_TOKEN_ID);
    padded.insert(padded.end(), ids.begin(), ids.end());
    padded.push_back(PAD_TOKEN_ID);

    // Step 5: Truncate to max length
    if (static_cast<int>(padded.size()) > MAX_TOKEN_LENGTH) {
        padded.resize(MAX_TOKEN_LENGTH);
        padded.back() = PAD_TOKEN_ID;
    }

    return padded;
}

// =============================================================================
// English Processing Methods
// =============================================================================

std::string KokoroPhonemizer::englishToIPA(const std::string& text) const {
    if (!espeak_available_) {
        std::cerr << "[KokoroPhonemizer] espeak-ng not available, skipping English: " << text << std::endl;
        return "";
    }

    if (text.empty()) return "";

    // Escape single quotes for shell safety
    std::string escaped = text;
    std::string::size_type pos = 0;
    while ((pos = escaped.find("'", pos)) != std::string::npos) {
        escaped.replace(pos, 1, "'\"'\"'");
        pos += 5;
    }

    // Call espeak-ng for IPA conversion
    std::string command = "echo '" + escaped + "' | espeak-ng -q --ipa=3 -v en-us";

    std::unique_ptr<FILE, PcloseDeleter> pipe(popen(command.c_str(), "r"));
    if (!pipe) {
        std::cerr << "[KokoroPhonemizer] Failed to run espeak-ng" << std::endl;
        return "";
    }

    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        result += buffer;
    }

    int exit_status = pclose(pipe.release());
    if (exit_status != 0 || result.empty()) {
        return "";
    }

    // Clean espeak output and convert to Kokoro-compatible IPA
    result = cleanEspeakIPA(result);
    result = text::convertToGruutEnUs(result);

    return result;
}

std::string KokoroPhonemizer::cleanEspeakIPA(const std::string& ipa) const {
    std::string result;
    result.reserve(ipa.size());

    size_t i = 0;
    bool last_was_space = false;

    while (i < ipa.size()) {
        unsigned char c = static_cast<unsigned char>(ipa[i]);

        // Skip newlines and carriage returns
        if (c == '\n' || c == '\r') {
            i++;
            continue;
        }

        // Determine UTF-8 character byte length
        int char_len = 1;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        if (i + char_len > ipa.size()) break;

        std::string ch = ipa.substr(i, char_len);

        // Skip syllable boundary dot
        if (ch == ".") {
            i += char_len;
            continue;
        }

        // Skip zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
        if (ch == "\xe2\x80\x8b" || ch == "\xe2\x80\x8c" ||
            ch == "\xe2\x80\x8d" || ch == "\xef\xbb\xbf") {
            i += char_len;
            continue;
        }

        // Handle spaces: collapse consecutive spaces
        if (ch == " ") {
            if (!last_was_space && !result.empty()) {
                result += ' ';
                last_was_space = true;
            }
            i += char_len;
            continue;
        }

        result += ch;
        last_was_space = false;
        i += char_len;
    }

    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    // Trim leading space
    if (!result.empty() && result.front() == ' ') {
        result.erase(result.begin());
    }

    return result;
}

// =============================================================================
// Private Methods
// =============================================================================

PinyinParts KokoroPhonemizer::parsePinyin(const std::string& pinyin) const {
    PinyinParts parts;

    if (pinyin.empty()) return parts;

    std::string py = pinyin;

    // Extract tone number from the end
    char last = py.back();
    if (last >= '1' && last <= '5') {
        parts.tone = last - '0';
        py.pop_back();
    }

    if (py.empty()) return parts;

    // Match longest initial first
    for (const auto& ini : initials_ordered_) {
        if (py.size() >= ini.size() && py.substr(0, ini.size()) == ini) {
            parts.initial = ini;
            parts.final_ = py.substr(ini.size());
            return parts;
        }
    }

    // No initial matched -- zero-initial syllable (e.g. "a", "o", "e", "an")
    parts.final_ = py;
    return parts;
}

std::string KokoroPhonemizer::toneToArrow(int tone) const {
    switch (tone) {
        case 1: return "\xe2\x86\x92";  // U+2192
        case 2: return "\xe2\x86\x97";  // U+2197
        case 3: return "\xe2\x86\x93";  // U+2193
        case 4: return "\xe2\x86\x98";  // U+2198
        case 5: return "";               // neutral tone
        default: return "";
    }
}

std::string KokoroPhonemizer::pinyinToIPA(const std::string& pinyin) const {
    PinyinParts parts = parsePinyin(pinyin);

    std::string ipa;

    // Special cases for retroflex/dental sibilant + "i"
    if (parts.final_ == "i") {
        if (parts.initial == "zh" || parts.initial == "ch" ||
            parts.initial == "sh" || parts.initial == "r") {
            auto ini_it = initial_to_ipa_.find(parts.initial);
            if (ini_it != initial_to_ipa_.end()) {
                ipa = ini_it->second;
            }
            ipa += "\xc9\xbb";  // syllabic retroflex vowel
            ipa += toneToArrow(parts.tone);
            return ipa;
        }
        if (parts.initial == "z" || parts.initial == "c" || parts.initial == "s") {
            auto ini_it = initial_to_ipa_.find(parts.initial);
            if (ini_it != initial_to_ipa_.end()) {
                ipa = ini_it->second;
            }
            ipa += "\xc9\xb9";  // syllabic dental vowel
            ipa += toneToArrow(parts.tone);
            return ipa;
        }
    }

    // Handle "j/q/x + u" -> actually "j/q/x + u-umlaut"
    if ((parts.initial == "j" || parts.initial == "q" || parts.initial == "x") &&
        !parts.final_.empty() && parts.final_[0] == 'u') {
        std::string adjusted_final = "v" + parts.final_.substr(1);
        auto fin_it = final_to_ipa_.find(adjusted_final);
        if (fin_it != final_to_ipa_.end()) {
            auto ini_it = initial_to_ipa_.find(parts.initial);
            if (ini_it != initial_to_ipa_.end()) {
                ipa = ini_it->second;
            }
            ipa += fin_it->second;
            ipa += toneToArrow(parts.tone);
            return ipa;
        }
    }

    // General case: look up initial + final
    if (!parts.initial.empty()) {
        auto ini_it = initial_to_ipa_.find(parts.initial);
        if (ini_it != initial_to_ipa_.end()) {
            ipa = ini_it->second;
        }
    }

    if (!parts.final_.empty()) {
        auto fin_it = final_to_ipa_.find(parts.final_);
        if (fin_it != final_to_ipa_.end()) {
            ipa += fin_it->second;
        } else {
            // Fallback: try individual characters
            for (char c : parts.final_) {
                std::string sc(1, c);
                auto it = final_to_ipa_.find(sc);
                if (it != final_to_ipa_.end()) {
                    ipa += it->second;
                } else {
                    ipa += sc;  // Pass through as-is
                }
            }
        }
    }

    // Append tone arrow
    ipa += toneToArrow(parts.tone);

    return ipa;
}

std::vector<int64_t> KokoroPhonemizer::ipaToTokenIds(const std::string& ipa) const {
    // Per-Unicode-character lookup, matching official Kokoro tokenizer
    std::vector<int64_t> ids;
    ids.reserve(ipa.size());

    size_t i = 0;
    while (i < ipa.size()) {
        // Determine UTF-8 character byte length
        unsigned char c = static_cast<unsigned char>(ipa[i]);
        int char_len = 1;
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        if (i + char_len <= ipa.size()) {
            std::string token = ipa.substr(i, char_len);
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            }
            // Not found -- skip silently (matches official behavior)
        }
        i += char_len;
    }

    return ids;
}

}  // namespace tts
