/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KOKORO_PHONEMIZER_HPP
#define KOKORO_PHONEMIZER_HPP

#include <cstdint>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration for cpp-pinyin
namespace Pinyin {
class Pinyin;
}  // namespace Pinyin

namespace tts {

// =============================================================================
// KokoroPhonemizer - Kokoro TTS phonemizer
// =============================================================================
//
// Converts Chinese/English/mixed text to Kokoro token IDs.
// Chinese: cpp-pinyin → Pinyin-to-IPA mapping (Misaki G2P compatible).
// English: espeak-ng → IPA cleanup → Gruut en-US normalization.
// Has a hardcoded 114-entry IPA vocabulary (Kokoro's fixed vocab).
//

struct PinyinParts {
    std::string initial;  // 声母
    std::string final_;   // 韵母
    int tone = 5;         // 声调 1-5 (5=轻声)
};

class KokoroPhonemizer {
public:
    static constexpr int PAD_TOKEN_ID = 0;
    static constexpr int MAX_TOKEN_LENGTH = 512;

    KokoroPhonemizer();
    ~KokoroPhonemizer();

    /// @brief Initialize cpp-pinyin converter (must be called before textToTokenIds)
    void initPinyin();

    /// @brief Convert text to Kokoro token IDs (padded with 0 at start/end)
    /// @param text Input Chinese/English/mixed text
    /// @return Token IDs vector, padded [0, ...ids..., 0]
    std::vector<int64_t> textToTokenIds(const std::string& text) const;

    /// @brief Get vocabulary size
    int getVocabSize() const { return static_cast<int>(vocab_.size()); }

    /// @brief Check if espeak-ng is available (kept for optional English fallback)
    static bool isEspeakAvailable();

private:
    /// @brief Convert a single pinyin syllable to IPA
    std::string pinyinToIPA(const std::string& pinyin) const;

    /// @brief Convert English text to IPA via espeak-ng + Gruut normalization
    std::string englishToIPA(const std::string& text) const;

    /// @brief Clean espeak-ng IPA output (remove syllable dots, zero-width chars, etc.)
    std::string cleanEspeakIPA(const std::string& ipa) const;

    /// @brief Parse pinyin string into initial + final + tone
    PinyinParts parsePinyin(const std::string& pinyin) const;

    /// @brief Convert tone number (1-5) to arrow marker
    std::string toneToArrow(int tone) const;

    /// @brief Convert IPA string to token IDs
    std::vector<int64_t> ipaToTokenIds(const std::string& ipa) const;

    // Initialize vocabulary
    void initVocab();

    // IPA token -> ID mapping (hardcoded Kokoro vocabulary)
    std::unordered_map<std::string, int64_t> vocab_;

    // cpp-pinyin converter
    std::unique_ptr<Pinyin::Pinyin> pinyin_converter_;

    // Whether espeak-ng is available for English processing
    bool espeak_available_ = false;

    // Pinyin -> IPA mapping tables
    static const std::unordered_map<std::string, std::string> initial_to_ipa_;
    static const std::unordered_map<std::string, std::string> final_to_ipa_;

    // Ordered initials list for longest-match parsing
    static const std::vector<std::string> initials_ordered_;
};

}  // namespace tts

#endif  // KOKORO_PHONEMIZER_HPP
