/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "text/phoneme_utils.hpp"

#include <string>
#include <utility>
#include <vector>

namespace tts {
namespace text {

namespace {

void ReplaceAll(std::string* text, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = text->find(from, pos)) != std::string::npos) {
        text->replace(pos, from.length(), to);
        pos += to.length();
    }
}

std::string RemoveZeroWidthJoiner(const std::string& ipa) {
    std::string text = ipa;
    ReplaceAll(&text, "\xe2\x80\x8d", "");
    return text;
}

void ApplyReplacements(
    std::string* text,
    const std::vector<std::pair<std::string, std::string>>& replacements) {
    for (const auto& rep : replacements) {
        ReplaceAll(text, rep.first, rep.second);
    }
}

}  // namespace

// =============================================================================
// IPA phoneme conversion
// =============================================================================

std::string convertToGruutEnUs(const std::string& ipa) {
    std::string text = RemoveZeroWidthJoiner(ipa);

    // R-colored vowels (standard IPA -> Gruut US decomposed)
    std::vector<std::pair<std::string, std::string>> replacements = {
        // nurse
        {"\xc9\x9d", "\xc9\x9c\xc9\xb9"},
        // letter
        {"\xc9\x9a", "\xc9\x99\xc9\xb9"},

        // Diphthongs (diphthong -> single uppercase letter)
        // Must process longer patterns first
        {"e\xc9\xaa", "A"},   // face
        {"a\xc9\xaa", "I"},   // price
        {"\xc9\x94\xc9\xaa", "Y"},   // choice
        {"o\xca\x8a", "O"},   // goat (American)
        {"\xc9\x99\xca\x8a", "O"},   // goat (British compatibility)
        {"\xc9\x9b\xca\x8a", "O"},   // goat variant
        {"a\xca\x8a", "W"},   // mouth

        // Affricates
        {"t\xca\x83", "\xca\xa7"},   // cheese
        {"d\xca\x92", "\xca\xa4"},   // joy

        // Consonant normalization
        {"g", "\xc9\xa1"},    // Standard g -> Script g (U+0261)
        {"r", "\xc9\xb9"},    // Standard r -> Turned r (U+0279)
    };

    ApplyReplacements(&text, replacements);

    return text;
}

std::string convertToMatchaEnUs(const std::string& ipa) {
    std::string text = RemoveZeroWidthJoiner(ipa);

    // Keep diphthongs and affricates decomposed because Matcha English tokens
    // include e/ɪ/a/ʊ/t/ʃ/d/ʒ, but not A/I/O/W/Y/ʧ/ʤ.
    std::vector<std::pair<std::string, std::string>> replacements = {
        // nurse
        {"\xc9\x9d", "\xc9\x9c\xc9\xb9"},
        // letter
        {"\xc9\x9a", "\xc9\x99\xc9\xb9"},

        // Consonant normalization
        {"g", "\xc9\xa1"},    // Standard g -> Script g (U+0261)
        {"r", "\xc9\xb9"},    // Standard r -> Turned r (U+0279)
    };

    ApplyReplacements(&text, replacements);

    return text;
}

}  // namespace text
}  // namespace tts
