/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KOKORO_VOICE_MANAGER_HPP
#define KOKORO_VOICE_MANAGER_HPP

#include <string>
#include <vector>

namespace tts {

// =============================================================================
// KokoroVoiceManager - Kokoro voice style vector manager
// =============================================================================
//
// Loads .bin files containing raw float32 style vectors, reshaped as (N, 256).
// Provides style vector lookup by token length.
//

class KokoroVoiceManager {
public:
    static constexpr int STYLE_DIM = 256;

    KokoroVoiceManager() = default;
    ~KokoroVoiceManager() = default;

    /// @brief Load a voice .bin file
    /// @param voice_path Path to the .bin file
    /// @return true if loaded successfully
    bool loadVoice(const std::string& voice_path);

    /// @brief Get style vector for given token length
    /// @param token_len Number of tokens in the input
    /// @return Style vector of size STYLE_DIM (256)
    std::vector<float> getStyleVector(int token_len) const;

    /// @brief Check if a voice is loaded
    bool isLoaded() const { return !style_data_.empty(); }

    /// @brief Get number of style rows
    int getNumRows() const { return num_rows_; }

private:
    std::vector<float> style_data_;  // Raw float32 data, (N * 256)
    int num_rows_ = 0;
};

}  // namespace tts

#endif  // KOKORO_VOICE_MANAGER_HPP
