/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KOKORO_MODEL_DOWNLOADER_HPP
#define KOKORO_MODEL_DOWNLOADER_HPP

#include <string>

namespace tts {

// =============================================================================
// KokoroModelDownloader - Kokoro model auto-downloader
// =============================================================================
//
// Downloads Kokoro v1.0 model and voice files.
// Default source: ModelScope (accessible in China).
// Set env KOKORO_MIRROR=huggingface to use HuggingFace instead.
// Cache directory: ~/.cache/models/tts/kokoro-tts/
//

class KokoroModelDownloader {
public:
    // Mirror sources
    static constexpr const char* MS_BASE_URL =
        "https://modelscope.cn/models/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main";
    static constexpr const char* HF_BASE_URL =
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main";

    // Model files
    static constexpr const char* MODEL_URL_PATH = "onnx/model_quantized.onnx";  // Remote quantized model
    static constexpr const char* MODEL_FILE = "kokoro-v1.0.q.onnx";             // Local filename
    static constexpr const char* DEFAULT_VOICE = "zf_xiaobei.bin";

    KokoroModelDownloader();
    ~KokoroModelDownloader() = default;

    /// @brief Ensure model and voice files exist, download if necessary
    /// @param voice Voice name (without .bin extension)
    /// @return true if all files are ready
    bool ensureModelsExist(const std::string& voice = "default");

    /// @brief Get cache directory path
    std::string getCacheDir() const { return cache_dir_; }

private:
    std::string cache_dir_;  // ~/.cache/models/tts/kokoro-tts/

    std::string getBaseUrl() const;
    bool ensureCacheDir();
    bool downloadFile(const std::string& url, const std::string& dest_path);
    bool downloadModel();
    bool downloadVoice(const std::string& voice);
};

}  // namespace tts

#endif  // KOKORO_MODEL_DOWNLOADER_HPP
