/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef TTS_MODEL_DOWNLOADER_HPP
#define TTS_MODEL_DOWNLOADER_HPP

#include <string>
#include <vector>

namespace tts {

class TTSModelDownloader {
public:
    // Model file names
    static constexpr const char* MATCHA_ZH_MODEL = "matcha-icefall-zh-baker/model-steps-3.q.onnx";
    static constexpr const char* MATCHA_ZH_LEXICON = "matcha-icefall-zh-baker/lexicon.txt";
    static constexpr const char* MATCHA_ZH_TOKENS = "matcha-icefall-zh-baker/tokens.txt";
    static constexpr const char* MATCHA_ZH_DICT_DIR = "matcha-icefall-zh-baker/dict";
    static constexpr const char* MATCHA_EN_MODEL = "matcha-icefall-en_US-ljspeech/model-steps-3.q.onnx";
    static constexpr const char* MATCHA_EN_TOKENS = "matcha-icefall-en_US-ljspeech/tokens.txt";
    static constexpr const char* MATCHA_EN_DATA_DIR = "matcha-icefall-en_US-ljspeech/espeak-ng-data";
    static constexpr const char* VOCOS_VOCODER = "vocos-22khz-univ.q.onnx";
    static constexpr const char* VOCOS_VOCODER_16K = "vocos-16khz-univ.q.onnx";

    // zh-en bilingual model (downloaded like zh and en models)
    static constexpr const char* MATCHA_ZH_EN_MODEL = "matcha-icefall-zh-en/model-steps-3.q.onnx";
    static constexpr const char* MATCHA_ZH_EN_TOKENS = "matcha-icefall-zh-en/vocab_tts.txt";

    // Third-party dependency repos
    static constexpr const char* CPPJIEBA_REPO = "https://gitee.com/spacemit-robotics/cppjieba.git";
    static constexpr const char* CPP_PINYIN_REPO = "https://gitee.com/spacemit-robotics/cpp-pinyin.git";

    TTSModelDownloader();
    ~TTSModelDownloader() = default;

    // Ensure all TTS models exist, download if necessary (legacy method)
    bool ensureModelsExist();

    // Ensure models for specific language exist, download if necessary
    bool ensureModelsExist(const std::string& language);

    // Ensure FST-based text normalization rule files exist
    // language: "zh" currently supported (English uses espeak-ng's built-in TN)
    // Files land under ~/.cache/models/tts/text_norm/v1/<lang>/
    bool ensureTextNormFiles(const std::string& language);

    // Ensure cppjieba is cloned to ~/.cache/thirdparty/cppjieba
    bool ensureCppJieba();

    // Ensure cpp-pinyin is cloned to ~/.cache/thirdparty/cpp-pinyin
    bool ensureCppPinyin();

    // Get cppjieba dictionary path (~/.cache/thirdparty/cppjieba/dict)
    std::string getCppJiebaPath() const;

    // Get cpp-pinyin dictionary path (~/.cache/thirdparty/cpp-pinyin/res/dict)
    std::string getCppPinyinPath() const;

    // Get full path to a model file
    std::string getModelPath(const std::string& filename) const;

    // Check if a model file exists locally
    bool modelExists(const std::string& filename) const;

    // Download a specific model
    bool downloadModel(const std::string& filename);

    // Download and extract the complete TTS model tar.gz package (legacy method)
    // bool downloadAndExtractTarGz();

    // Download and extract language-specific models
    bool downloadAndExtractLanguageModels(const std::string& language);

    // Get cache directory
    std::string getCacheDir() const { return cache_dir_; }

private:
    std::string cache_dir_;
    std::string base_cache_dir_;  // ~/.cache/

    // Get download URL for a model
    std::string getDownloadUrl(const std::string& filename) const;

    // Create cache directory if it doesn't exist
    bool ensureCacheDir();

    // Download file from URL
    bool downloadFile(const std::string& url, const std::string& dest_path);

    // Extract tar.gz archive
    bool extractTarGz(const std::string& archive_path, const std::string& dest_dir);

    // Download vocoder model
    bool downloadVocoder();

    // Download and extract language-specific model archive
    bool downloadLanguageModel(const std::string& language);

    // Get language-specific model URLs
    std::string getLanguageModelUrl(const std::string& language) const;

    // Git clone a repository
    bool gitClone(const std::string& repo_url, const std::string& dest_dir);
};

}  // namespace tts

#endif  // TTS_MODEL_DOWNLOADER_HPP
