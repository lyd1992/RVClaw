/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/matcha/tts_model_downloader.hpp"

#include <curl/curl.h>

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace tts {

TTSModelDownloader::TTSModelDownloader() {
    // Set cache directory to ~/.cache/models/tts/matcha-tts/
    const char* home = std::getenv("HOME");
    if (home) {
        base_cache_dir_ = std::string(home) + "/.cache/";
        cache_dir_ = std::string(home) + "/.cache/models/tts/matcha-tts/";
    } else {
        base_cache_dir_ = "./.cache/";
        cache_dir_ = "./.cache/models/tts/matcha-tts/";
    }
}

bool TTSModelDownloader::ensureModelsExist() {
    // Legacy method - defaults to Chinese
    return ensureModelsExist("zh");
}

bool TTSModelDownloader::ensureModelsExist(const std::string& language) {
    if (!ensureCacheDir()) {
        return false;
    }

    // First ensure vocoder exists (shared between languages)
    if (language == "zh-en") {
        // zh-en uses 16kHz vocoder
        std::string vocoder_path = getModelPath(VOCOS_VOCODER_16K);
        if (!fs::exists(vocoder_path)) {
            std::string url = "https://archive.spacemit.com/spacemit-ai/model_zoo/tts/vocoder/vocos-16khz-univ.q.onnx";
            std::cout << "Downloading 16kHz vocoder from " << url << "..." << std::endl;
            if (!downloadFile(url, vocoder_path)) {
                std::cerr << "Failed to download 16kHz vocoder" << std::endl;
                return false;
            }
        }

        // Check zh-en model files
        std::string model_path = getModelPath(MATCHA_ZH_EN_MODEL);
        std::string tokens_path = getModelPath(MATCHA_ZH_EN_TOKENS);

        if (!fs::exists(model_path) || !fs::exists(tokens_path)) {
            if (!downloadLanguageModel("zh-en")) {
                std::cerr << "Failed to download Chinese-English bilingual TTS models" << std::endl;
                return false;
            }
        }
        if (!fs::exists(model_path) || !fs::exists(tokens_path)) {
            std::cerr << "Missing Chinese-English bilingual TTS files after download: "
                << MATCHA_ZH_EN_MODEL << std::endl;
            return false;
        }
    } else {
        // zh and en use 22kHz vocoder
        std::string vocoder_path = getModelPath(VOCOS_VOCODER);
        if (!fs::exists(vocoder_path)) {
            if (!downloadVocoder()) {
                std::cerr << "Failed to download vocoder" << std::endl;
                return false;
            }
        }

        // Check language-specific models
        if (language == "zh") {
            std::string model_path = getModelPath(MATCHA_ZH_MODEL);
            std::string lexicon_path = getModelPath(MATCHA_ZH_LEXICON);
            std::string tokens_path = getModelPath(MATCHA_ZH_TOKENS);
            std::string dict_path = getModelPath(MATCHA_ZH_DICT_DIR);

            if (!fs::exists(model_path) || !fs::exists(lexicon_path) ||
                !fs::exists(tokens_path) || !fs::exists(dict_path)) {
                if (!downloadLanguageModel("zh")) {
                    std::cerr << "Failed to download Chinese TTS models" << std::endl;
                    return false;
                }
            }
            if (!fs::exists(model_path) || !fs::exists(lexicon_path) ||
                !fs::exists(tokens_path) || !fs::exists(dict_path)) {
                std::cerr << "Missing Chinese TTS files after download: "
                    << MATCHA_ZH_MODEL << std::endl;
                return false;
            }
        } else if (language == "en") {
            std::string model_path = getModelPath(MATCHA_EN_MODEL);
            std::string tokens_path = getModelPath(MATCHA_EN_TOKENS);
            std::string data_dir = getModelPath(MATCHA_EN_DATA_DIR);

            if (!fs::exists(model_path) || !fs::exists(tokens_path) || !fs::exists(data_dir)) {
                if (!downloadLanguageModel("en")) {
                    std::cerr << "Failed to download English TTS models" << std::endl;
                    return false;
                }
            }
            if (!fs::exists(model_path) || !fs::exists(tokens_path) || !fs::exists(data_dir)) {
                std::cerr << "Missing English TTS files after download: "
                    << MATCHA_EN_MODEL << std::endl;
                return false;
            }
        } else {
            std::cerr << "Unsupported language: " << language << std::endl;
            return false;
        }
    }

    std::cout << "All TTS models for " << language << " are ready!" << std::endl;
    return true;
}

std::string TTSModelDownloader::getModelPath(const std::string& filename) const {
    return cache_dir_ + filename;
}

bool TTSModelDownloader::modelExists(const std::string& filename) const {
    std::string path = getModelPath(filename);
    return fs::exists(path) && fs::is_regular_file(path);
}

bool TTSModelDownloader::downloadAndExtractLanguageModels(const std::string& language) {
    return downloadLanguageModel(language);
}

std::string TTSModelDownloader::getDownloadUrl(const std::string& filename) const {
    // This function is kept for compatibility but not used in the simplified approach
    return "";
}

bool TTSModelDownloader::downloadModel(const std::string& filename) {
    // This function is kept for compatibility but not used in the simplified approach
    return false;
}

bool TTSModelDownloader::ensureCacheDir() {
    try {
        if (!fs::exists(cache_dir_)) {
            fs::create_directories(cache_dir_);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create cache directory: " << e.what() << std::endl;
        return false;
    }
}

// CURL write callback
static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    size_t total_size = size * nmemb;
    file->write(static_cast<const char*>(contents), total_size);
    return total_size;
}

// CURL progress callback
static int progressCallback(void* clientp, double dltotal, double dlnow,
                            double ultotal, double ulnow) {
    if (dltotal > 0) {
        int progress = static_cast<int>((dlnow / dltotal) * 100);
        std::cout << "\rDownload progress: " << progress << "%" << std::flush;
    }
    return 0;
}

bool TTSModelDownloader::downloadFile(const std::string& url, const std::string& dest_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL" << std::endl;
        return false;
    }

    std::ofstream file(dest_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << dest_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    // 使用新的进度回调函数（CURL 7.32.0+）
    #if LIBCURL_VERSION_NUM >= 0x072000
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
    #else
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progressCallback);
    #endif
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    file.close();

    if (res != CURLE_OK) {
        std::cerr << "\nDownload failed: " << curl_easy_strerror(res) << std::endl;
        fs::remove(dest_path);
        return false;
    }

    std::cout << "\nDownload completed!" << std::endl;
    return true;
}

bool TTSModelDownloader::extractTarGz(const std::string& archive_path, const std::string& dest_dir) {
    // Use system tar command for extraction
    std::string command = "tar -xzf \"" + archive_path + "\" -C \"" + dest_dir + "\"";
    int result = std::system(command.c_str());

    if (result != 0) {
        std::cerr << "Failed to extract archive using tar command" << std::endl;
        return false;
    }

    return true;
}

bool TTSModelDownloader::downloadVocoder() {
    std::string url = "https://archive.spacemit.com/spacemit-ai/model_zoo/tts/vocoder/vocos-22khz-univ.q.onnx";
    std::string dest_path = getModelPath(VOCOS_VOCODER);

    std::cout << "Downloading vocoder from " << url << "..." << std::endl;
    if (!downloadFile(url, dest_path)) {
        std::cerr << "Failed to download vocoder" << std::endl;
        return false;
    }

    std::cout << "Vocoder downloaded successfully!" << std::endl;
    return true;
}

bool TTSModelDownloader::downloadLanguageModel(const std::string& language) {
    std::string url = getLanguageModelUrl(language);
    if (url.empty()) {
        std::cerr << "No URL available for language: " << language << std::endl;
        return false;
    }

    std::string archive_name;
    if (language == "zh") {
        archive_name = "matcha-icefall-zh-baker.tar.gz";
    } else if (language == "en") {
        archive_name = "matcha-icefall-en_US-ljspeech.tar.gz";
    } else if (language == "zh-en") {
        archive_name = "matcha-icefall-zh-en.tar.gz";
    } else {
        std::cerr << "Unknown language: " << language << std::endl;
        return false;
    }
    std::string archive_path = cache_dir_ + archive_name;

    std::cout << "Downloading " << language << " TTS models from " << url << "..." << std::endl;
    if (!downloadFile(url, archive_path)) {
        std::cerr << "Failed to download " << language << " TTS models archive" << std::endl;
        return false;
    }

    // Extract without stripping components to preserve directory structure
    std::cout << "Extracting " << language << " TTS models..." << std::endl;
    std::string command = "tar -xzf \"" + archive_path + "\" -C \"" + cache_dir_ + "\"";
    int result = std::system(command.c_str());

    if (result != 0) {
        std::cerr << "Failed to extract " << language << " TTS models archive" << std::endl;
        fs::remove(archive_path);
        return false;
    }

    // Clean up archive
    fs::remove(archive_path);
    std::cout << language << " TTS models downloaded and extracted successfully!" << std::endl;
    return true;
}

std::string TTSModelDownloader::getLanguageModelUrl(const std::string& language) const {
    if (language == "zh") {
        return "https://archive.spacemit.com/spacemit-ai/model_zoo/tts/matcha-tts/matcha-icefall-zh-baker.tar.gz";
    } else if (language == "en") {
        return "https://archive.spacemit.com/spacemit-ai/model_zoo/tts/matcha-tts/matcha-icefall-en_US-ljspeech.tar.gz";
    } else if (language == "zh-en") {
        return "https://archive.spacemit.com/spacemit-ai/model_zoo/tts/matcha-tts/matcha-icefall-zh-en.tar.gz";
    }
    return "";
}

bool TTSModelDownloader::gitClone(const std::string& repo_url, const std::string& dest_dir) {
    std::cout << "Cloning " << repo_url << " to " << dest_dir << "..." << std::endl;

    // Full clone (no --depth to ensure all submodules work)
    std::string command = "git clone \"" + repo_url + "\" \"" + dest_dir + "\" 2>&1";
    int result = std::system(command.c_str());

    if (result != 0) {
        std::cerr << "Failed to clone repository: " << repo_url << std::endl;
        return false;
    }

    std::cout << "Repository cloned successfully!" << std::endl;
    return true;
}

bool TTSModelDownloader::ensureTextNormFiles(const std::string& language) {
    // 目前仅中文有配套 FST; 英文走 espeak-ng 内置 TN, 不需要额外文件。
    if (language != "zh") {
        return false;
    }

    // 目录: ~/.cache/models/tts/text_norm/v1/zh/
    std::string target_dir = base_cache_dir_ + "models/tts/text_norm/v1/zh/";
    try {
        fs::create_directories(target_dir);
    } catch (const std::exception& e) {
        std::cerr << "[TextNormFiles] Failed to create directory " << target_dir
            << ": " << e.what() << std::endl;
        return false;
    }

    // 源: archive.spacemit.com/spacemit-ai/model_zoo/tts/text_norm/v1/zh/
    static constexpr const char* kBaseUrl =
        "https://archive.spacemit.com/spacemit-ai/model_zoo/tts/text_norm/v1/zh/";

    // date.fst / phone.fst / number.fst 是必需的, new_heteronym.fst 可选
    const std::vector<std::string> required = {"date.fst", "phone.fst", "number.fst"};

    for (const auto& name : required) {
        std::string dest = target_dir + name;
        if (fs::exists(dest) && fs::file_size(dest) > 0) {
            continue;
        }
        std::string url = kBaseUrl + name;
        std::cout << "[TextNormFiles] Downloading " << url << " ..." << std::endl;
        if (!downloadFile(url, dest)) {
            std::cerr << "[TextNormFiles] Failed to download " << name << std::endl;
            // 清掉可能的半截文件
            if (fs::exists(dest)) fs::remove(dest);
            return false;
        }
    }
    return true;
}

bool TTSModelDownloader::ensureCppJieba() {
    std::string thirdparty_dir = base_cache_dir_ + "thirdparty";
    fs::create_directories(thirdparty_dir);
    std::string cppjieba_dir = thirdparty_dir + "/cppjieba";
    std::string dict_file = cppjieba_dir + "/dict/jieba.dict.utf8";

    if (fs::exists(dict_file)) {
        return true;
    }

    // Remove incomplete directory if exists
    if (fs::exists(cppjieba_dir)) {
        std::cout << "Removing incomplete cppjieba directory..." << std::endl;
        fs::remove_all(cppjieba_dir);
    }

    // cppjieba 需要 --recursive 克隆子模块 (limonp)
    std::cout << "Cloning " << CPPJIEBA_REPO << " to " << cppjieba_dir << "..." << std::endl;
    std::string command =
        "git clone --recursive \"" + std::string(CPPJIEBA_REPO) + "\" \"" + cppjieba_dir + "\" 2>&1";
    int result = std::system(command.c_str());

    if (result != 0) {
        std::cerr << "Failed to clone cppjieba" << std::endl;
        return false;
    }

    std::cout << "cppjieba cloned successfully!" << std::endl;
    return true;
}

bool TTSModelDownloader::ensureCppPinyin() {
    std::string thirdparty_dir = base_cache_dir_ + "thirdparty";
    fs::create_directories(thirdparty_dir);
    std::string cpp_pinyin_dir = thirdparty_dir + "/cpp-pinyin";
    std::string dict_file = cpp_pinyin_dir + "/res/dict/mandarin";

    if (fs::exists(dict_file)) {
        return true;
    }

    // Remove incomplete directory if exists
    if (fs::exists(cpp_pinyin_dir)) {
        std::cout << "Removing incomplete cpp-pinyin directory..." << std::endl;
        fs::remove_all(cpp_pinyin_dir);
    }

    return gitClone(CPP_PINYIN_REPO, cpp_pinyin_dir);
}

std::string TTSModelDownloader::getCppJiebaPath() const {
    return base_cache_dir_ + "thirdparty/cppjieba/dict";
}

std::string TTSModelDownloader::getCppPinyinPath() const {
    return base_cache_dir_ + "thirdparty/cpp-pinyin/res/dict";
}

}  // namespace tts
