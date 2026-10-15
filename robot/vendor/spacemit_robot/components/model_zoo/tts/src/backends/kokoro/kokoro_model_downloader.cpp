/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#include "backends/kokoro/kokoro_model_downloader.hpp"

#include <curl/curl.h>

#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace tts {

KokoroModelDownloader::KokoroModelDownloader() {
    const char* home = std::getenv("HOME");
    if (home) {
        cache_dir_ = std::string(home) + "/.cache/models/tts/kokoro-tts/";
    } else {
        cache_dir_ = "./.cache/models/tts/kokoro-tts/";
    }
}

std::string KokoroModelDownloader::getBaseUrl() const {
    const char* mirror = std::getenv("KOKORO_MIRROR");
    if (mirror && std::string(mirror) == "huggingface") {
        return HF_BASE_URL;
    }
    return MS_BASE_URL;
}

bool KokoroModelDownloader::ensureModelsExist(const std::string& voice) {
    if (!ensureCacheDir()) {
        return false;
    }

    // Ensure ONNX model exists
    if (!downloadModel()) {
        return false;
    }

    // Ensure voice file exists
    if (!downloadVoice(voice)) {
        return false;
    }

    std::cout << "[Kokoro] All models are ready!" << std::endl;
    return true;
}

bool KokoroModelDownloader::ensureCacheDir() {
    try {
        if (!fs::exists(cache_dir_)) {
            fs::create_directories(cache_dir_);
        }
        // Also create voices subdirectory
        std::string voices_dir = cache_dir_ + "voices/";
        if (!fs::exists(voices_dir)) {
            fs::create_directories(voices_dir);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Kokoro] Failed to create cache directory: " << e.what() << std::endl;
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
    (void)clientp;
    (void)ultotal;
    (void)ulnow;
    if (dltotal > 0) {
        int progress = static_cast<int>((dlnow / dltotal) * 100);
        double dl_mb = dlnow / (1024.0 * 1024.0);
        double total_mb = dltotal / (1024.0 * 1024.0);
        std::cout << "\r[Kokoro] Download progress: " << progress << "% ("
            << std::fixed << std::setprecision(1) << dl_mb << "/"
            << total_mb << " MB)" << std::flush;
    }
    return 0;
}

bool KokoroModelDownloader::downloadFile(const std::string& url, const std::string& dest_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[Kokoro] Failed to initialize CURL" << std::endl;
        return false;
    }

    std::ofstream file(dest_path, std::ios::binary);
    if (!file) {
        std::cerr << "[Kokoro] Failed to open file for writing: " << dest_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

#if LIBCURL_VERSION_NUM >= 0x072000
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
#else
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progressCallback);
#endif
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "kokoro-tts/1.0");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    CURLcode res = curl_easy_perform(curl);

    int64_t http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_easy_cleanup(curl);
    file.close();

    if (res != CURLE_OK) {
        std::cerr << "\n[Kokoro] Download failed: " << curl_easy_strerror(res) << std::endl;
        fs::remove(dest_path);
        return false;
    }

    if (http_code != 200) {
        std::cerr << "\n[Kokoro] HTTP error " << http_code << " for " << url << std::endl;
        fs::remove(dest_path);
        return false;
    }

    std::cout << std::endl;
    return true;
}

bool KokoroModelDownloader::downloadModel() {
    std::string model_path = cache_dir_ + MODEL_FILE;
    if (fs::exists(model_path) && fs::file_size(model_path) > 1024) {
        return true;
    }
    // Remove corrupted/incomplete file
    if (fs::exists(model_path)) {
        fs::remove(model_path);
    }

    std::string url = getBaseUrl() + "/" + MODEL_URL_PATH;
    std::cout << "[Kokoro] Downloading model from " << url << " ..." << std::endl;

    if (!downloadFile(url, model_path)) {
        std::cerr << "[Kokoro] Failed to download model" << std::endl;
        return false;
    }

    std::cout << "[Kokoro] Model downloaded successfully!" << std::endl;
    return true;
}

bool KokoroModelDownloader::downloadVoice(const std::string& voice) {
    // Determine the actual voice file to download
    std::string voice_file = voice;
    if (voice == "default") {
        voice_file = DEFAULT_VOICE;  // zf_xiaobei.bin
    } else if (voice.find(".bin") == std::string::npos) {
        voice_file = voice + ".bin";
    }

    std::string voice_path = cache_dir_ + "voices/" + voice_file;

    // Remove corrupted/incomplete voice file
    if (fs::exists(voice_path) && fs::file_size(voice_path) <= 1024) {
        fs::remove(voice_path);
    }

    // Download the voice file if it doesn't exist
    if (!fs::exists(voice_path)) {
        std::string url = getBaseUrl() + "/voices/" + voice_file;
        std::cout << "[Kokoro] Downloading voice '" << voice_file << "' from " << url << " ..." << std::endl;

        if (!downloadFile(url, voice_path)) {
            std::cerr << "[Kokoro] Failed to download voice: " << voice_file << std::endl;
            return false;
        }

        std::cout << "[Kokoro] Voice downloaded successfully!" << std::endl;
    }

    // If voice is "default", also create default.bin as a copy
    if (voice == "default") {
        std::string default_path = cache_dir_ + "voices/default.bin";
        if (!fs::exists(default_path)) {
            try {
                fs::copy_file(voice_path, default_path);
            } catch (const std::exception& e) {
                std::cerr << "[Kokoro] Failed to create default.bin: " << e.what() << std::endl;
                return false;
            }
        }
    }

    return true;
}

}  // namespace tts
