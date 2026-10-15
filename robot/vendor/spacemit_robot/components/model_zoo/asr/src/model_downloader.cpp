/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "model_downloader.hpp"

#include <curl/curl.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <string>

namespace asr {

ModelDownloader::ModelDownloader(const Config& config)
    : config_(config), model_dir_(expandPath(config.model_dir)) {}

std::string ModelDownloader::expandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') return path;
    const char* home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE");
    return home ? (std::string(home) + path.substr(1)) : path;
}

std::string ModelDownloader::filePath(const std::string& filename) const {
    return model_dir_ + "/" + filename;
}

bool ModelDownloader::fileExists(const std::string& path) const {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

bool ModelDownloader::ensure() {
    // Check if all required files exist
    bool all_exist = true;
    for (const auto& file : config_.required_files) {
        if (!fileExists(filePath(file))) {
            std::cout << "[ModelDownloader] Missing: " << file << std::endl;
            all_exist = false;
        }
    }

    if (all_exist) {
        std::cout << "[ModelDownloader] All models available in " << model_dir_ << std::endl;
        return true;
    }

    // Download
    std::cout << "[ModelDownloader] Downloading to " << model_dir_ << " ..." << std::endl;
    if (!download()) {
        return false;
    }

    // Verify after download
    for (const auto& file : config_.required_files) {
        if (!fileExists(filePath(file))) {
            std::cerr << "[ModelDownloader] Still missing after download: " << file << std::endl;
            return false;
        }
    }
    return true;
}

bool ModelDownloader::download() {
    std::filesystem::create_directories(model_dir_);

    std::string archive_path = model_dir_ + "/" + config_.archive_name;

    if (!downloadFile(config_.url, archive_path)) {
        return false;
    }

    if (!extractArchive(archive_path)) {
        return false;
    }

    std::filesystem::remove(archive_path);
    return true;
}

// ---------------------------------------------------------------------------
// curl helpers
// ---------------------------------------------------------------------------

static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* file = static_cast<std::ofstream*>(userp);
    size_t total = size * nmemb;
    file->write(static_cast<const char*>(contents), total);
    return total;
}

bool ModelDownloader::downloadFile(
        const std::string& url, const std::string& output_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ModelDownloader] Failed to init curl" << std::endl;
        return false;
    }

    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ModelDownloader] Cannot open: " << output_path << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

    CURLcode res = curl_easy_perform(curl);

    int64_t response_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    curl_easy_cleanup(curl);
    file.close();

    if (res != CURLE_OK) {
        std::cerr << "[ModelDownloader] Download failed: "
                << curl_easy_strerror(res) << std::endl;
        std::filesystem::remove(output_path);
        return false;
    }

    if (response_code != 200) {
        std::cerr << "[ModelDownloader] HTTP error: " << response_code << std::endl;
        std::filesystem::remove(output_path);
        return false;
    }

    std::cout << "[ModelDownloader] Downloaded: " << output_path << std::endl;
    return true;
}

bool ModelDownloader::extractArchive(const std::string& archive_path) {
    std::string temp_dir = model_dir_ + "/temp_extract";

    if (std::filesystem::exists(temp_dir)) {
        std::filesystem::remove_all(temp_dir);
    }
    std::filesystem::create_directories(temp_dir);

    std::string cmd = "tar -xzf \"" + archive_path + "\" -C \"" + temp_dir + "\"";
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "[ModelDownloader] Extract failed" << std::endl;
        std::filesystem::remove_all(temp_dir);
        return false;
    }

    // Use configured subdirectory, or fall back to temp_dir itself
    std::string src_dir = temp_dir;
    if (!config_.archive_subdir.empty()) {
        std::string subdir = temp_dir + "/" + config_.archive_subdir;
        if (std::filesystem::exists(subdir)) {
            src_dir = subdir;
        }
    }

    // Copy files to model directory
    for (const auto& entry : std::filesystem::directory_iterator(src_dir)) {
        std::string dest = model_dir_ + "/" + entry.path().filename().string();
        try {
            if (entry.is_regular_file()) {
                std::filesystem::copy_file(entry.path(), dest,
                    std::filesystem::copy_options::overwrite_existing);
            } else if (entry.is_directory()) {
                if (!std::filesystem::exists(dest)) {
                    std::filesystem::copy(entry.path(), dest,
                        std::filesystem::copy_options::recursive);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ModelDownloader] Warning: failed to copy "
                    << entry.path().filename().string()
                    << ": " << e.what() << std::endl;
        }
    }

    std::filesystem::remove_all(temp_dir);

    std::cout << "[ModelDownloader] Extracted successfully" << std::endl;
    return true;
}

}  // namespace asr
