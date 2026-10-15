/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODEL_DOWNLOADER_HPP
#define MODEL_DOWNLOADER_HPP

#include <string>
#include <vector>

namespace asr {

/**
 * @class ModelDownloader
 * @brief Generic model download and verification utility
 *
 * Used by all backends to ensure model files exist at runtime.
 * Downloads from URL and extracts tar.gz archive if needed.
 */
class ModelDownloader {
public:
    struct Config {
        std::string model_dir;                      // e.g. ~/.cache/models/asr/sensevoice
        std::string url;                            // download URL
        std::string archive_name;                   // e.g. sensevoice.tar.gz
        std::string archive_subdir;                 // subdirectory inside archive (optional)
        std::vector<std::string> required_files;    // files to verify
    };

    explicit ModelDownloader(const Config& config);

    /// Check required_files; download + extract if any missing. Returns true on success.
    bool ensure();

    /// Expanded model directory path
    std::string modelDir() const { return model_dir_; }

    /// Full path to a file inside model directory
    std::string filePath(const std::string& filename) const;

    /// Expand ~ to $HOME
    static std::string expandPath(const std::string& path);

private:
    Config config_;
    std::string model_dir_;  // expanded

    bool download();
    bool downloadFile(const std::string& url, const std::string& output_path);
    bool extractArchive(const std::string& archive_path);
    bool fileExists(const std::string& path) const;
};

}  // namespace asr

#endif  // MODEL_DOWNLOADER_HPP
