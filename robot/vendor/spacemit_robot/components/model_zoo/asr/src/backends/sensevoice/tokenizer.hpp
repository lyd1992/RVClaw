/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file tokenizer.hpp
 * @brief Token decoder for SenseVoice ASR output
 *
 * Converts token IDs to text with post-processing.
 */

#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "onnx_compat.hpp"  // Use compat header for RISC-V support

namespace asr {
namespace sensevoice {

/**
 * @class Tokenizer
 * @brief Converts token IDs to readable text
 *
 * Supports:
 * - Vocabulary-based token decoding
 * - Optional ONNX decoder model
 * - Post-processing (remove special tokens, normalize)
 */
class Tokenizer {
public:
    struct DecodeResult {
        std::string text;
        std::string emotion;
    };

    /**
     * @brief Configuration
     */
    struct Config {
        std::string vocab_file;          ///< Path to vocabulary file (tokens.txt)
        std::string decoder_model_path;  ///< Path to ONNX decoder (optional)
    };

    explicit Tokenizer(const Config& config);
    ~Tokenizer();

    // Non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    /**
     * @brief Initialize the tokenizer
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Release resources
     */
    void shutdown();

    /**
     * @brief Decode token IDs to text
     * @param token_ids Vector of token IDs
     * @return Decoded text
     */
    std::string decode(const std::vector<int>& token_ids);

    /**
     * @brief Decode token IDs to text plus SenseVoice metadata.
     * @param token_ids Vector of token IDs
     * @return Clean text and optional emotion label
     */
    DecodeResult decodeWithMetadata(const std::vector<int>& token_ids);

    /**
     * @brief Encode text to token IDs
     * @param text Input text
     * @return Vector of token IDs
     */
    std::vector<int> encode(const std::string& text);

    /**
     * @brief Get token for ID
     * @param id Token ID
     * @return Token string
     */
    std::string idToToken(int id) const;

    /**
     * @brief Get ID for token
     * @param token Token string
     * @return Token ID
     */
    int tokenToId(const std::string& token) const;

    /**
     * @brief Get vocabulary size
     */
    size_t getVocabSize() const { return vocab_size_; }

    /**
     * @brief Get blank token ID (for CTC)
     */
    int getBlankId() const { return blank_id_; }

    int getUnkId() const { return unk_id_; }

private:
    Config config_;
    size_t vocab_size_ = 0;

    // Special token IDs
    int blank_id_ = 0;
    int unk_id_ = 1;
    int bos_id_ = 2;
    int eos_id_ = 3;

    // Vocabulary
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    // Optional ONNX decoder
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> decoder_session_;
    Ort::MemoryInfo memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Internal methods
    bool loadVocabulary();
    bool initializeDecoder();
    void cleanupDecoder();

    std::string postProcess(const std::string& text);
    std::vector<std::string> splitUTF8(const std::string& text);
    std::string joinTokens(const std::vector<std::string>& tokens);
};

}  // namespace sensevoice
}  // namespace asr

#endif  // TOKENIZER_HPP
