/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file sensevoice_model.hpp
 * @brief SenseVoice ASR Model - ONNX-based speech recognition model
 *
 * This class wraps the SenseVoice ONNX model for speech-to-text inference.
 * It handles feature extraction, model inference, and CTC decoding.
 */

#ifndef SENSEVOICE_MODEL_HPP
#define SENSEVOICE_MODEL_HPP

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <mutex>

#include "onnx_compat.hpp"  // Use compat header for RISC-V support

namespace asr {
namespace sensevoice {

// Forward declarations
class FeatureExtractor;
class Tokenizer;
class HotwordScorer;

/**
 * @class SenseVoiceModel
 * @brief SenseVoice ASR model wrapper for ONNX Runtime inference
 *
 * Example usage:
 * @code
 *   SenseVoiceModel::Config config;
 *   config.model_path = "~/.cache/models/asr/sensevoice/model_quant_optimized.onnx";
 *   config.vocab_path = "~/.cache/models/asr/sensevoice/tokens.txt";
 *
 *   SenseVoiceModel model(config);
 *   if (model.initialize()) {
 *       std::string text = model.recognize(audio_data);
 *   }
 * @endcode
 */
class SenseVoiceModel {
public:
    struct RecognitionOutput {
        std::string text;
        std::string emotion;
    };

    /**
     * @brief Model configuration
     */
    struct Config {
        std::string model_path;         ///< Path to ONNX model file
        std::string cmvn_path;          ///< Path to CMVN file (am.mvn)
        std::string vocab_path;         ///< Path to vocabulary file (tokens.txt)
        std::string decoder_path;      ///< Path to decoder model (optional)

        int batch_size = 1;             ///< Batch size for inference
        int sample_rate = 16000;        ///< Expected audio sample rate
        int num_threads = 2;            ///< Number of inference threads

        std::string language = "auto";  ///< Language code: zh, en, ja, ko, yue, auto
        bool use_itn = true;           ///< Enable Inverse Text Normalization
        std::string provider = "spacemit";  ///< "cpu" or "spacemit"
    };

    /**
     * @brief Inference statistics
     */
    struct InferenceStats {
        double feature_time_ms = 0;     ///< Feature extraction time
        double inference_time_ms = 0;   ///< ONNX inference time
        double decode_time_ms = 0;      ///< Token decoding time
        double total_time_ms = 0;       ///< Total processing time
        double audio_duration_ms = 0;   ///< Input audio duration
        double rtf = 0;                 ///< Real-Time Factor
    };

    explicit SenseVoiceModel(const Config& config);
    ~SenseVoiceModel();

    // Non-copyable
    SenseVoiceModel(const SenseVoiceModel&) = delete;
    SenseVoiceModel& operator=(const SenseVoiceModel&) = delete;

    /**
     * @brief Initialize the model
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Release resources
     */
    void shutdown();

    /**
     * @brief Check if model is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Recognize speech from audio samples
     * @param audio Audio samples (float, normalized to [-1, 1])
     * @return Recognized text
     */
    std::string recognize(const std::vector<float>& audio);

    /**
     * @brief Recognize speech from audio samples
     * @param audio Pointer to audio samples
     * @param length Number of samples
     * @return Recognized text
     */
    std::string recognize(const float* audio, size_t length);

    /**
     * @brief Recognize speech and return SenseVoice metadata.
     * @param audio Audio samples (float, normalized to [-1, 1])
     * @return Recognized text and optional emotion label
     */
    RecognitionOutput recognizeWithMetadata(const std::vector<float>& audio);

    /**
     * @brief Recognize speech and return SenseVoice metadata.
     * @param audio Pointer to audio samples
     * @param length Number of samples
     * @return Recognized text and optional emotion label
     */
    RecognitionOutput recognizeWithMetadata(const float* audio, size_t length);

    /**
     * @brief Batch recognition
     * @param audio_batch Vector of audio samples
     * @return Vector of recognized texts
     */
    std::vector<std::string> recognizeBatch(
        const std::vector<std::vector<float>>& audio_batch);

    /**
     * @brief Get last inference statistics
     */
    const InferenceStats& getLastStats() const { return last_stats_; }

    /**
     * @brief Set language for recognition
     * @param language Language code
     */
    void setLanguage(const std::string& language);

    /**
     * @brief Enable/disable verbose logging
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }

    /**
     * @brief Set hotwords for CTC biased decoding
     */
    void setHotwords(const std::vector<std::string>& hotwords, float boost);

private:
    Config config_;
    bool initialized_ = false;
    bool verbose_ = false;
    InferenceStats last_stats_;

    // ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    // Sub-components
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    std::unique_ptr<Tokenizer> tokenizer_;

    // Model metadata
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Language mapping
    std::map<std::string, int> language_map_;
    std::map<std::string, int> textnorm_map_;
    int blank_id_ = 0;

    // Hotword biasing
    std::unique_ptr<HotwordScorer> hotword_scorer_;
    mutable std::mutex hotword_mutex_;

    // Internal methods
    bool initializeSession();
    bool loadComponents();
    void initializeLanguageMaps();
    void cleanupSession();

    RecognitionOutput recognizeInternal(
        const float* audio, size_t length, bool include_metadata);
    std::vector<int> decodeCTC(const float* logits, int seq_len, int vocab_size);
    int getLanguageId(const std::string& language);
    int getTextnormId(bool use_itn);
};

}  // namespace sensevoice
}  // namespace asr

#endif  // SENSEVOICE_MODEL_HPP
