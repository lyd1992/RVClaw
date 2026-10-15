/* Copyright (C) 2025 SpacemiT Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0 */

#ifndef KOKORO_BACKEND_HPP
#define KOKORO_BACKEND_HPP

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backends/kokoro/kokoro_phonemizer.hpp"
#include "backends/kokoro/kokoro_voice_manager.hpp"
#include "backends/tts_backend.hpp"

namespace tts {

// =============================================================================
// KokoroBackend - Kokoro v1.0 TTS Backend
// =============================================================================
//
// End-to-end TTS backend using a single ONNX model.
// Input: token_ids + style_vector + speed → Output: 24kHz audio waveform
//
// Unlike Matcha-TTS (acoustic model + vocoder), Kokoro is a single model
// that directly outputs audio, so it inherits ITtsBackend directly.
//

class KokoroBackend : public ITtsBackend {
public:
    static constexpr int SAMPLE_RATE = 24000;
    static constexpr int MAX_TOKEN_LENGTH = 512;

    KokoroBackend();
    ~KokoroBackend() override;

    // -------------------------------------------------------------------------
    // ITtsBackend interface
    // -------------------------------------------------------------------------

    ErrorInfo initialize(const TtsConfig& config) override;
    void shutdown() override;
    bool isInitialized() const override;

    BackendType getType() const override;
    std::string getName() const override;
    std::string getVersion() const override;
    bool supportsStreaming() const override;
    int getNumSpeakers() const override;
    int getSampleRate() const override;

    ErrorInfo synthesize(const std::string& text, SynthesisResult& result) override;

    ErrorInfo setSpeed(float speed) override;

private:
    /// @brief Get model directory (expand ~)
    std::string getModelDir() const;

    /// @brief Run ONNX inference
    std::vector<float> runInference(const std::vector<int64_t>& token_ids,
                                    const std::vector<float>& style_vector,
                                    float speed);

    // Components
    KokoroPhonemizer phonemizer_;
    KokoroVoiceManager voice_manager_;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    // State
    TtsConfig config_;
    bool initialized_ = false;
    float current_speed_ = 1.0f;

    // Thread safety
    mutable std::mutex inference_mutex_;
};

}  // namespace tts

#endif  // KOKORO_BACKEND_HPP
