/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file sensevoice_model.cpp
 * @brief SenseVoice ASR Model implementation
 */

#include "backends/sensevoice/sensevoice_model.hpp"

#ifdef USE_SPACEMIT_EP
#include "spacemit_ort_env.h"
#endif

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/sensevoice/feature_extractor.hpp"
#include "backends/sensevoice/hotword_scorer.hpp"
#include "backends/sensevoice/tokenizer.hpp"

namespace asr {
namespace sensevoice {

SenseVoiceModel::SenseVoiceModel(const Config& config)
    : config_(config)
    , memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
{
    initializeLanguageMaps();
}

SenseVoiceModel::~SenseVoiceModel() {
    shutdown();
}

bool SenseVoiceModel::initialize() {
    if (initialized_) {
        return true;
    }

    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SenseVoiceModel");

        if (!initializeSession()) {
            std::cerr << "[SenseVoiceModel] Failed to initialize ONNX session" << std::endl;
            return false;
        }

        if (!loadComponents()) {
            std::cerr << "[SenseVoiceModel] Failed to load components" << std::endl;
            return false;
        }

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[SenseVoiceModel] Initialization error: " << e.what() << std::endl;
        return false;
    }
}

bool SenseVoiceModel::initializeSession() {
    try {
        Ort::SessionOptions session_options;

        // EP 用自己的线程池（SPACEMIT_EP_INTRA_THREAD_NUM），ORT intra_op 设 1
#ifdef USE_SPACEMIT_EP
        if (config_.provider == "spacemit") {
            session_options.SetIntraOpNumThreads(1);
        } else {
#endif
            session_options.SetIntraOpNumThreads(config_.num_threads);
#ifdef USE_SPACEMIT_EP
        }
#endif
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

#ifdef USE_SPACEMIT_EP
        if (config_.provider == "spacemit") {
            std::unordered_map<std::string, std::string> ep_options = {
                {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(config_.num_threads)}
            };
            Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(session_options, ep_options);
            if (status.IsOK()) {
                std::cout << "[ASR] SpaceMIT EP initialized (threads=" << config_.num_threads << ")" << std::endl;
            } else {
                std::cerr << "[ASR] SpaceMIT EP init failed: "
                    << status.GetErrorMessage() << ", fallback to CPU" << std::endl;
            }
        } else {
#endif
            // CPU 模式才禁用 memory arena（EP 不兼容这两个选项）
            session_options.DisableCpuMemArena();
            session_options.DisableMemPattern();
#ifdef USE_SPACEMIT_EP
        }
#endif

        session_ = std::make_unique<Ort::Session>(*env_, config_.model_path.c_str(), session_options);

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session_->GetInputCount();
        input_names_.reserve(num_inputs);
        input_shapes_.reserve(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.release());
            input_shapes_.push_back(
                session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        size_t num_outputs = session_->GetOutputCount();
        output_names_.reserve(num_outputs);
        output_shapes_.reserve(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.release());
            output_shapes_.push_back(
                session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[SenseVoiceModel] Session init error: " << e.what() << std::endl;
        return false;
    }
}

bool SenseVoiceModel::loadComponents() {
    // Initialize feature extractor
    FeatureExtractor::Config fe_config;
    fe_config.sample_rate = config_.sample_rate;
    fe_config.cmvn_file = config_.cmvn_path;

    feature_extractor_ = std::make_unique<FeatureExtractor>(fe_config);
    if (!feature_extractor_->initialize()) {
        std::cerr << "[SenseVoiceModel] Failed to initialize feature extractor" << std::endl;
        return false;
    }

    // Initialize tokenizer
    Tokenizer::Config tok_config;
    tok_config.vocab_file = config_.vocab_path;
    tok_config.decoder_model_path = config_.decoder_path;

    tokenizer_ = std::make_unique<Tokenizer>(tok_config);
    if (!tokenizer_->initialize()) {
        std::cerr << "[SenseVoiceModel] Failed to initialize tokenizer" << std::endl;
        return false;
    }

    blank_id_ = tokenizer_->getBlankId();

    return true;
}

void SenseVoiceModel::shutdown() {
    cleanupSession();
    feature_extractor_.reset();
    tokenizer_.reset();
    initialized_ = false;
}

void SenseVoiceModel::cleanupSession() {
    session_.reset();
    env_.reset();

    for (auto name : input_names_) {
        delete[] name;
    }
    input_names_.clear();

    for (auto name : output_names_) {
        delete[] name;
    }
    output_names_.clear();
}

void SenseVoiceModel::initializeLanguageMaps() {
    language_map_["auto"] = 0;
    language_map_["zh"] = 3;
    language_map_["en"] = 4;
    language_map_["yue"] = 7;
    language_map_["ja"] = 11;
    language_map_["ko"] = 12;
    language_map_["nospeech"] = 13;

    textnorm_map_["withitn"] = 14;
    textnorm_map_["woitn"] = 15;
}

std::string SenseVoiceModel::recognize(const std::vector<float>& audio) {
    return recognize(audio.data(), audio.size());
}

std::string SenseVoiceModel::recognize(const float* audio, size_t length) {
    return recognizeInternal(audio, length, false).text;
}

SenseVoiceModel::RecognitionOutput SenseVoiceModel::recognizeWithMetadata(
    const std::vector<float>& audio) {
    return recognizeWithMetadata(audio.data(), audio.size());
}

SenseVoiceModel::RecognitionOutput SenseVoiceModel::recognizeWithMetadata(
    const float* audio, size_t length) {
    return recognizeInternal(audio, length, true);
}

SenseVoiceModel::RecognitionOutput SenseVoiceModel::recognizeInternal(
    const float* audio, size_t length, bool include_metadata) {
    if (!initialized_) {
        std::cerr << "[SenseVoiceModel] Model not initialized" << std::endl;
        return {};
    }

    try {
        auto total_start = std::chrono::high_resolution_clock::now();

        // Feature extraction
        auto fe_start = std::chrono::high_resolution_clock::now();
        auto features = feature_extractor_->extract(audio, length);
        auto fe_end = std::chrono::high_resolution_clock::now();

        if (features.empty()) {
            return {};
        }

        size_t feature_dim = features[0].size();
        size_t seq_len = features.size();

        // Flatten features
        auto flat_start = std::chrono::high_resolution_clock::now();
        std::vector<float> flat_features;
        flat_features.reserve(seq_len * feature_dim);
        for (const auto& frame : features) {
            flat_features.insert(flat_features.end(), frame.begin(), frame.end());
        }
        auto flat_end = std::chrono::high_resolution_clock::now();

        // Prepare tensors
        std::vector<int64_t> feature_shape = {
            config_.batch_size,
            static_cast<int64_t>(seq_len),
            static_cast<int64_t>(feature_dim)
        };
        std::vector<int64_t> scalar_shape = {config_.batch_size};

        std::vector<int32_t> feat_len = {static_cast<int32_t>(seq_len)};
        std::vector<int32_t> lang_id = {getLanguageId(config_.language)};
        std::vector<int32_t> norm_id = {getTextnormId(config_.use_itn)};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, flat_features.data(), flat_features.size(),
            feature_shape.data(), feature_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, feat_len.data(), feat_len.size(),
            scalar_shape.data(), scalar_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, lang_id.data(), lang_id.size(),
            scalar_shape.data(), scalar_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, norm_id.data(), norm_id.size(),
            scalar_shape.data(), scalar_shape.size()));

        // Run inference
        auto inf_start = std::chrono::high_resolution_clock::now();
        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), inputs.data(), inputs.size(),
            output_names_.data(), output_names_.size());
        auto inf_end = std::chrono::high_resolution_clock::now();

        // Decode
        auto dec_start = std::chrono::high_resolution_clock::now();
        float* logits = outputs[0].GetTensorMutableData<float>();
        auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        int out_seq_len = static_cast<int>(logits_shape[1]);
        int vocab_size = static_cast<int>(logits_shape[2]);

        auto token_ids = decodeCTC(logits, out_seq_len, vocab_size);
        RecognitionOutput decoded;
        if (include_metadata) {
            auto metadata = tokenizer_->decodeWithMetadata(token_ids);
            decoded.text = metadata.text;
            decoded.emotion = metadata.emotion;
        } else {
            decoded.text = tokenizer_->decode(token_ids);
        }
        auto dec_end = std::chrono::high_resolution_clock::now();

        auto total_end = std::chrono::high_resolution_clock::now();

        // Update stats
        last_stats_.feature_time_ms =
            std::chrono::duration<double, std::milli>(fe_end - fe_start).count();
        last_stats_.inference_time_ms =
            std::chrono::duration<double, std::milli>(inf_end - inf_start).count();
        last_stats_.decode_time_ms =
            std::chrono::duration<double, std::milli>(dec_end - dec_start).count();
        last_stats_.total_time_ms =
            std::chrono::duration<double, std::milli>(total_end - total_start).count();
        last_stats_.audio_duration_ms =
            (static_cast<double>(length) / config_.sample_rate) * 1000.0;
        last_stats_.rtf = last_stats_.total_time_ms / last_stats_.audio_duration_ms;

        // Calculate flatten time
        double flatten_time_ms =
            std::chrono::duration<double, std::milli>(flat_end - flat_start).count();

        // Always print performance breakdown
        std::cout << "\n=== Performance Breakdown ===" << std::endl;
        std::cout << "Feature extraction: " << std::fixed << std::setprecision(2)
                << last_stats_.feature_time_ms / 1000.0 << "s ("
                << std::setprecision(2)
                << (last_stats_.feature_time_ms / last_stats_.total_time_ms * 100)
                << "%)" << std::endl;
        std::cout << "Data flattening: " << std::fixed << std::setprecision(2)
                << flatten_time_ms / 1000.0 << "s ("
                << std::setprecision(2)
                << (flatten_time_ms / last_stats_.total_time_ms * 100)
                << "%)" << std::endl;
        std::cout << "ONNX inference: " << std::fixed << std::setprecision(2)
                << last_stats_.inference_time_ms / 1000.0 << "s ("
                << std::setprecision(2)
                << (last_stats_.inference_time_ms / last_stats_.total_time_ms * 100)
                << "%)" << std::endl;
        std::cout << "Token decoding: " << std::fixed << std::setprecision(2)
                << last_stats_.decode_time_ms / 1000.0 << "s ("
                << std::setprecision(2)
                << (last_stats_.decode_time_ms / last_stats_.total_time_ms * 100)
                << "%)" << std::endl;
        std::cout << "Total (model): " << std::fixed << std::setprecision(2)
                << last_stats_.total_time_ms / 1000.0 << "s, Audio: "
                << last_stats_.audio_duration_ms / 1000.0 << "s, RTF: "
                << std::setprecision(3) << last_stats_.rtf << std::endl;

        return {decoded.text, decoded.emotion};
    } catch (const std::exception& e) {
        std::cerr << "[SenseVoiceModel] Inference error: " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::string> SenseVoiceModel::recognizeBatch(
    const std::vector<std::vector<float>>& audio_batch)
{
    std::vector<std::string> results;
    results.reserve(audio_batch.size());

    for (const auto& audio : audio_batch) {
        results.push_back(recognize(audio));
    }

    return results;
}

std::vector<int> SenseVoiceModel::decodeCTC(const float* logits, int seq_len, int vocab_size) {
    std::vector<int> tokens;
    tokens.reserve(seq_len / 2);

    std::lock_guard<std::mutex> lock(hotword_mutex_);
    bool has_scorer = hotword_scorer_ && !hotword_scorer_->empty();
    if (has_scorer) {
        hotword_scorer_->reset();
    }

    int prev_token = -1;
    for (int t = 0; t < seq_len; ++t) {
        const float* frame = logits + t * vocab_size;
        int max_token = 0;
        float max_prob = frame[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (frame[v] > max_prob) {
                max_prob = frame[v];
                max_token = v;
            }
        }
        if (has_scorer) {
            hotword_scorer_->applyBias(
                    frame, vocab_size, &max_token, &max_prob);
        }

        if (max_token != blank_id_ && max_token != prev_token) {
            tokens.push_back(max_token);
            if (has_scorer) {
                hotword_scorer_->advanceFrame(max_token);
            }
        }
        prev_token = max_token;
    }

    return tokens;
}

void SenseVoiceModel::setLanguage(const std::string& language) {
    config_.language = language;
}

int SenseVoiceModel::getLanguageId(const std::string& language) {
    auto it = language_map_.find(language);
    return it != language_map_.end() ? it->second : language_map_["auto"];
}

int SenseVoiceModel::getTextnormId(bool use_itn) {
    return use_itn ? textnorm_map_["withitn"] : textnorm_map_["woitn"];
}

void SenseVoiceModel::setHotwords(const std::vector<std::string>& hotwords, float boost) {
    std::lock_guard<std::mutex> lock(hotword_mutex_);
    if (hotwords.empty()) {
        hotword_scorer_.reset();
        return;
    }
    if (!tokenizer_) return;
    hotword_scorer_ = std::make_unique<HotwordScorer>();
    hotword_scorer_->build(hotwords, *tokenizer_, boost);
}

}  // namespace sensevoice
}  // namespace asr
