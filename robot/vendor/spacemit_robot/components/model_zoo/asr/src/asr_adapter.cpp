/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpacemiT ASR Engine Adapter Implementation
 *
 * 适配层实现，将 asr::ASREngine 封装为 SpacemiT::AsrEngine 接口。
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "asr_service.h"
#include "asr_engine.hpp"
#include "asr_config.hpp"
#include "asr_callback.hpp"
#include "asr_types.hpp"

namespace SpacemiT {

// =============================================================================
// RecognitionResult::Impl
// =============================================================================

struct RecognitionResult::Impl {
    std::string request_id;
    std::vector<Sentence> sentences;
    int audio_duration_ms = 0;
    int processing_time_ms = 0;
    float rtf = 0.0f;
    bool is_final = true;

    void fromInternal(const asr::RecognitionResult& internal) {
        request_id = internal.request_id;
        audio_duration_ms = static_cast<int>(internal.audio_duration_ms);
        processing_time_ms = static_cast<int>(internal.processing_time_ms);
        rtf = internal.rtf;

        sentences.clear();
        for (const auto& s : internal.sentences) {
            Sentence sentence;
            sentence.text = s.text;
            sentence.begin_time = s.begin_time_ms;
            sentence.end_time = s.end_time_ms;
            sentence.confidence = s.confidence;
            sentence.emotion = s.emotion;
            sentences.push_back(std::move(sentence));

            // 使用最后一个句子的 is_final 状态
            is_final = s.is_final;
        }
    }
};

// =============================================================================
// RecognitionResult Implementation
// =============================================================================

RecognitionResult::RecognitionResult()
    : impl_(std::make_unique<Impl>()) {
}

RecognitionResult::~RecognitionResult() = default;

RecognitionResult::RecognitionResult(RecognitionResult&&) noexcept = default;
RecognitionResult& RecognitionResult::operator=(RecognitionResult&&) noexcept = default;

void RecognitionResult::setFromInternal(const asr::RecognitionResult& internal) {
    impl_->fromInternal(internal);
}

void RecognitionResult::setFinal(bool is_final) {
    impl_->is_final = is_final;
}

Sentence RecognitionResult::GetSentence() const {
    if (impl_->sentences.empty()) {
        return Sentence{};
    }
    return impl_->sentences[0];
}

std::vector<Sentence> RecognitionResult::GetSentences() const {
    return impl_->sentences;
}

bool RecognitionResult::IsSentenceEnd() const {
    return impl_->is_final;
}

std::string RecognitionResult::GetRequestId() const {
    return impl_->request_id;
}

std::string RecognitionResult::GetText() const {
    std::string result;
    for (const auto& s : impl_->sentences) {
        result += s.text;
    }
    return result;
}

std::string RecognitionResult::GetEmotion() const {
    for (const auto& s : impl_->sentences) {
        if (!s.emotion.empty()) {
            return s.emotion;
        }
    }
    return "";
}

bool RecognitionResult::IsEmpty() const {
    return impl_->sentences.empty() || GetText().empty();
}

int RecognitionResult::GetAudioDuration() const {
    return impl_->audio_duration_ms;
}

int RecognitionResult::GetProcessingTime() const {
    return impl_->processing_time_ms;
}

float RecognitionResult::GetRTF() const {
    return impl_->rtf;
}

// Forward declaration
class CallbackAdapter;

// =============================================================================
// AsrEngine::Impl (must be defined before CallbackAdapter body)
// =============================================================================

struct AsrEngine::Impl {
    std::unique_ptr<asr::ASREngine> engine;
    std::shared_ptr<AsrEngineCallback> user_callback;
    std::unique_ptr<CallbackAdapter> callback_adapter;
    std::string engine_name;
    std::string model_dir;
    bool initialized = false;

    // 公开配置（SpacemiT::AsrConfig）
    AsrConfig public_config;

    // 缓存最后的结果用于 GetResponse()
    std::mutex result_mutex;
    asr::RecognitionResult last_result;
    int first_package_delay_ms = 0;
    int last_package_delay_ms = 0;

    // 流式模式下的音频缓冲
    std::vector<int16_t> audio_buffer;

    std::string generateRequestId() {
        static thread_local std::mt19937 gen(std::random_device{}());
        static thread_local std::uniform_int_distribution<> dis(0, 999);

        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << "req_" << ms << "_" << dis(gen);
        return oss.str();
    }

    // 语言字符串转枚举
    static asr::Language languageFromString(const std::string& lang) {
        if (lang == "zh" || lang == "ZH") return asr::Language::ZH;
        if (lang == "en" || lang == "EN") return asr::Language::EN;
        if (lang == "ja" || lang == "JA") return asr::Language::JA;
        if (lang == "ko" || lang == "KO") return asr::Language::KO;
        if (lang == "yue" || lang == "YUE") return asr::Language::YUE;
        if (lang == "auto" || lang == "AUTO") return asr::Language::AUTO;
        return asr::Language::AUTO;
    }

    // 语言枚举转字符串
    static std::string languageToString(asr::Language lang) {
        switch (lang) {
            case asr::Language::ZH: return "zh";
            case asr::Language::EN: return "en";
            case asr::Language::JA: return "ja";
            case asr::Language::KO: return "ko";
            case asr::Language::YUE: return "yue";
            case asr::Language::AUTO: return "auto";
            default: return "zh";
        }
    }

    std::string resultToJson(const asr::RecognitionResult& result) {
        std::ostringstream json;
        json << std::fixed << std::setprecision(3);
        json << "{\n";
        json << "    \"request_id\": \"" << result.request_id << "\",\n";
        json << "    \"sentences\": [\n";

        for (size_t i = 0; i < result.sentences.size(); ++i) {
            const auto& s = result.sentences[i];
            json << "        {\n";
            json << "            \"text\": \"" << escapeJson(s.text) << "\",\n";
            json << "            \"begin_time\": " << s.begin_time_ms << ",\n";
            json << "            \"end_time\": " << s.end_time_ms << ",\n";
            json << "            \"is_final\": " << (s.is_final ? "true" : "false") << ",\n";
            if (!s.emotion.empty()) {
                json << "            \"emotion\": \"" << escapeJson(s.emotion) << "\",\n";
            }
            json << "            \"confidence\": " << s.confidence << "\n";
            json << "        }";
            if (i < result.sentences.size() - 1) json << ",";
            json << "\n";
        }

        json << "    ],\n";
        json << "    \"audio_duration_ms\": " << result.audio_duration_ms << ",\n";
        json << "    \"processing_time_ms\": " << result.processing_time_ms << ",\n";
        json << "    \"rtf\": " << result.rtf << ",\n";
        json << "    \"first_package_delay_ms\": " << first_package_delay_ms << ",\n";
        json << "    \"last_package_delay_ms\": " << last_package_delay_ms << "\n";
        json << "}";

        return json.str();
    }

    static std::string escapeJson(const std::string& s) {
        std::string result;
        result.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '"':  result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\b': result += "\\b";  break;
                case '\f': result += "\\f";  break;
                case '\n': result += "\\n";  break;
                case '\r': result += "\\r";  break;
                case '\t': result += "\\t";  break;
                default:   result += c;      break;
            }
        }
        return result;
    }
};

// =============================================================================
// CallbackAdapter - 回调适配器
// =============================================================================

class CallbackAdapter : public asr::IASRCallback {
public:
    explicit CallbackAdapter(std::shared_ptr<AsrEngineCallback> callback,
                            AsrEngine::Impl* impl = nullptr)
        : callback_(std::move(callback)), impl_(impl) {
    }

    void setImpl(AsrEngine::Impl* impl) {
        impl_ = impl;
    }

    void onStart() override {
        if (callback_) {
            callback_->OnOpen();
        }
    }

    void onResult(const asr::RecognitionResult& result) override {
        // 缓存结果供 GetResponse() 使用
        if (impl_) {
            std::lock_guard<std::mutex> lock(impl_->result_mutex);
            impl_->last_result = result;
        }

        if (callback_) {
            auto spacemit_result = std::make_shared<RecognitionResult>();
            spacemit_result->setFromInternal(result);
            callback_->OnEvent(spacemit_result);
        }
    }

    void onSentence(const asr::SentenceResult& sentence, bool is_final) override {
        // 缓存句子结果供 GetResponse() 使用
        if (impl_) {
            std::lock_guard<std::mutex> lock(impl_->result_mutex);
            asr::SentenceResult sr;
            sr.text = sentence.text;
            sr.begin_time_ms = sentence.begin_time_ms;
            sr.end_time_ms = sentence.end_time_ms;
            sr.confidence = sentence.confidence;
            sr.is_final = is_final;
            impl_->last_result.sentences.clear();
            impl_->last_result.sentences.push_back(sr);
        }

        if (callback_) {
            auto result = std::make_shared<RecognitionResult>();
            Sentence s;
            s.text = sentence.text;
            s.begin_time = sentence.begin_time_ms;
            s.end_time = sentence.end_time_ms;
            s.confidence = sentence.confidence;
            result->impl_->sentences.push_back(std::move(s));
            result->impl_->is_final = is_final;
            callback_->OnEvent(result);
        }
    }

    void onComplete() override {
        if (callback_) {
            callback_->OnComplete();
        }
    }

    void onError(const asr::ErrorInfo& error) override {
        if (callback_) {
            auto result = std::make_shared<RecognitionResult>();
            // 将错误信息放入 request_id 或通过其他方式传递
            result->impl_->request_id =
                "error:" + std::to_string(static_cast<int>(error.code));
            Sentence s;
            s.text = error.message;
            result->impl_->sentences.push_back(std::move(s));
            callback_->OnError(result);
        }
    }

    void onClose() override {
        if (callback_) {
            callback_->OnClose();
        }
    }

private:
    std::shared_ptr<AsrEngineCallback> callback_;
    AsrEngine::Impl* impl_ = nullptr;
};

// =============================================================================
// AsrEngine Implementation
// =============================================================================

AsrEngine::AsrEngine(const std::string& engine, const std::string& model_dir)
    : impl_(std::make_unique<Impl>()) {

    impl_->engine_name = engine;
    impl_->model_dir = model_dir;

    // 保存公开配置
    impl_->public_config.engine = engine.empty() ? "sensevoice" : engine;
    impl_->public_config.model_dir = model_dir;

    // 创建内部引擎
    impl_->engine = std::make_unique<asr::ASREngine>();

    // 构建配置
    asr::ASRConfig config;

    if (engine == "sensevoice" || engine.empty()) {
        // 使用 SenseVoice 后端
        std::string dir = model_dir.empty() ? "~/.cache/models/asr/sensevoice" : model_dir;
        config = asr::ASRConfig::sensevoice(dir);
    } else {
        // 未来支持其他引擎
        // 目前默认使用 SenseVoice
        std::string dir = model_dir.empty() ? "~/.cache/models/asr/sensevoice" : model_dir;
        config = asr::ASRConfig::sensevoice(dir);
    }

    config.language = asr::Language::AUTO;

    // 初始化引擎
    auto err = impl_->engine->initialize(config);
    impl_->initialized = err.isOk();

    if (impl_->initialized) {
        // 设置内部回调以捕获流式结果
        impl_->callback_adapter =
            std::make_unique<CallbackAdapter>(nullptr, impl_.get());
        impl_->engine->setCallback(impl_->callback_adapter.get());
    }
}

AsrEngine::AsrEngine(const AsrConfig& config)
    : impl_(std::make_unique<Impl>()) {

    // 保存公开配置
    impl_->public_config = config;
    impl_->engine_name = config.engine;
    impl_->model_dir = config.model_dir;

    // 创建内部引擎
    impl_->engine = std::make_unique<asr::ASREngine>();

    // 构建内部配置
    asr::ASRConfig internal_config;

    if (config.engine == "qwen3-asr") {
        internal_config.backend = asr::BackendType::QWEN3_ASR;
        // Pass llama-server params via extra_params
        internal_config.extra_params["endpoint"] = config.endpoint;
        internal_config.extra_params["model"] = config.model;
        internal_config.extra_params["timeout"] = std::to_string(config.timeout);
    } else if (config.engine == "zipformer") {
        std::string dir =
            config.model_dir.empty() ? "~/.cache/models/asr/zipformer" : config.model_dir;
        internal_config = asr::ASRConfig::zipformer(dir);
    } else if (config.engine == "sensevoice" || config.engine.empty()) {
        std::string dir =
            config.model_dir.empty() ? "~/.cache/models/asr/sensevoice" : config.model_dir;
        internal_config = asr::ASRConfig::sensevoice(dir);
    } else {
        std::string dir =
            config.model_dir.empty() ? "~/.cache/models/asr/sensevoice" : config.model_dir;
        internal_config = asr::ASRConfig::sensevoice(dir);
    }

    // 应用公开配置
    internal_config.language = Impl::languageFromString(config.language);
    internal_config.punctuation_enabled = config.punctuation;
    internal_config.enable_emotion = config.enable_emotion;
    internal_config.sample_rate = config.sample_rate;
    if (!config.provider.empty()) {
        internal_config.extra_params["provider"] = config.provider;
    }

    // 转发热词配置
    internal_config.hotwords = config.hotwords;
    internal_config.hotword_boost = config.hotword_boost;
    internal_config.hotword_file = config.hotword_file;

    // 初始化引擎
    auto err = impl_->engine->initialize(internal_config);
    impl_->initialized = err.isOk();

    if (impl_->initialized) {
        impl_->callback_adapter =
            std::make_unique<CallbackAdapter>(nullptr, impl_.get());
        impl_->engine->setCallback(impl_->callback_adapter.get());
    }
}

AsrEngine::~AsrEngine() {
    // Clear callback before shutdown to prevent dangling pointer access
    if (impl_->engine) {
        impl_->engine->setCallback(nullptr);
    }
    impl_->callback_adapter.reset();

    if (impl_->engine) {
        impl_->engine->shutdown();
    }
}

std::shared_ptr<RecognitionResult> AsrEngine::Call(
    const std::string& file_path, const std::string& phrase_id) {
    if (!impl_->initialized || !impl_->engine) {
        return nullptr;
    }

    // phrase_id 暂时忽略
    (void)phrase_id;

    // 调用内部引擎识别
    auto internal_result = impl_->engine->recognizeFile(file_path);

    // 更新缓存
    {
        std::lock_guard<std::mutex> lock(impl_->result_mutex);
        impl_->last_result = internal_result;
        impl_->first_package_delay_ms =
            static_cast<int>(impl_->engine->getFirstPacketLatency());
        impl_->last_package_delay_ms =
            static_cast<int>(impl_->engine->getLastPacketLatency());
    }

    // 转换为 SpacemiT 结果
    auto result = std::make_shared<RecognitionResult>();
    result->setFromInternal(internal_result);
    return result;
}

void AsrEngine::SetCallback(std::shared_ptr<AsrEngineCallback> callback) {
    impl_->user_callback = std::move(callback);

    // 创建新的适配器，同时保留 impl_ 指针以捕获结果
    impl_->callback_adapter =
        std::make_unique<CallbackAdapter>(impl_->user_callback, impl_.get());
    impl_->engine->setCallback(impl_->callback_adapter.get());
}

void AsrEngine::Start(const std::string& phrase_id) {
    if (!impl_->initialized || !impl_->engine) {
        return;
    }

    // phrase_id 暂时忽略
    (void)phrase_id;

    // 清空音频缓冲
    impl_->audio_buffer.clear();

    // 启动流式识别
    impl_->engine->start();
}

void AsrEngine::SendAudioFrame(const std::vector<uint8_t>& data) {
    if (!impl_->initialized || !impl_->engine) {
        return;
    }

    if (data.empty()) {
        return;
    }

    // 将 uint8_t 数据转换为 int16_t（假设输入是 16bit PCM）
    // data.size() 应该是偶数（每个样本 2 字节）
    size_t samples = data.size() / sizeof(int16_t);
    const int16_t* audio_ptr = reinterpret_cast<const int16_t*>(data.data());

    // 发送音频数据
    impl_->engine->sendAudio(audio_ptr, samples);
}

void AsrEngine::Flush() {
    if (!impl_->initialized || !impl_->engine) {
        return;
    }

    // 刷新缓冲区并立即识别（不关闭会话）
    impl_->engine->flush();
}

void AsrEngine::Stop() {
    if (!impl_->initialized || !impl_->engine) {
        return;
    }

    // 停止流式识别
    impl_->engine->stop();

    // 更新延迟信息
    {
        std::lock_guard<std::mutex> lock(impl_->result_mutex);
        impl_->first_package_delay_ms =
            static_cast<int>(impl_->engine->getFirstPacketLatency());
        impl_->last_package_delay_ms =
            static_cast<int>(impl_->engine->getLastPacketLatency());
    }
}

std::string AsrEngine::GetLastRequestId() {
    if (impl_->engine) {
        return impl_->engine->getLastRequestId();
    }
    return "";
}

int AsrEngine::GetFirstPackageDelay() {
    std::lock_guard<std::mutex> lock(impl_->result_mutex);
    return impl_->first_package_delay_ms;
}

int AsrEngine::GetLastPackageDelay() {
    std::lock_guard<std::mutex> lock(impl_->result_mutex);
    return impl_->last_package_delay_ms;
}

std::string AsrEngine::GetResponse() {
    std::lock_guard<std::mutex> lock(impl_->result_mutex);
    return impl_->resultToJson(impl_->last_result);
}

bool AsrEngine::IsInitialized() const {
    return impl_->initialized;
}

std::string AsrEngine::GetEngineName() const {
    return impl_->engine_name;
}

std::shared_ptr<RecognitionResult> AsrEngine::Recognize(
    const std::vector<int16_t>& audio, int sample_rate) {
    if (!impl_->initialized || !impl_->engine) {
        return nullptr;
    }

    // 调用内部引擎识别
    auto chunk = asr::AudioChunk::fromPCM16(audio.data(), audio.size(), sample_rate, 1);
    auto internal_result = impl_->engine->recognize(chunk);

    // 更新缓存
    {
        std::lock_guard<std::mutex> lock(impl_->result_mutex);
        impl_->last_result = internal_result;
        impl_->first_package_delay_ms =
            static_cast<int>(impl_->engine->getFirstPacketLatency());
        impl_->last_package_delay_ms =
            static_cast<int>(impl_->engine->getLastPacketLatency());
    }

    // 转换为 SpacemiT 结果
    auto result = std::make_shared<RecognitionResult>();
    result->setFromInternal(internal_result);
    return result;
}

std::shared_ptr<RecognitionResult> AsrEngine::Recognize(
    const std::vector<float>& audio, int sample_rate) {
    if (!impl_->initialized || !impl_->engine) {
        return nullptr;
    }

    // 调用内部引擎识别
    auto chunk = asr::AudioChunk::fromPCMFloat(audio.data(), audio.size(), sample_rate, 1);
    auto internal_result = impl_->engine->recognize(chunk);

    // 更新缓存
    {
        std::lock_guard<std::mutex> lock(impl_->result_mutex);
        impl_->last_result = internal_result;
        impl_->first_package_delay_ms =
            static_cast<int>(impl_->engine->getFirstPacketLatency());
        impl_->last_package_delay_ms =
            static_cast<int>(impl_->engine->getLastPacketLatency());
    }

    // 转换为 SpacemiT 结果
    auto result = std::make_shared<RecognitionResult>();
    result->setFromInternal(internal_result);
    return result;
}

void AsrEngine::SetLanguage(const std::string& language) {
    if (!impl_->initialized || !impl_->engine) {
        return;
    }

    impl_->public_config.language = language;
    impl_->engine->setLanguage(Impl::languageFromString(language));
}

void AsrEngine::SetPunctuation(bool enabled) {
    impl_->public_config.punctuation = enabled;
}

void AsrEngine::SetHotwords(const std::vector<std::string>& hotwords, float boost) {
    impl_->public_config.hotwords = hotwords;
    impl_->public_config.hotword_boost = boost;
    if (impl_->initialized && impl_->engine) {
        impl_->engine->updateHotwords(hotwords);
    }
}

void AsrEngine::LoadHotwordFile(const std::string& file_path, float default_boost) {
    std::vector<std::string> words;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            words.push_back(line.substr(0, tab_pos));
        } else {
            words.push_back(line);
        }
    }
    SetHotwords(words, default_boost);
    impl_->public_config.hotword_file = file_path;
}

AsrConfig AsrEngine::GetConfig() const {
    return impl_->public_config;
}

}  // namespace SpacemiT
