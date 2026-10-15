/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <portaudio.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "internal/audio_stream.hpp"

// ============================================================================
// PortAudio Initialization Reference Counter
// Ensures Pa_Initialize/Pa_Terminate are called correctly across all streams
// ============================================================================
namespace {
    constexpr size_t kDefaultCallbackBufferFrames = 4096;

    static int g_pa_ref_count = 0;
    static std::mutex g_pa_mutex;

    size_t callbackBufferFrames(int frames_per_buffer) {
        return frames_per_buffer > 0 ?
            static_cast<size_t>(frames_per_buffer) : kDefaultCallbackBufferFrames;
    }

    #ifdef __riscv
    bool isMisalignedFloat(const void* ptr) {
        return ptr && (reinterpret_cast<uintptr_t>(ptr) & 0x03);
    }
    #endif

    PaSampleFormat toPortAudioFormat(AudioSampleFormat format) {
        switch (format) {
            case AudioSampleFormat::INT16:
                return paInt16;
            case AudioSampleFormat::INT32:
                return paInt32;
            case AudioSampleFormat::FLOAT32:
            default:
                return paFloat32;
        }
    }

    size_t bytesPerSample(AudioSampleFormat format) {
        switch (format) {
            case AudioSampleFormat::INT16:
                return sizeof(int16_t);
            case AudioSampleFormat::INT32:
                return sizeof(int32_t);
            case AudioSampleFormat::FLOAT32:
            default:
                return sizeof(float);
        }
    }

    int finishBlockingWrite(PaError err, size_t frames) {
        if (err == paOutputUnderflowed) {
            static std::atomic<int> underflow_count{0};
            int count = ++underflow_count;
            if (count == 1 || count % 50 == 0) {
                std::cerr << "[AudioOutputStream] Write underflowed (count="
                    << count << ")" << std::endl;
            }
        }
        if (err != paNoError && err != paOutputUnderflowed) {
            std::cerr << "[AudioOutputStream] Write failed: "
                << Pa_GetErrorText(err) << std::endl;
            return -1;
        }

        return static_cast<int>(frames);
    }

    bool ensurePortAudioInitialized() {
        std::lock_guard<std::mutex> lock(g_pa_mutex);
        if (g_pa_ref_count == 0) {
            // 在 Pa_Initialize() 周围把 fd 2 重定向到 /dev/null 来压制 ALSA 警告。
            // 之前实现用的是 freopen("/dev/null") + freopen("/dev/tty")，但
            // 在没有 controlling tty 的环境（daemon、systemd、非交互 ssh、cron）
            // 下后一个 freopen 会失败，stderr FILE 永久失效，导致后续所有
            // std::cerr 输出全部丢失。改为 dup/dup2 操作 fd，不动 stderr FILE。
            #ifndef __APPLE__
            int saved_stderr = dup(fileno(stderr));
            int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                fflush(stderr);
                dup2(devnull, fileno(stderr));
                ::close(devnull);
            }
            #endif

            PaError err = Pa_Initialize();

            #ifndef __APPLE__
            if (saved_stderr >= 0) {
                fflush(stderr);
                dup2(saved_stderr, fileno(stderr));
                ::close(saved_stderr);
            }
            #endif

            if (err != paNoError) {
                std::cerr << "[AudioStream] Failed to initialize PortAudio: "
                            << Pa_GetErrorText(err) << std::endl;
                return false;
            }
        }
        g_pa_ref_count++;
        return true;
    }

    void releasePortAudio() {
        std::lock_guard<std::mutex> lock(g_pa_mutex);
        if (g_pa_ref_count > 0) {
            g_pa_ref_count--;
            if (g_pa_ref_count == 0) {
                Pa_Terminate();
            }
        }
    }
}  // namespace

// ============================================================================
// AudioInputStream Implementation
// ============================================================================

AudioInputStream::AudioInputStream()
    : stream_(nullptr)
    , callback_(nullptr)
    , user_data_(nullptr)
    , actual_sample_rate_(0)
    , actual_channels_(0)
    , device_index_(-1)
    , actual_format_(AudioSampleFormat::FLOAT32)
    , is_running_(false)
    , is_open_(false) {
}

AudioInputStream::~AudioInputStream() {
    close();
}

AudioInputStream::AudioInputStream(AudioInputStream&& other) noexcept
    : stream_(other.stream_)
    , callback_(std::move(other.callback_))
    , user_data_(other.user_data_)
    , actual_sample_rate_(other.actual_sample_rate_)
    , actual_channels_(other.actual_channels_)
    , device_index_(other.device_index_)
    , actual_format_(other.actual_format_)
    , float_buffer_(std::move(other.float_buffer_))
    , is_running_(other.is_running_.load())
    , is_open_(other.is_open_.load()) {
    other.stream_ = nullptr;
    other.is_running_.store(false);
    other.is_open_.store(false);
}

AudioInputStream& AudioInputStream::operator=(AudioInputStream&& other) noexcept {
    if (this != &other) {
        close();
        stream_ = other.stream_;
        callback_ = std::move(other.callback_);
        user_data_ = other.user_data_;
        actual_sample_rate_ = other.actual_sample_rate_;
        actual_channels_ = other.actual_channels_;
        device_index_ = other.device_index_;
        actual_format_ = other.actual_format_;
        float_buffer_ = std::move(other.float_buffer_);
        is_running_.store(other.is_running_.load());
        is_open_.store(other.is_open_.load());

        other.stream_ = nullptr;
        other.is_running_.store(false);
        other.is_open_.store(false);
    }
    return *this;
}

void AudioInputStream::setCallback(AudioInputCallback callback, void* user_data) {
    callback_ = callback;
    user_data_ = user_data;
}

int AudioInputStream::paCallback(const void* input_buffer, void* output_buffer,
                                unsigned long frames_per_buffer,  // NOLINT(runtime/int)
                                const PaStreamCallbackTimeInfo* time_info,
                                PaStreamCallbackFlags status_flags,
                                void* user_data) {
    (void)output_buffer;
    (void)time_info;
    (void)status_flags;

    AudioInputStream* self = static_cast<AudioInputStream*>(user_data);
    if (!self || !self->callback_ || !input_buffer) {
        return paContinue;
    }

    const size_t samples = frames_per_buffer * self->actual_channels_;
    const float* input = nullptr;
    if (self->actual_format_ == AudioSampleFormat::INT16) {
        const uint8_t* pcm = static_cast<const uint8_t*>(input_buffer);
        if (self->float_buffer_.size() < samples) {
            return paContinue;
        }
        for (size_t i = 0; i < samples; ++i) {
            int16_t sample = 0;
            std::memcpy(&sample, pcm + i * sizeof(sample), sizeof(sample));
            self->float_buffer_[i] = static_cast<float>(sample) / 32768.0f;
        }
        input = self->float_buffer_.data();
    } else if (self->actual_format_ == AudioSampleFormat::INT32) {
        const uint8_t* pcm = static_cast<const uint8_t*>(input_buffer);
        if (self->float_buffer_.size() < samples) {
            return paContinue;
        }
        for (size_t i = 0; i < samples; ++i) {
            int32_t sample = 0;
            std::memcpy(&sample, pcm + i * sizeof(sample), sizeof(sample));
            self->float_buffer_[i] = static_cast<float>(sample) / 2147483648.0f;
        }
        input = self->float_buffer_.data();
    } else {
        #ifdef __riscv
        if (isMisalignedFloat(input_buffer)) {
            if (self->float_buffer_.size() < samples) {
                return paContinue;
            }
            std::memcpy(self->float_buffer_.data(), input_buffer,
                samples * sizeof(float));
            input = self->float_buffer_.data();
        } else
        #endif
        {
        input = static_cast<const float*>(input_buffer);
        }
    }

    self->callback_(input, frames_per_buffer, self->actual_channels_, self->user_data_);

    return paContinue;
}

bool AudioInputStream::open(const AudioInputConfig& config) {
    if (is_open_.load()) {
        std::cerr << "[AudioInputStream] Stream already open" << std::endl;
        return false;
    }

    if (!ensurePortAudioInitialized()) {
        return false;
    }

    // Find device
    int device = config.device_index;
    if (config.device_name_hint != nullptr && config.device_name_hint[0] != '\0') {
        device = findDeviceByName(config.device_name_hint);
        if (device < 0) {
            std::cerr << "[AudioInputStream] Device not found: " << config.device_name_hint << std::endl;
            releasePortAudio();
            return false;
        }
    }

    if (device < 0) {
        device = Pa_GetDefaultInputDevice();
    }

    if (device == paNoDevice) {
        std::cerr << "[AudioInputStream] No default input device" << std::endl;
        releasePortAudio();
        return false;
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(device);
    if (!deviceInfo) {
        std::cerr << "[AudioInputStream] Failed to get device info" << std::endl;
        releasePortAudio();
        return false;
    }

    std::cout << "[AudioInputStream] Opening device " << device << ": " << deviceInfo->name
                << " (max inputs: " << deviceInfo->maxInputChannels << ")" << std::endl;

    // Verify channel count
    int channels = config.channels;
    if (channels <= 0) {
        std::cerr << "[AudioInputStream] Invalid channel count: "
            << channels << std::endl;
        releasePortAudio();
        return false;
    }
    if (channels > deviceInfo->maxInputChannels) {
        std::cerr << "[AudioInputStream] Requested " << channels << " channels but device only has "
                    << deviceInfo->maxInputChannels << std::endl;
        releasePortAudio();
        return false;
    }

    // Setup stream parameters
    PaStreamParameters inputParams;
    inputParams.device = device;
    inputParams.channelCount = channels;
    inputParams.sampleFormat = toPortAudioFormat(config.format);
    inputParams.suggestedLatency = deviceInfo->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;

    // Open stream
    PaError err = Pa_OpenStream(
        reinterpret_cast<PaStream**>(&stream_),
        &inputParams,
        nullptr,  // No output
        config.sample_rate,
        config.frames_per_buffer,
        paClipOff,
        paCallback,
        this);

    if (err != paNoError) {
        std::cerr << "[AudioInputStream] Failed to open stream: " << Pa_GetErrorText(err) << std::endl;
        releasePortAudio();
        return false;
    }

    device_index_ = device;
    actual_sample_rate_ = config.sample_rate;
    actual_channels_ = channels;
    actual_format_ = config.format;
    bool needs_callback_buffer = actual_format_ != AudioSampleFormat::FLOAT32;
    #ifdef __riscv
    needs_callback_buffer = true;
    #endif
    if (needs_callback_buffer) {
        const size_t samples =
            callbackBufferFrames(config.frames_per_buffer) * actual_channels_;
        float_buffer_.resize(samples);
    }
    is_open_.store(true);

    std::cout << "[AudioInputStream] Opened: " << actual_sample_rate_ << "Hz, "
                << actual_channels_ << " channels, "
                << (actual_format_ == AudioSampleFormat::INT16 ? "int16" :
                    (actual_format_ == AudioSampleFormat::INT32 ? "int32" : "float32"))
                << std::endl;
    return true;
}

void AudioInputStream::close() {
    if (!is_open_.load()) {
        return;
    }

    stop();

    if (stream_) {
        Pa_CloseStream(static_cast<PaStream*>(stream_));
        stream_ = nullptr;
    }

    is_open_.store(false);
    releasePortAudio();
    std::cout << "[AudioInputStream] Closed" << std::endl;
}

bool AudioInputStream::start() {
    if (!is_open_.load()) {
        std::cerr << "[AudioInputStream] Cannot start: stream not open" << std::endl;
        return false;
    }

    if (is_running_.load()) {
        return true;  // Already running
    }

    PaError err = Pa_StartStream(static_cast<PaStream*>(stream_));
    if (err != paNoError) {
        std::cerr << "[AudioInputStream] Failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    is_running_.store(true);
    std::cout << "[AudioInputStream] Started" << std::endl;
    return true;
}

bool AudioInputStream::stop() {
    if (!is_running_.load()) {
        return true;  // Already stopped
    }

    PaError err = Pa_StopStream(static_cast<PaStream*>(stream_));
    if (err != paNoError) {
        std::cerr << "[AudioInputStream] Failed to stop stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    is_running_.store(false);
    std::cout << "[AudioInputStream] Stopped" << std::endl;
    return true;
}

bool AudioInputStream::isRunning() const {
    return is_running_.load();
}

bool AudioInputStream::isOpen() const {
    return is_open_.load();
}

void AudioInputStream::listDevices(std::vector<std::string>& names, std::vector<int>& indices) {
    names.clear();
    indices.clear();

    if (!ensurePortAudioInitialized()) {
        return;
    }

    int numDevices = Pa_GetDeviceCount();
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxInputChannels > 0) {
            names.push_back(info->name);
            indices.push_back(i);
        }
    }

    releasePortAudio();
}

int AudioInputStream::findDeviceByName(const char* name_hint) {
    if (!name_hint || name_hint[0] == '\0') {
        return -1;
    }

    if (!ensurePortAudioInitialized()) {
        return -1;
    }

    int result = -1;
    int numDevices = Pa_GetDeviceCount();
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxInputChannels > 0) {
            if (strstr(info->name, name_hint) != nullptr) {
                result = i;
                break;
            }
        }
    }

    releasePortAudio();
    return result;
}


// ============================================================================
// AudioOutputStream Implementation
// ============================================================================

AudioOutputStream::AudioOutputStream()
    : stream_(nullptr)
    , callback_(nullptr)
    , user_data_(nullptr)
    , actual_sample_rate_(0)
    , actual_channels_(0)
    , device_index_(-1)
    , actual_format_(AudioSampleFormat::FLOAT32)
    , is_running_(false)
    , is_open_(false)
    , use_callback_mode_(false) {
}

AudioOutputStream::~AudioOutputStream() {
    close();
}

AudioOutputStream::AudioOutputStream(AudioOutputStream&& other) noexcept
    : stream_(other.stream_)
    , callback_(std::move(other.callback_))
    , user_data_(other.user_data_)
    , actual_sample_rate_(other.actual_sample_rate_)
    , actual_channels_(other.actual_channels_)
    , device_index_(other.device_index_)
    , actual_format_(other.actual_format_)
    , is_running_(other.is_running_.load())
    , is_open_(other.is_open_.load())
    , use_callback_mode_(other.use_callback_mode_)
    , float_buffer_(std::move(other.float_buffer_))
    , int16_buffer_(std::move(other.int16_buffer_))
    , int32_buffer_(std::move(other.int32_buffer_)) {
    other.stream_ = nullptr;
    other.is_running_.store(false);
    other.is_open_.store(false);
}

AudioOutputStream& AudioOutputStream::operator=(AudioOutputStream&& other) noexcept {
    if (this != &other) {
        close();
        stream_ = other.stream_;
        callback_ = std::move(other.callback_);
        user_data_ = other.user_data_;
        actual_sample_rate_ = other.actual_sample_rate_;
        actual_channels_ = other.actual_channels_;
        device_index_ = other.device_index_;
        actual_format_ = other.actual_format_;
        is_running_.store(other.is_running_.load());
        is_open_.store(other.is_open_.load());
        use_callback_mode_ = other.use_callback_mode_;
        float_buffer_ = std::move(other.float_buffer_);
        int16_buffer_ = std::move(other.int16_buffer_);
        int32_buffer_ = std::move(other.int32_buffer_);

        other.stream_ = nullptr;
        other.is_running_.store(false);
        other.is_open_.store(false);
    }
    return *this;
}

void AudioOutputStream::setCallback(AudioOutputCallback callback, void* user_data) {
    callback_ = callback;
    user_data_ = user_data;
    use_callback_mode_ = (callback != nullptr);
}

int AudioOutputStream::paCallback(const void* input_buffer, void* output_buffer,
                                    unsigned long frames_per_buffer,  // NOLINT(runtime/int)
                                    const PaStreamCallbackTimeInfo* time_info,
                                    PaStreamCallbackFlags status_flags,
                                    void* user_data) {
    (void)input_buffer;
    (void)time_info;
    (void)status_flags;

    AudioOutputStream* self = static_cast<AudioOutputStream*>(user_data);
    if (!self) {
        return paAbort;
    }

    if (!self->callback_ || !output_buffer) {
        // Fill with silence if no callback
        if (output_buffer) {
            std::memset(output_buffer, 0,
                frames_per_buffer * self->actual_channels_ *
                    bytesPerSample(self->actual_format_));
        }
        return paContinue;
    }

    size_t frames_written = 0;
    if (self->actual_format_ == AudioSampleFormat::INT16) {
        const size_t samples = frames_per_buffer * self->actual_channels_;
        if (self->float_buffer_.size() < samples) {
            std::memset(output_buffer, 0,
                samples * bytesPerSample(self->actual_format_));
            return paContinue;
        }
        frames_written = self->callback_(self->float_buffer_.data(), frames_per_buffer,
            self->actual_channels_, self->user_data_);
        frames_written = std::min(frames_written, static_cast<size_t>(frames_per_buffer));

        uint8_t* output = static_cast<uint8_t*>(output_buffer);
        const size_t written_samples = frames_written * self->actual_channels_;
        for (size_t i = 0; i < written_samples; ++i) {
            int16_t sample = static_cast<int16_t>(
                std::clamp(self->float_buffer_[i], -1.0f, 1.0f) * 32767.0f);
            std::memcpy(output + i * sizeof(sample), &sample, sizeof(sample));
        }
    } else if (self->actual_format_ == AudioSampleFormat::INT32) {
        const size_t samples = frames_per_buffer * self->actual_channels_;
        if (self->float_buffer_.size() < samples) {
            std::memset(output_buffer, 0,
                samples * bytesPerSample(self->actual_format_));
            return paContinue;
        }
        frames_written = self->callback_(self->float_buffer_.data(), frames_per_buffer,
            self->actual_channels_, self->user_data_);
        frames_written = std::min(frames_written, static_cast<size_t>(frames_per_buffer));

        uint8_t* output = static_cast<uint8_t*>(output_buffer);
        const size_t written_samples = frames_written * self->actual_channels_;
        for (size_t i = 0; i < written_samples; ++i) {
            int32_t sample = static_cast<int32_t>(
                std::clamp(self->float_buffer_[i], -1.0f, 1.0f) * 2147483647.0f);
            std::memcpy(output + i * sizeof(sample), &sample, sizeof(sample));
        }
    } else {
        float* output = static_cast<float*>(output_buffer);
        bool copy_output = false;
        #ifdef __riscv
        if (isMisalignedFloat(output_buffer)) {
            const size_t samples = frames_per_buffer * self->actual_channels_;
            if (self->float_buffer_.size() < samples) {
                std::memset(output_buffer, 0, samples * sizeof(float));
                return paContinue;
            }
            output = self->float_buffer_.data();
            copy_output = true;
        }
        #endif
        frames_written = self->callback_(output, frames_per_buffer,
            self->actual_channels_, self->user_data_);
        frames_written = std::min(frames_written, static_cast<size_t>(frames_per_buffer));
        if (copy_output && frames_written > 0) {
            std::memcpy(output_buffer, self->float_buffer_.data(),
                frames_written * self->actual_channels_ * sizeof(float));
        }
    }

    // If callback returned less than requested, fill rest with silence
    if (frames_written < frames_per_buffer) {
        size_t remaining = frames_per_buffer - frames_written;
        uint8_t* output_bytes = static_cast<uint8_t*>(output_buffer);
        const size_t offset = frames_written * self->actual_channels_ *
            bytesPerSample(self->actual_format_);
        std::memset(output_bytes + offset, 0,
            remaining * self->actual_channels_ * bytesPerSample(self->actual_format_));

        // If callback returned 0, signal end of playback
        if (frames_written == 0) {
            return paComplete;
        }
    }

    return paContinue;
}

bool AudioOutputStream::open(const AudioOutputConfig& config) {
    if (is_open_.load()) {
        std::cerr << "[AudioOutputStream] Stream already open" << std::endl;
        return false;
    }

    if (!ensurePortAudioInitialized()) {
        return false;
    }

    // Find device
    int device = config.device_index;
    if (config.device_name_hint != nullptr && config.device_name_hint[0] != '\0') {
        device = findDeviceByName(config.device_name_hint);
        if (device < 0) {
            std::cerr << "[AudioOutputStream] Device not found: " << config.device_name_hint << std::endl;
            releasePortAudio();
            return false;
        }
    }

    if (device < 0) {
        device = Pa_GetDefaultOutputDevice();
    }

    if (device == paNoDevice) {
        std::cerr << "[AudioOutputStream] No default output device" << std::endl;
        releasePortAudio();
        return false;
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(device);
    if (!deviceInfo) {
        std::cerr << "[AudioOutputStream] Failed to get device info" << std::endl;
        releasePortAudio();
        return false;
    }

    std::cout << "[AudioOutputStream] Opening device " << device << ": " << deviceInfo->name
                << " (max outputs: " << deviceInfo->maxOutputChannels << ")" << std::endl;

    // Verify channel count
    int channels = config.channels;
    if (channels <= 0) {
        std::cerr << "[AudioOutputStream] Invalid channel count: "
            << channels << std::endl;
        releasePortAudio();
        return false;
    }
    if (channels > deviceInfo->maxOutputChannels) {
        std::cerr << "[AudioOutputStream] Requested " << channels << " channels but device only has "
                    << deviceInfo->maxOutputChannels << std::endl;
        releasePortAudio();
        return false;
    }

    // Setup stream parameters
    PaStreamParameters outputParams;
    outputParams.device = device;
    outputParams.channelCount = channels;
    outputParams.sampleFormat = toPortAudioFormat(config.format);
    outputParams.suggestedLatency = deviceInfo->defaultLowOutputLatency;
    outputParams.hostApiSpecificStreamInfo = nullptr;

    // Open stream
    PaError err;
    if (use_callback_mode_) {
        // Callback mode
        err = Pa_OpenStream(
            reinterpret_cast<PaStream**>(&stream_),
            nullptr,  // No input
            &outputParams,
            config.sample_rate,
            config.frames_per_buffer,
            paClipOff,
            paCallback,
            this);
    } else {
        // Blocking write mode (no callback)
        err = Pa_OpenStream(
            reinterpret_cast<PaStream**>(&stream_),
            nullptr,  // No input
            &outputParams,
            config.sample_rate,
            config.frames_per_buffer,
            paClipOff,
            nullptr,  // No callback for blocking mode
            nullptr);
    }

    if (err != paNoError) {
        std::cerr << "[AudioOutputStream] Failed to open stream: " << Pa_GetErrorText(err) << std::endl;
        releasePortAudio();
        return false;
    }

    device_index_ = device;
    actual_sample_rate_ = config.sample_rate;
    actual_channels_ = channels;
    actual_format_ = config.format;
    bool needs_callback_buffer = actual_format_ != AudioSampleFormat::FLOAT32;
    #ifdef __riscv
    needs_callback_buffer = true;
    #endif
    if (use_callback_mode_ && needs_callback_buffer) {
        const size_t samples =
            callbackBufferFrames(config.frames_per_buffer) * actual_channels_;
        float_buffer_.resize(samples);
    }
    is_open_.store(true);

    std::cout << "[AudioOutputStream] Opened: " << actual_sample_rate_ << "Hz, "
                << actual_channels_ << " channels, "
                << (actual_format_ == AudioSampleFormat::INT16 ? "int16, " :
                    (actual_format_ == AudioSampleFormat::INT32 ? "int32, " : "float32, "))
                << (use_callback_mode_ ? "callback mode" : "write mode") << std::endl;
    return true;
}

void AudioOutputStream::close() {
    if (!is_open_.load()) {
        return;
    }

    stop();

    if (stream_) {
        Pa_CloseStream(static_cast<PaStream*>(stream_));
        stream_ = nullptr;
    }

    is_open_.store(false);
    releasePortAudio();
    std::cout << "[AudioOutputStream] Closed" << std::endl;
}

bool AudioOutputStream::start() {
    if (!is_open_.load()) {
        std::cerr << "[AudioOutputStream] Cannot start: stream not open" << std::endl;
        return false;
    }

    if (is_running_.load()) {
        return true;  // Already running
    }

    PaError err = Pa_StartStream(static_cast<PaStream*>(stream_));
    if (err != paNoError) {
        std::cerr << "[AudioOutputStream] Failed to start stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    is_running_.store(true);
    std::cout << "[AudioOutputStream] Started" << std::endl;
    return true;
}

bool AudioOutputStream::stop() {
    if (!is_running_.load()) {
        return true;  // Already stopped
    }

    PaError err = Pa_StopStream(static_cast<PaStream*>(stream_));
    if (err != paNoError) {
        std::cerr << "[AudioOutputStream] Failed to stop stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    is_running_.store(false);
    std::cout << "[AudioOutputStream] Stopped" << std::endl;
    return true;
}

bool AudioOutputStream::abort() {
    if (!is_running_.load()) {
        return true;  // Already stopped
    }

    PaError err = Pa_AbortStream(static_cast<PaStream*>(stream_));
    if (err != paNoError) {
        std::cerr << "[AudioOutputStream] Failed to abort stream: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }

    is_running_.store(false);
    std::cout << "[AudioOutputStream] Aborted" << std::endl;
    return true;
}

int AudioOutputStream::write(const float* data, size_t frames) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!is_open_.load() || use_callback_mode_ || !data || frames == 0) {
        return -1;
    }

    if (!is_running_.load()) {
        // Auto-start if not running
        if (!start()) {
            return -1;
        }
    }

    if (actual_format_ == AudioSampleFormat::INT16) {
        size_t total_samples = frames * actual_channels_;
        if (int16_buffer_.size() < total_samples) {
            int16_buffer_.resize(total_samples);
        }
        for (size_t i = 0; i < total_samples; ++i) {
            int16_buffer_[i] = static_cast<int16_t>(
                std::clamp(data[i], -1.0f, 1.0f) * 32767.0f);
        }
        PaError err = Pa_WriteStream(static_cast<PaStream*>(stream_),
            int16_buffer_.data(), frames);
        return finishBlockingWrite(err, frames);
    } else if (actual_format_ == AudioSampleFormat::INT32) {
        size_t total_samples = frames * actual_channels_;
        if (int32_buffer_.size() < total_samples) {
            int32_buffer_.resize(total_samples);
        }
        for (size_t i = 0; i < total_samples; ++i) {
            int32_buffer_[i] = static_cast<int32_t>(
                std::clamp(data[i], -1.0f, 1.0f) * 2147483647.0f);
        }
        PaError err = Pa_WriteStream(static_cast<PaStream*>(stream_),
            int32_buffer_.data(), frames);
        return finishBlockingWrite(err, frames);
    }

    PaError err = Pa_WriteStream(static_cast<PaStream*>(stream_), data, frames);
    return finishBlockingWrite(err, frames);
}

int AudioOutputStream::writeInt16(const int16_t* data, size_t frames) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (!is_open_.load() || use_callback_mode_ || !data || frames == 0) {
        return -1;
    }

    if (!is_running_.load()) {
        if (!start()) {
            return -1;
        }
    }

    if (actual_format_ == AudioSampleFormat::INT16) {
        PaError err = Pa_WriteStream(static_cast<PaStream*>(stream_), data, frames);
        return finishBlockingWrite(err, frames);
    } else if (actual_format_ == AudioSampleFormat::INT32) {
        size_t total_samples = frames * actual_channels_;
        if (int32_buffer_.size() < total_samples) {
            int32_buffer_.resize(total_samples);
        }
        for (size_t i = 0; i < total_samples; ++i) {
            int32_buffer_[i] =
                static_cast<int32_t>(static_cast<int64_t>(data[i]) * 65536);
        }
        PaError err = Pa_WriteStream(static_cast<PaStream*>(stream_),
            int32_buffer_.data(), frames);
        return finishBlockingWrite(err, frames);
    }

    // Convert int16 to float using pre-allocated buffer
    size_t total_samples = frames * actual_channels_;

    // Resize only if needed (amortized O(1))
    if (float_buffer_.size() < total_samples) {
        float_buffer_.resize(total_samples);
    }

    for (size_t i = 0; i < total_samples; i++) {
        float_buffer_[i] = static_cast<float>(data[i]) / 32768.0f;
    }

    PaError err = Pa_WriteStream(static_cast<PaStream*>(stream_),
        float_buffer_.data(), frames);
    return finishBlockingWrite(err, frames);
}

bool AudioOutputStream::isRunning() const {
    return is_running_.load();
}

bool AudioOutputStream::isOpen() const {
    return is_open_.load();
}

void AudioOutputStream::listDevices(std::vector<std::string>& names, std::vector<int>& indices) {
    names.clear();
    indices.clear();

    if (!ensurePortAudioInitialized()) {
        return;
    }

    int numDevices = Pa_GetDeviceCount();
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxOutputChannels > 0) {
            names.push_back(info->name);
            indices.push_back(i);
        }
    }

    releasePortAudio();
}

int AudioOutputStream::findDeviceByName(const char* name_hint) {
    if (!name_hint || name_hint[0] == '\0') {
        return -1;
    }

    if (!ensurePortAudioInitialized()) {
        return -1;
    }

    int result = -1;
    int numDevices = Pa_GetDeviceCount();
    for (int i = 0; i < numDevices; i++) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (info && info->maxOutputChannels > 0) {
            if (strstr(info->name, name_hint) != nullptr) {
                result = i;
                break;
            }
        }
    }

    releasePortAudio();
    return result;
}


// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

// Wrapper structures
struct AudioInputStreamHandle {
    AudioInputStream* stream;
    AudioInputCallbackC c_callback;
    void* c_user_data;
};

struct AudioOutputStreamHandle {
    AudioOutputStream* stream;
    AudioOutputCallbackC c_callback;
    void* c_user_data;
};

// Input stream C callback wrapper
static void inputCallbackWrapper(const float* data, size_t frames,
                                int channels, void* user_data) {
    AudioInputStreamHandle* handle = static_cast<AudioInputStreamHandle*>(user_data);
    if (handle && handle->c_callback) {
        handle->c_callback(data, frames, channels, handle->c_user_data);
    }
}

// Output stream C callback wrapper
static size_t outputCallbackWrapper(float* data, size_t frames,
                                    int channels, void* user_data) {
    AudioOutputStreamHandle* handle = static_cast<AudioOutputStreamHandle*>(user_data);
    if (handle && handle->c_callback) {
        return handle->c_callback(data, frames, channels, handle->c_user_data);
    }
    return 0;
}

// ============== AudioInputStream C API ==============

AudioInputStreamHandle* audio_input_create(void) {
    AudioInputStreamHandle* handle = new (std::nothrow) AudioInputStreamHandle();
    if (!handle) return nullptr;

    handle->stream = new (std::nothrow) AudioInputStream();
    if (!handle->stream) {
        delete handle;
        return nullptr;
    }

    handle->c_callback = nullptr;
    handle->c_user_data = nullptr;
    return handle;
}

void audio_input_destroy(AudioInputStreamHandle* handle) {
    if (handle) {
        delete handle->stream;
        delete handle;
    }
}

void audio_input_set_callback(AudioInputStreamHandle* handle,
                                AudioInputCallbackC callback,
                                void* user_data) {
    if (!handle || !handle->stream) return;

    handle->c_callback = callback;
    handle->c_user_data = user_data;

    if (callback) {
        handle->stream->setCallback(inputCallbackWrapper, handle);
    } else {
        handle->stream->setCallback(nullptr, nullptr);
    }
}

int audio_input_open(AudioInputStreamHandle* handle,
                    int sample_rate, int channels,
                    int frames_per_buffer, int device_index) {
    if (!handle || !handle->stream) return 0;

    AudioInputConfig config;
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device_index = device_index;

    return handle->stream->open(config) ? 1 : 0;
}

int audio_input_open_by_name(AudioInputStreamHandle* handle,
                            int sample_rate, int channels,
                            int frames_per_buffer,
                            const char* device_name_hint) {
    if (!handle || !handle->stream) return 0;

    AudioInputConfig config;
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device_name_hint = device_name_hint;

    return handle->stream->open(config) ? 1 : 0;
}

void audio_input_close(AudioInputStreamHandle* handle) {
    if (handle && handle->stream) {
        handle->stream->close();
    }
}

int audio_input_start(AudioInputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->start() ? 1 : 0;
}

int audio_input_stop(AudioInputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->stop() ? 1 : 0;
}

int audio_input_is_running(AudioInputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->isRunning() ? 1 : 0;
}

int audio_input_get_sample_rate(AudioInputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->getSampleRate();
}

int audio_input_get_channels(AudioInputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->getChannels();
}

int audio_input_find_device(const char* name_hint) {
    return AudioInputStream::findDeviceByName(name_hint);
}

// ============== AudioOutputStream C API ==============

AudioOutputStreamHandle* audio_output_create(void) {
    AudioOutputStreamHandle* handle = new (std::nothrow) AudioOutputStreamHandle();
    if (!handle) return nullptr;

    handle->stream = new (std::nothrow) AudioOutputStream();
    if (!handle->stream) {
        delete handle;
        return nullptr;
    }

    handle->c_callback = nullptr;
    handle->c_user_data = nullptr;
    return handle;
}

void audio_output_destroy(AudioOutputStreamHandle* handle) {
    if (handle) {
        delete handle->stream;
        delete handle;
    }
}

void audio_output_set_callback(AudioOutputStreamHandle* handle,
                                AudioOutputCallbackC callback,
                                void* user_data) {
    if (!handle || !handle->stream) return;

    handle->c_callback = callback;
    handle->c_user_data = user_data;

    if (callback) {
        handle->stream->setCallback(outputCallbackWrapper, handle);
    } else {
        handle->stream->setCallback(nullptr, nullptr);
    }
}

int audio_output_open(AudioOutputStreamHandle* handle,
                        int sample_rate, int channels,
                        int frames_per_buffer, int device_index) {
    if (!handle || !handle->stream) return 0;

    AudioOutputConfig config;
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device_index = device_index;

    return handle->stream->open(config) ? 1 : 0;
}

int audio_output_open_by_name(AudioOutputStreamHandle* handle,
                                int sample_rate, int channels,
                                int frames_per_buffer,
                                const char* device_name_hint) {
    if (!handle || !handle->stream) return 0;

    AudioOutputConfig config;
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device_name_hint = device_name_hint;

    return handle->stream->open(config) ? 1 : 0;
}

void audio_output_close(AudioOutputStreamHandle* handle) {
    if (handle && handle->stream) {
        handle->stream->close();
    }
}

int audio_output_start(AudioOutputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->start() ? 1 : 0;
}

int audio_output_stop(AudioOutputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->stop() ? 1 : 0;
}

int audio_output_abort(AudioOutputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->abort() ? 1 : 0;
}

int audio_output_write(AudioOutputStreamHandle* handle,
                        const float* data, size_t frames) {
    if (!handle || !handle->stream) return -1;
    return handle->stream->write(data, frames);
}

int audio_output_write_int16(AudioOutputStreamHandle* handle,
                            const int16_t* data, size_t frames) {
    if (!handle || !handle->stream) return -1;
    return handle->stream->writeInt16(data, frames);
}

int audio_output_is_running(AudioOutputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->isRunning() ? 1 : 0;
}

int audio_output_get_sample_rate(AudioOutputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->getSampleRate();
}

int audio_output_get_channels(AudioOutputStreamHandle* handle) {
    if (!handle || !handle->stream) return 0;
    return handle->stream->getChannels();
}

int audio_output_find_device(const char* name_hint) {
    return AudioOutputStream::findDeviceByName(name_hint);
}

}  // extern "C"
