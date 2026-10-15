/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio_base.hpp"
#include "internal/audio_stream.hpp"

namespace SpacemitAudio {

// ============================================================================
// Global Configuration
// ============================================================================

static AudioConfig g_config;
static std::mutex g_config_mutex;

void Init(const AudioConfig& config) {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    g_config = config;
}

void Init(int sample_rate, int channels, int chunk_size,
            int capture_device, int player_device) {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    if (sample_rate > 0) g_config.sample_rate = sample_rate;
    if (channels > 0) g_config.channels = channels;
    if (chunk_size > 0) g_config.chunk_size = chunk_size;
    if (capture_device >= -1) g_config.capture_device = capture_device;
    if (player_device >= -1) g_config.player_device = player_device;
}

AudioConfig GetConfig() {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    return g_config;
}

// ============================================================================
// AudioCapture Implementation
// ============================================================================

struct AudioCapture::Impl {
    AudioInputStream stream;
    Callback user_callback;
    int device_index = -1;
    int sample_rate = 16000;
    int channels = 1;
    int chunk_size = 3200;  // bytes
    std::vector<uint8_t> buffer;  // accumulation buffer

    void onAudioData(const float* data, size_t frames, int ch) {
        if (!user_callback) return;

        // Convert float -> int16 -> uint8_t bytes
        size_t samples = frames * ch;
        size_t bytes_needed = samples * sizeof(int16_t);

        // Append to buffer
        size_t old_size = buffer.size();
        buffer.resize(old_size + bytes_needed);

        int16_t* out = reinterpret_cast<int16_t*>(buffer.data() + old_size);
        for (size_t i = 0; i < samples; ++i) {
            float sample = data[i];
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
            out[i] = static_cast<int16_t>(sample * 32767.0f);
        }

        // Deliver chunks of chunk_size bytes
        while (buffer.size() >= static_cast<size_t>(chunk_size)) {
            user_callback(buffer.data(), chunk_size);
            buffer.erase(buffer.begin(), buffer.begin() + chunk_size);
        }
    }
};

AudioCapture::AudioCapture(int device_index)
    : impl_(std::make_unique<Impl>()) {
    // Use global config if device_index is -1
    if (device_index == -1) {
        impl_->device_index = GetConfig().capture_device;
    } else {
        impl_->device_index = device_index;
    }
}

AudioCapture::~AudioCapture() {
    Close();
}

void AudioCapture::SetCallback(Callback cb) {
    impl_->user_callback = std::move(cb);
}

bool AudioCapture::Start(int sample_rate, int channels, int chunk_size) {
    // Use global config for unspecified parameters
    AudioConfig cfg = GetConfig();
    if (sample_rate <= 0) sample_rate = cfg.sample_rate;
    if (channels <= 0) channels = cfg.channels;
    if (chunk_size <= 0) chunk_size = cfg.chunk_size;

    impl_->sample_rate = sample_rate;
    impl_->channels = channels;
    impl_->chunk_size = chunk_size;
    impl_->buffer.clear();

    // Calculate frames per buffer based on chunk_size
    // chunk_size bytes = (frames * channels * 2) bytes (PCM16)
    int bytes_per_frame = channels * sizeof(int16_t);
    int frames_per_buffer = chunk_size / bytes_per_frame;
    if (frames_per_buffer < 64) frames_per_buffer = 64;

    // Set callback wrapper
    impl_->stream.setCallback(
        [this](const float* data, size_t frames, int ch, void*) {
            impl_->onAudioData(data, frames, ch);
        },
        nullptr);

    // Configure and open stream
    AudioInputConfig config;
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device_index = impl_->device_index;
    config.format = AudioSampleFormat::INT16;

    if (!impl_->stream.open(config)) {
        return false;
    }

    return impl_->stream.start();
}

void AudioCapture::Stop() {
    impl_->stream.stop();
}

void AudioCapture::Close() {
    impl_->stream.close();
    impl_->buffer.clear();
}

bool AudioCapture::IsRunning() const {
    return impl_->stream.isRunning();
}

std::vector<std::pair<int, std::string>> AudioCapture::ListDevices() {
    std::vector<std::string> names;
    std::vector<int> indices;
    AudioInputStream::listDevices(names, indices);

    std::vector<std::pair<int, std::string>> result;
    for (size_t i = 0; i < names.size(); ++i) {
        result.emplace_back(indices[i], names[i]);
    }
    return result;
}

// ============================================================================
// AudioPlayer Implementation
// ============================================================================

struct AudioPlayer::Impl {
    AudioOutputStream stream;
    int device_index = -1;
    int sample_rate = 16000;
    int channels = 1;
};

AudioPlayer::AudioPlayer(int device_index)
    : impl_(std::make_unique<Impl>()) {
    // Use global config if device_index is -1
    if (device_index == -1) {
        impl_->device_index = GetConfig().player_device;
    } else {
        impl_->device_index = device_index;
    }
}

AudioPlayer::~AudioPlayer() {
    Close();
}

bool AudioPlayer::Start(int sample_rate, int channels) {
    return Start(sample_rate, channels, 256);
}

bool AudioPlayer::Start(int sample_rate, int channels, int frames_per_buffer) {
    // Use global config for unspecified parameters
    AudioConfig cfg = GetConfig();
    if (sample_rate <= 0) sample_rate = cfg.sample_rate;
    if (channels <= 0) channels = cfg.channels;
    if (frames_per_buffer <= 0) frames_per_buffer = 256;

    impl_->sample_rate = sample_rate;
    impl_->channels = channels;

    AudioOutputConfig config;
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device_index = impl_->device_index;
    config.format = AudioSampleFormat::INT16;

    if (!impl_->stream.open(config)) {
        return false;
    }

    return impl_->stream.start();
}

bool AudioPlayer::Write(const std::vector<uint8_t>& data) {
    return Write(data.data(), data.size());
}

bool AudioPlayer::Write(const uint8_t* data, size_t size) {
    if (!impl_->stream.isOpen() || size == 0) {
        return false;
    }

    // Convert uint8_t (PCM16 bytes) -> int16_t -> frames
    size_t samples = size / sizeof(int16_t);
    size_t frames = samples / impl_->channels;

    if (frames == 0) return true;

    const int16_t* pcm = reinterpret_cast<const int16_t*>(data);
    return impl_->stream.writeInt16(pcm, frames) > 0;
}

bool AudioPlayer::PlayFile(const std::string& file_path) {
    // Simple WAV file reader
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "[AudioPlayer] Cannot open file: " << file_path << std::endl;
        return false;
    }

    // Read WAV header
    char riff[4];
    file.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) {
        std::cerr << "[AudioPlayer] Not a valid WAV file" << std::endl;
        return false;
    }

    file.seekg(4, std::ios::cur);  // Skip file size

    char wave[4];
    file.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) {
        std::cerr << "[AudioPlayer] Not a valid WAV file" << std::endl;
        return false;
    }

    // Find fmt chunk
    int16_t audio_format = 0;
    int16_t num_channels = 0;
    int32_t wav_sample_rate = 0;
    int16_t bits_per_sample = 0;

    while (file) {
        char chunk_id[4];
        int32_t chunk_size;
        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&wav_sample_rate), 4);
            file.seekg(6, std::ios::cur);  // Skip byte rate and block align
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

            // Skip rest of fmt chunk if any
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            // Found data chunk
            break;
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (audio_format != 1) {  // PCM
        std::cerr << "[AudioPlayer] Only PCM WAV is supported" << std::endl;
        return false;
    }

    if (bits_per_sample != 16) {
        std::cerr << "[AudioPlayer] Only 16-bit WAV is supported" << std::endl;
        return false;
    }

    // Start playback stream with WAV parameters
    bool was_open = impl_->stream.isOpen();
    if (!was_open) {
        if (!Start(wav_sample_rate, num_channels)) {
            return false;
        }
    }

    // Read and play audio data
    const size_t buffer_size = 4096;
    std::vector<uint8_t> buffer(buffer_size);

    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), buffer_size);
        size_t bytes_read = file.gcount();
        if (bytes_read > 0) {
            Write(buffer.data(), bytes_read);
        }
    }

    // Wait a bit for playback to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!was_open) {
        Stop();
        Close();
    }

    return true;
}

void AudioPlayer::Stop() {
    impl_->stream.stop();
}

void AudioPlayer::Close() {
    impl_->stream.close();
}

bool AudioPlayer::IsRunning() const {
    return impl_->stream.isRunning();
}

std::vector<std::pair<int, std::string>> AudioPlayer::ListDevices() {
    std::vector<std::string> names;
    std::vector<int> indices;
    AudioOutputStream::listDevices(names, indices);

    std::vector<std::pair<int, std::string>> result;
    for (size_t i = 0; i < names.size(); ++i) {
        result.emplace_back(indices[i], names[i]);
    }
    return result;
}

}  // namespace SpacemitAudio
