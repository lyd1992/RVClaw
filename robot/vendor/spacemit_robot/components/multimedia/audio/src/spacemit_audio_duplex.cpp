/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpacemitAudio Full-Duplex Implementation
 */

#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "audio_duplex.hpp"
#include "internal/audio_duplex_stream.hpp"

namespace SpacemitAudio {

// ============================================================================
// AudioDuplex Implementation
// ============================================================================

struct AudioDuplex::Impl {
    AudioDuplexStream stream;
    Callback user_callback;
    CallbackEx user_callback_ex;
    int input_device_index = -1;
    int output_device_index = -1;
    int sample_rate = 48000;
    int channels = 1;
    int input_channels = 1;
    int output_channels = 1;

    void onAudioData(const float* input, float* output, size_t frames, int ch) {
        if (user_callback) {
            user_callback(input, output, frames, ch);
        } else if (output) {
            // Fill with silence if no callback
            std::memset(output, 0, frames * ch * sizeof(float));
        }
    }

    void onAudioDataEx(const float* input, float* output, size_t frames,
        int in_ch, int out_ch) {
        if (user_callback_ex) {
            user_callback_ex(input, output, frames, in_ch, out_ch);
        } else if (output) {
            std::memset(output, 0, frames * out_ch * sizeof(float));
        }
    }
};

AudioDuplex::AudioDuplex(int input_device, int output_device)
    : impl_(std::make_unique<Impl>()) {
    impl_->input_device_index = input_device;
    impl_->output_device_index = output_device;
}

AudioDuplex::~AudioDuplex() {
    Close();
}

void AudioDuplex::SetCallback(Callback cb) {
    impl_->user_callback = std::move(cb);
    impl_->user_callback_ex = nullptr;
}

void AudioDuplex::SetCallbackEx(CallbackEx cb) {
    impl_->user_callback = nullptr;
    impl_->user_callback_ex = std::move(cb);
}

bool AudioDuplex::Start(int sample_rate, int channels, int frames_per_buffer) {
    return Start(sample_rate, channels, channels, frames_per_buffer);
}

bool AudioDuplex::Start(int sample_rate, int input_channels, int output_channels,
                        int frames_per_buffer) {
    impl_->sample_rate = sample_rate;
    impl_->channels = input_channels;
    impl_->input_channels = input_channels;
    impl_->output_channels = output_channels;

    if (input_channels != output_channels && !impl_->user_callback_ex) {
        std::cerr << "[AudioDuplex] SetCallbackEx is required when input/output "
            << "channel counts differ (input=" << input_channels
            << ", output=" << output_channels << ")" << std::endl;
        return false;
    }

    // Set stream callback
    if (impl_->user_callback_ex) {
        impl_->stream.setCallbackEx(
            [this](const float* input, float* output, size_t frames,
                int in_ch, int out_ch, void*) {
                impl_->onAudioDataEx(input, output, frames, in_ch, out_ch);
            },
            nullptr);
    } else {
        impl_->stream.setCallback(
            [this](const float* input, float* output, size_t frames, int ch, void*) {
                impl_->onAudioData(input, output, frames, ch);
            },
            nullptr);
    }

    // Configure and open stream
    AudioDuplexConfig config;
    config.sample_rate = sample_rate;
    config.channels = input_channels;
    config.input_channels = input_channels;
    config.output_channels = output_channels;
    config.frames_per_buffer = frames_per_buffer;
    config.input_device_index = impl_->input_device_index;
    config.output_device_index = impl_->output_device_index;

    if (!impl_->stream.open(config)) {
        return false;
    }

    return impl_->stream.start();
}

void AudioDuplex::Stop() {
    impl_->stream.stop();
}

void AudioDuplex::Close() {
    impl_->stream.close();
}

bool AudioDuplex::IsRunning() const {
    return impl_->stream.isRunning();
}

int AudioDuplex::GetSampleRate() const {
    return impl_->stream.getSampleRate();
}

int AudioDuplex::GetChannels() const {
    return impl_->stream.getChannels();
}

int AudioDuplex::GetInputChannels() const {
    return impl_->stream.getInputChannels();
}

int AudioDuplex::GetOutputChannels() const {
    return impl_->stream.getOutputChannels();
}

int AudioDuplex::GetInputDevice() const {
    return impl_->stream.getInputDeviceIndex();
}

int AudioDuplex::GetOutputDevice() const {
    return impl_->stream.getOutputDeviceIndex();
}

std::vector<std::pair<int, std::string>> AudioDuplex::ListInputDevices() {
    std::vector<std::string> names;
    std::vector<int> indices;
    AudioDuplexStream::listInputDevices(names, indices);

    std::vector<std::pair<int, std::string>> result;
    for (size_t i = 0; i < names.size(); ++i) {
        result.emplace_back(indices[i], names[i]);
    }
    return result;
}

std::vector<std::pair<int, std::string>> AudioDuplex::ListOutputDevices() {
    std::vector<std::string> names;
    std::vector<int> indices;
    AudioDuplexStream::listOutputDevices(names, indices);

    std::vector<std::pair<int, std::string>> result;
    for (size_t i = 0; i < names.size(); ++i) {
        result.emplace_back(indices[i], names[i]);
    }
    return result;
}

}  // namespace SpacemitAudio
