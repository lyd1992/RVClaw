/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUDIO_DUPLEX_STREAM_HPP
#define AUDIO_DUPLEX_STREAM_HPP

/**
 * AudioDuplexStream - Full-duplex audio stream
 *
 * Provides synchronized input/output audio processing in a single callback,
 * which is essential for acoustic echo cancellation (AEC).
 *
 * Usage:
 *   AudioDuplexStream stream;
 *   stream.setCallback([](const float* input, float* output, size_t frames,
 *                         int channels, void* user_data) {
 *       // Process input (microphone)
 *       // Fill output (speaker)
 *   }, nullptr);
 *
 *   AudioDuplexConfig config;
 *   config.sample_rate = 48000;
 *   stream.open(config);
 *   stream.start();
 */

#include <functional>
#include <atomic>
#include <vector>
#include <string>

// ============================================================================
// Configuration
// ============================================================================

struct AudioDuplexConfig {
    int sample_rate = 48000;           // Sample rate (Hz), 48000 recommended for AEC
    int channels = 1;                  // Number of channels (1=mono, 2=stereo)
    int input_channels = 0;            // 0 = use channels
    int output_channels = 0;           // 0 = use channels
#ifdef __linux__
    int frames_per_buffer = 960;       // Linux: 20ms @ 48kHz (避免 ALSA underrun)
#else
    int frames_per_buffer = 480;       // macOS: 10ms @ 48kHz
#endif
    int input_device_index = -1;       // Input device (-1 = default)
    int output_device_index = -1;      // Output device (-1 = default)
    const char* input_device_name = nullptr;   // Alternative: find by name
    const char* output_device_name = nullptr;  // Alternative: find by name
};

// ============================================================================
// Callback Type
// ============================================================================

/**
 * Full-duplex audio callback
 *
 * @param input   Microphone input samples (read-only)
 * @param output  Speaker output buffer (must be filled by callback)
 * @param frames  Number of frames in this callback
 * @param channels Number of channels
 * @param user_data User-provided context pointer
 *
 * Note: Both input and output have (frames * channels) samples.
 *       Input and output are synchronized in time, making this ideal for AEC.
 */
using AudioDuplexCallback = std::function<void(
    const float* input,
    float* output,
    size_t frames,
    int channels,
    void* user_data
)>;

using AudioDuplexCallbackEx = std::function<void(
    const float* input,
    float* output,
    size_t frames,
    int input_channels,
    int output_channels,
    void* user_data
)>;

// ============================================================================
// AudioDuplexStream Class
// ============================================================================

class AudioDuplexStream {
public:
    AudioDuplexStream();
    ~AudioDuplexStream();

    // Non-copyable, movable
    AudioDuplexStream(const AudioDuplexStream&) = delete;
    AudioDuplexStream& operator=(const AudioDuplexStream&) = delete;
    AudioDuplexStream(AudioDuplexStream&& other) noexcept;
    AudioDuplexStream& operator=(AudioDuplexStream&& other) noexcept;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /**
     * Set the audio callback
     * Must be called before open()
     */
    void setCallback(AudioDuplexCallback callback, void* user_data = nullptr);
    void setCallbackEx(AudioDuplexCallbackEx callback, void* user_data = nullptr);

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    /**
     * Open the full-duplex stream
     * @param config Stream configuration
     * @return true on success
     */
    bool open(const AudioDuplexConfig& config);

    /**
     * Close the stream and release resources
     */
    void close();

    /**
     * Start audio processing
     * @return true on success
     */
    bool start();

    /**
     * Stop audio processing
     * @return true on success
     */
    bool stop();

    // -------------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------------

    bool isOpen() const { return is_open_.load(); }
    bool isRunning() const { return is_running_.load(); }

    int getSampleRate() const { return actual_sample_rate_; }
    int getChannels() const { return actual_channels_; }
    int getInputChannels() const { return actual_input_channels_; }
    int getOutputChannels() const { return actual_output_channels_; }
    int getInputDeviceIndex() const { return input_device_index_; }
    int getOutputDeviceIndex() const { return output_device_index_; }

    // -------------------------------------------------------------------------
    // Static Utilities
    // -------------------------------------------------------------------------

    /**
     * List available input devices
     */
    static void listInputDevices(std::vector<std::string>& names,
                                    std::vector<int>& indices);

    /**
     * List available output devices
     */
    static void listOutputDevices(std::vector<std::string>& names,
                                    std::vector<int>& indices);

    /**
     * Find input device by name hint
     * @return Device index, or -1 if not found
     */
    static int findInputDeviceByName(const char* name_hint);

    /**
     * Find output device by name hint
     * @return Device index, or -1 if not found
     */
    static int findOutputDeviceByName(const char* name_hint);

private:
    // PortAudio callback (static)
    static int paCallback(const void* input_buffer,
                            void* output_buffer,
                            unsigned long frames_per_buffer,  // NOLINT(runtime/int)
                            const struct PaStreamCallbackTimeInfo* time_info,
                            unsigned long status_flags,  // NOLINT(runtime/int)
                            void* user_data);

    void* stream_;                     // PaStream*
    AudioDuplexCallback callback_;
    AudioDuplexCallbackEx callback_ex_;
    void* user_data_;

    int actual_sample_rate_;
    int actual_channels_;
    int actual_input_channels_;
    int actual_output_channels_;
    int input_device_index_;
    int output_device_index_;

    std::atomic<bool> is_running_;
    std::atomic<bool> is_open_;

    // RISC-V alignment buffers for unaligned PortAudio data
    std::vector<float> aligned_input_buffer_;
    std::vector<float> aligned_output_buffer_;
    int frames_per_buffer_;
};

// ============================================================================
// C API (for cross-language bindings)
// ============================================================================

extern "C" {

struct AudioDuplexStreamHandle;

typedef void (*AudioDuplexCallbackC)(
    const float* input,
    float* output,
    size_t frames,
    int channels,
    void* user_data
);

AudioDuplexStreamHandle* audio_duplex_create(void);
void audio_duplex_destroy(AudioDuplexStreamHandle* handle);

void audio_duplex_set_callback(AudioDuplexStreamHandle* handle,
                                AudioDuplexCallbackC callback,
                                void* user_data);

int audio_duplex_open(AudioDuplexStreamHandle* handle,
        int sample_rate,
        int channels,
        int frames_per_buffer,
        int input_device_index,
        int output_device_index);

void audio_duplex_close(AudioDuplexStreamHandle* handle);

int audio_duplex_start(AudioDuplexStreamHandle* handle);
int audio_duplex_stop(AudioDuplexStreamHandle* handle);

int audio_duplex_is_running(AudioDuplexStreamHandle* handle);
int audio_duplex_get_sample_rate(AudioDuplexStreamHandle* handle);
int audio_duplex_get_channels(AudioDuplexStreamHandle* handle);

}  // extern "C"

#endif  // AUDIO_DUPLEX_STREAM_HPP
