/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef AUDIO_STREAM_HPP
#define AUDIO_STREAM_HPP

// C-compatible headers (for C API)
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
// C++ specific headers
#include <cstdint>
#include <cstddef>
#include <functional>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

// Forward declarations for PortAudio types (to avoid including portaudio.h in header)
struct PaStreamCallbackTimeInfo;
typedef unsigned long PaStreamCallbackFlags;  // NOLINT(runtime/int)

// ============================================================================
// Audio Stream Components - PortAudio-based Input/Output Streams
// ============================================================================
// Provides generic, callback-based audio streaming functionality.
// - AudioInputStream: Captures audio from microphone or loopback devices
// - AudioOutputStream: Plays audio to speakers
// Both support continuous streaming with user-defined callbacks.
// ============================================================================

/**
 * Audio sample format enumeration
 */
enum class AudioSampleFormat {
    FLOAT32,    // 32-bit floating point [-1.0, 1.0]
    INT16,      // 16-bit signed integer
    INT32       // 32-bit signed integer
};

// ============================================================================
// AudioInputStream - Generic Audio Input Stream Component
// ============================================================================

/**
 * Callback type for receiving audio data
 * @param data Pointer to audio samples (format depends on configuration)
 * @param frames Number of frames (samples per channel)
 * @param channels Number of channels
 * @param user_data User-provided context pointer
 */
using AudioInputCallback = std::function<void(const float* data, size_t frames,
                                                int channels, void* user_data)>;

/**
 * AudioInputStream configuration
 */
struct AudioInputConfig {
    int sample_rate = 48000;         // Sample rate in Hz
    int channels = 2;                // Number of channels (1=mono, 2=stereo)
    int frames_per_buffer = 512;     // Frames per callback (0 for automatic)
    int device_index = -1;           // Device index (-1 for default)
    AudioSampleFormat format = AudioSampleFormat::FLOAT32;

    // Optional: device name substring to search for (e.g., "hw:1,0")
    const char* device_name_hint = nullptr;
};

/**
 * AudioInputStream - Captures audio from input devices
 *
 * Usage:
 *   AudioInputStream stream;
 *   AudioInputConfig config;
 *   config.sample_rate = 48000;
 *   config.channels = 2;
 *
 *   stream.setCallback([](const float* data, size_t frames, int ch, void* ud) {
 *       // Process audio data here
 *   });
 *
 *   stream.open(config);
 *   stream.start();
 *   // ... streaming ...
 *   stream.stop();
 *   stream.close();
 */
class AudioInputStream {
public:
    AudioInputStream();
    ~AudioInputStream();

    // Disable copy
    AudioInputStream(const AudioInputStream&) = delete;
    AudioInputStream& operator=(const AudioInputStream&) = delete;

    // Enable move
    AudioInputStream(AudioInputStream&& other) noexcept;
    AudioInputStream& operator=(AudioInputStream&& other) noexcept;

    /**
     * Set the audio callback function
     * Called from the audio thread - should be non-blocking!
     * @param callback Function to receive audio data
     * @param user_data Optional user data pointer passed to callback
     */
    void setCallback(AudioInputCallback callback, void* user_data = nullptr);

    /**
     * Open the audio stream with specified configuration
     * @param config Stream configuration
     * @return true on success, false on failure
     */
    bool open(const AudioInputConfig& config);

    /**
     * Close the audio stream
     */
    void close();

    /**
     * Start streaming (begins calling the callback)
     * @return true on success, false on failure
     */
    bool start();

    /**
     * Stop streaming (stops calling the callback)
     * @return true on success, false on failure
     */
    bool stop();

    /**
     * Check if stream is currently running
     */
    bool isRunning() const;

    /**
     * Check if stream is open
     */
    bool isOpen() const;

    /**
     * Get the actual sample rate (may differ from requested)
     */
    int getSampleRate() const { return actual_sample_rate_; }

    /**
     * Get the actual number of channels
     */
    int getChannels() const { return actual_channels_; }

    /**
     * Get the device index being used
     */
    int getDeviceIndex() const { return device_index_; }

    /**
     * List available input devices
     * @param names Output vector of device names
     * @param indices Output vector of device indices
     */
    static void listDevices(std::vector<std::string>& names, std::vector<int>& indices);

    /**
     * Find device index by name substring
     * @param name_hint Substring to search for in device names
     * @return Device index, or -1 if not found
     */
    static int findDeviceByName(const char* name_hint);

private:
    // PortAudio callback (static because PortAudio uses C callback)
    static int paCallback(const void* input_buffer, void* output_buffer,
                        unsigned long frames_per_buffer,  // NOLINT(runtime/int)
                        const PaStreamCallbackTimeInfo* time_info,
                        PaStreamCallbackFlags status_flags,
                        void* user_data);

    void* stream_ = nullptr;  // PaStream*
    AudioInputCallback callback_;
    void* user_data_ = nullptr;

    int actual_sample_rate_ = 0;
    int actual_channels_ = 0;
    int device_index_ = -1;
    AudioSampleFormat actual_format_ = AudioSampleFormat::FLOAT32;
    std::vector<float> float_buffer_;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_open_{false};
};


// ============================================================================
// AudioOutputStream - Generic Audio Output Stream Component
// ============================================================================

/**
 * Callback type for providing audio data for playback
 * @param data Pointer to buffer to fill with audio samples
 * @param frames Number of frames to provide
 * @param channels Number of channels
 * @param user_data User-provided context pointer
 * @return Number of frames actually written (0 to end playback)
 */
using AudioOutputCallback = std::function<size_t(float* data, size_t frames,
                                                int channels, void* user_data)>;

/**
 * AudioOutputStream configuration
 */
struct AudioOutputConfig {
    int sample_rate = 48000;         // Sample rate in Hz
    int channels = 2;                // Number of channels (1=mono, 2=stereo)
    int frames_per_buffer = 1024;    // Frames per callback (0 for automatic)
    int device_index = -1;           // Device index (-1 for default)
    AudioSampleFormat format = AudioSampleFormat::FLOAT32;

    // Optional: device name substring to search for (e.g., "hw:0,0")
    const char* device_name_hint = nullptr;
};

/**
 * AudioOutputStream - Plays audio to output devices
 *
 * Supports two modes:
 * 1. Callback mode: Set a callback to continuously provide audio data
 * 2. Write mode: Call write() to queue audio data for playback
 *
 * Usage (Callback mode):
 *   AudioOutputStream stream;
 *   AudioOutputConfig config;
 *
 *   stream.setCallback([](float* data, size_t frames, int ch, void* ud) -> size_t {
 *       // Fill buffer with audio data
 *       return frames; // Return 0 to stop
 *   });
 *
 *   stream.open(config);
 *   stream.start();
 *   // ... playback continues until callback returns 0 ...
 *   stream.stop();
 *   stream.close();
 *
 * Usage (Write mode):
 *   AudioOutputStream stream;
 *   stream.open(config);
 *   stream.start();
 *
 *   float buffer[1024];
 *   // Fill buffer...
 *   stream.write(buffer, 512); // Write 512 frames
 *
 *   stream.stop();
 *   stream.close();
 */
class AudioOutputStream {
public:
    AudioOutputStream();
    ~AudioOutputStream();

    // Disable copy
    AudioOutputStream(const AudioOutputStream&) = delete;
    AudioOutputStream& operator=(const AudioOutputStream&) = delete;

    // Enable move
    AudioOutputStream(AudioOutputStream&& other) noexcept;
    AudioOutputStream& operator=(AudioOutputStream&& other) noexcept;

    /**
     * Set the audio callback function (for callback mode)
     * Called from the audio thread - should be non-blocking!
     * @param callback Function to provide audio data
     * @param user_data Optional user data pointer passed to callback
     */
    void setCallback(AudioOutputCallback callback, void* user_data = nullptr);

    /**
     * Open the audio stream with specified configuration
     * @param config Stream configuration
     * @return true on success, false on failure
     */
    bool open(const AudioOutputConfig& config);

    /**
     * Close the audio stream
     */
    void close();

    /**
     * Start streaming
     * @return true on success, false on failure
     */
    bool start();

    /**
     * Stop streaming
     * @return true on success, false on failure
     */
    bool stop();

    /**
     * Abort streaming immediately (don't wait for buffer to drain)
     * @return true on success, false on failure
     */
    bool abort();

    /**
     * Write audio data to the stream (blocking)
     * Used for write mode - alternative to callback mode
     * @param data Audio samples (interleaved for multi-channel)
     * @param frames Number of frames to write
     * @return Number of frames actually written, or -1 on error
     */
    int write(const float* data, size_t frames);

    /**
     * Write int16 audio data to the stream (blocking)
     * Converts to float internally
     * @param data Audio samples (interleaved for multi-channel)
     * @param frames Number of frames to write
     * @return Number of frames actually written, or -1 on error
     */
    int writeInt16(const int16_t* data, size_t frames);

    /**
     * Check if stream is currently running
     */
    bool isRunning() const;

    /**
     * Check if stream is open
     */
    bool isOpen() const;

    /**
     * Get the actual sample rate
     */
    int getSampleRate() const { return actual_sample_rate_; }

    /**
     * Get the actual number of channels
     */
    int getChannels() const { return actual_channels_; }

    /**
     * Get the device index being used
     */
    int getDeviceIndex() const { return device_index_; }

    /**
     * List available output devices
     * @param names Output vector of device names
     * @param indices Output vector of device indices
     */
    static void listDevices(std::vector<std::string>& names, std::vector<int>& indices);

    /**
     * Find device index by name substring
     * @param name_hint Substring to search for in device names
     * @return Device index, or -1 if not found
     */
    static int findDeviceByName(const char* name_hint);

private:
    // PortAudio callback (static)
    static int paCallback(const void* input_buffer, void* output_buffer,
                        unsigned long frames_per_buffer,  // NOLINT(runtime/int)
                        const PaStreamCallbackTimeInfo* time_info,
                        PaStreamCallbackFlags status_flags,
                        void* user_data);

    void* stream_ = nullptr;  // PaStream*
    AudioOutputCallback callback_;
    void* user_data_ = nullptr;

    int actual_sample_rate_ = 0;
    int actual_channels_ = 0;
    int device_index_ = -1;
    AudioSampleFormat actual_format_ = AudioSampleFormat::FLOAT32;

    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_open_{false};
    bool use_callback_mode_ = false;
    std::mutex write_mutex_;
    std::vector<float> float_buffer_;
    std::vector<int16_t> int16_buffer_;
    std::vector<int32_t> int32_buffer_;
};

#endif  // __cplusplus

// ============================================================================
// C API for Audio Streams
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct AudioInputStreamHandle AudioInputStreamHandle;
typedef struct AudioOutputStreamHandle AudioOutputStreamHandle;

// C callback types
typedef void (*AudioInputCallbackC)(const float* data, size_t frames,
                                    int channels, void* user_data);
typedef size_t (*AudioOutputCallbackC)(float* data, size_t frames,
                                        int channels, void* user_data);

// ============== AudioInputStream C API ==============

/**
 * Create an audio input stream
 * @return Handle to the stream, or NULL on failure
 */
AudioInputStreamHandle* audio_input_create(void);

/**
 * Destroy an audio input stream
 */
void audio_input_destroy(AudioInputStreamHandle* handle);

/**
 * Set the audio callback
 */
void audio_input_set_callback(AudioInputStreamHandle* handle,
                                AudioInputCallbackC callback,
                                void* user_data);

/**
 * Open the stream
 * @param sample_rate Sample rate in Hz
 * @param channels Number of channels
 * @param frames_per_buffer Frames per callback
 * @param device_index Device index (-1 for default)
 * @return 1 on success, 0 on failure
 */
int audio_input_open(AudioInputStreamHandle* handle,
                    int sample_rate, int channels,
                    int frames_per_buffer, int device_index);

/**
 * Open the stream with device name hint
 */
int audio_input_open_by_name(AudioInputStreamHandle* handle,
                            int sample_rate, int channels,
                            int frames_per_buffer,
                            const char* device_name_hint);

/**
 * Close the stream
 */
void audio_input_close(AudioInputStreamHandle* handle);

/**
 * Start streaming
 */
int audio_input_start(AudioInputStreamHandle* handle);

/**
 * Stop streaming
 */
int audio_input_stop(AudioInputStreamHandle* handle);

/**
 * Check if running
 */
int audio_input_is_running(AudioInputStreamHandle* handle);

/**
 * Get sample rate
 */
int audio_input_get_sample_rate(AudioInputStreamHandle* handle);

/**
 * Get channels
 */
int audio_input_get_channels(AudioInputStreamHandle* handle);

/**
 * Find device by name
 */
int audio_input_find_device(const char* name_hint);

// ============== AudioOutputStream C API ==============

/**
 * Create an audio output stream
 */
AudioOutputStreamHandle* audio_output_create(void);

/**
 * Destroy an audio output stream
 */
void audio_output_destroy(AudioOutputStreamHandle* handle);

/**
 * Set the audio callback (for callback mode)
 */
void audio_output_set_callback(AudioOutputStreamHandle* handle,
                                AudioOutputCallbackC callback,
                                void* user_data);

/**
 * Open the stream
 */
int audio_output_open(AudioOutputStreamHandle* handle,
                        int sample_rate, int channels,
                        int frames_per_buffer, int device_index);

/**
 * Open the stream with device name hint
 */
int audio_output_open_by_name(AudioOutputStreamHandle* handle,
                                int sample_rate, int channels,
                                int frames_per_buffer,
                                const char* device_name_hint);

/**
 * Close the stream
 */
void audio_output_close(AudioOutputStreamHandle* handle);

/**
 * Start streaming
 */
int audio_output_start(AudioOutputStreamHandle* handle);

/**
 * Stop streaming
 */
int audio_output_stop(AudioOutputStreamHandle* handle);

/**
 * Abort streaming immediately
 */
int audio_output_abort(AudioOutputStreamHandle* handle);

/**
 * Write audio data (blocking, for write mode)
 * @return Number of frames written, or -1 on error
 */
int audio_output_write(AudioOutputStreamHandle* handle,
                        const float* data, size_t frames);

/**
 * Write int16 audio data (blocking, for write mode)
 * @return Number of frames written, or -1 on error
 */
int audio_output_write_int16(AudioOutputStreamHandle* handle,
                            const int16_t* data, size_t frames);

/**
 * Check if running
 */
int audio_output_is_running(AudioOutputStreamHandle* handle);

/**
 * Get sample rate
 */
int audio_output_get_sample_rate(AudioOutputStreamHandle* handle);

/**
 * Get channels
 */
int audio_output_get_channels(AudioOutputStreamHandle* handle);

/**
 * Find device by name
 */
int audio_output_find_device(const char* name_hint);

#ifdef __cplusplus
}
#endif

#endif  // AUDIO_STREAM_HPP
