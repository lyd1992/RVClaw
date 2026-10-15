/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * SpacemitAudio Demo - 录音、播放示例
 *
 * Usage:
 *   audio_demo list
 *   audio_demo record OUT.wav --duration 5 --rate 16000 --channels 2 --device -1
 *   audio_demo play IN.wav --rate 48000 --channels 1 --device 1
 *   audio_demo roundtrip OUT.wav --duration 2 --rate 16000 --channels 1 --play-channels 1
 */

#include "audio_base.hpp"
#include "audio_resampler.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using SpacemitAudio::AudioCapture;
using SpacemitAudio::AudioPlayer;

static std::atomic<bool> g_running{true};

struct RecordOptions {
    std::string output_file;
    int duration = 3;
    int sample_rate = 16000;
    int channels = 1;
    int device = -1;
};

struct PlayOptions {
    std::string input_file;
    int sample_rate = 48000;
    int channels = -1;
    int device = -1;
};

struct RoundtripOptions {
    std::string output_file;
    int duration = 2;
    int sample_rate = 16000;
    int channels = 1;
    int input_device = -1;
    int output_device = -1;
    int play_rate = 48000;
    int play_channels = -1;
};

void signalHandler(int) {
    g_running = false;
}

bool isHelpOption(const std::string& arg) {
    return arg == "-h" || arg == "--help";
}

bool containsHelp(int argc, char* argv[], int start_index) {
    for (int i = start_index; i < argc; ++i) {
        if (isHelpOption(argv[i])) {
            return true;
        }
    }
    return false;
}

bool parseInt(const std::string& value, const std::string& name, int* output) {
    if (value.empty()) {
        std::cerr << "Missing value for " << name << std::endl;
        return false;
    }

    errno = 0;
    char* end = nullptr;
    int64_t parsed = std::strtoll(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' ||
            parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid integer for " << name << ": " << value << std::endl;
        return false;
    }

    *output = static_cast<int>(parsed);
    return true;
}

bool parsePositiveInt(const std::string& value, const std::string& name, int* output) {
    if (!parseInt(value, name, output)) {
        return false;
    }
    if (*output <= 0) {
        std::cerr << name << " must be greater than 0" << std::endl;
        return false;
    }
    return true;
}

bool parseIntOption(int argc,
                    char* argv[],
                    int* index,
                    const std::string& option,
                    int* output,
                    bool positive = false) {
    if (*index + 1 >= argc) {
        std::cerr << "Missing value for " << option << std::endl;
        return false;
    }

    ++(*index);
    if (positive) {
        return parsePositiveInt(argv[*index], option, output);
    }
    return parseInt(argv[*index], option, output);
}

void listDevices() {
    std::cout << "=== Input Devices ===" << std::endl;
    for (auto& [idx, name] : AudioCapture::ListDevices()) {
        std::cout << "  [" << idx << "] " << name << std::endl;
    }

    std::cout << "\n=== Output Devices ===" << std::endl;
    for (auto& [idx, name] : AudioPlayer::ListDevices()) {
        std::cout << "  [" << idx << "] " << name << std::endl;
    }
}

void writeWavHeader(std::ofstream& file,
                    int sample_rate,
                    int channels,
                    uint32_t data_size) {
    uint32_t byte_rate = static_cast<uint32_t>(sample_rate * channels * 2);
    uint16_t block_align = static_cast<uint16_t>(channels * 2);
    uint32_t file_size = 36 + data_size;
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t ch = static_cast<uint16_t>(channels);
    uint16_t bits_per_sample = 16;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&ch), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
}

struct WavData {
    int sample_rate = 0;
    int channels = 0;
    std::vector<int16_t> samples;
};

bool readWavFile(const std::string& filename, WavData* wav) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    char riff[4];
    file.read(riff, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0) {
        std::cerr << "Not a valid WAV file" << std::endl;
        return false;
    }

    file.seekg(4, std::ios::cur);

    char wave[4];
    file.read(wave, 4);
    if (std::memcmp(wave, "WAVE", 4) != 0) {
        std::cerr << "Not a valid WAV file" << std::endl;
        return false;
    }

    bool have_fmt = false;
    bool have_data = false;
    uint16_t audio_format = 0;
    uint16_t bits_per_sample = 0;

    while (file && (!have_fmt || !have_data)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (!file.read(chunk_id, 4)) break;
        if (!file.read(reinterpret_cast<char*>(&chunk_size), 4)) break;

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t channels = 0;
            uint32_t sample_rate = 0;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.seekg(6, std::ios::cur);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
            wav->channels = channels;
            wav->sample_rate = static_cast<int>(sample_rate);
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!have_fmt) {
                std::cerr << "Invalid WAV file: data chunk before fmt chunk" << std::endl;
                return false;
            }
            wav->samples.resize(chunk_size / sizeof(int16_t));
            file.read(reinterpret_cast<char*>(wav->samples.data()), chunk_size);
            wav->samples.resize(static_cast<size_t>(file.gcount()) / sizeof(int16_t));
            have_data = true;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }

        if (chunk_size % 2 != 0) {
            file.seekg(1, std::ios::cur);
        }
    }

    if (!have_fmt || !have_data) {
        std::cerr << "Invalid WAV file: missing fmt or data chunk" << std::endl;
        return false;
    }
    if (audio_format != 1 || bits_per_sample != 16) {
        std::cerr << "Only 16-bit PCM WAV is supported" << std::endl;
        return false;
    }
    if (wav->sample_rate <= 0 || wav->channels <= 0) {
        std::cerr << "Invalid WAV parameters" << std::endl;
        return false;
    }

    return true;
}

std::vector<int16_t> resamplePcm16(
    const std::vector<int16_t>& input,
    int input_rate,
    int output_rate,
    int channels) {
    if (input_rate == output_rate || input.empty()) {
        return input;
    }

    size_t frames = input.size() / static_cast<size_t>(channels);
    if (frames < 2) {
        return input;
    }

    std::vector<float> input_float(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input_float[i] = static_cast<float>(input[i]) / 32768.0f;
    }

    Resampler::Config config;
    config.input_sample_rate = input_rate;
    config.output_sample_rate = output_rate;
    config.channels = channels;
    config.method = (output_rate > input_rate) ?
        ResampleMethod::LINEAR_UPSAMPLE : ResampleMethod::LINEAR_DOWNSAMPLE;

    Resampler resampler(config);
    if (!resampler.initialize()) {
        return {};
    }

    std::vector<float> output_float = resampler.process(input_float);
    std::vector<int16_t> output(output_float.size());
    for (size_t i = 0; i < output_float.size(); ++i) {
        float sample = std::clamp(output_float[i], -1.0f, 1.0f);
        output[i] = static_cast<int16_t>(sample * 32767.0f);
    }

    return output;
}

std::vector<int16_t> convertChannelsPcm16(
    const std::vector<int16_t>& input,
    int input_channels,
    int output_channels) {
    if (input_channels == output_channels || input.empty()) {
        return input;
    }

    size_t frames = input.size() / static_cast<size_t>(input_channels);
    std::vector<int16_t> output(frames * static_cast<size_t>(output_channels));

    for (size_t frame = 0; frame < frames; ++frame) {
        size_t input_base = frame * static_cast<size_t>(input_channels);
        size_t output_base = frame * static_cast<size_t>(output_channels);

        if (output_channels == 1) {
            int total = 0;
            for (int ch = 0; ch < input_channels; ++ch) {
                total += input[input_base + static_cast<size_t>(ch)];
            }
            output[output_base] = static_cast<int16_t>(total / input_channels);
            continue;
        }

        if (input_channels == 1) {
            int16_t value = input[input_base];
            for (int ch = 0; ch < output_channels; ++ch) {
                output[output_base + static_cast<size_t>(ch)] = value;
            }
            continue;
        }

        for (int ch = 0; ch < output_channels; ++ch) {
            int input_ch = ch < input_channels ? ch : input_channels - 1;
            output[output_base + static_cast<size_t>(ch)] =
                input[input_base + static_cast<size_t>(input_ch)];
        }
    }

    return output;
}

bool recordAudio(const RecordOptions& options) {
    std::vector<uint8_t> all_data;

    AudioCapture capture(options.device);
    capture.SetCallback([&](const uint8_t* data, size_t size) {
        all_data.insert(all_data.end(), data, data + size);
        std::cout << "\rRecording... " << all_data.size() << " bytes" << std::flush;
    });

    std::cout << "Recording " << options.duration << "s to "
        << options.output_file << "..." << std::endl;
    std::cout << "Config: " << options.sample_rate << "Hz, "
        << options.channels << "ch, device=" << options.device << std::endl;

    g_running = true;
    if (!capture.Start(options.sample_rate, options.channels)) {
        std::cerr << "Failed to start capture" << std::endl;
        return false;
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(options.duration);
    while (g_running && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    capture.Stop();
    capture.Close();
    std::cout << std::endl;

    if (all_data.size() > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Recorded data is too large for WAV" << std::endl;
        return false;
    }

    std::ofstream file(options.output_file, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open output file: " << options.output_file << std::endl;
        return false;
    }

    writeWavHeader(
        file,
        options.sample_rate,
        options.channels,
        static_cast<uint32_t>(all_data.size()));
    file.write(
        reinterpret_cast<const char*>(all_data.data()),
        static_cast<std::streamsize>(all_data.size()));
    if (!file) {
        std::cerr << "Failed to write WAV file: " << options.output_file << std::endl;
        return false;
    }

    std::cout << "Saved " << all_data.size() << " bytes to "
        << options.output_file << std::endl;
    return true;
}

bool playAudio(const PlayOptions& options) {
    std::cout << "Playing " << options.input_file << "..." << std::endl;

    WavData wav;
    if (!readWavFile(options.input_file, &wav)) {
        return false;
    }

    int playback_channels = options.channels > 0 ? options.channels : wav.channels;
    std::vector<int16_t> playback_samples = wav.samples;
    if (wav.channels != playback_channels) {
        std::cout << "Converting channels " << wav.channels << "ch -> "
            << playback_channels << "ch..." << std::endl;
        playback_samples = convertChannelsPcm16(playback_samples, wav.channels, playback_channels);
    }

    if (wav.sample_rate != options.sample_rate) {
        std::cout << "Resampling " << wav.sample_rate << "Hz -> "
            << options.sample_rate << "Hz..." << std::endl;
        playback_samples = resamplePcm16(
            playback_samples,
            wav.sample_rate,
            options.sample_rate,
            playback_channels);
        if (playback_samples.empty() && !wav.samples.empty()) {
            std::cerr << "Failed to resample audio" << std::endl;
            return false;
        }
    }

    AudioPlayer player(options.device);
    if (!player.Start(options.sample_rate, playback_channels)) {
        player.Close();
        if (options.channels <= 0 && playback_channels != 1) {
            std::cerr << "Failed to start player with WAV channels; "
                << "retrying with 1ch" << std::endl;
            playback_samples = convertChannelsPcm16(playback_samples, playback_channels, 1);
            playback_channels = 1;
        }
        if (!player.Start(options.sample_rate, playback_channels)) {
            std::cerr << "Failed to start player" << std::endl;
            return false;
        }
    }

    bool ok = true;
    const size_t chunk_samples = 4096 * static_cast<size_t>(playback_channels);
    for (size_t offset = 0; offset < playback_samples.size(); offset += chunk_samples) {
        size_t samples = std::min(chunk_samples, playback_samples.size() - offset);
        const uint8_t* data =
            reinterpret_cast<const uint8_t*>(playback_samples.data() + offset);
        if (!player.Write(data, samples * sizeof(int16_t))) {
            std::cerr << "Failed to play audio" << std::endl;
            ok = false;
            break;
        }
    }

    player.Stop();
    player.Close();

    if (!ok) {
        return false;
    }

    std::cout << "Done" << std::endl;
    return true;
}

bool roundtripAudio(const RoundtripOptions& options) {
    RecordOptions record_options;
    record_options.output_file = options.output_file;
    record_options.duration = options.duration;
    record_options.sample_rate = options.sample_rate;
    record_options.channels = options.channels;
    record_options.device = options.input_device;

    if (!recordAudio(record_options)) {
        return false;
    }

    PlayOptions play_options;
    play_options.input_file = options.output_file;
    play_options.sample_rate = options.play_rate;
    play_options.channels = options.play_channels;
    play_options.device = options.output_device;
    return playAudio(play_options);
}

void printUsage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " <command> [args]\n"
        << "\nExamples:\n"
        << "  " << prog << " list\n"
        << "  " << prog
        << " record OUT.wav --duration 5 --rate 16000 --channels 2 --device -1\n"
        << "  " << prog << " play IN.wav --rate 48000 --channels 1 --device 1\n"
        << "  " << prog
        << " roundtrip OUT.wav --duration 2 --rate 16000 --channels 1 "
        << "--play-channels 1 --input-device -1 --output-device -1\n"
        << "\nCommands:\n"
        << "  list       List input and output devices\n"
        << "  record     Record audio to a WAV file\n"
        << "  play       Play a WAV file\n"
        << "  roundtrip  Record audio, save it, then play it\n"
        << "\nRun `" << prog << " <command> --help` for command options.\n";
}

void printListHelp(const char* prog) {
    std::cout << "Usage:\n"
        << "  " << prog << " list\n"
        << "\nList input and output audio devices.\n";
}

void printRecordHelp(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " record OUT.wav [options]\n"
        << "\nExample:\n"
        << "  " << prog
        << " record OUT.wav --duration 5 --rate 16000 --channels 2 --device -1\n"
        << "\nOptions:\n"
        << "  --duration, -t <secs>  Recording duration in seconds (default: 3)\n"
        << "  --rate, -s <hz>        Recording sample rate (default: 16000)\n"
        << "  --channels, -c <num>   Recording channels (default: 1)\n"
        << "  --device, -d <idx>     Input device index (default: -1, auto)\n"
        << "  --help, -h             Show this help\n";
}

void printPlayHelp(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " play IN.wav [options]\n"
        << "\nExample:\n"
        << "  " << prog << " play IN.wav --rate 48000 --channels 1 --device 1\n"
        << "\nOptions:\n"
        << "  --rate, -s <hz>     Target playback sample rate (default: 48000)\n"
        << "  --channels, -c <n>  Target playback channels (default: WAV channels)\n"
        << "  --device, -d <idx>  Output device index (default: -1, auto)\n"
        << "  --help, -h          Show this help\n";
}

void printRoundtripHelp(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " roundtrip OUT.wav [options]\n"
        << "\nExample:\n"
        << "  " << prog
        << " roundtrip OUT.wav --duration 2 --rate 16000 --channels 1 "
        << "--play-rate 48000 --play-channels 1\n"
        << "\nOptions:\n"
        << "  --duration, -t <secs>  Recording duration in seconds (default: 2)\n"
        << "  --rate, -s <hz>        Recording sample rate (default: 16000)\n"
        << "  --channels, -c <num>   Recording channels (default: 1)\n"
        << "  --input-device <idx>   Input device index (default: -1, auto)\n"
        << "  --output-device <idx>  Output device index (default: -1, auto)\n"
        << "  --play-rate <hz>       Target playback sample rate (default: 48000)\n"
        << "  --play-channels <num>  Target playback channels (default: WAV channels)\n"
        << "  --help, -h             Show this help\n";
}

bool parseRecordOptions(int argc, char* argv[], RecordOptions* options) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" || arg == "-t") {
            if (!parseIntOption(argc, argv, &i, arg, &options->duration, true)) return false;
        } else if (arg == "--rate" || arg == "-s") {
            if (!parseIntOption(argc, argv, &i, arg, &options->sample_rate, true)) return false;
        } else if (arg == "--channels" || arg == "-c") {
            if (!parseIntOption(argc, argv, &i, arg, &options->channels, true)) return false;
        } else if (arg == "--device" || arg == "-d") {
            if (!parseIntOption(argc, argv, &i, arg, &options->device)) return false;
        } else if (options->output_file.empty()) {
            options->output_file = arg;
        } else {
            std::cerr << "Unexpected record argument: " << arg << std::endl;
            return false;
        }
    }

    if (options->output_file.empty()) {
        std::cerr << "record requires OUT.wav" << std::endl;
        return false;
    }

    return true;
}

bool parsePlayOptions(int argc, char* argv[], PlayOptions* options) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rate" || arg == "-s") {
            if (!parseIntOption(argc, argv, &i, arg, &options->sample_rate, true)) return false;
        } else if (arg == "--channels" || arg == "-c") {
            if (!parseIntOption(argc, argv, &i, arg, &options->channels, true)) return false;
        } else if (arg == "--device" || arg == "-d") {
            if (!parseIntOption(argc, argv, &i, arg, &options->device)) return false;
        } else if (options->input_file.empty()) {
            options->input_file = arg;
        } else {
            std::cerr << "Unexpected play argument: " << arg << std::endl;
            return false;
        }
    }

    if (options->input_file.empty()) {
        std::cerr << "play requires IN.wav" << std::endl;
        return false;
    }

    return true;
}

bool parseRoundtripOptions(int argc, char* argv[], RoundtripOptions* options) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--duration" || arg == "-t") {
            if (!parseIntOption(argc, argv, &i, arg, &options->duration, true)) return false;
        } else if (arg == "--rate" || arg == "-s") {
            if (!parseIntOption(argc, argv, &i, arg, &options->sample_rate, true)) return false;
        } else if (arg == "--channels" || arg == "-c") {
            if (!parseIntOption(argc, argv, &i, arg, &options->channels, true)) return false;
        } else if (arg == "--input-device") {
            if (!parseIntOption(argc, argv, &i, arg, &options->input_device)) return false;
        } else if (arg == "--output-device") {
            if (!parseIntOption(argc, argv, &i, arg, &options->output_device)) return false;
        } else if (arg == "--play-rate") {
            if (!parseIntOption(argc, argv, &i, arg, &options->play_rate, true)) return false;
        } else if (arg == "--play-channels") {
            if (!parseIntOption(argc, argv, &i, arg, &options->play_channels, true)) return false;
        } else if (options->output_file.empty()) {
            options->output_file = arg;
        } else {
            std::cerr << "Unexpected roundtrip argument: " << arg << std::endl;
            return false;
        }
    }

    if (options->output_file.empty()) {
        std::cerr << "roundtrip requires OUT.wav" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signalHandler);

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string command = argv[1];
    if (isHelpOption(command)) {
        printUsage(argv[0]);
        return 0;
    }

    if (command == "list") {
        if (containsHelp(argc, argv, 2)) {
            printListHelp(argv[0]);
            return 0;
        }
        if (argc != 2) {
            std::cerr << "list does not accept options" << std::endl;
            return 1;
        }
        listDevices();
        return 0;
    }

    if (command == "record") {
        if (containsHelp(argc, argv, 2)) {
            printRecordHelp(argv[0]);
            return 0;
        }
        RecordOptions options;
        if (!parseRecordOptions(argc, argv, &options)) {
            printRecordHelp(argv[0]);
            return 1;
        }
        return recordAudio(options) ? 0 : 1;
    }

    if (command == "play") {
        if (containsHelp(argc, argv, 2)) {
            printPlayHelp(argv[0]);
            return 0;
        }
        PlayOptions options;
        if (!parsePlayOptions(argc, argv, &options)) {
            printPlayHelp(argv[0]);
            return 1;
        }
        return playAudio(options) ? 0 : 1;
    }

    if (command == "roundtrip") {
        if (containsHelp(argc, argv, 2)) {
            printRoundtripHelp(argv[0]);
            return 0;
        }
        RoundtripOptions options;
        if (!parseRoundtripOptions(argc, argv, &options)) {
            printRoundtripHelp(argv[0]);
            return 1;
        }
        return roundtripAudio(options) ? 0 : 1;
    }

    std::cerr << "Unknown command: " << command << std::endl;
    printUsage(argv[0]);
    return 1;
}
