cmake_minimum_required(VERSION 3.16)
project(ASR VERSION 1.0.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Position Independent Code for all targets (required for RISC-V shared libs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# Build Options
# -------------------------------------------------------
# BUILD_STREAM_DEMO: 编译麦克风流式识别示例和 Python 绑定
#
# 需要额外依赖:
#   1. PortAudio: brew install portaudio (macOS) / apt install portaudio19-dev (Linux)
#   2. audio 组件: SDK 编译时通过 package.xml 依赖自动构建安装到 staging
#
# 使用方法:
#   SDK 编译时默认开启；独立编译时: cmake .. -DBUILD_STREAM_DEMO=ON
#
# 检测是否在 SDK build 流程内编译（envsetup.sh 已 source 则 SROBOTIS_OUTPUT_ROOT 被 export）。
# 仅文件存在不算——源码可能在仓库里，但用户裸 cmake 时 audio 组件并不一定已装，
# 此时不应默认打开 BUILD_STREAM_DEMO。
if(DEFINED ENV{SROBOTIS_OUTPUT_ROOT} AND NOT "$ENV{SROBOTIS_OUTPUT_ROOT}" STREQUAL "")
    set(_IN_SDK_BUILD ON)
else()
    set(_IN_SDK_BUILD OFF)
endif()

option(BUILD_STREAM_DEMO "Build stream demo and Python bindings (requires audio component)" ${_IN_SDK_BUILD})
option(BUILD_PYTHON_BINDINGS "Build Python bindings (requires BUILD_STREAM_DEMO=ON)" ON)
option(ASR_MODEL_FETCH_OFF "Disable automatic model downloading at configure time" ON)

# -------------------------------------------------------
# Find Dependencies
# -------------------------------------------------------
find_package(PkgConfig REQUIRED)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    /opt/homebrew/include/onnxruntime
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
    /usr/local/lib
    /usr/lib
    /opt/homebrew/lib
)

option(USE_SPACEMIT_EP "Enable SpaceMIT Execution Provider for ASR" ON)

# libsndfile
pkg_check_modules(SNDFILE REQUIRED sndfile)

# libcurl
pkg_check_modules(CURL REQUIRED libcurl)

# FFTW3
if(APPLE)
    find_library(FFTW3_LIBRARIES NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(FFTW3_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
else()
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
endif()

# libsamplerate (optional)
pkg_check_modules(SAMPLERATE samplerate)

# kaldi-native-fbank (for Zipformer feature extraction)
# gitee mirror 优先 + GitHub 回退; 内部 kissfft 预下到 ${CMAKE_BINARY_DIR}/
# 跳过 knf 自带的 FetchContent GitHub 访问. 详见 cmake/kaldi_native_fbank.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/kaldi_native_fbank.cmake)

# -------------------------------------------------------
# Fetch ASR Models
# -------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchSenseVoiceModel.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchZipformerModel.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchQwen3AsrModel.cmake)

# -------------------------------------------------------
# Audio Component (optional, for stream demo)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    find_library(SPACEMIT_AUDIO_LIB spacemit_audio)
    find_library(AUDIO_RESAMPLER_LIB audio_resampler)
    find_path(AUDIO_INC_DIR audio_base.hpp)
    pkg_check_modules(PORTAUDIO portaudio-2.0)

    if(NOT SPACEMIT_AUDIO_LIB OR NOT AUDIO_RESAMPLER_LIB OR NOT AUDIO_INC_DIR OR NOT PORTAUDIO_FOUND)
        message(WARNING "Audio component or PortAudio not found, disabling BUILD_STREAM_DEMO")
        set(BUILD_STREAM_DEMO OFF)
    endif()
endif()

# -------------------------------------------------------
# SenseVoice Backend Library
# -------------------------------------------------------
set(SENSEVOICE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/sensevoice_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/feature_extractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/hotword_scorer.cpp
)

add_library(sensevoice STATIC ${SENSEVOICE_SOURCES})

target_include_directories(sensevoice PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(sensevoice
    PUBLIC ${ONNXRUNTIME_LIB}
    PUBLIC ${FFTW3_LIBRARIES}
    PRIVATE pthread
    PRIVATE m
)

# -------------------------------------------------------
# Zipformer Backend Library
# -------------------------------------------------------
set(ZIPFORMER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/zipformer/zipformer_ctc_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/zipformer/zipformer_feature_extractor.cpp
)

add_library(zipformer STATIC ${ZIPFORMER_SOURCES})

target_include_directories(zipformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${KALDI_FBANK_INCLUDE_DIRS}
)

target_link_libraries(zipformer
    PUBLIC ${ONNXRUNTIME_LIB}
    PUBLIC kaldi-native-fbank-core
)

set_target_properties(zipformer PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# -------------------------------------------------------
# ASR Framework Library
# -------------------------------------------------------
set(ASR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model_downloader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/sensevoice/sensevoice_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/qwen3_asr/qwen3_asr_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/zipformer/zipformer_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/asr_presets.cpp
)

add_library(asr STATIC ${ASR_SOURCES})
add_library(stt ALIAS asr)

target_include_directories(asr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CURL_INCLUDE_DIRS}
)

# Conditionally add resampler include
if(BUILD_STREAM_DEMO)
    target_include_directories(asr PUBLIC ${AUDIO_INC_DIR})
    target_link_libraries(asr PUBLIC ${AUDIO_RESAMPLER_LIB} ${SAMPLERATE_LIBRARIES})
endif()

target_link_libraries(asr
    PUBLIC sensevoice
    PUBLIC zipformer
    PUBLIC ${SNDFILE_LIBRARIES}
    PRIVATE ${CURL_LIBRARIES}
    PRIVATE pthread
)

target_link_directories(asr PRIVATE ${CURL_LIBRARY_DIRS})

target_link_directories(asr PUBLIC ${SNDFILE_LIBRARY_DIRS})

set_target_properties(asr PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

set_target_properties(sensevoice PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "ASR Framework library enabled")

# -------------------------------------------------------
# Install rules
# -------------------------------------------------------
install(TARGETS asr sensevoice zipformer
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES $<TARGET_FILE:asr> DESTINATION lib RENAME libstt.a)
install(FILES include/asr_service.h DESTINATION include)

if(TARGET kaldi-native-fbank-core)
    install(TARGETS kaldi-native-fbank-core
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

foreach(_knf_runtime_dep
        kissfft-float
        kissfft-double
        kissfft-int16_t
        kissfft)
    if(TARGET ${_knf_runtime_dep})
        install(TARGETS ${_knf_runtime_dep}
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
        )
    endif()
endforeach()

# -------------------------------------------------------
# ASR File Demo Executable (always built)
# -------------------------------------------------------
add_executable(asr_file_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/asr_file_demo.cpp)

target_include_directories(asr_file_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(asr_file_demo
    PRIVATE asr
    PRIVATE ${SNDFILE_LIBRARIES}
    PRIVATE pthread
    PRIVATE m
)

target_link_directories(asr_file_demo PRIVATE ${SNDFILE_LIBRARY_DIRS})

set_target_properties(asr_file_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS asr_file_demo RUNTIME DESTINATION bin)

if(USE_SPACEMIT_EP)
    find_path(SPACEMIT_ORT_INCLUDE_DIR
        NAMES spacemit_ort_env.h
        PATHS ${ONNXRUNTIME_INCLUDE_DIR} /usr/include /usr/local/include
        PATH_SUFFIXES onnxruntime
    )
    find_library(SPACEMIT_EP_LIB
        NAMES spacemit_ep
        PATHS /usr/local/lib /usr/lib
    )
    if(SPACEMIT_ORT_INCLUDE_DIR AND SPACEMIT_EP_LIB)
        target_compile_definitions(sensevoice PRIVATE USE_SPACEMIT_EP)
        target_include_directories(sensevoice PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
        target_link_libraries(sensevoice PUBLIC ${SPACEMIT_EP_LIB})
        target_compile_definitions(zipformer PRIVATE USE_SPACEMIT_EP)
        target_include_directories(zipformer PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
        target_link_libraries(zipformer PUBLIC ${SPACEMIT_EP_LIB})
        target_compile_definitions(asr_file_demo PRIVATE USE_SPACEMIT_EP)
        target_include_directories(asr_file_demo PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
        message(STATUS "SpaceMIT EP enabled: ${SPACEMIT_ORT_INCLUDE_DIR}, lib: ${SPACEMIT_EP_LIB}")
    else()
        message(WARNING "SpaceMIT EP not found (header=${SPACEMIT_ORT_INCLUDE_DIR}, lib=${SPACEMIT_EP_LIB}), USE_SPACEMIT_EP ignored")
    endif()
endif()

message(STATUS "ASR file demo executable enabled")

# -------------------------------------------------------
# ASR Stream Demo Executable (optional)
# -------------------------------------------------------
if(BUILD_STREAM_DEMO)
    add_executable(asr_stream_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/asr_stream_demo.cpp)

    target_include_directories(asr_stream_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${AUDIO_INC_DIR}
    )

    target_link_libraries(asr_stream_demo
        PRIVATE asr
        PRIVATE ${SPACEMIT_AUDIO_LIB}
        PRIVATE ${AUDIO_RESAMPLER_LIB}
        PRIVATE ${PORTAUDIO_LIBRARIES}
        PRIVATE ${SAMPLERATE_LIBRARIES}
        PRIVATE ${SNDFILE_LIBRARIES}
        PRIVATE pthread
        PRIVATE m
    )

    target_link_directories(asr_stream_demo PRIVATE
        ${SNDFILE_LIBRARY_DIRS}
        ${PORTAUDIO_LIBRARY_DIRS}
        ${SAMPLERATE_LIBRARY_DIRS}
    )

    if(SAMPLERATE_FOUND)
        target_compile_definitions(asr_stream_demo PRIVATE USE_LIBSAMPLERATE)
    endif()

    if(USE_SPACEMIT_EP AND SPACEMIT_ORT_INCLUDE_DIR)
        target_compile_definitions(asr_stream_demo PRIVATE USE_SPACEMIT_EP)
        target_include_directories(asr_stream_demo PRIVATE ${SPACEMIT_ORT_INCLUDE_DIR})
    endif()

    set_target_properties(asr_stream_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    install(TARGETS asr_stream_demo RUNTIME DESTINATION bin)

    message(STATUS "ASR stream demo executable enabled")
else()
    message(STATUS "ASR stream demo disabled (BUILD_STREAM_DEMO=OFF)")
    message(STATUS "  To enable: cmake .. -DBUILD_STREAM_DEMO=ON")
    message(STATUS "  Requires: audio component + PortAudio")
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    # Find Python first (respects -DPython3_EXECUTABLE=...)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE} (${Python3_VERSION})")
    endif()

    # Try to find pybind11
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Try using Python's pybind11
        if(Python3_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE PYBIND11_RESULT
            )
            if(PYBIND11_RESULT EQUAL 0)
                list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
                find_package(pybind11 CONFIG QUIET)
            endif()
        endif()
    endif()

    if(pybind11_FOUND)
        pybind11_add_module(_spacemit_asr ${CMAKE_CURRENT_SOURCE_DIR}/python/asr_bindings.cpp)

        target_include_directories(_spacemit_asr PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

        target_link_libraries(_spacemit_asr PRIVATE asr)

        set(ASR_PYTHON_BUILD_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python/spacemit_asr)
        file(MAKE_DIRECTORY ${ASR_PYTHON_BUILD_PACKAGE_DIR})

        file(GLOB ASR_PYTHON_PACKAGE_SOURCES CONFIGURE_DEPENDS
             ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_asr/*.py)
        set(ASR_PYTHON_BUILD_PACKAGE_FILES)
        foreach(ASR_PYTHON_PACKAGE_SOURCE IN LISTS ASR_PYTHON_PACKAGE_SOURCES)
            get_filename_component(ASR_PYTHON_PACKAGE_FILE ${ASR_PYTHON_PACKAGE_SOURCE} NAME)
            set(ASR_PYTHON_PACKAGE_OUTPUT ${ASR_PYTHON_BUILD_PACKAGE_DIR}/${ASR_PYTHON_PACKAGE_FILE})
            add_custom_command(
                OUTPUT ${ASR_PYTHON_PACKAGE_OUTPUT}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${ASR_PYTHON_PACKAGE_SOURCE}
                        ${ASR_PYTHON_PACKAGE_OUTPUT}
                DEPENDS ${ASR_PYTHON_PACKAGE_SOURCE}
                VERBATIM
            )
            list(APPEND ASR_PYTHON_BUILD_PACKAGE_FILES ${ASR_PYTHON_PACKAGE_OUTPUT})
        endforeach()

        add_custom_target(spacemit_asr_python_package ALL
            DEPENDS ${ASR_PYTHON_BUILD_PACKAGE_FILES}
        )
        add_dependencies(_spacemit_asr spacemit_asr_python_package)

        set_target_properties(_spacemit_asr PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${ASR_PYTHON_BUILD_PACKAGE_DIR}
        )

        message(STATUS "Python bindings enabled (_spacemit_asr module for Python ${Python3_VERSION})")

        # Get Python site-packages path
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Install target for pip
        install(TARGETS _spacemit_asr LIBRARY DESTINATION .)

        # Custom target: install Python package into the active environment.
        add_custom_target(asr-install-python
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/spacemit_asr
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_asr ${PYTHON_SITE_PACKAGES}/spacemit_asr
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_asr> ${PYTHON_SITE_PACKAGES}/spacemit_asr/
            DEPENDS _spacemit_asr
            COMMENT "Installing spacemit_asr to ${PYTHON_SITE_PACKAGES}/spacemit_asr/"
        )
        add_custom_target(stt-install-python
            DEPENDS asr-install-python
            COMMENT "Compatibility alias for asr-install-python"
        )
    else()
        message(STATUS "Python bindings disabled (pybind11 not found)")
        message(STATUS "  Install with: pip install pybind11 OR brew install pybind11")
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "=== ASR Build Configuration ===")
message(STATUS "  BUILD_STREAM_DEMO:      ${BUILD_STREAM_DEMO}")
message(STATUS "  BUILD_PYTHON_BINDINGS:  ${BUILD_PYTHON_BINDINGS}")
message(STATUS "")
