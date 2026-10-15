cmake_minimum_required(VERSION 3.16)
project(VAD VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Position Independent Code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# Build Options
# -------------------------------------------------------
option(BUILD_VAD_EXAMPLES "Build VAD example programs" ON)
option(BUILD_VAD_PYTHON "Build Python bindings" ON)

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

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found!\n"
        "Install ONNX Runtime:\n"
        "  macOS: brew install onnxruntime\n"
        "  Linux: Download from https://github.com/microsoft/onnxruntime/releases"
    )
endif()

# libcurl (for model auto-download)
pkg_check_modules(CURL REQUIRED libcurl)

# -------------------------------------------------------
# Silero Backend Library
# -------------------------------------------------------
set(SILERO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/silero/silero_backend.cpp
)

add_library(silero_vad STATIC ${SILERO_SOURCES})

# Map to portable _Float16 for RISC-V
add_definitions(-D__fp16=_Float16)

target_include_directories(silero_vad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(silero_vad
    PUBLIC ${ONNXRUNTIME_LIB}
    PRIVATE pthread ${CURL_LIBRARIES}
)

set_target_properties(silero_vad PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# -------------------------------------------------------
# VAD Framework Library
# -------------------------------------------------------
set(VAD_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vad_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vad_backend_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vad_presets.cpp
)

add_library(vad STATIC ${VAD_SOURCES})

target_include_directories(vad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vad
    PUBLIC silero_vad
    PRIVATE pthread
)

set_target_properties(vad PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "VAD Framework library enabled")

# -------------------------------------------------------
# Install rules
# -------------------------------------------------------
install(TARGETS vad silero_vad
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES include/vad_service.h DESTINATION include)

# -------------------------------------------------------
# VAD Demo 可执行程序
# -------------------------------------------------------
# BUILD_STREAM_DEMO=ON 时编译流式检测示例，需 PortAudio 与 audio 组件。
option(BUILD_VAD_EXAMPLES "Build VAD file demo" ON)
option(BUILD_STREAM_DEMO "Build VAD stream demo (requires audio component + PortAudio)" OFF)

if(BUILD_VAD_EXAMPLES)
    add_executable(vad_simple_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_demo.cpp)

    target_include_directories(vad_simple_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(vad_simple_demo
        PRIVATE vad
        PRIVATE pthread
        PRIVATE m
    )

    set_target_properties(vad_simple_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    install(TARGETS vad_simple_demo RUNTIME DESTINATION bin)
endif()

# Stream demo (if BUILD_STREAM_DEMO and audio component and source file available)
set(AUDIO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../audio)
set(STREAM_DEMO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/examples/stream_demo.cpp)
if(BUILD_STREAM_DEMO AND EXISTS ${AUDIO_DIR} AND EXISTS ${STREAM_DEMO_SRC})
    pkg_check_modules(PORTAUDIO portaudio-2.0)
    if(PORTAUDIO_FOUND)
        add_executable(vad_stream_demo ${STREAM_DEMO_SRC})

            target_include_directories(vad_stream_demo PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${AUDIO_DIR}/inc
                ${PORTAUDIO_INCLUDE_DIRS}
            )

            target_link_libraries(vad_stream_demo
                PRIVATE vad
                PRIVATE ${PORTAUDIO_LIBRARIES}
                PRIVATE pthread
                PRIVATE m
            )

            target_link_directories(vad_stream_demo PRIVATE ${PORTAUDIO_LIBRARY_DIRS})

            set_target_properties(vad_stream_demo PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            )

        install(TARGETS vad_stream_demo RUNTIME DESTINATION bin)
        message(STATUS "VAD stream demo enabled")
    endif()
else()
    if(BUILD_STREAM_DEMO)
        message(STATUS "VAD stream demo disabled (audio/stream_demo not found or PortAudio missing)")
    endif()
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
if(BUILD_VAD_PYTHON)
    set(VAD_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/python/vad_bindings.cpp)
    if(NOT EXISTS ${VAD_BINDINGS_SRC})
        message(STATUS "Python bindings disabled (vad_bindings.cpp not found)")
        set(BUILD_VAD_PYTHON OFF)
    endif()
endif()

if(BUILD_VAD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE} (${Python3_VERSION})")
    endif()

    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND AND Python3_FOUND)
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

    if(pybind11_FOUND)
        pybind11_add_module(_spacemit_vad ${CMAKE_CURRENT_SOURCE_DIR}/python/vad_bindings.cpp)

        target_include_directories(_spacemit_vad PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )

        target_link_libraries(_spacemit_vad PRIVATE vad)

        # Only set default output dir if not specified externally (e.g., by setup.py)
        if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
            set_target_properties(_spacemit_vad PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
            )
        endif()

        message(STATUS "Python bindings enabled (_spacemit_vad module for Python ${Python3_VERSION})")

        set(VAD_WHEEL_BUILD_ROOT "${CMAKE_BINARY_DIR}/python_wheel")
        set(VAD_WHEEL_OUTPUT_DIR "${VAD_WHEEL_BUILD_ROOT}/dist")
        set(VAD_WHEEL_CMAKE_BUILD_DIR "${VAD_WHEEL_BUILD_ROOT}/cmake")
        set(VAD_WHEEL_SOURCE_DIR "${VAD_WHEEL_BUILD_ROOT}/src")
        if(DEFINED ENV{SROBOTIS_OUTPUT_ROOT} AND NOT "$ENV{SROBOTIS_OUTPUT_ROOT}" STREQUAL "")
            set(_vad_output_root "$ENV{SROBOTIS_OUTPUT_ROOT}")
        elseif(DEFINED ENV{OUTPUT_ROOT} AND NOT "$ENV{OUTPUT_ROOT}" STREQUAL "")
            set(_vad_output_root "$ENV{OUTPUT_ROOT}")
        else()
            get_filename_component(_vad_output_root "${CMAKE_INSTALL_PREFIX}/.." ABSOLUTE)
        endif()
        set(VAD_DIST_INSTALL_DIR "${_vad_output_root}/dist")
        set(VAD_WHEEL_STAMP "${VAD_WHEEL_BUILD_ROOT}/wheel.stamp")
        file(MAKE_DIRECTORY "${VAD_WHEEL_OUTPUT_DIR}")

        add_custom_command(
            OUTPUT ${VAD_WHEEL_STAMP}
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${VAD_WHEEL_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E rm -rf ${VAD_WHEEL_SOURCE_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${VAD_WHEEL_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${VAD_WHEEL_CMAKE_BUILD_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python ${VAD_WHEEL_SOURCE_DIR}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${VAD_WHEEL_SOURCE_DIR}/README.md
            COMMAND ${CMAKE_COMMAND} -E chdir ${VAD_WHEEL_SOURCE_DIR}
                ${CMAKE_COMMAND} -E env
                SPACEMIT_PREBUILT_EXTENSION=$<TARGET_FILE:_spacemit_vad>
                SPACEMIT_CMAKE_BUILD_DIR=${VAD_WHEEL_CMAKE_BUILD_DIR}
                ${Python3_EXECUTABLE} -m pip wheel . --no-deps --wheel-dir ${VAD_WHEEL_OUTPUT_DIR}
            COMMAND ${CMAKE_COMMAND} -E touch ${VAD_WHEEL_STAMP}
            DEPENDS
                _spacemit_vad
                ${CMAKE_CURRENT_SOURCE_DIR}/README.md
                ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py
                ${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml
                ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_vad/__init__.py
            VERBATIM
            COMMENT "Building spacemit_vad wheel into ${VAD_WHEEL_OUTPUT_DIR}"
        )

        add_custom_target(vad-python-wheel ALL DEPENDS ${VAD_WHEEL_STAMP})

        # Get Python site-packages path
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Install target
        add_custom_target(vad-install-python
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/spacemit_vad
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_vad ${PYTHON_SITE_PACKAGES}/spacemit_vad
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_vad> ${PYTHON_SITE_PACKAGES}/spacemit_vad/
            DEPENDS _spacemit_vad
            COMMENT "Installing spacemit_vad to ${PYTHON_SITE_PACKAGES}/spacemit_vad/"
        )

        install(
            DIRECTORY ${VAD_WHEEL_OUTPUT_DIR}/
            DESTINATION ${VAD_DIST_INSTALL_DIR}
            FILES_MATCHING PATTERN "*.whl"
        )
    else()
        message(STATUS "Python bindings disabled (pybind11 not found)")
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "=== VAD Build Configuration ===")
message(STATUS "  BUILD_VAD_EXAMPLES:  ${BUILD_VAD_EXAMPLES}")
message(STATUS "  BUILD_VAD_PYTHON:    ${BUILD_VAD_PYTHON}")
message(STATUS "  ONNX Runtime:        ${ONNXRUNTIME_LIB}")
if(BUILD_VAD_PYTHON AND pybind11_FOUND)
message(STATUS "  Wheel output:        ${VAD_WHEEL_OUTPUT_DIR}")
message(STATUS "  Wheel install dir:   ${VAD_DIST_INSTALL_DIR}")
endif()
message(STATUS "")
