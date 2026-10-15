cmake_minimum_required(VERSION 3.16)

# audio component: SpacemitAudio - Audio capture and playback library
# Provides: audio_resampler, spacemit_audio
# Optional: Python bindings (_spacemit_audio)

project(audio VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# Options
# -------------------------------------------------------
option(AUDIO_BUILD_PYTHON "Build Python bindings" ON)
option(AUDIO_BUILD_EXAMPLES "Build examples" ON)
option(AUDIO_SHARED "Build as shared libraries" OFF)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# -------------------------------------------------------
# Detect build mode: standalone vs component
# -------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(AUDIO_STANDALONE ON)
    message(STATUS "[audio] Standalone build mode")
else()
    set(AUDIO_STANDALONE OFF)
    message(STATUS "[audio] Component build mode")
endif()

# -------------------------------------------------------
# Find Dependencies
# -------------------------------------------------------
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(SAMPLERATE samplerate)

# -------------------------------------------------------
# PortAudio: build from source (ALSA-only, no JACK/PulseAudio)
# Avoids Pa_Initialize failures when system PortAudio has
# PulseAudio/JACK backends but the services are unreachable.
# -------------------------------------------------------
include(FetchContent)
set(PA_USE_ALSA ON CACHE BOOL "" FORCE)
set(PA_USE_JACK OFF CACHE BOOL "" FORCE)
set(PA_USE_PULSEAUDIO OFF CACHE BOOL "" FORCE)
set(PA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(portaudio
    GIT_REPOSITORY https://github.com/PortAudio/portaudio.git
    GIT_TAG v19.7.0
)
FetchContent_MakeAvailable(portaudio)

set(PORTAUDIO_FOUND TRUE)
set(PORTAUDIO_LIBRARIES portaudio_static)
set(PORTAUDIO_INCLUDE_DIRS ${portaudio_SOURCE_DIR}/include)

# Suppress warnings from third-party PortAudio source
target_compile_options(portaudio_static PRIVATE -w)
target_compile_options(portaudio PRIVATE -w)

# -------------------------------------------------------
# Resampler Library (no dependencies)
# -------------------------------------------------------
set(RESAMPLER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resampler_rvv.cpp
)

if(AUDIO_SHARED)
    add_library(audio_resampler SHARED ${RESAMPLER_SOURCES})
else()
    add_library(audio_resampler STATIC ${RESAMPLER_SOURCES})
endif()

target_include_directories(audio_resampler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(audio_resampler PUBLIC Threads::Threads)

if(SAMPLERATE_FOUND)
    target_compile_definitions(audio_resampler PUBLIC USE_LIBSAMPLERATE)
    target_include_directories(audio_resampler PRIVATE ${SAMPLERATE_INCLUDE_DIRS})
    target_link_libraries(audio_resampler PUBLIC ${SAMPLERATE_LIBRARIES})
    target_link_directories(audio_resampler PUBLIC ${SAMPLERATE_LIBRARY_DIRS})
    message(STATUS "[audio] libsamplerate enabled")
endif()

# Namespace alias
add_library(components::audio_resampler ALIAS audio_resampler)

message(STATUS "[audio] audio_resampler enabled")

# -------------------------------------------------------
# SpacemitAudio Library (main API, requires PortAudio)
# -------------------------------------------------------
if(PORTAUDIO_FOUND)
    set(SPACEMIT_AUDIO_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spacemit_audio.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spacemit_audio_duplex.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audio_stream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/audio_duplex_stream.cpp
    )

    if(AUDIO_SHARED)
        add_library(spacemit_audio SHARED ${SPACEMIT_AUDIO_SOURCES})
    else()
        add_library(spacemit_audio STATIC ${SPACEMIT_AUDIO_SOURCES})
    endif()

    target_include_directories(spacemit_audio
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
            ${PORTAUDIO_INCLUDE_DIRS}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include/internal
    )

    target_link_libraries(spacemit_audio
        PUBLIC
            ${PORTAUDIO_LIBRARIES}
            Threads::Threads
    )

    # Namespace alias
    add_library(components::spacemit_audio ALIAS spacemit_audio)

    message(STATUS "[audio] spacemit_audio enabled")
endif()

# -------------------------------------------------------
# Examples
# -------------------------------------------------------
if(AUDIO_BUILD_EXAMPLES AND PORTAUDIO_FOUND)
    add_executable(audio_demo
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/audio_demo.cpp
    )

    target_link_libraries(audio_demo PRIVATE spacemit_audio audio_resampler)

    set_target_properties(audio_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    message(STATUS "[audio] audio_demo enabled")
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
set(PYTHON_BINDINGS_ENABLED OFF)

if(AUDIO_BUILD_PYTHON AND PORTAUDIO_FOUND)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        find_package(pybind11 CONFIG QUIET)

        if(NOT pybind11_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_pip_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE pybind11_result
                ERROR_QUIET
            )
            if(pybind11_result EQUAL 0)
                set(pybind11_DIR ${pybind11_pip_DIR})
                find_package(pybind11 CONFIG QUIET)
            endif()
        endif()

        if(pybind11_FOUND)
            pybind11_add_module(_spacemit_audio
                ${CMAKE_CURRENT_SOURCE_DIR}/python/src/bindings.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/spacemit_audio.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/spacemit_audio_duplex.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/audio_stream.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/audio_duplex_stream.cpp
            )

            target_include_directories(_spacemit_audio PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/include/internal
                ${PORTAUDIO_INCLUDE_DIRS}
            )

            target_link_libraries(_spacemit_audio PRIVATE ${PORTAUDIO_LIBRARIES})

            set(PYTHON_BINDINGS_ENABLED ON)
            message(STATUS "[audio] Python bindings enabled (${Python3_EXECUTABLE})")

            # Generate __init__.py in build directory (avoid dependency on source tree)
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/spacemit_audio/__init__.py"
                "from ._spacemit_audio import AudioCapture, AudioPlayer, init, get_config\n"
                "\n"
                "__all__ = ['AudioCapture', 'AudioPlayer', 'init', 'get_config']\n"
                "__version__ = '${PROJECT_VERSION}'\n"
            )
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
                OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            add_custom_target(audio-install-python
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/spacemit_audio
                COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_SITE_PACKAGES}/spacemit_audio
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/spacemit_audio/__init__.py ${PYTHON_SITE_PACKAGES}/spacemit_audio/
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_audio> ${PYTHON_SITE_PACKAGES}/spacemit_audio/
                DEPENDS _spacemit_audio
                COMMENT "Installing spacemit_audio to ${PYTHON_SITE_PACKAGES}/spacemit_audio/"
            )
        else()
            message(STATUS "[audio] Python bindings disabled (pybind11 not found)")
        endif()
    else()
        message(STATUS "[audio] Python bindings disabled (Python3 not found)")
    endif()
endif()

# -------------------------------------------------------
# Install Targets
# -------------------------------------------------------

# Determine install destination based on build mode
if(AUDIO_STANDALONE)
    # Standalone: install to build/install or CMAKE_INSTALL_PREFIX
    set(AUDIO_INSTALL_LIBDIR lib)
    set(AUDIO_INSTALL_BINDIR bin)
    set(AUDIO_INSTALL_INCLUDEDIR include)
    set(AUDIO_INSTALL_PYTHONDIR lib/python)
else()
    # Component: use standard destinations (will go to output/sdk/)
    set(AUDIO_INSTALL_LIBDIR lib)
    set(AUDIO_INSTALL_BINDIR bin)
    set(AUDIO_INSTALL_INCLUDEDIR include)
    set(AUDIO_INSTALL_PYTHONDIR lib/python)
endif()

# Install resampler
install(
    TARGETS audio_resampler
    EXPORT ComponentsTargets
    ARCHIVE DESTINATION ${AUDIO_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${AUDIO_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${AUDIO_INSTALL_BINDIR}
)

install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/audio_resampler.hpp
    DESTINATION ${AUDIO_INSTALL_INCLUDEDIR}
)

# Install spacemit_audio (if PortAudio available)
if(PORTAUDIO_FOUND)
    install(
        TARGETS spacemit_audio
        EXPORT ComponentsTargets
        ARCHIVE DESTINATION ${AUDIO_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${AUDIO_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${AUDIO_INSTALL_BINDIR}
    )

    install(
        FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/audio_base.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/include/audio_duplex.hpp
        DESTINATION ${AUDIO_INSTALL_INCLUDEDIR}
    )

    # Install example
    if(AUDIO_BUILD_EXAMPLES)
        install(
            TARGETS audio_demo
            RUNTIME DESTINATION ${AUDIO_INSTALL_BINDIR}
        )
    endif()

    # Install Python bindings
    if(PYTHON_BINDINGS_ENABLED)
        install(
            TARGETS _spacemit_audio
            LIBRARY DESTINATION ${AUDIO_INSTALL_PYTHONDIR}/spacemit_audio
        )

        install(
            FILES ${CMAKE_CURRENT_BINARY_DIR}/spacemit_audio/__init__.py
            DESTINATION ${AUDIO_INSTALL_PYTHONDIR}/spacemit_audio
        )
    endif()
endif()

# -------------------------------------------------------
# Standalone: also output to source tree for development
# -------------------------------------------------------
if(AUDIO_STANDALONE AND PYTHON_BINDINGS_ENABLED)
    # Use CMAKE_LIBRARY_OUTPUT_DIRECTORY if set (e.g., by pip wheel build)
    # Otherwise output to source tree for local development
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(_spacemit_audio PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_audio
        )
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "[audio] Build Summary:")
message(STATUS "  Mode:           ${AUDIO_STANDALONE}")
message(STATUS "  audio_resampler: ON")
message(STATUS "  spacemit_audio:     ${PORTAUDIO_FOUND}")
message(STATUS "  Examples:        ${AUDIO_BUILD_EXAMPLES}")
message(STATUS "  Python:          ${PYTHON_BINDINGS_ENABLED}")
message(STATUS "")
