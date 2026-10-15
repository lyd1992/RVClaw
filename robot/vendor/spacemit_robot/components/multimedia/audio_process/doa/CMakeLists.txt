cmake_minimum_required(VERSION 3.16)

# audio_process component: Sound source localization (GCC-PHAT)
# Provides: sound_locator
# Optional: Python bindings (_spacemit_audio_process), ssl_demo example

project(audio_process VERSION 1.0.0 LANGUAGES CXX)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(AUDIO_PROCESS_RISCV_CXX_FLAGS)
set(AUDIO_PROCESS_RISCV_PYTHON_CXX_FLAGS)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv|RISCV")
    list(APPEND AUDIO_PROCESS_RISCV_CXX_FLAGS
        -mstrict-align
        -fno-strict-aliasing
        -fno-aggressive-loop-optimizations
    )
    list(APPEND AUDIO_PROCESS_RISCV_PYTHON_CXX_FLAGS
        -O2
        -fno-strict-overflow
        ${AUDIO_PROCESS_RISCV_CXX_FLAGS}
    )
    message(STATUS "[audio_process] RISC-V conservative C++ flags: ${AUDIO_PROCESS_RISCV_CXX_FLAGS}")
    message(STATUS "[audio_process] RISC-V Python extension flags: ${AUDIO_PROCESS_RISCV_PYTHON_CXX_FLAGS}")
endif()

# -------------------------------------------------------
# Options
# -------------------------------------------------------
option(AUDIO_PROCESS_BUILD_PYTHON "Build Python bindings" ON)
option(AUDIO_PROCESS_BUILD_EXAMPLES "Build examples" OFF)
option(AUDIO_PROCESS_SHARED "Build as shared library" OFF)

# -------------------------------------------------------
# Detect build mode: standalone vs component
# -------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(AUDIO_PROCESS_STANDALONE ON)
    message(STATUS "[audio_process] Standalone build mode")
else()
    set(AUDIO_PROCESS_STANDALONE OFF)
    message(STATUS "[audio_process] Component build mode")
endif()

# -------------------------------------------------------
# Find Dependencies
# -------------------------------------------------------
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# FFTW3f (float version, required)
if(APPLE)
    find_library(FFTW3F_LIBRARIES NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib)
    find_path(FFTW3F_INCLUDE_DIRS NAMES fftw3.h PATHS /opt/homebrew/include /usr/local/include)
else()
    pkg_check_modules(FFTW3F REQUIRED fftw3f)
endif()

if(NOT FFTW3F_LIBRARIES)
    message(FATAL_ERROR "[audio_process] FFTW3f not found. Install libfftw3-dev.")
endif()

# -------------------------------------------------------
# SoundLocator Library
# -------------------------------------------------------
set(SOUND_LOCATOR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sound_locator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gcc_phat_pair.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multi_sound_locator.cpp
)

if(AUDIO_PROCESS_SHARED)
    add_library(sound_locator SHARED ${SOUND_LOCATOR_SOURCES})
else()
    add_library(sound_locator STATIC ${SOUND_LOCATOR_SOURCES})
endif()

target_include_directories(sound_locator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(sound_locator
    PUBLIC Threads::Threads
    PRIVATE ${FFTW3F_LIBRARIES} m
)
if(AUDIO_PROCESS_RISCV_CXX_FLAGS)
    target_compile_options(sound_locator PRIVATE ${AUDIO_PROCESS_RISCV_CXX_FLAGS})
endif()
target_include_directories(sound_locator PRIVATE ${FFTW3F_INCLUDE_DIRS})
if(FFTW3F_LIBRARY_DIRS)
    target_link_directories(sound_locator PRIVATE ${FFTW3F_LIBRARY_DIRS})
endif()

# Namespace alias
add_library(components::sound_locator ALIAS sound_locator)

message(STATUS "[audio_process] sound_locator enabled")

# -------------------------------------------------------
# Examples
# -------------------------------------------------------
if(AUDIO_PROCESS_BUILD_EXAMPLES)
    # Unified ssl_demo (handles 2-ch via SoundLocator and 3+ch via
    # MultiSoundLocator; channel count selected with `-c N`).
    add_executable(ssl_demo
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/ssl_demo.cpp
    )

    target_link_libraries(ssl_demo PRIVATE sound_locator m)

    # Try to find spacemit_audio for live-capture mode
    find_library(SPACE_AUDIO_LIB NAMES spacemit_audio)
    find_path(SPACE_AUDIO_INC NAMES audio_base.hpp)

    if(SPACE_AUDIO_LIB AND SPACE_AUDIO_INC)
        target_compile_definitions(ssl_demo PRIVATE HAS_SPACEMIT_AUDIO)
        target_include_directories(ssl_demo PRIVATE ${SPACE_AUDIO_INC})
        target_link_libraries(ssl_demo PRIVATE ${SPACE_AUDIO_LIB})

        # PortAudio (transitive dep of the static libspacemit_audio). Prefer
        # the ALSA-only PortAudio the audio component bundled + installed next
        # to libspacemit_audio: the system pkg-config PortAudio is commonly
        # built with the JACK host API, whose Pa_Initialize() returns
        # "Unanticipated host error" on boards with no JACK server. Link the
        # static archive so there is no runtime .so resolution ambiguity
        # (mirrors how the audio component links audio_demo). Fall back to
        # pkg-config only when the bundled archive is absent.
        get_filename_component(_sa_libdir "${SPACE_AUDIO_LIB}" DIRECTORY)
        find_library(PORTAUDIO_BUNDLED
            NAMES libportaudio.a portaudio
            HINTS "${_sa_libdir}" NO_DEFAULT_PATH)
        if(PORTAUDIO_BUNDLED)
            find_package(Threads REQUIRED)
            find_library(ASOUND_LIB NAMES asound)
            if(NOT ASOUND_LIB)
                message(FATAL_ERROR
                    "[audio_process] ssl_demo uses static PortAudio but libasound "
                    "was not found; install libasound2-dev or disable examples")
            endif()
            target_link_libraries(ssl_demo PRIVATE ${PORTAUDIO_BUNDLED} Threads::Threads)
            target_link_libraries(ssl_demo PRIVATE ${ASOUND_LIB})
            message(STATUS "[audio_process] ssl_demo: portaudio=${PORTAUDIO_BUNDLED} (bundled ALSA-only)")
        else()
            pkg_check_modules(PORTAUDIO portaudio-2.0)
            if(PORTAUDIO_FOUND)
                target_link_libraries(ssl_demo PRIVATE ${PORTAUDIO_LIBRARIES})
                target_link_directories(ssl_demo PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
            endif()
            message(WARNING "[audio_process] ssl_demo: using system pkg-config PortAudio "
                "(bundled not found); live mode may hit JACK Pa_Initialize errors")
        endif()
        message(STATUS "[audio_process] ssl_demo: live mode enabled (spacemit_audio found)")
    else()
        message(STATUS "[audio_process] ssl_demo: live mode disabled (spacemit_audio not found)")
    endif()

    set_target_properties(ssl_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    message(STATUS "[audio_process] ssl_demo (unified 2-ch + 3+ch) enabled")
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
set(PYTHON_BINDINGS_ENABLED OFF)

if(AUDIO_PROCESS_BUILD_PYTHON)
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
            pybind11_add_module(_spacemit_audio_process NO_EXTRAS
                ${CMAKE_CURRENT_SOURCE_DIR}/python/src/bindings.cpp
                ${SOUND_LOCATOR_SOURCES}
            )

            # Keep the CMake-built extension aligned with setup.py/wheel builds.
            target_link_libraries(_spacemit_audio_process PRIVATE Threads::Threads ${FFTW3F_LIBRARIES} m)
            target_include_directories(_spacemit_audio_process PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${FFTW3F_INCLUDE_DIRS}
            )
            if(FFTW3F_LIBRARY_DIRS)
                target_link_directories(_spacemit_audio_process PRIVATE ${FFTW3F_LIBRARY_DIRS})
            endif()
            if(AUDIO_PROCESS_RISCV_PYTHON_CXX_FLAGS)
                target_compile_options(_spacemit_audio_process PRIVATE ${AUDIO_PROCESS_RISCV_PYTHON_CXX_FLAGS})
            endif()

            set(PYTHON_BINDINGS_ENABLED ON)
            message(STATUS "[audio_process] Python bindings enabled (${Python3_EXECUTABLE})")

            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/spacemit_audio_process/__init__.py"
                "from ._spacemit_audio_process import (\n"
                "    SoundLocator, SoundLocatorConfig,\n"
                "    MicrophonePosition, MultiSoundLocatorConfig,\n"
                "    MultiSoundLocatorResult, MultiSoundLocator,\n"
                ")\n"
                "\n"
                "__all__ = [\n"
                "    \"SoundLocator\", \"SoundLocatorConfig\",\n"
                "    \"MicrophonePosition\", \"MultiSoundLocatorConfig\",\n"
                "    \"MultiSoundLocatorResult\", \"MultiSoundLocator\",\n"
                "]\n"
                "__version__ = \"${PROJECT_VERSION}\"\n"
            )
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib'))"
                OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            add_custom_target(audio_process-install-python
                COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/spacemit_audio_process
                COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_SITE_PACKAGES}/spacemit_audio_process
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/spacemit_audio_process/__init__.py ${PYTHON_SITE_PACKAGES}/spacemit_audio_process/
                COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_spacemit_audio_process> ${PYTHON_SITE_PACKAGES}/spacemit_audio_process/
                DEPENDS _spacemit_audio_process
                COMMENT "Installing spacemit_audio_process to ${PYTHON_SITE_PACKAGES}/spacemit_audio_process/"
            )
        else()
            message(STATUS "[audio_process] Python bindings disabled (pybind11 not found)")
        endif()
    else()
        message(STATUS "[audio_process] Python bindings disabled (Python3 not found)")
    endif()
endif()

# -------------------------------------------------------
# Install Targets
# -------------------------------------------------------
set(AP_INSTALL_LIBDIR lib)
set(AP_INSTALL_BINDIR bin)
set(AP_INSTALL_INCLUDEDIR include)
set(AP_INSTALL_PYTHONDIR lib/python)

install(
    TARGETS sound_locator
    EXPORT ComponentsTargets
    ARCHIVE DESTINATION ${AP_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${AP_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${AP_INSTALL_BINDIR}
)

install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/doa_service.h
    DESTINATION ${AP_INSTALL_INCLUDEDIR}
)

if(AUDIO_PROCESS_BUILD_EXAMPLES)
    install(
        TARGETS ssl_demo
        RUNTIME DESTINATION ${AP_INSTALL_BINDIR}
    )
endif()

if(PYTHON_BINDINGS_ENABLED)
    install(
        TARGETS _spacemit_audio_process
        LIBRARY DESTINATION ${AP_INSTALL_PYTHONDIR}/spacemit_audio_process
    )

    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/spacemit_audio_process/__init__.py
        DESTINATION ${AP_INSTALL_PYTHONDIR}/spacemit_audio_process
    )
endif()

# -------------------------------------------------------
# Standalone: output Python module to source tree
# -------------------------------------------------------
if(AUDIO_PROCESS_STANDALONE AND PYTHON_BINDINGS_ENABLED)
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set_target_properties(_spacemit_audio_process PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/spacemit_audio_process
        )
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "[audio_process] Build Summary:")
message(STATUS "  Mode:            ${AUDIO_PROCESS_STANDALONE}")
message(STATUS "  sound_locator:   ON")
message(STATUS "  Examples:        ${AUDIO_PROCESS_BUILD_EXAMPLES}")
message(STATUS "  Python:          ${PYTHON_BINDINGS_ENABLED}")
message(STATUS "")
