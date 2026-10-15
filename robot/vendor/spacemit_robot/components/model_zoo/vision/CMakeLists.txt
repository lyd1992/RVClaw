cmake_minimum_required(VERSION 3.18)
project(ModelZoo VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Project Configuration
# ============================================================================

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS on)

# ============================================================================
# RPATH / Runtime Library Search Paths
# ============================================================================
# Make binaries runnable from both build/ and installed (staging) layouts.
# - Build tree: use link paths so local runs find OpenCV/onnxruntime/spacemit libs.
# - Install tree: search libs relative to the binary (../lib).
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

set(_modelzoo_build_rpath "")
if(DEFINED OpenCV_INSTALL_DIR)
    list(APPEND _modelzoo_build_rpath "${OpenCV_INSTALL_DIR}/lib")
endif()
if(DEFINED SPACEMIT_DIR)
    list(APPEND _modelzoo_build_rpath "${SPACEMIT_DIR}/lib")
endif()
set(CMAKE_BUILD_RPATH "${_modelzoo_build_rpath}")
set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN")

# Build Options
option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings (_vision_service_cpp) for the spacemit_vision wheel" ON)
option(BUILD_PYTHON_WHEEL "Also build the spacemit_vision wheel as part of the default build (needs python -m build)" ON)
option(VISION_WITH_MPP "Enable MPP camera capture backend (UVC/VI) in examples" OFF)

# ============================================================================
# Dependencies Configuration
# ============================================================================

if(NOT DEFINED OpenCV_DIR)
    # Bianbu 4.x ships OpenCV 5 under opencv5/; older images used opencv4/
    if(EXISTS "/opt/opencv-spacemit/lib/cmake/opencv5/OpenCVConfig.cmake")
        set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv5")
    elseif(EXISTS "/opt/opencv-spacemit/lib/cmake/opencv4/OpenCVConfig.cmake")
        set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv4")
    else()
        set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv5")
    endif()
endif()
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs highgui dnn
    NO_DEFAULT_PATH
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "  Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
    set(MODELZOO_OPENCV_PRINTED TRUE CACHE INTERNAL "OpenCV info already printed")
else()
    message(FATAL_ERROR "OpenCV not found! Check OpenCV_DIR or install OpenCV.")
endif()


# Eigen3 (system)
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
endif()

# yaml-cpp Configuration
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp found: ${yaml-cpp_VERSION}")
    message(STATUS "  Includes: ${yaml-cpp_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${yaml-cpp_LIBRARIES}")
else()
    message(FATAL_ERROR "yaml-cpp not found! Please install yaml-cpp.")
endif()

# Library Dependencies
set(LIBS
    ${OpenCV_LIBS}
    onnxruntime
    spacemit_ep
    yaml-cpp
)

# OpenCV 5 moved contour/hull/perspective APIs out of imgproc into the new
# opencv_geometry module (see opencv2/geometry/2d.hpp); on OpenCV 4 those APIs
# live in imgproc, which is already linked via OpenCV_LIBS.
if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
    # Derive the lib dir from OpenCV_DIR (…/lib/cmake/opencvN -> …/lib) so a
    # custom OpenCV_DIR also works; keep the vendor default as fallback.
    get_filename_component(_opencv_lib_dir "${OpenCV_DIR}/../.." ABSOLUTE)
    find_library(OPENCV_GEOMETRY_LIBRARY
        NAMES opencv_geometry
        PATHS "${_opencv_lib_dir}"
              /opt/opencv-spacemit/lib
        NO_DEFAULT_PATH
        REQUIRED
    )
    list(APPEND LIBS ${OPENCV_GEOMETRY_LIBRARY})
    message(STATUS "OpenCV geometry: ${OPENCV_GEOMETRY_LIBRARY}")
endif()

if(VISION_WITH_MPP)
    if(NOT DEFINED STAGING_DIR)
        set(STAGING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../output/staging")
    endif()
    set(VISION_MPP_LIB "${STAGING_DIR}/lib/libmpp.so")
    if(NOT EXISTS "${VISION_MPP_LIB}")
        message(FATAL_ERROR
            "VISION_WITH_MPP=ON but ${VISION_MPP_LIB} not found. "
            "Build MPP first or pass -DSTAGING_DIR=<output/staging>.")
    endif()
    set(VISION_MPP_RPATH "${STAGING_DIR}/lib")
    set(VISION_MPP_INCLUDE_DIRS
        "${STAGING_DIR}/include"
        "${STAGING_DIR}/include/sys"
        "${STAGING_DIR}/include/uvc"
        "${STAGING_DIR}/include/vdec"
        "${STAGING_DIR}/include/v2d"
        "${STAGING_DIR}/include/vi"
    )
    message(STATUS "Vision MPP enabled (staging: ${STAGING_DIR})")
else()
    message(STATUS "Vision MPP disabled (pass -DVISION_WITH_MPP=ON to enable)")
endif()

# ============================================================================
# Source Files
# ============================================================================

file(GLOB_RECURSE VISION_CORE_SOURCES "src/core/cpp/*.cpp")
set(VISION_SERVICE_SOURCES
    src/vision_service.cpp
)
file(GLOB_RECURSE CV_COMMON_SOURCES "src/common/cpp/*.cpp")
file(GLOB DEPLOY_TOP_SOURCES CONFIGURE_DEPENDS
    "src/deploy/*/cpp/*.cpp"
    "src/deploy/*/cpp/*.cc"
)
file(GLOB_RECURSE DEPLOY_NESTED_SOURCES CONFIGURE_DEPENDS
    "src/deploy/*/cpp/src/*.cpp"
    "src/deploy/*/cpp/src/*.cc"
)

# ============================================================================
# Library Target (output: libvision.so)
# ============================================================================

add_library(vision SHARED
    ${VISION_SERVICE_SOURCES}
    ${VISION_CORE_SOURCES}
    ${CV_COMMON_SOURCES}
    ${DEPLOY_TOP_SOURCES}
    ${DEPLOY_NESTED_SOURCES}
)

target_link_libraries(vision
    ${LIBS}
    Eigen3::Eigen
)

# ONNX Runtime float16: use __fp16 if available, else define for dependency headers
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_QUIET TRUE)
check_cxx_source_compiles(
    "int main() { __fp16 x = 0; (void)x; return 0; }"
    HAVE_NATIVE_FP16
)
unset(CMAKE_REQUIRED_QUIET)
if(NOT HAVE_NATIVE_FP16)
    target_compile_definitions(vision PRIVATE __fp16=_Float16)
    message(STATUS "vision: defining __fp16=_Float16 for dependency headers (no native __fp16)")
endif()

file(GLOB DEPLOY_CPP_DIRS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/*/cpp"
)
file(GLOB DEPLOY_INTERNAL_INCLUDE_DIRS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/*/cpp/include"
)

target_include_directories(vision PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy
    ${DEPLOY_CPP_DIRS}
    ${DEPLOY_INTERNAL_INCLUDE_DIRS}
    Eigen3::Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${OpenCV_INCLUDE_DIRS}
    ${SPACEMIT_DIR}/include
    ${yaml-cpp_INCLUDE_DIR}
)

# Public API: only vision_service.h is exposed; internal headers are not installed.
target_include_directories(vision PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(VISION_WITH_MPP)
    target_compile_definitions(vision PRIVATE
        VISION_HAS_MPP=1
        VISION_MPP_STAGING_LIB="${VISION_MPP_RPATH}"
    )
    target_include_directories(vision PRIVATE ${VISION_MPP_INCLUDE_DIRS})
    target_link_libraries(vision
        "${VISION_MPP_LIB}"
        ${CMAKE_DL_LIBS}
    )
    set_property(TARGET vision APPEND PROPERTY BUILD_RPATH "${VISION_MPP_RPATH}")
    set_property(TARGET vision APPEND PROPERTY INSTALL_RPATH "${VISION_MPP_RPATH}")
endif()

# ============================================================================
# Python Bindings (pybind11 extension: _vision_service_cpp)
# ============================================================================
# Compiles the pybind11 module directly here (no separate bindings CMakeLists).
# The resulting extension and libvision.so are packaged into the spacemit_vision
# wheel by src/python/build_wheel.sh.
#
# Default ON. To keep a pure C++ build working on machines without a Python dev
# environment / pybind11, a missing toolchain only WARNs and skips the module
# instead of failing configuration. Use -DBUILD_PYTHON_BINDINGS=OFF to skip.
if(BUILD_PYTHON_BINDINGS)
    # Locate the pybind binding source (post-migration src/python, or current bindings/).
    set(_pybind_src "")
    foreach(_cand
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python/pybind/vision_service_pybind.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/vision_service_pybind.cpp")
        if(EXISTS "${_cand}")
            set(_pybind_src "${_cand}")
            break()
        endif()
    endforeach()

    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)

    set(_pybind11_cmake_dir "")
    if(Python3_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_query
            ERROR_QUIET
        )
    endif()

    if(NOT _pybind_src)
        message(WARNING "BUILD_PYTHON_BINDINGS=ON but no vision_service_pybind.cpp found; skipping Python bindings.")
    elseif(NOT Python3_FOUND)
        message(WARNING "BUILD_PYTHON_BINDINGS=ON but Python3 development files not found; skipping Python bindings. Install python3-dev or set -DBUILD_PYTHON_BINDINGS=OFF.")
    elseif(NOT _pybind11_query EQUAL 0)
        message(WARNING "BUILD_PYTHON_BINDINGS=ON but pybind11 not found for ${Python3_EXECUTABLE}; skipping Python bindings. Install with: ${Python3_EXECUTABLE} -m pip install pybind11")
    else()
        list(PREPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
        find_package(pybind11 CONFIG REQUIRED)

        pybind11_add_module(_vision_service_cpp "${_pybind_src}")
        target_link_libraries(_vision_service_cpp PRIVATE vision)
        target_include_directories(_vision_service_cpp PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
        )
        # Output next to libvision.so location expectations of build_wheel.sh.
        set_target_properties(_vision_service_cpp PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
            BUILD_RPATH "${CMAKE_BINARY_DIR}"
            INSTALL_RPATH "\$ORIGIN"
        )
        message(STATUS "Python bindings: ${_pybind_src}")
        message(STATUS "  Python:        ${Python3_EXECUTABLE} (${Python3_VERSION})")
        message(STATUS "  Output dir:    ${CMAKE_BINARY_DIR}/python")

        # Convenience target: build the spacemit_vision wheel after the extension
        # is built. Runs src/python/build_wheel.sh, which copies the extension +
        # libvision.so into the package and invokes `python -m build`.
        #
        # With BUILD_PYTHON_WHEEL=ON (default) the wheel is part of the default
        # build, so a plain `cmake --build build` also produces src/python/dist/*.whl
        # — no separate ./build_wheel.sh step needed. Turn it OFF (or build only the
        # `_vision_service_cpp` target) on environments without `python -m build`.
        #   cmake --build build                       # builds C++, ext, and wheel
        #   cmake --build build --target python_wheel # wheel only
        set(_build_wheel_sh "${CMAKE_CURRENT_SOURCE_DIR}/src/python/build_wheel.sh")
        if(EXISTS "${_build_wheel_sh}")
            # Probe whether `python -m build` is available so that missing the
            # packaging frontend doesn't silently break a plain `cmake --build`.
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -m build --version
                RESULT_VARIABLE _build_mod_ok
                OUTPUT_QUIET ERROR_QUIET
            )
            if(BUILD_PYTHON_WHEEL AND _build_mod_ok EQUAL 0)
                set(_wheel_all "ALL")
            else()
                set(_wheel_all "")
                if(BUILD_PYTHON_WHEEL AND NOT _build_mod_ok EQUAL 0)
                    message(WARNING
                        "BUILD_PYTHON_WHEEL=ON but `${Python3_EXECUTABLE} -m build` not found; "
                        "python_wheel target registered but excluded from ALL. "
                        "Install with: ${Python3_EXECUTABLE} -m pip install build")
                endif()
            endif()
            add_custom_target(python_wheel ${_wheel_all}
                COMMAND "${CMAKE_COMMAND}" -E env
                        "PYTHON_BIN=${Python3_EXECUTABLE}"
                        "${_build_wheel_sh}" "${CMAKE_BINARY_DIR}"
                DEPENDS _vision_service_cpp
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/python"
                COMMENT "Building spacemit_vision wheel"
                USES_TERMINAL)
        endif()
    endif()
endif()

# ============================================================================
# Subdirectories
# ============================================================================

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Applications (applications/<name>/cpp/ and config/)
if(BUILD_EXAMPLES)
    add_subdirectory(applications)
endif()

# Tests and benchmarks
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/benchmarks)
    add_subdirectory(tests/cpp)
endif()

# ============================================================================
# Installation (public API: library + vision_service.h only)
# ============================================================================

install(TARGETS vision
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Only expose vision_service.h; internal headers are not installed.
install(FILES include/vision_service.h
    DESTINATION include
)

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "==================== ModelZoo Configuration ====================")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  C++ Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build examples:       ${BUILD_EXAMPLES}")
message(STATUS "  Build tests:           ${BUILD_TESTS}")
message(STATUS "  Build python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build python wheel:    ${BUILD_PYTHON_WHEEL}")
message(STATUS "  OpenCV version:        ${OpenCV_VERSION}")
message(STATUS "  OpenCV libs:           ${OpenCV_LIBS}")
message(STATUS "  SpaceMIT dir:         ${SPACEMIT_DIR}")
message(STATUS "================================================================")
message(STATUS "")