# kaldifst: k2-fsa 维护的 FST 文本规范化运行时。
# 源码拉到 ~/.cache/thirdparty/kaldifst, 内嵌 vendored OpenFst (csukuangfj fork)。
#
# 关键细节:
#  - kaldifst 的 CMake 用 FetchContent 下载 OpenFst 压缩包。离线环境需把
#    openfst-sherpa-onnx-2024-06-13.tar.gz 放到 ~/Downloads/ 或 /tmp/ 下。
#  - CMAKE_POLICY_VERSION_MINIMUM=3.5 必须提前设置, 否则 OpenFst 的老
#    cmake_minimum_required 在 CMake 4.x 下会报错。
#  - TTS_USE_FST=OFF 可禁用整个 FST 路径, text_normalizer 退化为 identity。
#    用于离线环境或 FST 编译失败时保底 SDK build 不中断。

if(DEFINED _KALDIFST_LOADED)
  return()
endif()
set(_KALDIFST_LOADED ON)

option(TTS_USE_FST "Enable FST-based text normalization (kaldifst)" ON)

if(NOT TTS_USE_FST)
  message(STATUS "[tts] TTS_USE_FST=OFF, skipping kaldifst (text normalization will fall back to identity)")
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/FetchThirdParty.cmake")

set(_KALDIFST_GIT_REPO_GITEE  "https://gitee.com/spacemit-robotics/kaldifst.git")
set(_KALDIFST_GIT_REPO_GITHUB "https://github.com/k2-fsa/kaldifst.git")
set(_KALDIFST_GIT_REF         "v1.7.12")

# 先尝试 gitee 镜像 (国内网络友好), 失败则回落到 GitHub 上游。
# 这里手动做 git clone 而不用 fetch_thirdparty, 因为后者在 clone 失败时直接 FATAL_ERROR 没有重试窗口。
if(DEFINED ENV{SROBOTIS_THIRDPARTY_CACHE})
  set(_KALDIFST_CACHE_ROOT "$ENV{SROBOTIS_THIRDPARTY_CACHE}")
elseif(DEFINED ENV{HOME})
  set(_KALDIFST_CACHE_ROOT "$ENV{HOME}/.cache/thirdparty")
else()
  set(_KALDIFST_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/thirdparty")
endif()
set(KALDIFST_DIR "${_KALDIFST_CACHE_ROOT}/kaldifst")

if(NOT EXISTS "${KALDIFST_DIR}")
  file(MAKE_DIRECTORY "${_KALDIFST_CACHE_ROOT}")
  message(STATUS "[kaldifst] Trying gitee mirror: ${_KALDIFST_GIT_REPO_GITEE}")
  execute_process(
    COMMAND git clone --depth 1 "${_KALDIFST_GIT_REPO_GITEE}" -b "${_KALDIFST_GIT_REF}" "${KALDIFST_DIR}"
    RESULT_VARIABLE _clone_res
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT _clone_res EQUAL 0)
    message(STATUS "[kaldifst] gitee failed, falling back to GitHub: ${_KALDIFST_GIT_REPO_GITHUB}")
    file(REMOVE_RECURSE "${KALDIFST_DIR}")
    execute_process(
      COMMAND git clone --depth 1 "${_KALDIFST_GIT_REPO_GITHUB}" -b "${_KALDIFST_GIT_REF}" "${KALDIFST_DIR}"
      RESULT_VARIABLE _clone_res
    )
    if(NOT _clone_res EQUAL 0)
      message(FATAL_ERROR "[kaldifst] failed to clone from both gitee and GitHub")
    endif()
  endif()
endif()

# 把 OpenFst 压缩包预置到 kaldifst 检索路径, 避免 add_subdirectory 时触发 FetchContent 从 GitHub 下载。
set(_OPENFST_TARBALL "openfst-sherpa-onnx-2024-06-13.tar.gz")
set(_OPENFST_ARCHIVE_URL "https://archive.spacemit.com/spacemit-ai/thirdparty/${_OPENFST_TARBALL}")
set(_OPENFST_GITHUB_URL  "https://github.com/csukuangfj/openfst/archive/refs/tags/sherpa-onnx-2024-06-13.tar.gz")
set(_OPENFST_DEST "${CMAKE_BINARY_DIR}/${_OPENFST_TARBALL}")

# kaldifst/cmake/openfst.cmake 的搜索顺序 (按优先级):
#   $ENV{HOME}/Downloads/  <  ${CMAKE_SOURCE_DIR}/  <  ${CMAKE_BINARY_DIR}/  <  /tmp/
# 把 tarball 放到 ${CMAKE_BINARY_DIR}/ 即可被 kaldifst 识别并跳过网络下载。
if(NOT EXISTS "${_OPENFST_DEST}")
  # 依次尝试本地预置位置
  set(_existing_openfst "")
  foreach(_candidate
      "$ENV{HOME}/Downloads/${_OPENFST_TARBALL}"
      "/tmp/${_OPENFST_TARBALL}")
    if(EXISTS "${_candidate}")
      set(_existing_openfst "${_candidate}")
      break()
    endif()
  endforeach()

  if(_existing_openfst)
    message(STATUS "[kaldifst] Copying OpenFst tarball from ${_existing_openfst}")
    file(COPY "${_existing_openfst}" DESTINATION "${CMAKE_BINARY_DIR}")
  else()
    message(STATUS "[kaldifst] Downloading OpenFst tarball from ${_OPENFST_ARCHIVE_URL}")
    file(DOWNLOAD
      "${_OPENFST_ARCHIVE_URL}"
      "${_OPENFST_DEST}"
      STATUS _dl_status
      TLS_VERIFY OFF
      SHOW_PROGRESS)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(STATUS "[kaldifst] archive.spacemit.com download failed, falling back to GitHub: ${_OPENFST_GITHUB_URL}")
      file(REMOVE "${_OPENFST_DEST}")
      file(DOWNLOAD
        "${_OPENFST_GITHUB_URL}"
        "${_OPENFST_DEST}"
        STATUS _dl_status2
        TLS_VERIFY OFF
        SHOW_PROGRESS)
      list(GET _dl_status2 0 _dl_code2)
      if(NOT _dl_code2 EQUAL 0)
        file(REMOVE "${_OPENFST_DEST}")
        message(WARNING "[kaldifst] Failed to pre-download OpenFst tarball. kaldifst will attempt FetchContent directly (may fail offline).")
      endif()
    endif()
  endif()
endif()

# OpenFst 的老 CMakeLists 用了 cmake_minimum_required(2.x), CMake 4.x 会拒绝。
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# 精简 kaldifst 构建选项
set(KALDIFST_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDIFST_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)  # 静态链接, 匹配 libtts.a 的静态库方式

# kaldifst 的 CMakeLists 内部用 ${CMAKE_SOURCE_DIR}/cmake 去 include 'openfst', 当以
# add_subdirectory 方式嵌入时 ${CMAKE_SOURCE_DIR} 指向外层 tts 项目, 找不到 openfst.cmake。
# 手动把 kaldifst/cmake 加到 CMAKE_MODULE_PATH 里绕过。
list(APPEND CMAKE_MODULE_PATH "${KALDIFST_DIR}/cmake" "${KALDIFST_DIR}/cmake/Modules")

add_subdirectory(${KALDIFST_DIR} ${CMAKE_BINARY_DIR}/kaldifst-build EXCLUDE_FROM_ALL)

# kaldifst 的 kaldifst/csrc/CMakeLists.txt 里用 include_directories(${CMAKE_SOURCE_DIR})
# 来支持源文件里的 #include "kaldifst/csrc/xxx.h", 当以 add_subdirectory 嵌入时 ${CMAKE_SOURCE_DIR}
# 指向外层 tts 项目, 这些 include 找不到 header。对 kaldifst_core target 补回正确路径。
if(TARGET kaldifst_core)
  target_include_directories(kaldifst_core PRIVATE "${KALDIFST_DIR}")
  set_target_properties(kaldifst_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(TARGET fst)
  set_target_properties(fst PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# 对外暴露 kaldifst 的 public header 搜索路径 (消费者以 #include "kaldifst/csrc/text-normalizer.h" 方式引用)
# OpenFst 由 kaldifst 的 FetchContent 拉到 ${CMAKE_BINARY_DIR}/_deps/openfst-src/, 不在 kaldifst-build 子目录下。
set(KALDIFST_INCLUDE_DIRS
  "${KALDIFST_DIR}"
  "${CMAKE_BINARY_DIR}/_deps/openfst-src/src/include"
  CACHE INTERNAL "kaldifst include dirs"
)
