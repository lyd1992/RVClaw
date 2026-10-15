# kaldi_native_fbank.cmake
# csukuangfj/kaldi-native-fbank for Zipformer feature extraction.
# Static link 进 _spacemit_asr.so, 消除 libkaldi-native-fbank-core.so 运行时依赖.
#
# 两层 GitHub 依赖, 都做了 gitee/archive 镜像兜底:
#  1. knf 主仓库: gitee mirror -> GitHub fallback (git clone)
#  2. knf 内部 cmake/kissfft.cmake 用 FetchContent 拉 kissfft zip
#     (URL_HASH 锁 commit febd4ca...). 这里把 zip 预下载到 ${CMAKE_BINARY_DIR}/,
#     命中 knf 的 possible_file_locations 逃生口 (knf cmake/kissfft.cmake:15-21),
#     跳过它的 FetchContent 网络访问.
#
# 镜像准备 (一次性, 由维护者完成):
#  1. fork csukuangfj/kaldi-native-fbank -> gitee.com/spacemit-robotics/kaldi-native-fbank
#  2. wget https://github.com/mborgerding/kissfft/archive/febd4caeed32e33ad8b2e0bb5ea77542c40f18ec.zip
#     直接转存到 archive.spacemit.com/spacemit-ai/thirdparty/
#     (字节必须与 GitHub 原始 zip 完全一致, knf 的 URL_HASH 校验会拒绝改过的 zip).

if(DEFINED _KALDI_NATIVE_FBANK_LOADED)
  return()
endif()
set(_KALDI_NATIVE_FBANK_LOADED ON)

set(_KNF_GIT_REPO_GITEE  "https://gitee.com/spacemit-robotics/kaldi-native-fbank.git")
set(_KNF_GIT_REPO_GITHUB "https://github.com/csukuangfj/kaldi-native-fbank.git")
# TODO: 首次成功 build 后 pin 一个上游 tag (如 v1.21.x), master 不可重现
set(_KNF_GIT_REF "master")

if(DEFINED ENV{SROBOTIS_THIRDPARTY_CACHE})
  set(_KNF_CACHE_ROOT "$ENV{SROBOTIS_THIRDPARTY_CACHE}")
elseif(DEFINED ENV{HOME})
  set(_KNF_CACHE_ROOT "$ENV{HOME}/.cache/thirdparty")
else()
  set(_KNF_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/thirdparty")
endif()
set(KNF_DIR "${_KNF_CACHE_ROOT}/kaldi-native-fbank")

if(NOT EXISTS "${KNF_DIR}")
  file(MAKE_DIRECTORY "${_KNF_CACHE_ROOT}")
  message(STATUS "[kaldi-native-fbank] Trying gitee mirror: ${_KNF_GIT_REPO_GITEE}")
  execute_process(
    COMMAND git clone --depth 1 "${_KNF_GIT_REPO_GITEE}" -b "${_KNF_GIT_REF}" "${KNF_DIR}"
    RESULT_VARIABLE _knf_clone_res
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT _knf_clone_res EQUAL 0)
    message(STATUS "[kaldi-native-fbank] gitee failed, falling back to GitHub: ${_KNF_GIT_REPO_GITHUB}")
    file(REMOVE_RECURSE "${KNF_DIR}")
    execute_process(
      COMMAND git clone --depth 1 "${_KNF_GIT_REPO_GITHUB}" -b "${_KNF_GIT_REF}" "${KNF_DIR}"
      RESULT_VARIABLE _knf_clone_res
    )
    if(NOT _knf_clone_res EQUAL 0)
      file(REMOVE_RECURSE "${KNF_DIR}")
      message(FATAL_ERROR "[kaldi-native-fbank] failed to clone from both gitee and GitHub")
    endif()
  endif()
endif()

# 预下载 kissfft zip, 让 knf 的 kissfft.cmake 命中 ${CMAKE_BINARY_DIR}/<zip>
# 文件名 / SHA256 必须严格匹配 knf cmake/kissfft.cmake L9-11
set(_KISSFFT_COMMIT "febd4caeed32e33ad8b2e0bb5ea77542c40f18ec")
set(_KISSFFT_ZIP    "kissfft-${_KISSFFT_COMMIT}.zip")
set(_KISSFFT_SHA256 "497103e664168ebe39580b757adbe616f6cf85a16572af581ca7bc42d0ab13fd")
set(_KISSFFT_ARCHIVE_URL "https://archive.spacemit.com/spacemit-ai/thirdparty/${_KISSFFT_ZIP}")
set(_KISSFFT_GITHUB_URL  "https://github.com/mborgerding/kissfft/archive/${_KISSFFT_COMMIT}.zip")
set(_KISSFFT_DEST "${CMAKE_BINARY_DIR}/${_KISSFFT_ZIP}")

if(NOT EXISTS "${_KISSFFT_DEST}")
  set(_existing_kissfft "")
  foreach(_candidate
      "$ENV{HOME}/Downloads/${_KISSFFT_ZIP}"
      "/tmp/${_KISSFFT_ZIP}")
    if(EXISTS "${_candidate}")
      set(_existing_kissfft "${_candidate}")
      break()
    endif()
  endforeach()

  if(_existing_kissfft)
    message(STATUS "[kaldi-native-fbank] Copying kissfft zip from ${_existing_kissfft}")
    file(COPY "${_existing_kissfft}" DESTINATION "${CMAKE_BINARY_DIR}")
  else()
    message(STATUS "[kaldi-native-fbank] Downloading kissfft from ${_KISSFFT_ARCHIVE_URL}")
    file(DOWNLOAD
      "${_KISSFFT_ARCHIVE_URL}"
      "${_KISSFFT_DEST}"
      EXPECTED_HASH SHA256=${_KISSFFT_SHA256}
      STATUS _kissfft_dl_status
      TLS_VERIFY OFF
      SHOW_PROGRESS)
    list(GET _kissfft_dl_status 0 _kissfft_dl_code)
    if(NOT _kissfft_dl_code EQUAL 0)
      message(STATUS "[kaldi-native-fbank] archive.spacemit.com failed, falling back to GitHub: ${_KISSFFT_GITHUB_URL}")
      file(REMOVE "${_KISSFFT_DEST}")
      file(DOWNLOAD
        "${_KISSFFT_GITHUB_URL}"
        "${_KISSFFT_DEST}"
        EXPECTED_HASH SHA256=${_KISSFFT_SHA256}
        STATUS _kissfft_dl_status2
        TLS_VERIFY OFF
        SHOW_PROGRESS)
      list(GET _kissfft_dl_status2 0 _kissfft_dl_code2)
      if(NOT _kissfft_dl_code2 EQUAL 0)
        file(REMOVE "${_KISSFFT_DEST}")
        message(WARNING "[kaldi-native-fbank] Failed to pre-download kissfft zip. knf will attempt FetchContent directly (may fail offline).")
      endif()
    endif()
  endif()
endif()

# 静态构建 knf, 链进 _spacemit_asr.so; 保存/恢复 BUILD_SHARED_LIBS 不污染外层
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(_KNF_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${KNF_DIR} ${CMAKE_BINARY_DIR}/kaldi-native-fbank-build)

set(BUILD_SHARED_LIBS ${_KNF_SAVED_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

set(KALDI_FBANK_INCLUDE_DIRS
  "${KNF_DIR}"
  "${KNF_DIR}/kaldi-native-fbank/csrc"
  CACHE INTERNAL "kaldi-native-fbank include dirs"
)

# 所有 knf / kissfft 静态库需 PIC 才能链进 shared library
foreach(_knf_pic_target
    kaldi-native-fbank-core
    kissfft-float
    kissfft-double
    kissfft-int16_t
    kissfft)
  if(TARGET ${_knf_pic_target})
    set_target_properties(${_knf_pic_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endforeach()
