# cpp-pinyin: Chinese pinyin conversion library.
# Fetched to ~/.cache/thirdparty/cpp-pinyin at configure time.

if(DEFINED _CPP_PINYIN_LOADED)
  return()
endif()
set(_CPP_PINYIN_LOADED ON)

include("${CMAKE_CURRENT_LIST_DIR}/FetchThirdParty.cmake")

set(_CPP_PINYIN_GIT_REPO "https://gitee.com/spacemit-robotics/cpp-pinyin.git")

fetch_thirdparty(NAME cpp-pinyin
  GIT_REPO "${_CPP_PINYIN_GIT_REPO}"
  OUT_SOURCE_DIR CPP_PINYIN_DIR)
