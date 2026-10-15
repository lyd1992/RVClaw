# cppjieba: Chinese word segmentation library.
# Fetched to ~/.cache/thirdparty/cppjieba at configure time.
# Requires --recursive for limonp submodule.

if(DEFINED _CPPJIEBA_LOADED)
  return()
endif()
set(_CPPJIEBA_LOADED ON)

include("${CMAKE_CURRENT_LIST_DIR}/FetchThirdParty.cmake")

set(_CPPJIEBA_GIT_REPO "https://gitee.com/spacemit-robotics/cppjieba.git")

fetch_thirdparty(NAME cppjieba
  GIT_REPO "${_CPPJIEBA_GIT_REPO}"
  GIT_RECURSIVE
  OUT_SOURCE_DIR CPPJIEBA_DIR)
