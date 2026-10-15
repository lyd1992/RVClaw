# FetchZipformerModel.cmake - Fetch Zipformer model to cache
# Sets: ZIPFORMER_MODEL_DIR

if(DEFINED _FETCH_ZIPFORMER_MODEL_LOADED)
  return()
endif()
set(_FETCH_ZIPFORMER_MODEL_LOADED ON)

set(_ZIPFORMER_MODEL_URL "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/zipformer.tar.gz")
set(_ZIPFORMER_MODEL_NAME "zipformer")
set(_ZIPFORMER_ARCHIVE_SUBDIR "zipformer")
set(_ZIPFORMER_REQUIRED_FILES "ctc-epoch-20-avg-1-chunk-16-left-128.q.onnx;tokens.txt")

# Cache root
if(DEFINED ENV{HOME})
  set(_ZIPFORMER_CACHE_ROOT "$ENV{HOME}/.cache/models/asr")
else()
  set(_ZIPFORMER_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/models/asr")
endif()

set(_ZIPFORMER_MODEL_DIR "${_ZIPFORMER_CACHE_ROOT}/${_ZIPFORMER_MODEL_NAME}")

# Check if model exists
set(_need_download OFF)
foreach(_file IN LISTS _ZIPFORMER_REQUIRED_FILES)
  if(NOT EXISTS "${_ZIPFORMER_MODEL_DIR}/${_file}")
    set(_need_download ON)
    break()
  endif()
endforeach()

if(_need_download)
  if(DEFINED ASR_MODEL_FETCH_OFF AND ASR_MODEL_FETCH_OFF)
    message(WARNING "Zipformer model not found at ${_ZIPFORMER_MODEL_DIR}, fetch disabled (ASR_MODEL_FETCH_OFF). Will attempt runtime download.")
    return()
  endif()

  message(STATUS "Fetching Zipformer model to ${_ZIPFORMER_MODEL_DIR} ...")
  file(MAKE_DIRECTORY "${_ZIPFORMER_MODEL_DIR}")

  set(_archive_path "${_ZIPFORMER_MODEL_DIR}/${_ZIPFORMER_MODEL_NAME}.tar.gz")

  file(DOWNLOAD
    "${_ZIPFORMER_MODEL_URL}"
    "${_archive_path}"
    SHOW_PROGRESS
    STATUS _download_status
    TLS_VERIFY OFF
  )

  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_error)
    message(FATAL_ERROR "Failed to download Zipformer model: ${_download_error}")
  endif()

  message(STATUS "Extracting Zipformer model...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_archive_path}"
    WORKING_DIRECTORY "${_ZIPFORMER_MODEL_DIR}"
    RESULT_VARIABLE _extract_result
  )

  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract Zipformer model")
  endif()

  # Move files from subdirectory if exists
  if(EXISTS "${_ZIPFORMER_MODEL_DIR}/${_ZIPFORMER_ARCHIVE_SUBDIR}")
    file(GLOB _subdir_files "${_ZIPFORMER_MODEL_DIR}/${_ZIPFORMER_ARCHIVE_SUBDIR}/*")
    foreach(_file IN LISTS _subdir_files)
      get_filename_component(_filename "${_file}" NAME)
      file(RENAME "${_file}" "${_ZIPFORMER_MODEL_DIR}/${_filename}")
    endforeach()
    file(REMOVE_RECURSE "${_ZIPFORMER_MODEL_DIR}/${_ZIPFORMER_ARCHIVE_SUBDIR}")
  endif()

  file(REMOVE "${_archive_path}")
  message(STATUS "Zipformer model ready at ${_ZIPFORMER_MODEL_DIR}")
else()
  message(STATUS "Zipformer model found at ${_ZIPFORMER_MODEL_DIR}")
endif()

# Export
set(ZIPFORMER_MODEL_DIR "${_ZIPFORMER_MODEL_DIR}" CACHE PATH "Zipformer model directory" FORCE)
