# FetchQwen3AsrModel.cmake - Fetch Qwen3-ASR model to cache
# Sets: QWEN3_ASR_MODEL_DIR
# Note: This model is used by llama-server, not directly by the ASR library

if(DEFINED _FETCH_QWEN3_ASR_MODEL_LOADED)
  return()
endif()
set(_FETCH_QWEN3_ASR_MODEL_LOADED ON)

set(_QWEN3_ASR_MODEL_URL "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/qwen3-asr-1.7B-dynq-q40.tar.gz")
set(_QWEN3_ASR_MODEL_NAME "qwen3-asr-1.7B-dynq-q40")
set(_QWEN3_ASR_ARCHIVE_SUBDIR "qwen3-asr-1.7B-dynq-q40")
set(_QWEN3_ASR_REQUIRED_FILES
  "Qwen3-ASR-1.7B-text-q40.gguf"
  "Qwen3-ASR-1.7B-encoder-frontend.dynq.onnx"
  "Qwen3-ASR-1.7B-encoder-backend.dynq.onnx"
  "Qwen3-ASR-1.7B-encoder-split-metadata.json"
  "config.json"
)

# Cache root
if(DEFINED ENV{HOME})
  set(_QWEN3_ASR_CACHE_ROOT "$ENV{HOME}/.cache/models/asr")
else()
  set(_QWEN3_ASR_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/models/asr")
endif()

set(_QWEN3_ASR_MODEL_DIR "${_QWEN3_ASR_CACHE_ROOT}/${_QWEN3_ASR_MODEL_NAME}")

# Check if model exists
set(_need_download OFF)
foreach(_file IN LISTS _QWEN3_ASR_REQUIRED_FILES)
  if(NOT EXISTS "${_QWEN3_ASR_MODEL_DIR}/${_file}")
    set(_need_download ON)
    break()
  endif()
endforeach()

if(_need_download)
  if(DEFINED ASR_MODEL_FETCH_OFF AND ASR_MODEL_FETCH_OFF)
    message(WARNING "Qwen3-ASR model not found at ${_QWEN3_ASR_MODEL_DIR}, fetch disabled (ASR_MODEL_FETCH_OFF). Will attempt runtime download.")
    return()
  endif()

  message(STATUS "Fetching Qwen3-ASR model to ${_QWEN3_ASR_MODEL_DIR} ...")
  file(MAKE_DIRECTORY "${_QWEN3_ASR_MODEL_DIR}")

  set(_archive_path "${_QWEN3_ASR_MODEL_DIR}/${_QWEN3_ASR_MODEL_NAME}.tar.gz")

  file(DOWNLOAD
    "${_QWEN3_ASR_MODEL_URL}"
    "${_archive_path}"
    SHOW_PROGRESS
    STATUS _download_status
    TLS_VERIFY OFF
  )

  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_error)
    message(FATAL_ERROR "Failed to download Qwen3-ASR model: ${_download_error}")
  endif()

  message(STATUS "Extracting Qwen3-ASR model...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_archive_path}"
    WORKING_DIRECTORY "${_QWEN3_ASR_MODEL_DIR}"
    RESULT_VARIABLE _extract_result
  )

  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract Qwen3-ASR model")
  endif()

  # Move files from subdirectory if exists
  if(EXISTS "${_QWEN3_ASR_MODEL_DIR}/${_QWEN3_ASR_ARCHIVE_SUBDIR}")
    file(GLOB _subdir_files "${_QWEN3_ASR_MODEL_DIR}/${_QWEN3_ASR_ARCHIVE_SUBDIR}/*")
    foreach(_file IN LISTS _subdir_files)
      get_filename_component(_filename "${_file}" NAME)
      file(RENAME "${_file}" "${_QWEN3_ASR_MODEL_DIR}/${_filename}")
    endforeach()
    file(REMOVE_RECURSE "${_QWEN3_ASR_MODEL_DIR}/${_QWEN3_ASR_ARCHIVE_SUBDIR}")
  endif()

  file(REMOVE "${_archive_path}")
  message(STATUS "Qwen3-ASR model ready at ${_QWEN3_ASR_MODEL_DIR}")
else()
  message(STATUS "Qwen3-ASR model found at ${_QWEN3_ASR_MODEL_DIR}")
endif()

# Export
set(QWEN3_ASR_MODEL_DIR "${_QWEN3_ASR_MODEL_DIR}" CACHE PATH "Qwen3-ASR model directory" FORCE)
