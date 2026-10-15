# FetchSenseVoiceModel.cmake - Fetch SenseVoice model to cache
# Sets: SENSEVOICE_MODEL_DIR

if(DEFINED _FETCH_SENSEVOICE_MODEL_LOADED)
  return()
endif()
set(_FETCH_SENSEVOICE_MODEL_LOADED ON)

set(_SENSEVOICE_MODEL_URL "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/sensevoice.tar.gz")
set(_SENSEVOICE_MODEL_NAME "sensevoice")
set(_SENSEVOICE_ARCHIVE_SUBDIR "sensevoice")
set(_SENSEVOICE_REQUIRED_FILES "model_quant_optimized.onnx;tokens.txt;am.mvn")

# Cache root
if(DEFINED ENV{HOME})
  set(_SENSEVOICE_CACHE_ROOT "$ENV{HOME}/.cache/models/asr")
else()
  set(_SENSEVOICE_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/models/asr")
endif()

set(_SENSEVOICE_MODEL_DIR "${_SENSEVOICE_CACHE_ROOT}/${_SENSEVOICE_MODEL_NAME}")

# Check if model exists
set(_need_download OFF)
foreach(_file IN LISTS _SENSEVOICE_REQUIRED_FILES)
  if(NOT EXISTS "${_SENSEVOICE_MODEL_DIR}/${_file}")
    set(_need_download ON)
    break()
  endif()
endforeach()

if(_need_download)
  if(DEFINED ASR_MODEL_FETCH_OFF AND ASR_MODEL_FETCH_OFF)
    message(WARNING "SenseVoice model not found at ${_SENSEVOICE_MODEL_DIR}, fetch disabled (ASR_MODEL_FETCH_OFF). Will attempt runtime download.")
    return()
  endif()

  message(STATUS "Fetching SenseVoice model to ${_SENSEVOICE_MODEL_DIR} ...")
  file(MAKE_DIRECTORY "${_SENSEVOICE_MODEL_DIR}")

  set(_archive_path "${_SENSEVOICE_MODEL_DIR}/${_SENSEVOICE_MODEL_NAME}.tar.gz")

  file(DOWNLOAD
    "${_SENSEVOICE_MODEL_URL}"
    "${_archive_path}"
    SHOW_PROGRESS
    STATUS _download_status
    TLS_VERIFY OFF
  )

  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_error)
    message(FATAL_ERROR "Failed to download SenseVoice model: ${_download_error}")
  endif()

  message(STATUS "Extracting SenseVoice model...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_archive_path}"
    WORKING_DIRECTORY "${_SENSEVOICE_MODEL_DIR}"
    RESULT_VARIABLE _extract_result
  )

  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract SenseVoice model")
  endif()

  # Move files from subdirectory if exists
  if(EXISTS "${_SENSEVOICE_MODEL_DIR}/${_SENSEVOICE_ARCHIVE_SUBDIR}")
    file(GLOB _subdir_files "${_SENSEVOICE_MODEL_DIR}/${_SENSEVOICE_ARCHIVE_SUBDIR}/*")
    foreach(_file IN LISTS _subdir_files)
      get_filename_component(_filename "${_file}" NAME)
      file(RENAME "${_file}" "${_SENSEVOICE_MODEL_DIR}/${_filename}")
    endforeach()
    file(REMOVE_RECURSE "${_SENSEVOICE_MODEL_DIR}/${_SENSEVOICE_ARCHIVE_SUBDIR}")
  endif()

  file(REMOVE "${_archive_path}")
  message(STATUS "SenseVoice model ready at ${_SENSEVOICE_MODEL_DIR}")
else()
  message(STATUS "SenseVoice model found at ${_SENSEVOICE_MODEL_DIR}")
endif()

# Export
set(SENSEVOICE_MODEL_DIR "${_SENSEVOICE_MODEL_DIR}" CACHE PATH "SenseVoice model directory" FORCE)
