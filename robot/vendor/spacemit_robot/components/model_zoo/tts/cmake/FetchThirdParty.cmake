# FetchThirdParty.cmake - Fetch third-party source to cache (component-owned for standalone build).
# Usage: include() from component root, then call fetch_thirdparty(...).

if(DEFINED _FETCH_THIRDPARTY_LOADED)
  return()
endif()
set(_FETCH_THIRDPARTY_LOADED ON)

# Cache root: env or default ~/.cache/thirdparty
if(DEFINED ENV{SROBOTIS_THIRDPARTY_CACHE})
  set(_FETCH_TP_CACHE_ROOT "$ENV{SROBOTIS_THIRDPARTY_CACHE}")
elseif(DEFINED ENV{HOME})
  set(_FETCH_TP_CACHE_ROOT "$ENV{HOME}/.cache/thirdparty")
else()
  set(_FETCH_TP_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/thirdparty")
endif()

# Optional: disable fetch (offline / use pre-populated cache only)
if(DEFINED SROBOTIS_THIRDPARTY_FETCH_OFF)
  set(_FETCH_TP_DISABLED "${SROBOTIS_THIRDPARTY_FETCH_OFF}")
else()
  set(_FETCH_TP_DISABLED OFF)
endif()

function(fetch_thirdparty)
  cmake_parse_arguments(ARG "GIT_RECURSIVE" "NAME;GIT_REPO;GIT_REF;GIT_COMMIT;SUBDIR;OUT_SOURCE_DIR" "" ${ARGN})
  if(NOT ARG_NAME OR NOT ARG_GIT_REPO)
    message(FATAL_ERROR "fetch_thirdparty: NAME and GIT_REPO are required")
  endif()
  set(_dir "${_FETCH_TP_CACHE_ROOT}/${ARG_NAME}")
  if(ARG_SUBDIR)
    set(_check_path "${_dir}/${ARG_SUBDIR}")
  else()
    set(_check_path "${_dir}")
  endif()

  set(_need_clone OFF)
  if(NOT EXISTS "${_dir}")
    set(_need_clone ON)
  endif()
  if(_need_clone AND _FETCH_TP_DISABLED)
    message(FATAL_ERROR "fetch_thirdparty: ${ARG_NAME} not found at ${_dir} and fetch is disabled (SROBOTIS_THIRDPARTY_FETCH_OFF)")
  endif()

  # Build clone flags
  set(_recursive_flag "")
  if(ARG_GIT_RECURSIVE)
    set(_recursive_flag "--recursive")
  endif()

  if(_need_clone)
    message(STATUS "Fetching ${ARG_NAME} to ${_dir} ...")
    file(MAKE_DIRECTORY "${_FETCH_TP_CACHE_ROOT}")
    if(ARG_GIT_COMMIT)
      execute_process(
        COMMAND git clone --depth 1 ${_recursive_flag} "${ARG_GIT_REPO}" "${_dir}"
        WORKING_DIRECTORY "${_FETCH_TP_CACHE_ROOT}"
        RESULT_VARIABLE _res
      )
      if(_res EQUAL 0)
        execute_process(COMMAND git checkout "${ARG_GIT_COMMIT}" WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _res)
      endif()
    elseif(ARG_GIT_REF)
      execute_process(
        COMMAND git clone --depth 1 ${_recursive_flag} "${ARG_GIT_REPO}" -b "${ARG_GIT_REF}" "${_dir}"
        WORKING_DIRECTORY "${_FETCH_TP_CACHE_ROOT}"
        RESULT_VARIABLE _res
      )
    else()
      # No GIT_REF or GIT_COMMIT: clone default branch
      execute_process(
        COMMAND git clone --depth 1 ${_recursive_flag} "${ARG_GIT_REPO}" "${_dir}"
        WORKING_DIRECTORY "${_FETCH_TP_CACHE_ROOT}"
        RESULT_VARIABLE _res
      )
    endif()
    if(NOT _res EQUAL 0)
      message(FATAL_ERROR "fetch_thirdparty: failed to clone ${ARG_NAME} (${_res})")
    endif()
  elseif(NOT _FETCH_TP_DISABLED)
    if(ARG_GIT_COMMIT)
      execute_process(COMMAND git fetch --depth 1 origin "${ARG_GIT_COMMIT}" WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _res)
      if(_res EQUAL 0)
        execute_process(COMMAND git checkout --detach FETCH_HEAD WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _res)
      endif()
      if(NOT _res EQUAL 0)
        message(WARNING "fetch_thirdparty: failed to update ${ARG_NAME} (${_res}), using existing cache")
      endif()
    elseif(ARG_GIT_REF)
      execute_process(COMMAND git fetch --depth 1 origin "${ARG_GIT_REF}" WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _res)
      if(_res EQUAL 0)
        execute_process(COMMAND git checkout -B "${ARG_GIT_REF}" FETCH_HEAD WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _res)
      endif()
      if(NOT _res EQUAL 0)
        message(WARNING "fetch_thirdparty: failed to update ${ARG_NAME} (${_res}), using existing cache")
      endif()
    endif()
    # No GIT_REF or GIT_COMMIT: skip update, use existing cache as-is
  endif()

  if(NOT EXISTS "${_check_path}")
    message(FATAL_ERROR "fetch_thirdparty: path does not exist: ${_check_path}")
  endif()

  if(ARG_OUT_SOURCE_DIR)
    if(ARG_SUBDIR)
      set(${ARG_OUT_SOURCE_DIR} "${_dir}/${ARG_SUBDIR}" PARENT_SCOPE)
    else()
      set(${ARG_OUT_SOURCE_DIR} "${_dir}" PARENT_SCOPE)
    endif()
  endif()
endfunction()
