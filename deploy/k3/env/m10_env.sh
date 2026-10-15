#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Usage: source /data/rvclaw-k3-lab/env/m10_env.sh\n' >&2
  exit 1
fi

export RVCLAW_LAB_ROOT="${RVCLAW_LAB_ROOT:-/data/rvclaw-k3-lab}"
export RVCLAW_ROS_WS="${RVCLAW_ROS_WS:-${RVCLAW_LAB_ROOT}/ros2_ws}"
export RVCLAW_LIDAR_SYSROOT="${RVCLAW_LIDAR_SYSROOT:-${RVCLAW_LAB_ROOT}/deps/sysroot}"

for required_path in \
  /opt/ros/humble/setup.bash \
  "${RVCLAW_ROS_WS}/install/setup.bash" \
  "${RVCLAW_LIDAR_SYSROOT}/usr/include/pcl-1.15"; do
  if [[ ! -r "${required_path}" ]]; then
    printf '[m10-env] Missing required path: %s\n' "${required_path}" >&2
    return 1
  fi
done

_m10_prepend_path() {
  local variable="$1"
  local value="$2"
  local current="${!variable:-}"

  case ":${current}:" in
    *":${value}:"*) ;;
    *)
      if [[ -n "${current}" ]]; then
        printf -v "${variable}" '%s:%s' "${value}" "${current}"
      else
        printf -v "${variable}" '%s' "${value}"
      fi
      export "${variable}"
      ;;
  esac
}

_m10_nounset_enabled=0
case "$-" in
  *u*)
    _m10_nounset_enabled=1
    set +u
    ;;
esac

source /opt/ros/humble/setup.bash

_m10_prepend_path CPATH "${RVCLAW_LIDAR_SYSROOT}/usr/include"
_m10_prepend_path CPATH "${RVCLAW_LIDAR_SYSROOT}/usr/include/pcl-1.15"
_m10_prepend_path CPATH "${RVCLAW_LIDAR_SYSROOT}/opt/ros/humble/include/pcl_conversions"
_m10_prepend_path CPATH "${RVCLAW_LIDAR_SYSROOT}/opt/ros/humble/include/pcl_msgs"
_m10_prepend_path LIBRARY_PATH "${RVCLAW_LIDAR_SYSROOT}/usr/lib/riscv64-linux-gnu"
_m10_prepend_path LD_LIBRARY_PATH "${RVCLAW_LIDAR_SYSROOT}/usr/lib/riscv64-linux-gnu"
_m10_prepend_path LD_LIBRARY_PATH "${RVCLAW_LIDAR_SYSROOT}/opt/ros/humble/lib"
_m10_prepend_path PKG_CONFIG_PATH "${RVCLAW_LIDAR_SYSROOT}/usr/lib/riscv64-linux-gnu/pkgconfig"
_m10_prepend_path CMAKE_PREFIX_PATH "${RVCLAW_LIDAR_SYSROOT}/usr"
_m10_prepend_path CMAKE_PREFIX_PATH "${RVCLAW_LIDAR_SYSROOT}/opt/ros/humble"

source "${RVCLAW_ROS_WS}/install/setup.bash"

if [[ ${_m10_nounset_enabled} -eq 1 ]]; then
  set -u
fi

printf '[m10-env] ROS=%s workspace=%s lidar_sysroot=%s\n' \
  "${ROS_DISTRO:-unset}" "${RVCLAW_ROS_WS}" "${RVCLAW_LIDAR_SYSROOT}"

unset _m10_nounset_enabled
unset -f _m10_prepend_path
