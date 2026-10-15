#!/usr/bin/env bash
# Start exactly one long-lived STM32, lidar, localization, Nav2, and guard stack.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ROOT=${RVCLAW_LAB_ROOT:-$DEFAULT_ROOT}
BASE_ROOT=${RVCLAW_BASE_ROOT:-/data/rvclaw-k3-lab}
LIDAR_SYSROOT=${RVCLAW_LIDAR_SYSROOT:-$BASE_ROOT/deps/sysroot}
LIDAR_LIB_DIR=${RVCLAW_LIDAR_LIB_DIR:-$LIDAR_SYSROOT/usr/lib/riscv64-linux-gnu}
MAP_FILE=${RVCLAW_MAP_FILE:-$ROOT/maps/slam-20260827-105711/map.yaml}
CACHE_FILE=${RVCLAW_LOCALIZATION_CACHE:-$ROOT/state/localization-pose.json}
START_RVIZ=${RVCLAW_AUTO_RVIZ:-false}

if [[ ${1:-} == --describe && $# -eq 1 ]]; then
  cat <<EOF
hardware_io_on_describe=false
runtime_model=one persistent hardware owner, one lidar owner, one Nav2 stack
map=$MAP_FILE
cache=$CACHE_FILE
localization_guard=/rvclaw/localization/status and /rvclaw/localization/valid
lidar_motor_start=automatic bounded retry after driver launch
motion_on_start=false
route_motion_started=false
rviz=$START_RVIZ
EOF
  exit 0
fi

if [[ ${1:-} != --run || $# -ne 1 ]]; then
  printf 'Usage: %s --describe | --run\n' "$0" >&2
  exit 2
fi

if [[ ! -f $MAP_FILE || ! -f $ROOT/ros2_ws/install/setup.bash ]]; then
  echo "persistent runtime requires a built workspace and an existing map" >&2
  exit 3
fi

if [[ ! -e $LIDAR_LIB_DIR/libpcl_common.so.1.15 ]]; then
  echo "persistent runtime requires the lidar PCL runtime at $LIDAR_LIB_DIR" >&2
  exit 5
fi

export LD_LIBRARY_PATH="$LIDAR_LIB_DIR:$LIDAR_SYSROOT/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

set +u
source /opt/ros/humble/setup.bash
if [[ $BASE_ROOT != "$ROOT" && -f $BASE_ROOT/ros2_ws/install/setup.bash ]]; then
  source "$BASE_ROOT/ros2_ws/install/setup.bash"
fi
source "$ROOT/ros2_ws/install/setup.bash"
set -u

existing_nodes=$(timeout 5s ros2 node list 2>/dev/null || true)
for node in /rvclaw_stm32_hardware /x10/lslidar_driver_node /amcl /controller_server /rvclaw_localization_runtime; do
  if grep -Fxq "$node" <<<"$existing_nodes"; then
    echo "refusing duplicate persistent runtime: node already exists: $node" >&2
    exit 4
  fi
done

mkdir -p -- "$(dirname -- "$CACHE_FILE")" "$ROOT/runs/persistent-runtime"
RUN_ID=$(date +%Y%m%d-%H%M%S)
RUN_DIR=$ROOT/runs/persistent-runtime/$RUN_ID
mkdir -p -- "$RUN_DIR"
printf '%s\n' "$MAP_FILE" >"$RUN_DIR/map-file.txt"
printf '%s\n' "$CACHE_FILE" >"$RUN_DIR/cache-file.txt"

start_lidar_motor() {
  local attempt
  for attempt in 1 2 3; do
    if timeout 8s ros2 topic pub --once /x10/motor_control \
      std_msgs/msg/Int8 '{data: 1}' \
      >"$RUN_DIR/lidar-start-topic-attempt-$attempt.log" 2>&1; then
      printf 'lidar motor start published on attempt %s\n' "$attempt" \
        >"$RUN_DIR/lidar-motor-start.result"
      return 0
    fi
    sleep 1
  done
  printf 'lidar motor start publish failed after 3 attempts\n' \
    >"$RUN_DIR/lidar-motor-start.result"
  return 1
}

start_lidar_motor &

exec ros2 launch rvclaw_nav2_bringup persistent_navigation.launch.py \
  map:="$MAP_FILE" cache_file:="$CACHE_FILE" \
  use_sim_time:=false autostart:=true \
  start_hardware:=true start_lidar:=true start_rviz:="$START_RVIZ" \
  laser_x:=0.07725 laser_y:=0.0 laser_z:=0.11144 \
  laser_roll:=0.0 laser_pitch:=0.0 laser_yaw:=0.0 \
  2>&1 | tee "$RUN_DIR/persistent-runtime.log"
