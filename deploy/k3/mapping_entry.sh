#!/usr/bin/env bash
# Voice supplies the mission root; mapping needs its own unchanged profile.
set -eo pipefail
ENGINEERING_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export RVCLAW_LAB_ROOT="$ENGINEERING_ROOT/runtime/k3/mapping"
export RVCLAW_BASE_ROOT="$ENGINEERING_ROOT/runtime/k3/base"
export RVCLAW_ROS_WS="$ENGINEERING_ROOT/robot/profiles/mapping/ros2_ws"
export RVCLAW_LIDAR_SYSROOT="$ENGINEERING_ROOT/runtime/k3/base/deps/sysroot"
exec bash "$RVCLAW_LAB_ROOT/scripts/slam_mapping_session.sh" "$@"
