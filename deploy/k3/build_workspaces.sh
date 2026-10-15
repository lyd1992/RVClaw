#!/usr/bin/env bash
# Build K3 artifacts in the engineering tree without loading the old workspace.
set -eo pipefail
ENGINEERING_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source /opt/ros/humble/setup.bash
export RVCLAW_LIDAR_SYSROOT="$ENGINEERING_ROOT/runtime/k3/base/deps/sysroot"
export CPATH="$RVCLAW_LIDAR_SYSROOT/usr/include/pcl-1.15:$RVCLAW_LIDAR_SYSROOT/usr/include:$RVCLAW_LIDAR_SYSROOT/opt/ros/humble/include/pcl_conversions:$RVCLAW_LIDAR_SYSROOT/opt/ros/humble/include/pcl_msgs:/opt/ros/humble/include/message_filters${CPATH:+:$CPATH}"
export LIBRARY_PATH="$RVCLAW_LIDAR_SYSROOT/usr/lib/riscv64-linux-gnu:$RVCLAW_LIDAR_SYSROOT/opt/ros/humble/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$RVCLAW_LIDAR_SYSROOT/usr:$RVCLAW_LIDAR_SYSROOT/opt/ros/humble${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
cd "$ENGINEERING_ROOT/robot/ros2_ws"
colcon build --base-paths src \
  --packages-select rvclaw_stm32_base lslidar_msgs lslidar_driver rvclaw_nav2_bringup \
  --executor sequential --cmake-args \
  "-DPCL_DIR=$RVCLAW_LIDAR_SYSROOT/usr/lib/riscv64-linux-gnu/cmake/pcl" \
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
source "$ENGINEERING_ROOT/robot/ros2_ws/install/setup.bash"
for profile in navigation mapping; do
  cd "$ENGINEERING_ROOT/robot/profiles/$profile/ros2_ws"
  colcon build --base-paths src --packages-select rvclaw_nav2_bringup \
    --executor sequential
done
