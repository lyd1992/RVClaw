#!/usr/bin/env bash
set -eo pipefail
# ROS setup files reference optional variables; tolerate inherited nounset
# while sourcing them, then restore strict undefined-variable checking.
set +u

ROOT="${RVCLAW_LAB_ROOT:-/data/rvclaw-k3-lab}"

if [[ "${1:-}" == "--describe" && $# -eq 1 ]]; then
  cat <<EOF
hardware_io=false
motion_commands=false
display=RViz2 via X11 forwarding
launch=rvclaw_nav2_bringup/visualization.launch.py
config=${ROOT}/ros2_ws/install/rvclaw_nav2_bringup/share/rvclaw_nav2_bringup/rviz/rvclaw_navigation.rviz
required_environment=DISPLAY or WAYLAND_DISPLAY
EOF
  exit 0
fi

if [[ "${1:-}" != "--run" || $# -ne 1 ]]; then
  printf 'Usage: %s --describe | --run\n' "$0" >&2
  exit 2
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  cat >&2 <<'EOF'
No graphical display is available in this SSH session.
Reconnect with MobaXterm X11 forwarding enabled, then run this command again.
The visualization launch is read-only and does not start LiDAR, Nav2, STM32,
or any velocity publisher.
EOF
  exit 3
fi

source /opt/ros/humble/setup.bash
source "${ROOT}/ros2_ws/install/setup.bash"
if ! command -v rviz2 >/dev/null 2>&1; then
  printf 'rviz2 is not installed. Install ros-humble-rviz2 after approval.\n' >&2
  exit 4
fi
set -u
exec ros2 launch rvclaw_nav2_bringup visualization.launch.py
