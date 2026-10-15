#!/usr/bin/env bash
# shellcheck shell=bash

set -eo pipefail

ROOT=/data/rvclaw-k3-lab
STM32_DEVICE=/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00
LIDAR_DEVICE=/dev/wheeltec_lidar
LIDAR_BAUD=500000
PROFILE=${RVCLAW_HEADING_PROFILE:-spin_left_90}
AUTO_RVIZ=${RVCLAW_AUTO_RVIZ:-auto}
MAP_FILE=${RVCLAW_MAP_FILE:-$ROOT/maps/slam-20260827-105711/map.yaml}
export RVCLAW_MAP_FILE="$MAP_FILE"
case "$PROFILE" in
  spin_left_90)
    ARM_TOKEN=WHEELS_GROUND_PS2_OFF_NAV2_SPIN_LEFT_90
    CLIENT=$ROOT/scripts/nav2_heading_spin_left_90.py
    CLIENT_TIMEOUT_S=30
    PASS_MARKER=NAV2_HEADING_SPIN_LEFT_90_PASS
    ;;
  spin_left_0350)
    ARM_TOKEN=WHEELS_GROUND_PS2_OFF_NAV2_SPIN_LEFT_0350
    CLIENT=$ROOT/scripts/nav2_heading_spin_left_0350.py
    CLIENT_TIMEOUT_S=20
    PASS_MARKER=NAV2_HEADING_SPIN_LEFT_PASS
    ;;
  global_localization_scan)
    ARM_TOKEN=WHEELS_GROUND_PS2_OFF_AMCL_GLOBAL_SCAN
    CLIENT=$ROOT/scripts/nav2_global_localization_scan.py
    CLIENT_TIMEOUT_S=100
    PASS_MARKER=AMCL_GLOBAL_LOCALIZATION_SCAN_PASS
    ;;
  *)
    echo "unsupported heading profile: $PROFILE" >&2
    exit 2
    ;;
esac

if [[ ${1:-} == --describe && $# -eq 1 ]]; then
  source /opt/ros/humble/setup.bash
  source "$ROOT/ros2_ws/install/setup.bash"
  exec python3 "$CLIENT" --describe
fi

if [[ $# -ne 4 || $1 != --trace-file || $3 != --arm-token ]]; then
  echo "usage: $0 --trace-file <workspace-json> --arm-token <token>" >&2
  exit 2
fi

TRACE_FILE=$2
PROVIDED_TOKEN=$4
if [[ $PROVIDED_TOKEN != "$ARM_TOKEN" ]]; then
  echo "exact heading profile arm token is required" >&2
  exit 2
fi
if [[ $TRACE_FILE != "$ROOT"/* ]]; then
  echo "trace file must be an absolute path under $ROOT" >&2
  exit 2
fi
if [[ ! -d $(dirname "$TRACE_FILE") || -e $TRACE_FILE ]]; then
  echo "trace directory is missing or trace file already exists" >&2
  exit 2
fi

RUN_DIR=$(dirname "$TRACE_FILE")
HARDWARE_LOG=$RUN_DIR/hardware.launch.log
TF_LOG=$RUN_DIR/tf.launch.log
ODOM_TF_LOG=$RUN_DIR/odom-tf.launch.log
LIDAR_LOG=$RUN_DIR/lidar.launch.log
HEADING_LOG=$RUN_DIR/heading.launch.log
LOCALIZATION_LOG=$RUN_DIR/localization.launch.log
RVIZ_LOG=$RUN_DIR/rviz.log
for output in "$HARDWARE_LOG" "$TF_LOG" "$ODOM_TF_LOG" "$LIDAR_LOG" "$HEADING_LOG" "$LOCALIZATION_LOG" "$RVIZ_LOG"; do
  if [[ -e $output ]]; then
    echo "launch log already exists: $output" >&2
    exit 2
  fi
done

source "$ROOT/env/m10_env.sh"
set -u

START_SENSOR_TRANSFORMS=true
if ros2 node list 2>/dev/null | grep -Fxq /rvclaw_base_to_laser; then
  if ! pgrep -af '[s]tatic_transform_publisher' \
      | grep -Fq -- '--x 0.07725 --y 0.0 --z 0.11144 --roll 0.0 --pitch 0.0 --yaw 0.0 --frame-id base_footprint --child-frame-id laser'; then
    echo "existing /rvclaw_base_to_laser does not match the reviewed transform" >&2
    exit 1
  fi
  START_SENSOR_TRANSFORMS=false
  echo "reusing verified existing /rvclaw_base_to_laser" >"$TF_LOG"
fi

if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
  echo "current user is not in the dialout group" >&2
  exit 1
fi

for device_spec in "$STM32_DEVICE:0002" "$LIDAR_DEVICE:0001"; do
  device=${device_spec%:*}
  expected_serial=${device_spec##*:}
  resolved=$(readlink -f "$device")
  if [[ ! -c $resolved || ! -r $resolved || ! -w $resolved ]]; then
    echo "inaccessible serial endpoint: $device -> $resolved" >&2
    exit 1
  fi
  actual_serial=$(udevadm info -q property -n "$resolved" | sed -n 's/^ID_SERIAL_SHORT=//p')
  if [[ $actual_serial != "$expected_serial" ]]; then
    echo "serial identity mismatch: $resolved expected=$expected_serial actual=$actual_serial" >&2
    exit 1
  fi
  if fuser "$resolved" >/dev/null 2>&1; then
    echo "serial endpoint is already owned: $resolved" >&2
    fuser -v "$resolved" >&2 || true
    exit 1
  fi
done

pgrep -af \
  '[n]av2_|[s]lam_toolbox|[r]vclaw_stm32_|[r]vclaw_navigation_|[r]vclaw_ekf|[r]vclaw_odom_tf|[l]slidar_driver_node|[b]ehavior_server|[l]ocal_costmap|[a]mcl|[m]ap_server' \
  | awk -v self_pid="$$" -v parent_pid="$PPID" \
      '$1 != self_pid && $1 != parent_pid' \
  >"$RUN_DIR/preexisting-processes.txt" || true
if [[ -s $RUN_DIR/preexisting-processes.txt ]]; then
  echo "another navigation or RVClaw process is already running" >&2
  cat "$RUN_DIR/preexisting-processes.txt" >&2
  exit 1
fi

tf_pid=
hardware_pid=
odom_tf_pid=
lidar_pid=
heading_pid=
localization_pid=
rviz_pid=
bag_pid=
cleanup_started=0
run_completed=0

stop_group() {
  local pid=$1
  if [[ -z $pid ]] || ! kill -0 -- "-$pid" 2>/dev/null; then
    return
  fi
  kill -INT -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 -- "-$pid" 2>/dev/null || return
    sleep 0.25
  done
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 0.5
  kill -KILL -- "-$pid" 2>/dev/null || true
}

start_rviz() {
  if [[ "$AUTO_RVIZ" =~ ^(0|false|off|no)$ ]]; then
    printf 'RViz auto-start disabled (RVCLAW_AUTO_RVIZ=%s)\n' "$AUTO_RVIZ" \
      >"$RVIZ_LOG"
    return 0
  fi
  if [[ -z ${DISPLAY:-} && -z ${WAYLAND_DISPLAY:-} ]]; then
    printf 'RViz skipped: DISPLAY/WAYLAND_DISPLAY is not set\n' >"$RVIZ_LOG"
    return 0
  fi
  setsid env QT_X11_NO_MITSHM=1 LIBGL_ALWAYS_SOFTWARE=1 \
    bash "$ROOT/scripts/rvclaw_visualization.sh" --run \
    >"$RVIZ_LOG" 2>&1 &
  rviz_pid=$!
  sleep 1
  if ! kill -0 -- "-$rviz_pid" 2>/dev/null; then
    echo "RViz failed to stay running; see $RVIZ_LOG" >&2
    rviz_pid=
  fi
}

lifecycle_transition_if_state() {
  local node=$1
  local expected_state=$2
  local transition=$3
  local output=$4
  local current_state=

  if ! current_state=$(timeout 6s ros2 lifecycle get "$node" 2>/dev/null); then
    timeout 6s ros2 lifecycle set "$node" "$transition" >"$output" 2>&1 || true
    return
  fi
  if ! grep -Eq "^${expected_state} \\[[0-9]+\\]$" <<<"$current_state"; then
    return
  fi
  timeout 6s ros2 lifecycle set "$node" "$transition" >"$output" 2>&1 || true
}

cleanup() {
  local status=$?
  local stm32_resolved=
  local lidar_resolved=

  if [[ $cleanup_started -eq 1 ]]; then
    return
  fi
  cleanup_started=1
  trap - EXIT INT TERM
  set +e

  if [[ $run_completed -ne 1 && $status -eq 0 ]]; then
    status=1
  fi

  timeout 2s ros2 topic pub --once \
    /rvclaw/control/emergency_stop std_msgs/msg/Bool '{data: true}' \
    >"$RUN_DIR/cleanup-estop.log" 2>&1
  timeout 2s ros2 topic pub --once \
    /rvclaw/control/deadman std_msgs/msg/Bool '{data: false}' \
    >"$RUN_DIR/cleanup-deadman.log" 2>&1
  timeout 3s ros2 service call \
    /rvclaw/control/set_armed std_srvs/srv/SetBool '{data: false}' \
    >"$RUN_DIR/cleanup-disarm.log" 2>&1

  if ros2 node list 2>/dev/null | grep -Fxq /x10/lslidar_driver_node; then
    timeout 4s ros2 topic pub --once \
      /x10/motor_control std_msgs/msg/Int8 '{data: 0}' \
      >"$RUN_DIR/lidar-stop-topic.log" 2>&1
    sleep 1
  fi

  lifecycle_transition_if_state \
    /behavior_server active deactivate \
    "$RUN_DIR/cleanup-behavior-deactivate.log"
  lifecycle_transition_if_state \
    /costmap/costmap active deactivate \
    "$RUN_DIR/cleanup-costmap-deactivate.log"
  lifecycle_transition_if_state \
    /behavior_server inactive cleanup \
    "$RUN_DIR/cleanup-behavior-cleanup.log"
  lifecycle_transition_if_state \
    /costmap/costmap inactive cleanup \
    "$RUN_DIR/cleanup-costmap-cleanup.log"

  stop_group "$heading_pid"
  stop_group "$rviz_pid"
  stop_group "$bag_pid"
  stop_group "$localization_pid"
  stop_group "$lidar_pid"
  stop_group "$odom_tf_pid"
  stop_group "$hardware_pid"
  stop_group "$tf_pid"
  wait "$heading_pid" 2>/dev/null || true
  wait "$rviz_pid" 2>/dev/null || true
  wait "$bag_pid" 2>/dev/null || true
  wait "$localization_pid" 2>/dev/null || true
  wait "$lidar_pid" 2>/dev/null || true
  wait "$odom_tf_pid" 2>/dev/null || true
  wait "$hardware_pid" 2>/dev/null || true
  wait "$tf_pid" 2>/dev/null || true

  if [[ -n $bag_pid ]] && ! timeout 10s ros2 bag info \
    "$RUN_DIR/diagnostics-bag" >"$RUN_DIR/bag-info.txt" 2>&1; then
    echo "global localization diagnostics bag validation failed" >&2
    status=1
  fi

  timeout --signal=INT --kill-after=2s 5s \
    python3 "$ROOT/scripts/m10_serial_probe.py" \
      --motor-action stop \
      --baud "$LIDAR_BAUD" \
      --duration 0.2 \
      --max-bytes 512 \
      >"$RUN_DIR/m10-raw-stop.log" 2>&1
  if ! grep -Fq '"stop_bytes_written": 188' "$RUN_DIR/m10-raw-stop.log"; then
    echo "M10P raw stop was not confirmed" >&2
    status=1
  fi

  stm32_resolved=$(readlink -f "$STM32_DEVICE" 2>/dev/null)
  lidar_resolved=$(readlink -f "$LIDAR_DEVICE" 2>/dev/null)
  for resolved in "$stm32_resolved" "$lidar_resolved"; do
    if [[ -n $resolved ]] && fuser "$resolved" >/dev/null 2>&1; then
      echo "serial endpoint remained owned: $resolved" >&2
      fuser -v "$resolved" >&2 || true
      status=1
    fi
  done

  if [[ $status -eq 0 && $run_completed -eq 1 ]]; then
    echo "$PASS_MARKER"
    echo "trace=$TRACE_FILE"
    echo "run_dir=$RUN_DIR"
  fi
  exit "$status"
}

trap cleanup EXIT INT TERM

if [[ $PROFILE == global_localization_scan ]]; then
  if [[ ! -f $MAP_FILE ]]; then
    echo "reviewed map is missing: $MAP_FILE" >&2
    exit 1
  fi
  setsid ros2 launch rvclaw_nav2_bringup localization.launch.py \
    map:="$MAP_FILE" autostart:=true use_sim_time:=false \
    start_sensor_transforms:="$START_SENSOR_TRANSFORMS" \
    laser_x:=0.07725 laser_y:=0.0 laser_z:=0.11144 \
    laser_roll:=0.0 laser_pitch:=0.0 laser_yaw:=0.0 \
    >"$LOCALIZATION_LOG" 2>&1 &
  localization_pid=$!
else
  if [[ $START_SENSOR_TRANSFORMS == true ]]; then
    setsid ros2 launch rvclaw_nav2_bringup sensor_transforms.launch.py \
      laser_x:=0.07725 laser_y:=0.0 laser_z:=0.11144 \
      laser_roll:=0.0 laser_pitch:=0.0 laser_yaw:=0.0 \
      >"$TF_LOG" 2>&1 &
    tf_pid=$!
  fi
fi

setsid ros2 launch rvclaw_stm32_base hardware.launch.py \
  enable_hardware_output:=true >"$HARDWARE_LOG" 2>&1 &
hardware_pid=$!

if [[ $PROFILE != global_localization_scan ]]; then
  setsid ros2 launch rvclaw_nav2_bringup odom_tf.launch.py \
    >"$ODOM_TF_LOG" 2>&1 &
  odom_tf_pid=$!
fi

if [[ $PROFILE == global_localization_scan ]]; then
  # Global localization uses the profile-local self-filter so the reviewed
  # chassis returns cannot make an in-place Spin look occupied. Normal
  # mapping/navigation keep the vendor lidar topic untouched.
  setsid ros2 launch rvclaw_nav2_bringup global_scan_lidar.launch.py \
    >"$LIDAR_LOG" 2>&1 &
else
  setsid ros2 launch lslidar_driver lsm10p_uart_launch.py \
    >"$LIDAR_LOG" 2>&1 &
fi
lidar_pid=$!

setsid ros2 launch rvclaw_nav2_bringup heading_gate.launch.py \
  >"$HEADING_LOG" 2>&1 &
heading_pid=$!

required_nodes=(
  /rvclaw_base_to_laser
  /rvclaw_stm32_hardware
  /rvclaw_odom_tf
  /x10/lslidar_driver_node
  /costmap/costmap
  /behavior_server
  /rvclaw_nav2_cmd_adapter
)
if [[ $PROFILE == global_localization_scan ]]; then
  required_nodes+=(/map_server /amcl /rvclaw_scan_self_filter)
fi
for required in "${required_nodes[@]}"; do
  ready=0
  for _ in $(seq 1 80); do
    if ros2 node list 2>/dev/null | grep -Fxq "$required"; then
      ready=1
      break
    fi
    sleep 0.25
  done
  if [[ $ready -ne 1 ]]; then
    echo "required node did not become ready: $required" >&2
    exit 1
  fi
done

ros2 node list | sort >"$RUN_DIR/nodes.txt"
ros2 topic list -t | sort >"$RUN_DIR/topics.txt"

# The M10P driver may need ~20 s to finish opening the UART and create the
# motor-control subscription on K3. Wait long enough for that bounded startup
# while still failing closed if the subscriber never appears.
timeout 30s ros2 topic pub --once \
  /x10/motor_control std_msgs/msg/Int8 '{data: 1}' \
  >"$RUN_DIR/lidar-start-topic.log" 2>&1

timeout 8s ros2 topic echo --once --qos-profile sensor_data /scan \
  >"$RUN_DIR/scan-preflight.txt"
timeout 8s ros2 topic echo --once --qos-profile sensor_data /odom \
  >"$RUN_DIR/raw-odom-preflight.txt"

for transition_spec in \
  '/costmap/costmap:configure:costmap-configure.txt' \
  '/behavior_server:configure:behavior-configure.txt' \
  '/costmap/costmap:activate:costmap-activate.txt' \
  '/behavior_server:activate:behavior-activate.txt'; do
  node=${transition_spec%%:*}
  remainder=${transition_spec#*:}
  transition=${remainder%%:*}
  output=${remainder##*:}
  timeout 8s ros2 lifecycle set "$node" "$transition" \
    | tee "$RUN_DIR/$output"
  if ! grep -Fq 'Transitioning successful' "$RUN_DIR/$output"; then
    echo "lifecycle transition failed: $node $transition" >&2
    exit 1
  fi
done

ros2 lifecycle get /costmap/costmap >"$RUN_DIR/costmap-state.txt"
ros2 lifecycle get /behavior_server >"$RUN_DIR/behavior-state.txt"
grep -Fq 'active' "$RUN_DIR/costmap-state.txt"
grep -Fq 'active' "$RUN_DIR/behavior-state.txt"

timeout 8s ros2 topic echo --once /costmap/costmap_raw \
  >"$RUN_DIR/costmap-preflight.txt"

if [[ $PROFILE == global_localization_scan ]]; then
  for lifecycle_node in /map_server /amcl; do
    # On K3, AMCL lifecycle service calls can be delayed while the lidar and
    # costmap are warming up. Use the same bounded 30 s window as the client
    # service calls instead of treating a slow response as an inactive node.
    if ! timeout 30s ros2 lifecycle get "$lifecycle_node" \
      | grep -Fq active; then
      echo "localization lifecycle node is not active: $lifecycle_node" >&2
      exit 1
    fi
  done
  setsid ros2 bag record --storage sqlite3 --max-cache-size 10485760 \
    --output "$RUN_DIR/diagnostics-bag" \
    /scan /odom /amcl_pose /particle_cloud /tf /tf_static \
    /rvclaw/nav2_cmd_vel /rvclaw/cmd_vel/autonomy \
    /rvclaw/control/guarded_cmd_vel /rvclaw/control/state \
    /rvclaw/control/armed /rvclaw/control/deadman /PowerVoltage \
    >"$RUN_DIR/bag-record.log" 2>&1 &
  bag_pid=$!
  sleep 2
  if ! kill -0 -- "-$bag_pid" 2>/dev/null; then
    echo "global localization diagnostics bag did not stay running" >&2
    exit 1
  fi
fi

start_rviz

ros2 action info /spin >"$RUN_DIR/spin-action-info.txt"
if ! grep -Eq 'Action servers: +1' "$RUN_DIR/spin-action-info.txt"; then
  echo "exactly one Spin action server was not observed" >&2
  cat "$RUN_DIR/spin-action-info.txt" >&2
  exit 1
fi
if ! grep -Eq 'Action clients: +0' "$RUN_DIR/spin-action-info.txt"; then
  echo "an unexpected Spin action client already exists" >&2
  cat "$RUN_DIR/spin-action-info.txt" >&2
  exit 1
fi

if [[ $PROFILE == global_localization_scan ]]; then
  echo "AMCL_GLOBAL_LOCALIZATION_SCAN_READY"
  echo "No motion has been commanded. Enter:"
  echo "CONFIRM_AMCL_GLOBAL_LOCALIZATION_SCAN_ARMING"
  read -r gate_input
  if [[ $gate_input != CONFIRM_AMCL_GLOBAL_LOCALIZATION_SCAN_ARMING ]]; then
    echo "AMCL global localization scan cancelled before arming" >&2
    exit 1
  fi
  echo "AMCL_GLOBAL_LOCALIZATION_SCAN_ARMING"
  echo "The chassis may begin rotating within 3 seconds."
  sleep 3
fi

set +e
if [[ $PROFILE == global_localization_scan ]]; then
  timeout "${CLIENT_TIMEOUT_S}s" python3 "$CLIENT" \
    --trace-file "$TRACE_FILE" \
    --arm-token "$PROVIDED_TOKEN"
else
  timeout 20s python3 "$CLIENT" \
    --trace-file "$TRACE_FILE" \
    --arm-token "$PROVIDED_TOKEN"
fi
client_status=$?
set -e
if [[ $client_status -ne 0 ]]; then
  echo "Nav2 heading client failed with status $client_status" >&2
  tail -n 100 "$HEADING_LOG" >&2 || true
  tail -n 100 "$HARDWARE_LOG" >&2 || true
  exit "$client_status"
fi

run_completed=1
