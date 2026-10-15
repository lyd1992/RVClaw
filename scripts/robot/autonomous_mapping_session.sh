#!/usr/bin/env bash
set -euo pipefail

lab_root="${RVCLAW_LAB_ROOT:-/data/rvclaw-k3-lab}"
arm_token="WHEELS_GROUND_AUTONOMOUS_MAPPING_LOW_SPEED"
confirm_token="CONFIRM_AUTONOMOUS_MAPPING_LOW_SPEED"
lidar_device="/dev/wheeltec_lidar"
stm32_device="/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00"
lidar_baud="500000"
max_runtime_s="${RVCLAW_AUTONOMOUS_MAX_RUNTIME_S:-60.0}"
max_goals="${RVCLAW_AUTONOMOUS_MAX_GOALS:-3}"
max_forward_speed_m_s="${RVCLAW_AUTONOMOUS_MAX_FORWARD_SPEED_M_S:-0.06}"
auto_rviz="${RVCLAW_AUTO_RVIZ:-auto}"

if [[ "${1:-}" == "--describe" ]]; then
  cat <<EOF
hardware_io=true_only_after_explicit_run_confirmation
motion_source=frontier_explorer_NavigateToPose
mapping=slam_toolbox_online_async
direct_cmd_vel_publishers=0
max_runtime_s=${max_runtime_s}
max_goals=${max_goals}
max_forward_speed_m_s=${max_forward_speed_m_s}
automatic_reverse=false
failure_policy=cancel_goal_disarm_zero_estop
arm_token=${arm_token}
confirmation=${confirm_token}
auto_rviz=${auto_rviz}
EOF
  exit 0
fi

if [[ "${1:-}" != "--run" || "${2:-}" != "--arm-token" || "${3:-}" != "${arm_token}" ]]; then
  printf 'Exact --run --arm-token value is required. Use --describe first.\n' >&2
  exit 2
fi
if [[ ! -t 0 ]]; then
  printf 'Autonomous mapping run requires an interactive terminal.\n' >&2
  exit 2
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${lab_root}/logs/autonomous-mapping-${timestamp}"
map_dir="${lab_root}/maps/autonomous-${timestamp}"
mkdir -p "${run_dir}/ros" "${map_dir}"
exec > >(tee "${run_dir}/run.log") 2>&1
export ROS_LOG_DIR="${run_dir}/ros"
export RCUTILS_COLORIZED_OUTPUT=0

mapping_pid=""
bag_pid=""
cleanup_started=0
run_completed=0

stop_group() {
  local pid="$1"
  if [[ -z "${pid}" ]] || ! kill -0 -- "-${pid}" 2>/dev/null; then
    return
  fi
  kill -INT -- "-${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 -- "-${pid}" 2>/dev/null || return
    sleep 0.25
  done
  kill -KILL -- "-${pid}" 2>/dev/null || true
}

safe_disarm() {
  timeout 5s ros2 service call /rvclaw/control/set_armed std_srvs/srv/SetBool \
    '{data: false}' >/dev/null 2>&1 || true
  timeout 5s ros2 topic pub --once /rvclaw/control/emergency_stop \
    std_msgs/msg/Bool '{data: true}' >/dev/null 2>&1 || true
  timeout 5s ros2 topic pub --once /rvclaw/cmd_vel/autonomy \
    geometry_msgs/msg/TwistStamped '{twist: {linear: {x: 0.0, y: 0.0}, angular: {z: 0.0}}}' >/dev/null 2>&1 || true
}

cleanup() {
  local status=$?
  if [[ ${cleanup_started} -eq 1 ]]; then
    return
  fi
  cleanup_started=1
  trap - EXIT INT TERM
  set +e
  safe_disarm
  if [[ ${run_completed} -eq 1 ]]; then
    timeout 20s ros2 run nav2_map_server map_saver_cli \
      -f "${map_dir}/map" --ros-args \
      -p save_map_timeout:=10.0 -p free_thresh_default:=0.25 \
      -p occupied_thresh_default:=0.65 >"${run_dir}/map-saver.log" 2>&1
    timeout 15s ros2 service call /slam_toolbox/serialize_map \
      slam_toolbox/srv/SerializePoseGraph "{filename: '${map_dir}/posegraph'}" \
      >"${run_dir}/serialize-map.log" 2>&1
    if [[ -s "${map_dir}/map.yaml" && -s "${map_dir}/map.pgm" ]]; then
      cp "${lab_root}/ros2_ws/src/rvclaw_nav2_bringup/config/slam_toolbox.yaml" "${map_dir}/slam_toolbox.yaml"
      sha256sum "${map_dir}"/* >"${map_dir}/SHA256SUMS"
    else
      status=1
    fi
  fi
  if ros2 node list 2>/dev/null | grep -Fxq /x10/lslidar_driver_node; then
    timeout 4s ros2 topic pub --once /x10/motor_control \
      std_msgs/msg/Int8 '{data: 0}' >/dev/null 2>&1
  fi
  stop_group "${bag_pid}"
  stop_group "${mapping_pid}"
  wait "${bag_pid}" 2>/dev/null || true
  wait "${mapping_pid}" 2>/dev/null || true
  printf '[autonomous-mapping] run_dir=%s\n' "${run_dir}"
  printf '[autonomous-mapping] map_dir=%s\n' "${map_dir}"
  if [[ ${status} -eq 0 && ${run_completed} -eq 1 ]]; then
    printf 'AUTONOMOUS_MAPPING_SESSION_PASS\n'
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

source "${lab_root}/env/m10_env.sh"
for device_spec in "${lidar_device}:0001" "${stm32_device}:0002"; do
  device="${device_spec%%:*}"
  serial="${device_spec##*:}"
  resolved="$(readlink -f "${device}" 2>/dev/null || true)"
  if [[ ! -c "${resolved}" || ! -r "${resolved}" ]]; then
    printf '[autonomous-mapping] inaccessible device: %s -> %s\n' "${device}" "${resolved}" >&2
    exit 2
  fi
  actual="$(udevadm info -q property -n "${resolved}" | sed -n 's/^ID_SERIAL_SHORT=//p')"
  if [[ "${actual}" != "${serial}" ]]; then
    printf '[autonomous-mapping] identity mismatch: %s expected=%s actual=%s\n' "${resolved}" "${serial}" "${actual}" >&2
    exit 2
  fi
  if fuser "${resolved}" >/dev/null 2>&1; then
    printf '[autonomous-mapping] device already occupied: %s\n' "${resolved}" >&2
    exit 2
  fi
done

existing_nodes="$(ros2 node list 2>/dev/null)"
for forbidden in /rvclaw_stm32_hardware /slam_toolbox /planner_server /controller_server /bt_navigator /x10/lslidar_driver_node; do
  if grep -Fxq "${forbidden}" <<<"${existing_nodes}"; then
    printf '[autonomous-mapping] forbidden existing node: %s\n' "${forbidden}" >&2
    exit 2
  fi
done

printf 'Field safety confirmation required. Confirm PS2 off, four wheels grounded, clear low-speed area, and operator beside black power switch.\n'
read -r -p "Type ${confirm_token}: " confirmation
if [[ "${confirmation}" != "${confirm_token}" ]]; then
  printf 'Autonomous mapping was not armed.\n' >&2
  exit 2
fi

setsid ros2 launch rvclaw_nav2_bringup autonomous_mapping.launch.py \
  use_sim_time:=false autostart:=true enable_hardware_output:=true \
  start_exploration:=true arm_token:="${arm_token}" \
  max_runtime_s:="${max_runtime_s}" max_goals:="${max_goals}" \
  max_forward_speed_m_s:="${max_forward_speed_m_s}" \
  laser_x:=0.07725 laser_y:=0.0 laser_z:=0.11144 \
  laser_roll:=0.0 laser_pitch:=0.0 laser_yaw:=0.0 \
  start_rviz:="$([[ "${auto_rviz}" == "1" || "${auto_rviz}" == "true" || ( "${auto_rviz}" == "auto" && -n "${DISPLAY:-}" ) ]] && printf true || printf false)" \
  >"${run_dir}/autonomous-launch.log" 2>&1 &
mapping_pid=$!

required_nodes=(/rvclaw_stm32_hardware /x10/lslidar_driver_node /rvclaw_scan_self_filter /slam_toolbox /planner_server /controller_server /bt_navigator /rvclaw_frontier_explorer)
for required in "${required_nodes[@]}"; do
  ready=0
  for _ in $(seq 1 80); do
    if ros2 node list 2>/dev/null | grep -Fxq "${required}"; then ready=1; break; fi
    sleep 0.25
  done
  if [[ ${ready} -ne 1 ]]; then
    printf '[autonomous-mapping] node did not become ready: %s\n' "${required}" >&2
    exit 3
  fi
done

printf '[autonomous-mapping] starting diagnostics bag\n'
setsid ros2 bag record --storage sqlite3 --max-cache-size 10485760 \
  --output "${run_dir}/diagnostics-bag" \
  /map /scan /odom /tf /tf_static /amcl_pose \
  /rvclaw/exploration/state /rvclaw/exploration/current_goal \
  /rvclaw/nav2_cmd_vel_raw /rvclaw/nav2_cmd_vel /rvclaw/cmd_vel/autonomy \
  /rvclaw/control/guarded_cmd_vel /rvclaw/control/state \
  /rvclaw/control/armed /rvclaw/control/estop_latched \
  >"${run_dir}/bag-record.log" 2>&1 &
bag_pid=$!
sleep 1
if ! kill -0 -- "-${bag_pid}" 2>/dev/null; then
  printf '[autonomous-mapping] diagnostics bag recorder failed\n' >&2
  exit 3
fi

timeout 4s ros2 topic pub --once /x10/motor_control \
  std_msgs/msg/Int8 '{data: 1}' >/dev/null
sleep 2
ros2 run rvclaw_nav2_bringup rvclaw-check-nav-readiness \
  --timeout 12 --minimum-messages 20 \
  --output "${run_dir}/readiness.json" | tee "${run_dir}/readiness.log"
# Release any stop assertion left by an earlier, safely terminated session
# before asking the hardware supervisor to clear its startup latch. Repeat the
# volatile command briefly so startup subscriber discovery cannot lose it.
timeout 3s ros2 topic pub --rate 10 --times 10 /rvclaw/control/emergency_stop \
  std_msgs/msg/Bool '{data: false}' >"${run_dir}/release-estop.log" 2>&1
release_status=$?
if [[ ${release_status} -ne 0 ]]; then
  printf '[autonomous-mapping] failed to release emergency stop (status=%s)\n' "${release_status}" >&2
  exit 3
fi
sleep 0.2
reset_output="$(timeout 5s ros2 service call /rvclaw/control/reset_latches std_srvs/srv/Trigger '{}' 2>&1 || true)"
printf '%s\n' "${reset_output}" | tee "${run_dir}/reset-latches.log"
if ! grep -Fq 'success=True' <<<"${reset_output}"; then
  printf '[autonomous-mapping] reset_latches was refused\n' >&2
  exit 3
fi
arm_output="$(timeout 5s ros2 service call /rvclaw/control/set_armed std_srvs/srv/SetBool '{data: true}' 2>&1 || true)"
printf '%s\n' "${arm_output}" | tee "${run_dir}/arm.log"
if ! grep -Fq 'success=True' <<<"${arm_output}"; then
  printf '[autonomous-mapping] set_armed was refused\n' >&2
  exit 3
fi
printf 'AUTONOMOUS_MAPPING_READY\n'

while kill -0 "${mapping_pid}" 2>/dev/null; do
  state_sample="$(timeout 2s ros2 topic echo --once /rvclaw/exploration/state 2>/dev/null || true)"
  if grep -Fq 'STOPPED' <<<"${state_sample}"; then
    run_completed=1
    break
  fi
  sleep 1
done
if [[ ${run_completed} -eq 1 ]]; then
  safe_disarm
  stop_group "${mapping_pid}"
  wait "${mapping_pid}" 2>/dev/null || true
else
  exit 1
fi
