#!/usr/bin/env bash
set -Eeuo pipefail

lab_root="${RVCLAW_LAB_ROOT:-/data/rvclaw-k3-lab}"
base_root="${RVCLAW_BASE_ROOT:-${lab_root}}"
arm_token="WHEELS_GROUND_PS2_MAPPING_SUPERVISED"
voice_arm_token="VOICE_WAKE_PS2_MAPPING_SUPERVISED"
lidar_device="/dev/wheeltec_lidar"
stm32_device="/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00"
lidar_baud="500000"
snapshot_interval_s="${RVCLAW_SLAM_SNAPSHOT_INTERVAL_S:-0}"
snapshot_outputs="disabled"
auto_rviz="${RVCLAW_AUTO_RVIZ:-auto}"
control_mode="interactive"
control_file="${RVCLAW_MAPPING_CONTROL_FILE:-}"
state_file="${RVCLAW_MAPPING_STATE_FILE:-}"
latest_file="${RVCLAW_MAPPING_LATEST_FILE:-${lab_root}/state/manual-mapping-latest.json}"
stationary_duration_s="${RVCLAW_MAPPING_STATIONARY_DURATION_S:-5}"
stationary_minimum_messages="${RVCLAW_MAPPING_STATIONARY_MIN_MESSAGES:-20}"
node_release_timeout_s="${RVCLAW_MAPPING_NODE_RELEASE_TIMEOUT_S:-30}"
if [[ ${snapshot_interval_s} =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ ${snapshot_interval_s} != 0 ]]; then
  snapshot_outputs="map-vNNN every ${snapshot_interval_s}s"
fi

if [[ "${1:-}" == "--describe" ]]; then
  cat <<EOF
hardware_io=false_until_run
motion_source=PS2_only
ros_velocity_publishers=0
mapping=slam_toolbox_online_async
map_outputs=map.yaml,map.pgm,posegraph
operator_supervision_required=true
arm_token=${arm_token}
voice_arm_token=${voice_arm_token}
voice_control=atomic_control_file
snapshot_interval_s=${snapshot_interval_s}
snapshot_outputs=${snapshot_outputs}
auto_rviz=${auto_rviz}
EOF
  exit 0
fi

case "${1:-}" in
  --run)
    if [[ "${2:-}" != "--arm-token" || "${3:-}" != "${arm_token}" || $# -ne 3 ]]; then
      printf 'Exact --run --arm-token value is required. Use --describe first.\n' >&2
      exit 2
    fi
    if [[ ! -t 0 ]]; then
      printf 'Mapping run requires an interactive terminal.\n' >&2
      exit 2
    fi
    ;;
  --voice-run)
    control_mode="voice"
    if [[ "${2:-}" != "--arm-token" || "${3:-}" != "${voice_arm_token}" || $# -ne 3 ]]; then
      printf 'Exact --voice-run --arm-token value is required. Use --describe first.\n' >&2
      exit 2
    fi
    if [[ -z "${control_file}" || -z "${state_file}" ]]; then
      printf 'Voice mapping requires RVCLAW_MAPPING_CONTROL_FILE and RVCLAW_MAPPING_STATE_FILE.\n' >&2
      exit 2
    fi
    if [[ "${control_file}" != /* || "${state_file}" != /* || "${latest_file}" != /* ]]; then
      printf 'Voice mapping control, state, and latest paths must be absolute.\n' >&2
      exit 2
    fi
    if [[ "${control_file}" == "${state_file}" ]]; then
      printf 'Voice mapping control and state paths must be different.\n' >&2
      exit 2
    fi
    ;;
  *)
    printf 'Usage: %s --describe | --run --arm-token TOKEN | --voice-run --arm-token TOKEN\n' "$0" >&2
    exit 2
    ;;
esac

if [[ ! ${stationary_duration_s} =~ ^[0-9]+([.][0-9]+)?$ || ${stationary_duration_s} == 0 ]]; then
  printf 'RVCLAW_MAPPING_STATIONARY_DURATION_S must be positive.\n' >&2
  exit 2
fi
if [[ ! ${stationary_minimum_messages} =~ ^[1-9][0-9]*$ ]]; then
  printf 'RVCLAW_MAPPING_STATIONARY_MIN_MESSAGES must be a positive integer.\n' >&2
  exit 2
fi
if [[ ! ${node_release_timeout_s} =~ ^[1-9][0-9]*$ ]]; then
  printf 'RVCLAW_MAPPING_NODE_RELEASE_TIMEOUT_S must be a positive integer.\n' >&2
  exit 2
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${RVCLAW_MAPPING_RUN_DIR:-${lab_root}/logs/slam-mapping-${timestamp}}"
map_dir="${RVCLAW_MAPPING_MAP_DIR:-${lab_root}/maps/slam-${timestamp}}"
mkdir -p "${run_dir}/ros" "${map_dir}"
exec > >(tee "${run_dir}/run.log") 2>&1

export ROS_LOG_DIR="${run_dir}/ros"
export RCUTILS_COLORIZED_OUTPUT=0

mapping_pid=""
telemetry_pid=""
lidar_pid=""
bag_pid=""
snapshot_pid=""
rviz_pid=""
cleanup_started=0
run_completed=0
terminal_state="FAILED"
failure_reason="mapping_startup_failed"
control_generation=0

report_error() {
  local status="$1"
  local line="$2"
  local command="$3"
  printf '[mapping] command_failed status=%s line=%s command=%q failure_reason=%s\n' \
    "${status}" "${line}" "${command}" "${failure_reason}" >&2
  return "${status}"
}

write_voice_state() {
  local state="$1"
  local error="${2:-}"
  local result_code="${3:-}"
  local temporary

  if [[ "${control_mode}" != "voice" ]]; then
    return 0
  fi
  mkdir -p "$(dirname -- "${state_file}")"
  temporary="${state_file}.tmp.$$"
  python3 - "${temporary}" "${state}" "${run_dir}" "${map_dir}" \
    "${error}" "${result_code}" "$$" "${control_generation}" <<'PY'
import json
import os
import sys
import time

target, state, run_dir, map_dir, error, result_code, pid, generation = sys.argv[1:]
payload = {
    "schema_version": 1,
    "state": state,
    "run_dir": run_dir,
    "map_dir": map_dir,
    "pid": int(pid),
    "generation": int(generation),
    "updated_at": time.time(),
    "error": error or None,
    "result_code": int(result_code) if result_code else None,
}
with open(target, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
  mv -f -- "${temporary}" "${state_file}"
}

write_latest_map() {
  local temporary

  mkdir -p "$(dirname -- "${latest_file}")"
  temporary="${latest_file}.tmp.$$"
  python3 - "${temporary}" "${run_dir}" "${map_dir}" <<'PY'
import json
import os
import sys
import time

target, run_dir, map_dir = sys.argv[1:]
payload = {
    "schema_version": 1,
    "created_at": time.time(),
    "run_dir": run_dir,
    "map_dir": map_dir,
    "map_yaml": os.path.join(map_dir, "map.yaml"),
    "map_image": os.path.join(map_dir, "map.pgm"),
    "posegraph": os.path.join(map_dir, "posegraph"),
    "status": "SAVED",
}
with open(target, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
  mv -f -- "${temporary}" "${latest_file}"
}

if [[ "${control_mode}" == "voice" ]]; then
  mkdir -p "$(dirname -- "${control_file}")"
  rm -f -- "${control_file}"
  write_voice_state "STARTING"
fi

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

cleanup() {
  local status=$?
  local lidar_resolved=""
  local stm32_resolved=""

  if [[ ${cleanup_started} -eq 1 ]]; then
    return
  fi
  cleanup_started=1
  trap - EXIT INT TERM ERR
  set +e

  if [[ -n "${snapshot_pid}" ]] && kill -0 "${snapshot_pid}" 2>/dev/null; then
    kill -TERM "${snapshot_pid}" 2>/dev/null || true
    wait "${snapshot_pid}" 2>/dev/null || true
  fi

  if [[ ${run_completed} -ne 1 && ${status} -eq 0 ]]; then
    status=1
  fi

  if ros2 node list 2>/dev/null | grep -Fxq /x10/lslidar_driver_node; then
    timeout 4s ros2 topic pub --once /x10/motor_control \
      std_msgs/msg/Int8 '{data: 0}' >/dev/null 2>&1
    sleep 2
  fi

  stop_group "${bag_pid}"
  wait "${bag_pid}" 2>/dev/null
  if [[ -n "${bag_pid}" ]] && ! timeout 10s ros2 bag info \
    "${run_dir}/diagnostics-bag" >"${run_dir}/bag-info.txt" 2>&1; then
    printf '[mapping] diagnostics bag validation failed\n' >&2
    status=1
  fi

  stop_group "${mapping_pid}"
  stop_group "${rviz_pid}"
  stop_group "${lidar_pid}"
  stop_group "${telemetry_pid}"
  wait "${mapping_pid}" 2>/dev/null
  wait "${rviz_pid}" 2>/dev/null
  wait "${lidar_pid}" 2>/dev/null
  wait "${telemetry_pid}" 2>/dev/null

  timeout --signal=INT --kill-after=2s 5s \
    python3 "${lab_root}/scripts/m10_serial_probe.py" \
      --motor-action stop \
      --baud "${lidar_baud}" \
      --duration 0.2 \
      --max-bytes 512 \
      >"${run_dir}/m10-raw-stop.log" 2>&1
  if ! grep -Fq '"stop_bytes_written": 188' "${run_dir}/m10-raw-stop.log"; then
    printf '[mapping] M10P raw stop was not confirmed\n' >&2
    status=1
  fi

  lidar_resolved="$(readlink -f "${lidar_device}" 2>/dev/null)"
  stm32_resolved="$(readlink -f "${stm32_device}" 2>/dev/null)"
  if [[ -n "${lidar_resolved}" ]] && fuser "${lidar_resolved}" >/dev/null 2>&1; then
    printf '[mapping] LiDAR serial remains occupied\n' >&2
    status=1
  fi
  if [[ -n "${stm32_resolved}" ]] && fuser "${stm32_resolved}" >/dev/null 2>&1; then
    printf '[mapping] STM32 serial remains occupied\n' >&2
    status=1
  fi

  printf '[mapping] run_dir=%s\n' "${run_dir}"
  printf '[mapping] map_dir=%s\n' "${map_dir}"
  if [[ ${status} -eq 0 ]]; then
    printf 'SLAM_MAPPING_SESSION_PASS\n'
  fi
  if [[ "${control_mode}" == "voice" ]]; then
    if [[ ${run_completed} -eq 1 && ${status} -eq 0 ]]; then
      terminal_state="COMPLETED"
      failure_reason=""
    elif [[ "${terminal_state}" != "CANCELLED" ]]; then
      terminal_state="FAILED"
    fi
    write_voice_state "${terminal_state}" "${failure_reason}" "${status}"
    rm -f -- "${control_file}"
  fi
  exit "${status}"
}

trap 'report_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap cleanup EXIT INT TERM
# The isolated candidate owns the ROS workspace, while the vendor LiDAR/PCL
# sysroot is shared by all candidates under the stable base root.  m10_env.sh
# otherwise derives both paths from RVCLAW_LAB_ROOT and would incorrectly look
# for a full deps/sysroot copy inside every timestamped candidate directory.
export RVCLAW_ROS_WS="${RVCLAW_ROS_WS:-${lab_root}/ros2_ws}"
export RVCLAW_LIDAR_SYSROOT="${RVCLAW_LIDAR_SYSROOT:-${base_root}/deps/sysroot}"
source "${base_root}/env/m10_env.sh"

for device_spec in "${lidar_device}:0001" "${stm32_device}:0002"; do
  device="${device_spec%%:*}"
  serial="${device_spec##*:}"
  resolved="$(readlink -f "${device}")"
  if [[ ! -c "${resolved}" || ! -r "${resolved}" ]]; then
    failure_reason="device_inaccessible:${device}"
    printf '[mapping] inaccessible device: %s -> %s\n' "${device}" "${resolved}" >&2
    exit 2
  fi
  actual="$(udevadm info -q property -n "${resolved}" | sed -n 's/^ID_SERIAL_SHORT=//p')"
  if [[ "${actual}" != "${serial}" ]]; then
    failure_reason="device_identity_mismatch:${device}"
    printf '[mapping] identity mismatch: %s expected=%s actual=%s\n' \
      "${resolved}" "${serial}" "${actual}" >&2
    exit 2
  fi
  if fuser "${resolved}" >/dev/null 2>&1; then
    failure_reason="device_still_occupied:${device}"
    printf '[mapping] device already occupied: %s\n' "${resolved}" >&2
    exit 2
  fi
done

# The ROS 2 CLI daemon can retain discovery records for stopped Nav2 nodes for
# roughly twenty seconds even after the systemd cgroup and serial devices are
# gone. Reset only that CLI discovery cache; it does not stop any ROS node.
# The bounded node-release loop below remains the safety authority if reset is
# unavailable or if a real process is still alive.
printf '[mapping] refreshing ROS graph discovery before exclusivity check\n'
timeout 5s ros2 daemon stop >/dev/null 2>&1 || true

forbidden_nodes=(
  /rvclaw_stm32_hardware
  /rvclaw_safety_supervisor
  /rvclaw_stm32_telemetry
  /slam_toolbox
  /amcl
  /controller_server
  /planner_server
  /bt_navigator
)
node_release_deadline=$((SECONDS + node_release_timeout_s))
node_release_wait_logged=0
while true; do
  existing_nodes="$(timeout 3s ros2 node list 2>/dev/null || true)"
  lingering_node=""
  for forbidden in "${forbidden_nodes[@]}"; do
    if grep -Fxq "${forbidden}" <<<"${existing_nodes}"; then
      lingering_node="${forbidden}"
      break
    fi
  done
  if [[ -z "${lingering_node}" ]]; then
    break
  fi
  if (( SECONDS >= node_release_deadline )); then
    failure_reason="navigation_node_not_released:${lingering_node}"
    printf '[mapping] forbidden existing node remained after %ss: %s\n' \
      "${node_release_timeout_s}" "${lingering_node}" >&2
    exit 2
  fi
  if [[ ${node_release_wait_logged} -eq 0 ]]; then
    printf '[mapping] waiting for stopped navigation nodes to leave the ROS graph: %s\n' \
      "${lingering_node}"
    node_release_wait_logged=1
  fi
  sleep 0.25
done

printf '[mapping] starting read-only STM32 telemetry\n'
setsid ros2 launch rvclaw_stm32_base telemetry.launch.py \
  >"${run_dir}/telemetry.log" 2>&1 &
telemetry_pid=$!

printf '[mapping] starting M10P driver\n'
setsid ros2 launch lslidar_driver lsm10p_uart_launch.py \
  >"${run_dir}/lidar.log" 2>&1 &
lidar_pid=$!

printf '[mapping] starting reviewed TF, raw odom TF, and slam_toolbox\n'
setsid ros2 launch rvclaw_nav2_bringup mapping.launch.py \
  use_sim_time:=false \
  laser_x:=0.07725 laser_y:=0.0 laser_z:=0.11144 \
  laser_roll:=0.0 laser_pitch:=0.0 laser_yaw:=0.0 \
  >"${run_dir}/mapping.log" 2>&1 &
mapping_pid=$!

required_nodes=(
  /rvclaw_stm32_telemetry
  /x10/lslidar_driver_node
  /rvclaw_base_to_laser
  /rvclaw_odom_tf
  /slam_toolbox
)
for required in "${required_nodes[@]}"; do
  ready=0
  for _ in $(seq 1 60); do
    if ros2 node list 2>/dev/null | grep -Fxq "${required}"; then
      ready=1
      break
    fi
    sleep 0.25
  done
  if [[ ${ready} -ne 1 ]]; then
    printf '[mapping] node did not become ready: %s\n' "${required}" >&2
    exit 3
  fi
done

ros2 node list | sort >"${run_dir}/nodes.txt"
ros2 topic list -t | sort >"${run_dir}/topics.txt"
ros2 node info /rvclaw_stm32_telemetry >"${run_dir}/telemetry-node-info.txt"
if grep -Eq '(^|/)(cmd_vel|rvclaw/nav2_cmd_vel|rvclaw/safe_cmd_vel)' \
  "${run_dir}/topics.txt"; then
  printf '[mapping] unexpected command topic exists\n' >&2
  exit 3
fi
if grep -Eq 'cmd_vel|safe_cmd' "${run_dir}/telemetry-node-info.txt"; then
  printf '[mapping] telemetry unexpectedly exposes a command interface\n' >&2
  exit 3
fi

printf '[mapping] starting read-only diagnostics bag\n'
setsid ros2 bag record \
  --storage sqlite3 \
  --max-cache-size 10485760 \
  --output "${run_dir}/diagnostics-bag" \
  /scan /odom /tf /tf_static \
  >"${run_dir}/bag-record.log" 2>&1 &
bag_pid=$!
sleep 1
if ! kill -0 -- "-${bag_pid}" 2>/dev/null; then
  printf '[mapping] diagnostics bag recorder did not stay running\n' >&2
  exit 3
fi

printf '[mapping] starting LiDAR motor\n'
timeout 4s ros2 topic pub --once /x10/motor_control \
  std_msgs/msg/Int8 '{data: 1}' >/dev/null
sleep 2

ros2 run rvclaw_nav2_bringup rvclaw-check-nav-readiness \
  --timeout 12 \
  --minimum-messages 20 \
  --output "${run_dir}/readiness.json" \
  | tee "${run_dir}/readiness.log"

rviz_log="${run_dir}/rviz.log"
if [[ "${auto_rviz}" =~ ^(0|false|off|no)$ ]]; then
  printf 'RViz auto-start disabled (RVCLAW_AUTO_RVIZ=%s)\n' \
    "${auto_rviz}" >"${rviz_log}"
elif [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  printf 'RViz skipped: DISPLAY/WAYLAND_DISPLAY is not set\n' >"${rviz_log}"
elif ! command -v rviz2 >/dev/null 2>&1; then
  printf 'RViz skipped: rviz2 is not installed\n' >"${rviz_log}"
else
  setsid bash "${lab_root}/scripts/rvclaw_visualization.sh" --run \
    >"${rviz_log}" 2>&1 &
  rviz_pid=$!
  sleep 1
  if kill -0 -- "-${rviz_pid}" 2>/dev/null; then
    printf '[mapping] RViz started; log=%s\n' "${rviz_log}"
  else
    printf '[mapping] RViz failed to stay running; see %s\n' "${rviz_log}" >&2
    rviz_pid=""
  fi
fi

printf 'SLAM_MAPPING_READY\n'
failure_reason="mapping_process_exited_before_save"
if [[ "${control_mode}" == "voice" ]]; then
  write_voice_state "READY"
  printf 'Use PS2 for slow forward-dominant mapping. Turn PS2 off, wait for a full stop, then say the mapping-finish command.\n'
else
  printf 'Use PS2 for slow forward-dominant mapping. Turn PS2 off and wait for a full stop before pressing Enter.\n'
fi
if [[ ${snapshot_interval_s} =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ ${snapshot_interval_s} != 0 ]]; then
  snapshot_loop() {
    local index=1
    local snapshot_dir
    while true; do
      sleep "${snapshot_interval_s}" || return 0
      snapshot_dir="${map_dir}/map-v$(printf '%03d' "${index}")"
      mkdir -p "${snapshot_dir}"
      timeout 20s ros2 run nav2_map_server map_saver_cli \
        -f "${snapshot_dir}/map" \
        --ros-args \
        -p save_map_timeout:=10.0 \
        -p free_thresh_default:=0.25 \
        -p occupied_thresh_default:=0.65
      if [[ -s "${snapshot_dir}/map.yaml" && -s "${snapshot_dir}/map.pgm" ]]; then
        sha256sum "${snapshot_dir}"/* >"${snapshot_dir}/SHA256SUMS"
        printf '[mapping] saved live snapshot: %s\n' "${snapshot_dir}"
      fi
      index=$((index + 1))
    done
  }
  snapshot_loop >"${run_dir}/snapshot-loop.log" 2>&1 &
  snapshot_pid=$!
  printf '[mapping] live map snapshots enabled: every %ss\n' "${snapshot_interval_s}"
fi

if [[ "${control_mode}" == "voice" ]]; then
  while true; do
    if [[ -s "${control_file}" ]]; then
      control_command="$(tr -d '\r\n[:space:]' <"${control_file}" | tr '[:lower:]' '[:upper:]')"
      rm -f -- "${control_file}"
      control_generation=$((control_generation + 1))
      case "${control_command}" in
        SAVE)
          write_voice_state "VERIFYING_STOP"
          if python3 "${lab_root}/scripts/navigation_stationary_probe.py" \
            --duration "${stationary_duration_s}" \
            --minimum-messages "${stationary_minimum_messages}" \
            --output "${run_dir}/pre-save-stationarity.json" \
            | tee "${run_dir}/pre-save-stationarity.log"; then
            break
          fi
          printf '[mapping] save request rejected because the chassis is not stationary\n' >&2
          write_voice_state "READY" "not_stationary"
          ;;
        CANCEL)
          terminal_state="CANCELLED"
          failure_reason="cancelled_by_voice_command"
          exit 130
          ;;
        *)
          printf '[mapping] ignored invalid voice control command: %s\n' "${control_command}" >&2
          write_voice_state "READY" "invalid_control_command"
          ;;
      esac
    fi
    if ! kill -0 -- "-${mapping_pid}" 2>/dev/null; then
      failure_reason="slam_process_exited_before_save"
      exit 3
    fi
    sleep 0.2
  done
else
  read -r _
  python3 "${lab_root}/scripts/navigation_stationary_probe.py" \
    --duration "${stationary_duration_s}" \
    --minimum-messages "${stationary_minimum_messages}" \
    --output "${run_dir}/pre-save-stationarity.json" \
    | tee "${run_dir}/pre-save-stationarity.log"
fi

if [[ -n "${snapshot_pid}" ]] && kill -0 "${snapshot_pid}" 2>/dev/null; then
  kill -TERM "${snapshot_pid}" 2>/dev/null || true
  wait "${snapshot_pid}" 2>/dev/null || true
  snapshot_pid=""
fi

write_voice_state "SAVING"

failure_reason="grid_map_save_failed"
if ! timeout 20s ros2 run nav2_map_server map_saver_cli \
  -f "${map_dir}/map" \
  --ros-args \
  -p save_map_timeout:=10.0 \
  -p free_thresh_default:=0.25 \
  -p occupied_thresh_default:=0.65 \
  | tee "${run_dir}/map-saver.log"; then
  exit 4
fi

failure_reason="posegraph_save_failed"
if ! timeout 15s ros2 service call /slam_toolbox/serialize_map \
  slam_toolbox/srv/SerializePoseGraph \
  "{filename: '${map_dir}/posegraph'}" \
  | tee "${run_dir}/serialize-map.log"; then
  exit 5
fi

failure_reason="saved_map_validation_failed"
test -s "${map_dir}/map.yaml"
test -s "${map_dir}/map.pgm"
grep -Eq 'result[=:][[:space:]]*0' "${run_dir}/serialize-map.log"
cp "${lab_root}/ros2_ws/src/rvclaw_nav2_bringup/config/slam_toolbox.yaml" \
  "${map_dir}/slam_toolbox.yaml"
cp "${lab_root}/ros2_ws/src/rvclaw_nav2_bringup/config/physical_measurements.yaml" \
  "${map_dir}/physical_measurements.yaml"
sha256sum "${map_dir}"/* >"${map_dir}/SHA256SUMS"
write_latest_map

failure_reason=""
run_completed=1
