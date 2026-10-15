#!/usr/bin/env bash
# Run a route only after the persistent localization guard reaches HOT_READY.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
ROOT=${RVCLAW_LAB_ROOT:-$DEFAULT_ROOT}
BASE_ROOT=${RVCLAW_BASE_ROOT:-/data/rvclaw-k3-lab}
SITE_DIR=${RVCLAW_SITE_DIR:-}
MAP_FILE=${RVCLAW_MAP_FILE:-}
CACHE_FILE=${RVCLAW_LOCALIZATION_CACHE:-$ROOT/state/localization-pose.json}
WAYPOINTS_FILE=${RVCLAW_NAV2_WAYPOINTS_FILE:-}
if [[ -n $SITE_DIR ]]; then
  # The map selector/UI supplies a site directory, never a code change.
  # Explicit file overrides are permitted only if the Python binding check
  # confirms that the waypoint catalog declares this very map.
  MAP_FILE=${MAP_FILE:-$SITE_DIR/map.yaml}
  WAYPOINTS_FILE=${WAYPOINTS_FILE:-$SITE_DIR/waypoints.yaml}
fi
ROUTE_NAMES=${RVCLAW_NAV2_WAYPOINT_ROUTE:-A,B,C,ORIGIN}
INSPECTION_CONFIG=${RVCLAW_INSPECTION_CONFIG:-}
CHECKPOINT_FILE=${RVCLAW_ROUTE_CHECKPOINT_FILE:-/data/RVClaw/runtime/inspection-route-checkpoint.json}
RESUME_CONTEXT=${RVCLAW_ROUTE_RESUME_CONTEXT:-}
FORCE_RELOCALIZATION=${RVCLAW_FORCE_RELOCALIZATION:-0}
ROUTE_ARM_TOKEN=WHEELS_GROUND_PS2_OFF_NAV2_WAYPOINT_ROUTE
CONTROL_NICE=${RVCLAW_CONTROL_NICE:--10}
INSPECTION_ARGS=()
RESUME_ARGS=()
if [[ -n $INSPECTION_CONFIG ]]; then
  INSPECTION_ARGS=(--inspection-config "$INSPECTION_CONFIG")
fi
if [[ -n $RESUME_CONTEXT ]]; then
  # This wrapper runs nav2_fast_localization.py to PASS immediately before the
  # route client.  Mark only this guarded path as allowed to rebase continuity
  # on the freshly verified current AMCL pose.
  RESUME_ARGS=(
    --resume-context "$RESUME_CONTEXT"
    --resume-localization-verified
  )
fi

if [[ $CONTROL_NICE != 0 && $CONTROL_NICE != -10 ]]; then
  echo "RVCLAW_CONTROL_NICE must be 0 or -10" >&2
  exit 2
fi

raise_process_priority() {
  local label=$1
  local pattern=$2
  local pid
  local observed_nice
  local -a pids=()
  local -a tids=()

  mapfile -t pids < <(pgrep -f "$pattern")
  if [[ ${#pids[@]} -ne 1 ]]; then
    echo "expected exactly one $label process, found ${#pids[@]}" >&2
    return 1
  fi
  pid=${pids[0]}
  for task_dir in /proc/"$pid"/task/*; do
    tids+=("${task_dir##*/}")
  done
  sudo -n renice -n "$CONTROL_NICE" -p "${tids[@]}"
  observed_nice=$(ps -o ni= -p "$pid" | tr -d ' ')
  if [[ $observed_nice != "$CONTROL_NICE" ]]; then
    echo "$label priority readback failed: pid=$pid nice=$observed_nice" >&2
    return 1
  fi
  echo "$label priority verified: pid=$pid nice=$observed_nice"
}

configure_control_priority() {
  if [[ $CONTROL_NICE == 0 ]]; then
    echo "control priority override disabled"
    return 0
  fi
  raise_process_priority \
    hardware_node \
    '/rvclaw_stm32_base/hardware_node([[:space:]]|$)'
  raise_process_priority \
    cmd_vel_adapter \
    '/rvclaw_nav2_bringup/cmd_vel_adapter([[:space:]]|$)'
  raise_process_priority \
    controller_server \
    '/nav2_controller/controller_server([[:space:]]|$)'
  raise_process_priority \
    planner_server \
    '/nav2_planner/planner_server([[:space:]]|$)'
  raise_process_priority \
    bt_navigator \
    '/nav2_bt_navigator/bt_navigator([[:space:]]|$)'
  raise_process_priority \
    behavior_server \
    '/nav2_behaviors/behavior_server([[:space:]]|$)'
  raise_process_priority \
    velocity_smoother \
    '/nav2_velocity_smoother/velocity_smoother([[:space:]]|$)'
  raise_process_priority \
    collision_monitor \
    '/nav2_collision_monitor/collision_monitor([[:space:]]|$)'
}

raise_route_priority() {
  local observed_nice

  if [[ $CONTROL_NICE == 0 ]]; then
    echo "route-client priority override disabled"
    return 0
  fi
  # exec below preserves the shell's nice value for the Python route client
  # and for its independent deadman keepalive thread. This keeps the 30 ms
  # heartbeat from being starved by RViz and vision inference bursts.
  sudo -n renice -n "$CONTROL_NICE" -p "$$"
  observed_nice=$(ps -o ni= -p "$$" | tr -d ' ')
  if [[ $observed_nice != "$CONTROL_NICE" ]]; then
    echo "route-client priority readback failed: pid=$$ nice=$observed_nice" >&2
    return 1
  fi
  echo "route-client priority verified: pid=$$ nice=$observed_nice"
}

if [[ ${1:-} == --describe && $# -eq 1 ]]; then
  cat <<EOF
hardware_stack_started=false
lidar_stack_started=false
nav2_stack_started=false
requires_existing_persistent_runtime=true
pre_route_gate=nav2_fast_localization.py with HOT_READY final state
ambiguous_static_match=bounded_global_scan_only_in_run_mode_then_revalidate
control_priority=$CONTROL_NICE for hardware_node, cmd_vel_adapter, controller_server, planner_server, bt_navigator, behavior_server, velocity_smoother, collision_monitor, and route_client
route=$ROUTE_NAMES
waypoints=$WAYPOINTS_FILE
inspection_config=${INSPECTION_CONFIG:-disabled}
motion_requires=--run and existing physical field supervision
EOF
  exit 0
fi

MODE=${1:-}
if [[ $MODE != --preflight && $MODE != --run && $MODE != --inspection-readiness ]] || [[ $# -ne 1 ]]; then
  printf 'Usage: %s --describe | --inspection-readiness | --preflight | --run\n' "$0" >&2
  exit 2
fi

if [[ -z $MAP_FILE || -z $WAYPOINTS_FILE || ! -f $MAP_FILE || ! -f $WAYPOINTS_FILE ]]; then
  echo "select an existing site map and waypoint catalog before navigation" >&2
  exit 2
fi

set +u
source /opt/ros/humble/setup.bash
if [[ $BASE_ROOT != "$ROOT" && -f $BASE_ROOT/ros2_ws/install/setup.bash ]]; then
  source "$BASE_ROOT/ros2_ws/install/setup.bash"
fi
source "$ROOT/ros2_ws/install/setup.bash"
set -u
export RVCLAW_MAP_FILE="$MAP_FILE"

# Fail before localization or any route action if the selected point catalog
# belongs to a different map. Studio/site selection changes data, not code.
python3 -c 'import sys; from rvclaw_nav2_bringup.site_bundle_contract import validate_map_waypoint_binding; validate_map_waypoint_binding(sys.argv[1], sys.argv[2])' \
  "$MAP_FILE" "$WAYPOINTS_FILE"

if [[ $MODE == --inspection-readiness ]]; then
  if [[ -z $INSPECTION_CONFIG ]]; then
    echo "RVCLAW_INSPECTION_CONFIG is required for --inspection-readiness" >&2
    exit 2
  fi
  exec python3 "$SCRIPT_DIR/rvclaw_waypoint_route_client.py" \
    --inspection-readiness-only \
    --waypoints-file "$WAYPOINTS_FILE" \
    --route "$ROUTE_NAMES" \
    "${INSPECTION_ARGS[@]}"
fi

# A matching YAML/catalog pair is insufficient when the running map_server
# still serves a previous site. This read-only check compares every /map cell
# and its geometry with the selected map before localization and motion.
python3 "$SCRIPT_DIR/verify_runtime_map.py" --map-file "$MAP_FILE"

mkdir -p -- "$ROOT/runs/persistent-route"
RUN_ID=$(date +%Y%m%d-%H%M%S)
FAST_TRACE=${RVCLAW_FAST_LOCALIZATION_TRACE_FILE:-$ROOT/runs/persistent-route/$RUN_ID-fast-localization.json}
ROUTE_TRACE=${RVCLAW_ROUTE_TRACE_FILE:-$ROOT/runs/persistent-route/$RUN_ID-route.json}

if [[ $FORCE_RELOCALIZATION == 1 ]]; then
  REPOSITION_RESULT=$(timeout 10s ros2 service call \
    /rvclaw/localization/mark_repositioned std_srvs/srv/Trigger '{}')
  if ! grep -Eiq 'success=(true|1)' <<<"${REPOSITION_RESULT// /}"; then
    echo "failed to mark manual reposition before route resume" >&2
    echo "$REPOSITION_RESULT" >&2
    exit 8
  fi
fi

LOCALIZATION_RC=0
python3 "$SCRIPT_DIR/nav2_fast_localization.py" \
  --map-file "$MAP_FILE" \
  --cache-file "$CACHE_FILE" \
  --trace-file "$FAST_TRACE" || LOCALIZATION_RC=$?

if [[ $LOCALIZATION_RC -eq 4 ]]; then
  # Exit 4 is MOTION_REQUIRED, not authorization to bypass localization.
  if ! python3 -c 'import json, sys; d=json.load(open(sys.argv[1], encoding="utf-8")); sys.exit(0 if d.get("status") == "MOTION_REQUIRED" and d.get("motion_authorized") is False else 1)' "$FAST_TRACE"; then
    echo "localization exit code and MOTION_REQUIRED trace disagree; refusing scan" >&2
    exit 4
  fi
  if [[ $MODE != --run ]]; then
    echo "localization needs movement; preflight never authorizes a scan" >&2
    exit 4
  fi
  GLOBAL_TRACE=$ROOT/runs/persistent-route/$RUN_ID-global-scan.json
  python3 "$SCRIPT_DIR/nav2_global_localization_scan.py" \
    --arm-token WHEELS_GROUND_PS2_OFF_AMCL_GLOBAL_SCAN \
    --trace-file "$GLOBAL_TRACE"
  python3 "$SCRIPT_DIR/nav2_fast_localization.py" \
    --map-file "$MAP_FILE" \
    --cache-file "$CACHE_FILE" \
    --trace-file "$ROOT/runs/persistent-route/$RUN_ID-post-scan-localization.json"
elif [[ $LOCALIZATION_RC -ne 0 ]]; then
  exit "$LOCALIZATION_RC"
fi

if [[ $MODE == --preflight ]]; then
  exec python3 "$SCRIPT_DIR/rvclaw_waypoint_route_client.py" \
    --preflight-only \
    --waypoints-file "$WAYPOINTS_FILE" \
    --route "$ROUTE_NAMES" \
    "${RESUME_ARGS[@]}" \
    "${INSPECTION_ARGS[@]}" \
    --trace-file "$ROUTE_TRACE"
fi

PRIORITY_LOG=$ROOT/runs/persistent-route/$RUN_ID-control-priority.log
if ! configure_control_priority >"$PRIORITY_LOG" 2>&1; then
  cat "$PRIORITY_LOG" >&2
  echo "failed to apply verified control-process priority" >&2
  exit 6
fi
if ! raise_route_priority >>"$PRIORITY_LOG" 2>&1; then
  cat "$PRIORITY_LOG" >&2
  echo "failed to apply verified route-client priority" >&2
  exit 7
fi
{
  echo "allowed_cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/$$/status)"
  echo "loadavg=$(cat /proc/loadavg)"
  if [[ -r /proc/pressure/cpu ]]; then
    echo "cpu_pressure=$(tr '\n' ';' </proc/pressure/cpu)"
  fi
} >>"$PRIORITY_LOG"
cat "$PRIORITY_LOG"

exec python3 "$SCRIPT_DIR/rvclaw_waypoint_route_client.py" \
  --arm-token "$ROUTE_ARM_TOKEN" \
  --waypoints-file "$WAYPOINTS_FILE" \
  --route "$ROUTE_NAMES" \
  --checkpoint-file "$CHECKPOINT_FILE" \
  "${RESUME_ARGS[@]}" \
  "${INSPECTION_ARGS[@]}" \
  --trace-file "$ROUTE_TRACE"
