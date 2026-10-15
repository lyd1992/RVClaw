#!/usr/bin/env python3
"""Navigate a reviewed absolute waypoint route with fail-stop semantics."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from uuid import uuid4
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PolygonStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import DriveOnHeading, Spin
from rclpy.action import ActionClient
from rcl_interfaces.srv import GetParameters, SetParametersAtomically
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(
    0,
    str(WORKSPACE_ROOT / "ros2_ws" / "src" / "rvclaw_nav2_bringup"),
)

import rvclaw_waypoint_client as WAYPOINT  # noqa: E402
import nav2_heading_spin_left_0350 as HEADING  # noqa: E402
from rvclaw_nav2_bringup import dynamic_obstacle_policy as DYNAMIC  # noqa: E402
from rvclaw_nav2_bringup.inspection_task_runtime import (  # noqa: E402
    InspectionExecutor,
    load_inspection_plan,
    plan_summary as inspection_plan_summary,
)
from rvclaw_nav2_bringup.navigation_recovery_policy import (  # noqa: E402
    PROMPTS as NAVIGATION_RECOVERY_PROMPTS,
    classify_navigation_failure,
    retry_is_safe,
)
from rvclaw_nav2_bringup.waypoint_catalog import (  # noqa: E402
    load_waypoints,
)
from rvclaw_nav2_bringup.site_bundle_contract import (  # noqa: E402
    validate_map_waypoint_binding,
)
from rvclaw_nav2_bringup.safe_egress_policy import (  # noqa: E402
    EGRESS_ACTION_TIME_ALLOWANCE_S,
    EGRESS_BODY_FRONT_M,
    EGRESS_DISTANCE_M,
    EGRESS_LASER_X_M,
    EGRESS_MIN_DISTANCE_M,
    EGRESS_RESULT_TIMEOUT_S,
    EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
    EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
    EGRESS_MODE_SIDE_NEAR,
    EGRESS_MODE_STANDARD,
    EGRESS_SIDE_NEAR_SAFETY_MARGIN_M,
    EGRESS_SPEED_M_S,
    detect_side_near_obstacle,
    evaluate_front_egress_scan,
    evaluate_runtime_egress_monitor,
    remaining_egress_distance,
    select_safe_egress,
)
from rvclaw_nav2_bringup.side_turn_departure_policy import (  # noqa: E402
    NOMINAL_ROTATION_RADIUS_M,
    TURN_REVIEW_MIN_ANGLE_RAD,
    check_nominal_static_rotation_disk,
    classify_side_turn_departure,
)
from rvclaw_nav2_bringup.staged_egress_policy import (  # noqa: E402
    MAX_STEPS as MAX_AUTONOMOUS_EGRESS_STEPS,
    step_budget,
    validate_chain_continuity,
    validate_completed_step,
)


ROUTE_ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_NAV2_WAYPOINT_ROUTE"
# This equipment-level qualification is still missing: the 2-D lidar plane
# cannot certify upper chassis protrusions or the 10 mm side-near margin.
# Keep this recovery branch inactive even if the scan/map gates pass. This is
# not a per-site parameter and must not be enabled by a map/voice command.
SIDE_NEAR_EGRESS_FIELD_CERTIFIED = False
DEFAULT_ROUTE = ("A", "B", "C", "ORIGIN")
ROUTE_MAX_DURATION_S = 600.0
# Obstacle waiting and Nav2 retries share the transit deadline. Extend only
# ORIGIN travel; inspection legs and the separate final-heading limit retain
# their existing budgets.
ORIGIN_TRANSIT_ACTION_RESULT_TIMEOUT_S = 240.0
# The reviewed 0.309 m x 0.255 m footprint has a circumscribed radius of
# about 0.20 m.  Include the 0.05 m ORIGIN XY tolerance plus 0.10 m for AMCL
# and costmap rasterization so a future return preflight does not mistake the
# robot currently covering ORIGIN for an obstacle at that future goal. Runtime
# planning and collision checks remain mandatory when C->ORIGIN is executed.
MAX_DEFERRED_RETURN_DISTANCE_M = 0.35
MAX_INITIAL_YAW_INPUT_DEG = 1000000.0
ORIGIN_HEADING_SKIP_TOLERANCE_RAD = math.radians(4.0)
ORIGIN_HEADING_TOLERANCE_RAD = math.radians(12.0)
# Meter framing is much less tolerant than the final ORIGIN pose.  A 12-degree
# terminal error was enough to push the phone/display to the edge of the camera
# image, so inspection alignment has its own strict bound.
INSPECTION_HEADING_COMPLETION_TOLERANCE_RAD = math.radians(10.0)
ORIGIN_HEADING_ACTION_TIME_ALLOWANCE_S = 10
ORIGIN_HEADING_ACTION_RESULT_TIMEOUT_S = 15.0
ORIGIN_HEADING_MIN_GUARDED_RAD_S = 0.20
ORIGIN_HEADING_MAX_GUARDED_RAD_S = 0.60
# A long in-place Spin can be aborted by Nav2's collision checker when the
# forward safety polygon is occupied.  Keep each relative Spin below the
# observed abort distance, while retaining the same Nav2 and hardware path.
ORIGIN_HEADING_SPIN_CHUNK_MAX_RAD = 0.45
ORIGIN_HEADING_MAX_CHUNKS = 12
ORIGIN_HEADING_MAX_TOTAL_RAD = math.pi + math.radians(30.0)
ORIGIN_GOAL_XY_TOLERANCE_M = 0.05
ORIGIN_GOAL_YAW_TOLERANCE_RAD = 0.25
ORIGIN_POST_ENTRY_MAX_ROTATION_RAD = ORIGIN_HEADING_MAX_TOTAL_RAD
ORIGIN_REPLAN_DISTANCE_M = 0.25
CONTINUOUS_ROUTE_MAX_POSITION_JUMP_M = 0.25
CONTINUOUS_ROUTE_MAX_YAW_JUMP_RAD = math.radians(30.0)
VERIFIED_RESUME_REBASE_KEY = "verified_resume_rebase_from_current_amcl"

# A lethal start cell can be a historical chassis return left in the static
# map.  The route may use a bounded forward escape only when the independent
# live-scan and static-map gates prove the newly swept corridor is clear.
NEAR_START_EGRESS_PATH_STATUS = GoalStatus.STATUS_ABORTED
NEAR_START_EGRESS_MIN_START_COST = WAYPOINT.BASE.MAXIMUM_START_COST + 1
# ``pre_arm_front_clearance_m`` is the raw forward range from the lidar, not
# the vehicle-front gap.  Convert the reviewed 0.30 m physical gap through
# the measured lidar/body geometry before admitting the adaptive policy.
NEAR_START_EGRESS_MIN_PHYSICAL_FRONT_GAP_M = 0.30
NEAR_START_EGRESS_MIN_FRONT_CLEARANCE_M = (
    EGRESS_BODY_FRONT_M
    + NEAR_START_EGRESS_MIN_PHYSICAL_FRONT_GAP_M
    - EGRESS_LASER_X_M
)

# Stop the position controller before final rotation. DWB's rotation critic
# latches its XY window while the non-stateful goal checker keeps reevaluating
# XY. A changed scan view can make these disagree and produce endless turns.
# Use the existing guarded Spin path, then independently validate final XY.
ORIGIN_NAVIGATION_MODE = "split_spin"
# Intermediate waypoints are position checkpoints.  Their catalog heading is
# retained as metadata. Runtime and preflight use a travel-bearing hint plus a
# dedicated goal checker that accepts any endpoint yaw after XY is reached.
INTERMEDIATE_NAVIGATION_MODE = "position_only"
POSITION_ONLY_BT_NAME = "navigate_to_pose_position_only.xml"
ORIGIN_FULL_POSE_BT_NAME = "navigate_to_pose_origin_heading.xml"
DEFAULT_ARRIVAL_EVENT_DIR = "/data/RVClaw/runtime/waypoint-arrivals"
DEFAULT_INSPECTION_RESULT_DIR = "/data/RVClaw/runs/route-inspections"
DEFAULT_ROUTE_CHECKPOINT_FILE = "/data/RVClaw/runtime/inspection-route-checkpoint.json"
DEFAULT_SPEECH_ACK_TIMEOUT_S = 12.0
# The safety-pause prompt takes longer than four seconds to speak.  Wait for
# the completed playback acknowledgement before reporting a false timeout.
RECOVERY_SPEECH_ACK_TIMEOUT_S = 12.0
SPEECH_ACK_POLL_INTERVAL_S = 0.05

AMCL_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def shortest_heading_delta(target_yaw: float, current_yaw: float) -> float:
    """Return the shortest signed rotation from current to target."""
    return WAYPOINT.BASE.normalize_angle(target_yaw - current_yaw)


def normalize_yaw_degrees(value: float) -> float:
    """Normalize any finite degree input to the ROS [-180, 180) convention."""
    if not math.isfinite(value):
        raise ValueError("initial yaw must be finite")
    if abs(value) > MAX_INITIAL_YAW_INPUT_DEG:
        raise ValueError(
            "initial yaw must be within +/-%.0f degrees"
            % MAX_INITIAL_YAW_INPUT_DEG
        )
    return math.degrees(WAYPOINT.BASE.normalize_angle(math.radians(value)))


def validate_initial_pose(x: float, y: float, yaw_deg: float):
    """Validate and normalize a user-supplied map-frame start pose."""
    values = (float(x), float(y), float(yaw_deg))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("initial pose values must be finite")
    # Metric coordinates belong to the selected map, whose extent and origin
    # are not fixed by this route client. The live map/localization preflight
    # rejects an impossible pose; a hardcoded +/-20 m box rejects valid sites.
    normalized_yaw_deg = normalize_yaw_degrees(values[2])
    return {
        "x": values[0],
        "y": values[1],
        "yaw_deg": normalized_yaw_deg,
        "yaw": math.radians(normalized_yaw_deg),
    }


def load_global_localization_pose(trace_path):
    """Read a converged pose from a passed AMCL global-scan trace."""
    path = Path(trace_path).expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("global localization trace is not readable") from exc
    if document.get("status") != "PASS":
        raise ValueError("global localization trace is not a PASS result")
    pose = document.get("localization", {}).get("final_pose")
    if not isinstance(pose, dict):
        raise ValueError("global localization trace has no final pose")
    try:
        result = validate_initial_pose(
            pose["x"], pose["y"], math.degrees(float(pose["yaw"]))
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("global localization final pose is invalid") from exc
    result["source"] = "amcl_global_localization_scan"
    return result


def _is_near_start_path_failure(node, path_error=None):
    """Recognize only the reviewed lethal-start/empty-path condition."""
    return (
        getattr(node, "start_cost", None) is not None
        and int(node.start_cost) >= NEAR_START_EGRESS_MIN_START_COST
        and getattr(node, "path_check_status", None)
        == NEAR_START_EGRESS_PATH_STATUS
        and getattr(node, "latest_amcl_pose", None) is not None
        and getattr(node, "pre_arm_front_clearance_m", None) is not None
        and float(node.pre_arm_front_clearance_m)
        >= NEAR_START_EGRESS_MIN_FRONT_CLEARANCE_M
    )


class SideNearTurnEgressRequired(RuntimeError):
    """The planner succeeded, but its departure turn is not sweep-clear."""


class SideNearTurnUnsafe(RuntimeError):
    """An uncertain near-side turn cannot be released automatically."""


def _active_map_file():
    """Never silently compare a new-map pose with the old demonstration map."""
    configured = os.environ.get("RVCLAW_MAP_FILE", "").strip()
    if not configured:
        raise RuntimeError("current map is not configured for route safety checks")
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("current route map is unavailable: %s" % path)
    return str(path)


def _evaluate_node_egress_gates(node):
    """Evaluate live and static gates using the node's newest observations."""
    if node.latest_scan is None or node.latest_amcl_pose is None:
        raise RuntimeError("near-start egress requires a fresh scan and AMCL pose")
    map_file = _active_map_file()
    from rvclaw_nav2_bringup.scan_map_consistency import load_static_map

    grid = load_static_map(map_file)
    pose = node.latest_amcl_pose
    candidates = [
        {
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
            "yaw_deg": math.degrees(pose["yaw"]),
        }
    ]
    side_obstacle = detect_side_near_obstacle(
        node.latest_scan.ranges,
        node.latest_scan.angle_min,
        node.latest_scan.angle_increment,
        node.latest_scan.range_min,
        node.latest_scan.range_max,
    )
    mode = EGRESS_MODE_SIDE_NEAR if side_obstacle["detected"] else EGRESS_MODE_STANDARD
    result = select_safe_egress(
        node.latest_scan.ranges,
        node.latest_scan.angle_min,
        node.latest_scan.angle_increment,
        node.latest_scan.range_min,
        node.latest_scan.range_max,
        grid,
        candidates,
        mode=mode,
    )
    if mode == EGRESS_MODE_SIDE_NEAR and not SIDE_NEAR_EGRESS_FIELD_CERTIFIED:
        result.update({
            "status": "FAIL",
            "decision": "REQUIRE_OPERATOR_REPOSITION",
            "selected_distance_m": None,
            "failures": list(result.get("failures", [])) + [
                "side-near escape is not certified for the complete robot envelope"
            ],
        })
    if side_obstacle["status"] != "PASS":
        failures = list(result.get("failures", []))
        failures.append("side obstacle is inside the side-near safety margin")
        result["status"] = "FAIL"
        result["decision"] = "REQUIRE_OPERATOR_REPOSITION"
        result["selected_distance_m"] = None
        result["failures"] = failures
    result["map_file"] = map_file
    result["pose"] = dict(pose)
    result["side_near_obstacle"] = side_obstacle
    return result


def _check_continuous_route_localization(node, continuity_anchor):
    """Validate pose continuity without repeating an ambiguous global scan search."""
    if continuity_anchor is None:
        return False
    if node.latest_amcl_pose is None:
        raise RuntimeError("continuous route localization requires a fresh AMCL pose")
    current = node.latest_amcl_pose
    if (
        isinstance(continuity_anchor, dict)
        and continuity_anchor.get(VERIFIED_RESUME_REBASE_KEY) is True
    ):
        # The persistent wrapper sets this marker only after the independent
        # fast-localization gate has returned PASS/HOT_READY.  A resumed segment
        # must therefore start from that freshly verified current pose, not from
        # the last completed waypoint: the robot may have travelled safely into
        # the failed segment before an obstacle forced a stop.
        result = {
            "status": "PASS",
            "mode": "VERIFIED_RESUME_REBASE",
            "failures": [],
            "anchor_pose": dict(current),
            "current_pose": dict(current),
            "position_jump_m": 0.0,
            "yaw_jump_rad": 0.0,
            "maximum_position_jump_m": CONTINUOUS_ROUTE_MAX_POSITION_JUMP_M,
            "maximum_yaw_jump_rad": CONTINUOUS_ROUTE_MAX_YAW_JUMP_RAD,
        }
        node.localization_gate_result = result
        node.localization_gate_scan_sample_count = node.scan_samples
        node.localization_gate_checked_monotonic = time.monotonic()
        node.localization_gate_pose = dict(current)
        node._record("verified_resume_localization_rebase", result)
        return True
    try:
        position_jump_m = math.hypot(
            float(current["x"]) - float(continuity_anchor["x"]),
            float(current["y"]) - float(continuity_anchor["y"]),
        )
        yaw_jump_rad = abs(
            WAYPOINT.BASE.normalize_angle(
                float(current["yaw"]) - float(continuity_anchor["yaw"])
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("continuous route localization anchor is invalid") from exc
    failures = []
    if position_jump_m > CONTINUOUS_ROUTE_MAX_POSITION_JUMP_M:
        failures.append(
            "AMCL position jumped %.3f m between route segments (limit %.3f m)"
            % (position_jump_m, CONTINUOUS_ROUTE_MAX_POSITION_JUMP_M)
        )
    if yaw_jump_rad > CONTINUOUS_ROUTE_MAX_YAW_JUMP_RAD:
        failures.append(
            "AMCL yaw jumped %.1f deg between route segments (limit %.1f deg)"
            % (
                math.degrees(yaw_jump_rad),
                math.degrees(CONTINUOUS_ROUTE_MAX_YAW_JUMP_RAD),
            )
        )
    result = {
        "status": "FAIL" if failures else "PASS",
        "mode": "CONTINUOUS_ROUTE",
        "failures": failures,
        "anchor_pose": dict(continuity_anchor),
        "current_pose": dict(current),
        "position_jump_m": position_jump_m,
        "yaw_jump_rad": yaw_jump_rad,
        "maximum_position_jump_m": CONTINUOUS_ROUTE_MAX_POSITION_JUMP_M,
        "maximum_yaw_jump_rad": CONTINUOUS_ROUTE_MAX_YAW_JUMP_RAD,
    }
    node.localization_gate_result = result
    node.localization_gate_scan_sample_count = node.scan_samples
    node.localization_gate_checked_monotonic = time.monotonic()
    node.localization_gate_pose = dict(current)
    node._record("continuous_route_localization_gate", result)
    if failures:
        raise RuntimeError(
            "continuous route localization gate failed: %s" % "; ".join(failures)
        )
    return True


class LiveObstacleWait:
    """One nonblocking prompt per sustained stop, rearmed after stable clearance."""

    def __init__(self):
        self.started = None
        self.announced = False
        self.clear_started = None
        self.last_announced = None

    def update(self, blocked, now, clear=True):
        if not blocked:
            self.started = None
            if not clear:
                self.clear_started = None
                return False
            if self.clear_started is None:
                self.clear_started = now
            if now - self.clear_started >= 2.0:
                self.announced = False
            return False
        self.clear_started = None
        if self.started is None:
            self.started = now
        if (not self.announced and now - self.started >= .6
                and (self.last_announced is None or now - self.last_announced >= 3.0)):
            self.announced = True
            self.last_announced = now
            return True
        return False


def scan_polygon_hit_count(scan, points, tx, ty, yaw):
    """Count real scan returns in the live CollisionMonitor polygon frame."""
    if len(points) < 3:
        return 0
    count = 0
    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance) or not scan.range_min <= distance <= scan.range_max:
            continue
        angle = scan.angle_min + index * scan.angle_increment + yaw
        x, y = tx + distance * math.cos(angle), ty + distance * math.sin(angle)
        inside = False
        previous = points[-1]
        for point in points:
            if ((point.y > y) != (previous.y > y)
                    and x < (previous.x - point.x) * (y - point.y)
                    / (previous.y - point.y) + point.x):
                inside = not inside
            previous = point
        count += int(inside)
    return count


class PositionOnlyWaypointClient(WAYPOINT.WaypointClient):
    """Navigate to a position without enforcing an endpoint heading."""

    def __init__(self, waypoint, start_pose_hint=None, continuity_anchor=None,
                 continuing_egress=False):
        super().__init__(waypoint)
        self.start_pose_hint = (
            None if start_pose_hint is None else dict(start_pose_hint)
        )
        self.continuity_anchor = (
            None if continuity_anchor is None else dict(continuity_anchor)
        )
        self.continuing_egress = bool(continuing_egress)
        self.side_turn_review = None
        self.live_obstacle_wait = LiveObstacleWait()
        self.live_obstacle_navigation_active = False
        self.live_obstacle_input_time = None
        self.live_obstacle_output = None
        self.live_obstacle_polygon = None
        self.live_obstacle_max_points = None
        self.live_obstacle_event_path = None
        self.live_obstacle_tf = Buffer()
        self.live_obstacle_tf_listener = TransformListener(self.live_obstacle_tf, self)
        self.live_obstacle_params = self.create_client(GetParameters, "/collision_monitor/get_parameters")
        self.create_subscription(Twist, "/rvclaw/nav2_cmd_vel_raw", self._on_live_obstacle_input, 10)
        self.create_subscription(PolygonStamped, "/rvclaw/collision_monitor/front_stop",
                                 self._on_live_obstacle_polygon, 10)
        self.create_timer(.10, self._check_live_obstacle_wait)
        self.position_only_behavior_tree = str(
            Path(get_package_share_directory("rvclaw_nav2_bringup"))
            / "behavior_trees"
            / POSITION_ONLY_BT_NAME
        )

    def _on_live_obstacle_input(self, message):
        self.live_obstacle_input_time = time.monotonic()

    def _on_live_obstacle_polygon(self, message):
        self.live_obstacle_polygon = message

    def _on_monitor_command(self, message):
        super()._on_monitor_command(message)
        self.live_obstacle_output = message

    def _load_live_obstacle_configuration(self):
        if self.live_obstacle_params.service_is_ready():
            request = GetParameters.Request()
            request.names = ["front_stop.max_points", "front_stop.enabled"]
            pending = self.live_obstacle_params.call_async(request)
            def configured(done):
                try:
                    values = done.result().values
                    if values[0].type == 2 and values[1].type == 1 and values[1].bool_value:
                        self.live_obstacle_max_points = values[0].integer_value
                except Exception as exc:
                    self._record("live_obstacle_notification_config_error", str(exc))
            pending.add_done_callback(configured)

    def _wait_navigation_result(self, future, timeout_s):
        self.live_obstacle_navigation_active = True
        self._load_live_obstacle_configuration()
        try:
            return super()._wait_navigation_result(future, timeout_s)
        finally:
            self.live_obstacle_navigation_active = False
            self._retire_live_obstacle_event()

    def _retire_live_obstacle_event(self):
        path = getattr(self, "live_obstacle_event_path", None)
        if path is not None:
            Path(path).unlink(missing_ok=True)
            speech_ack_path(path).unlink(missing_ok=True)
            self.live_obstacle_event_path = None

    def _check_live_obstacle_wait(self):
        now = time.monotonic()
        output = self.live_obstacle_output
        polygon = self.live_obstacle_polygon
        fresh = all(stamp is not None and 0. <= now - stamp <= .30 for stamp in (
            self.live_obstacle_input_time, self.latest_monitor_command_monotonic,
            self.latest_scan_monotonic,
        ))
        stopped = (output is not None and all(abs(value) <= 1e-4 for value in (
            output.linear.x, output.linear.y, output.angular.z,
        )) and abs(self.latest_feedback_x) <= WAYPOINT.BASE.MAX_POST_STOP_LINEAR_M_S
            and abs(self.latest_feedback_angular) <= WAYPOINT.BASE.MAX_POST_STOP_ANGULAR_RAD_S)
        active = (self.live_obstacle_navigation_active and self.armed is True
                  and self.state == "ACTIVE" and fresh and stopped
                  and (self.action_feedback_final_distance_m or 0.) > .15
                  and self.live_obstacle_max_points is not None and polygon is not None)
        hits = 0
        if active:
            try:
                transform = self.live_obstacle_tf.lookup_transform(
                    polygon.header.frame_id, self.latest_scan.header.frame_id, rclpy.time.Time())
                t = transform.transform
                yaw = WAYPOINT.BASE.quaternion_to_yaw(t.rotation)
                hits = scan_polygon_hit_count(self.latest_scan, polygon.polygon.points,
                                              t.translation.x, t.translation.y, yaw)
            except Exception:
                active = False  # A missing transform must never invent an obstacle.
        blocked = active and hits > self.live_obstacle_max_points
        # A STOP/SLOWDOWN toggle is not evidence that the obstruction cleared.
        # Rearm the prompt only after measured physical movement resumes.
        moving = (abs(self.latest_feedback_x) > .01
                  or abs(self.latest_feedback_angular) > .10)
        cleared = (fresh and self.armed is True and self.state == "ACTIVE" and moving)
        announce = self.live_obstacle_wait.update(blocked, now, clear=cleared)
        if not blocked:
            self._retire_live_obstacle_event()
        if announce:
            try:
                decision = {"code": "ROUTE_BLOCKED",
                            "started_speech_key": "navigation_obstacle_recovery_started"}
                self.live_obstacle_event_path = publish_navigation_recovery_event(
                    "navigation_recovery_started", self.waypoint, decision,
                    os.getenv("RVCLAW_ARRIVAL_EVENT_DIR", DEFAULT_ARRIVAL_EVENT_DIR),
                )
                self._record("live_obstacle_wait_announced", {"hits": hits, "debounce_s": .6})
            except Exception as exc:
                self._record("live_obstacle_notification_error", str(exc))

    def _compute_goal_pose(self):
        reference = getattr(self, "start_pose_hint", None) or self.latest_amcl_pose
        if reference is None:
            raise RuntimeError("position-only start pose is unavailable")
        delta_x = self.waypoint["x"] - reference["x"]
        delta_y = self.waypoint["y"] - reference["y"]
        if math.hypot(delta_x, delta_y) <= 1e-6:
            goal_yaw = reference["yaw"]
        else:
            goal_yaw = math.atan2(delta_y, delta_x)
        return {
            "x": self.waypoint["x"],
            "y": self.waypoint["y"],
            "yaw": goal_yaw,
        }

    def _configure_navigation_goal(self, goal):
        self._validate_position_only_behavior_tree()
        goal.behavior_tree = self.position_only_behavior_tree

    def _validate_position_only_behavior_tree(self):
        path = Path(self.position_only_behavior_tree)
        if not path.is_file():
            raise RuntimeError(
                "position-only behavior tree is unavailable: %s" % path
            )

    def _check_static_map_localization(self):
        _active_map_file()
        if _check_continuous_route_localization(self, self.continuity_anchor):
            return
        return super()._check_static_map_localization()

    def _pre_arm_gate(self, check_localization: bool = True):
        """Validate position while deliberately allowing the current heading."""
        WAYPOINT.BASE.Nav2ForwardGoalClient._pre_arm_gate(
            self,
            check_localization=check_localization,
        )
        if any(
            not math.isclose(
                self.goal_pose[key], self.waypoint[key], rel_tol=0.0, abs_tol=1e-9
            )
            for key in ("x", "y")
        ):
            raise RuntimeError("position-only waypoint goal was not preserved")
        self._validate_position_only_behavior_tree()

    def _preflight_path(self, start_pose=None):
        super()._preflight_path(start_pose=start_pose)
        if start_pose is not None:
            return  # Future route segments have no current live scan.
        # Planning and action-result callbacks can outlive the scan freshness
        # window in this client. Pump both scan subscriptions again instead of
        # treating an aged local callback as a lidar outage. The same bounded
        # wait and freshness limits still reject an actual stopped publisher.
        before_samples = self.scan_samples
        before_time = getattr(self, "latest_scan_monotonic", None)
        self._wait_for_fresh_scan()
        self._record("post_path_scan_refresh", {
            "samples_before": before_samples,
            "samples_after": self.scan_samples,
            "previous_receive_time": before_time,
            "latest_age_s": time.monotonic() - self.latest_scan_monotonic,
        })
        scan = self.latest_scan
        scan_time = getattr(self, "latest_scan_monotonic", None)
        if (scan is None or self.latest_amcl_pose is None
                or scan_time is None
                or time.monotonic() - scan_time > WAYPOINT.BASE.MAXIMUM_SCAN_AGE_S):
            raise SideNearTurnUnsafe("side turn review requires a fresh scan and pose")
        fields = (scan.ranges, scan.angle_min, scan.angle_increment,
                  scan.range_min, scan.range_max)
        side = detect_side_near_obstacle(*fields)
        self._record("side_detection", side)
        if side["status"] != "PASS":
            raise SideNearTurnUnsafe("side obstacle overlaps reviewed chassis margin")
        delta = self.initial_path_heading_delta_rad
        sharp = delta is not None and abs(delta) >= TURN_REVIEW_MIN_ANGLE_RAD
        if not sharp:
            return
        from rvclaw_nav2_bringup.scan_map_consistency import load_static_map
        from rvclaw_nav2_bringup.scan_trial_policy import evaluate_rotation_clearance
        map_file = _active_map_file()
        grid = load_static_map(map_file)
        rotation_scan = evaluate_rotation_clearance(
            *fields, minimum_radial_m=NOMINAL_ROTATION_RADIUS_M,
            minimum_coverage_deg=330.0,
        )
        rotation_map = check_nominal_static_rotation_disk(
            grid, self.latest_amcl_pose
        )
        review = classify_side_turn_departure(
            delta, side, rotation_scan, rotation_map,
            continuing_egress=self.continuing_egress,
        )
        self.side_turn_review = {
            **review, "side_detection": side,
            "scan_rotation": rotation_scan, "map_rotation": rotation_map,
            "heading_delta_deg": math.degrees(delta),
            "map_file": map_file,
        }
        self._record("side_turn_review", self.side_turn_review)
        if review["status"] == "REQUIRE_STRAIGHT_EGRESS":
            raise SideNearTurnEgressRequired(review["reason"])
        if review["status"] != "NORMAL":
            raise SideNearTurnUnsafe(review["reason"])

    def evaluate_navigation_evidence(self, metrics):
        """Position stage deliberately defers the final heading decision."""
        failures = super().evaluate_navigation_evidence(metrics)
        return [
            failure
            for failure in failures
            if not failure.startswith("final AMCL yaw error exceeded")
        ]


class OriginPoseWaypointClient(WAYPOINT.WaypointClient):
    """Navigate to ORIGIN with a dedicated checker and bounded final rotation."""

    def __init__(self, waypoint, continuity_anchor=None):
        super().__init__(waypoint)
        self.continuity_anchor = (
            None if continuity_anchor is None else dict(continuity_anchor)
        )
        self.origin_behavior_tree = str(
            Path(get_package_share_directory("rvclaw_nav2_bringup"))
            / "behavior_trees"
            / ORIGIN_FULL_POSE_BT_NAME
        )
        self.origin_rotation_guard_active = False
        self.origin_xy_entered = False
        self.origin_xy_entry_pose = None
        self.origin_rotation_last_yaw = None
        self.origin_post_entry_total_rotation_rad = 0.0
        self.origin_rotation_limit_exceeded = False

    def _check_static_map_localization(self):
        _active_map_file()
        if _check_continuous_route_localization(self, self.continuity_anchor):
            return
        return super()._check_static_map_localization()

    def _validate_origin_behavior_tree(self):
        path = Path(self.origin_behavior_tree)
        if not path.is_file():
            raise RuntimeError(
                "ORIGIN full-pose behavior tree is unavailable: %s" % path
            )
        try:
            tree = ET.parse(path)
        except (OSError, ET.ParseError) as exc:
            raise RuntimeError(
                "ORIGIN full-pose behavior tree is invalid: %s" % exc
            ) from exc
        if tree.findall(".//RateController"):
            raise RuntimeError(
                "ORIGIN behavior tree must not time-replan during terminal rotation"
            )
        distance_controllers = tree.findall(".//DistanceController")
        if len(distance_controllers) != 1:
            raise RuntimeError(
                "ORIGIN behavior tree requires one distance-based replanner"
            )
        try:
            replan_distance_m = float(distance_controllers[0].get("distance"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ORIGIN distance replanner has an invalid threshold"
            ) from exc
        if not math.isclose(
            replan_distance_m,
            ORIGIN_REPLAN_DISTANCE_M,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                "ORIGIN distance replanner threshold is not the reviewed %.2f m"
                % ORIGIN_REPLAN_DISTANCE_M
            )
        followers = tree.findall(".//FollowPath")
        if not followers or any(
            follower.get("goal_checker_id") != "origin_goal_checker"
            for follower in followers
        ):
            raise RuntimeError(
                "ORIGIN behavior tree does not select origin_goal_checker"
            )

    def _configure_navigation_goal(self, goal):
        self._validate_origin_behavior_tree()
        goal.behavior_tree = self.origin_behavior_tree

    def _pre_arm_gate(self, check_localization: bool = True):
        super()._pre_arm_gate(check_localization=check_localization)
        self._validate_origin_behavior_tree()

    def _start_origin_rotation_guard(self):
        self.origin_rotation_guard_active = True
        self.origin_xy_entered = False
        self.origin_xy_entry_pose = None
        self.origin_rotation_last_yaw = None
        self.origin_post_entry_total_rotation_rad = 0.0
        self.origin_rotation_limit_exceeded = False
        self._update_origin_rotation_guard()

    def _update_origin_rotation_guard(self):
        if not getattr(self, "origin_rotation_guard_active", False):
            return
        pose = getattr(self, "latest_amcl_pose", None)
        if pose is None:
            return
        goal = getattr(self, "goal_pose", None) or self.waypoint
        distance_m = math.hypot(pose["x"] - goal["x"], pose["y"] - goal["y"])
        if not self.origin_xy_entered:
            if distance_m > ORIGIN_GOAL_XY_TOLERANCE_M:
                return
            self.origin_xy_entered = True
            self.origin_xy_entry_pose = dict(pose)
            self.origin_rotation_last_yaw = pose["yaw"]
            return

        previous_yaw = self.origin_rotation_last_yaw
        self.origin_rotation_last_yaw = pose["yaw"]
        if previous_yaw is None:
            return
        delta = abs(shortest_heading_delta(pose["yaw"], previous_yaw))
        self.origin_post_entry_total_rotation_rad += delta
        if (
            self.origin_post_entry_total_rotation_rad
            > ORIGIN_POST_ENTRY_MAX_ROTATION_RAD
        ):
            self.origin_rotation_limit_exceeded = True

    def _on_amcl(self, message):
        super()._on_amcl(message)
        self._update_origin_rotation_guard()

    def _wait_navigation_result(self, future, timeout_s: float):
        """Fail-stop if terminal heading control accumulates more than 210 deg."""
        self._start_origin_rotation_guard()
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=WAYPOINT.BASE.INTERVAL_S)
                if getattr(self, "motion_map_guard_tripped", False):
                    raise RuntimeError("in-motion scan/map localization disagreement; stopped")
                if self.origin_rotation_limit_exceeded:
                    details = {
                        "xy_entered": self.origin_xy_entered,
                        "xy_entry_pose": self.origin_xy_entry_pose,
                        "total_rotation_rad": self.origin_post_entry_total_rotation_rad,
                        "total_rotation_deg": math.degrees(
                            self.origin_post_entry_total_rotation_rad
                        ),
                        "limit_rad": ORIGIN_POST_ENTRY_MAX_ROTATION_RAD,
                        "limit_deg": math.degrees(
                            ORIGIN_POST_ENTRY_MAX_ROTATION_RAD
                        ),
                    }
                    self._record("origin_rotation_guard_exceeded", details)
                    raise RuntimeError(
                        "ORIGIN post-entry rotation exceeded %.1f deg safety limit"
                        % math.degrees(ORIGIN_POST_ENTRY_MAX_ROTATION_RAD)
                    )
                if future.done():
                    return future.result()
                if self.armed is False or self.state in (
                    "WATCHDOG_LATCHED",
                    "ESTOP_LATCHED",
                ):
                    raise RuntimeError(
                        "hardware safety stop during ORIGIN NavigateToPose "
                        "(state=%s reason=%s)"
                        % (self.state, self.transition_reason)
                    )
            raise RuntimeError("timed out waiting for ORIGIN NavigateToPose result")
        finally:
            self.origin_rotation_guard_active = False

    def result(self, status, error):
        result = super().result(status, error)
        result.update(
            {
                "origin_behavior_tree": ORIGIN_FULL_POSE_BT_NAME,
                "origin_goal_xy_tolerance_m": ORIGIN_GOAL_XY_TOLERANCE_M,
                "origin_goal_yaw_tolerance_rad": ORIGIN_GOAL_YAW_TOLERANCE_RAD,
                "origin_xy_entered": self.origin_xy_entered,
                "origin_xy_entry_pose": self.origin_xy_entry_pose,
                "origin_post_entry_total_rotation_rad": (
                    self.origin_post_entry_total_rotation_rad
                ),
                "origin_post_entry_total_rotation_deg": math.degrees(
                    self.origin_post_entry_total_rotation_rad
                ),
                "origin_post_entry_max_rotation_rad": (
                    ORIGIN_POST_ENTRY_MAX_ROTATION_RAD
                ),
                "origin_rotation_limit_exceeded": (
                    self.origin_rotation_limit_exceeded
                ),
            }
        )
        return result


class SafeEgressClient(WAYPOINT.BASE.Nav2ForwardGoalClient):
    """Move a short distance forward to clear a historical lethal start.

    This is intentionally a separate action from NavigateToPose.  It never
    reverses, strafes, or spins, and it fails closed on every fresh-scan
    violation.  After the motion it forces fresh AMCL samples and an
    independent static-map match before the route is retried.
    """

    COLLISION_PROFILE_PARAMETERS = (
        "front_stop.enabled",
        "front_slow.enabled",
        "side_near_stop.enabled",
    )

    def __init__(self):
        super().__init__()
        self.egress_action = ActionClient(
            self,
            DriveOnHeading,
            "/drive_on_heading",
        )
        self.egress_feedback_samples = 0
        self.egress_feedback_distance_m = 0.0
        self.egress_feedback_invalid = False
        self.egress_runtime_scan_checks = 0
        self.egress_gate = None
        self.egress_distance_m = None
        self.egress_mode = EGRESS_MODE_STANDARD
        self.egress_start_raw_pose = None
        self.egress_final_raw_pose = None
        self.egress_start_amcl_pose = None
        self.egress_final_amcl_pose = None
        self.collision_profile_active = False
        self.collision_profile_restored = False
        self.collision_profile_before = None
        self.collision_profile_after = None
        self.collision_get_client = self.create_client(
            GetParameters, "/collision_monitor/get_parameters"
        )
        self.collision_set_client = self.create_client(
            SetParametersAtomically, "/collision_monitor/set_parameters_atomically"
        )

    def _collision_parameters(self):
        return self.COLLISION_PROFILE_PARAMETERS

    def _read_collision_profile(self):
        if not self.collision_get_client.wait_for_service(
            timeout_sec=WAYPOINT.BASE.SERVICE_TIMEOUT_S
        ):
            raise RuntimeError("collision monitor parameter service is unavailable")
        request = GetParameters.Request()
        request.names = list(self._collision_parameters())
        future = self.collision_get_client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=WAYPOINT.BASE.SERVICE_TIMEOUT_S
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("collision monitor parameter read timed out")
        values = future.result().values
        if len(values) != len(request.names):
            raise RuntimeError("collision monitor parameter read was incomplete")
        profile = {}
        for name, value in zip(request.names, values):
            if value.type != 1:
                raise RuntimeError("collision monitor parameter %s is not boolean" % name)
            profile[name] = bool(value.bool_value)
        return profile

    def _write_collision_profile(self, profile):
        if not self.collision_set_client.wait_for_service(
            timeout_sec=WAYPOINT.BASE.SERVICE_TIMEOUT_S
        ):
            raise RuntimeError("collision monitor atomic parameter service is unavailable")
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter(name=name, value=bool(profile[name])).to_parameter_msg()
            for name in self._collision_parameters()
        ]
        future = self.collision_set_client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=WAYPOINT.BASE.SERVICE_TIMEOUT_S
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("collision monitor atomic parameter write timed out")
        response = future.result()
        if not response.result.successful:
            raise RuntimeError(
                "collision monitor rejected profile: %s" % response.result.reason
            )
        observed = self._read_collision_profile()
        expected = {name: bool(profile[name]) for name in self._collision_parameters()}
        if observed != expected:
            raise RuntimeError("collision monitor profile readback mismatch")
        return observed

    def _enter_side_near_profile(self):
        before = self._read_collision_profile()
        expected = {
            "front_stop.enabled": True,
            "front_slow.enabled": True,
            "side_near_stop.enabled": False,
        }
        if before != expected:
            raise RuntimeError(
                "collision monitor was not in the ordinary safe profile before side-near mode"
            )
        self.collision_profile_before = before
        try:
            after = self._write_collision_profile(
                {
                    **before,
                    "front_stop.enabled": False,
                    "front_slow.enabled": False,
                    "side_near_stop.enabled": True,
                }
            )
        except BaseException:
            # Atomic writes should not partially apply, but fail closed if a
            # service reports an error after changing the profile.
            try:
                self._write_collision_profile(before)
            except BaseException as restore_exc:
                self.publish_estop(True)
                self._record("collision_profile_restore_error", str(restore_exc))
            raise
        self.collision_profile_after = after
        self.collision_profile_active = True
        self._record("collision_profile_entered", after)

    def _restore_collision_profile(self):
        if self.collision_profile_before is None:
            return
        if not self.collision_profile_active and self.collision_profile_restored:
            return
        profile = self.collision_profile_before or {
            "front_stop.enabled": True,
            "front_slow.enabled": True,
            "side_near_stop.enabled": False,
        }
        restored = self._write_collision_profile(profile)
        self.collision_profile_restored = True
        self.collision_profile_active = False
        self._record("collision_profile_restored", restored)

    def emergency_cleanup(self):
        cleanup_error = None
        try:
            super().emergency_cleanup()
        finally:
            try:
                self._restore_collision_profile()
            except BaseException as exc:
                cleanup_error = exc
                self.publish_estop(True)
                self._record("collision_profile_restore_error", str(exc))
        if cleanup_error is not None:
            raise RuntimeError(
                "collision monitor profile restoration failed: %s" % cleanup_error
            )

    def _on_egress_feedback(self, message):
        value = float(message.feedback.distance_traveled)
        if math.isfinite(value) and value >= 0.0:
            self.egress_feedback_samples += 1
            self.egress_feedback_distance_m = value
        else:
            self.egress_feedback_invalid = True
            self._record(
                "egress_feedback_invalid",
                {"distance_traveled": repr(value)},
            )

    def _wait_egress_result(self, future):
        deadline = time.monotonic() + EGRESS_RESULT_TIMEOUT_S
        previous_scan_samples = self.scan_samples
        response = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=WAYPOINT.BASE.INTERVAL_S)
            if self.egress_feedback_invalid:
                self._cancel_goal()
                raise RuntimeError(
                    "safe egress received invalid DriveOnHeading feedback"
                )
            if self.scan_samples > previous_scan_samples:
                previous_scan_samples = self.scan_samples
                if self.latest_scan_monotonic is None:
                    raise RuntimeError("egress received a scan without a timestamp")
                scan_age = time.monotonic() - self.latest_scan_monotonic
                remaining_distance_m = remaining_egress_distance(
                    self.egress_distance_m,
                    self.egress_feedback_distance_m,
                )
                scan_gate = evaluate_front_egress_scan(
                    self.latest_scan.ranges,
                    self.latest_scan.angle_min,
                    self.latest_scan.angle_increment,
                    self.latest_scan.range_min,
                    self.latest_scan.range_max,
                    distance_m=remaining_distance_m,
                    mode=self.egress_mode,
                )
                runtime = evaluate_runtime_egress_monitor(
                    scan_gate,
                    scan_age,
                    maximum_scan_age_s=EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
                )
                runtime["selected_distance_m"] = self.egress_distance_m
                runtime["distance_traveled_m"] = self.egress_feedback_distance_m
                runtime["remaining_distance_m"] = remaining_distance_m
                self.egress_runtime_scan_checks += 1
                self._record("egress_runtime_scan", runtime)
                if runtime["status"] != "PASS":
                    self._cancel_goal()
                    raise RuntimeError(
                        "safe egress runtime scan gate failed: %s"
                        % "; ".join(runtime["failures"])
                    )
            if self.armed is False or self.state in (
                "WATCHDOG_LATCHED",
                "ESTOP_LATCHED",
            ):
                raise RuntimeError(
                    "hardware safety stop during safe egress (state=%s reason=%s)"
                    % (self.state, self.transition_reason)
                )
            if future.done():
                response = future.result()
                if self.egress_runtime_scan_checks >= EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS:
                    return response
        raise RuntimeError("timed out waiting for safe egress result")

    def run_egress(self):
        self.wait_for(
            lambda: self.transport_attached is True,
            WAYPOINT.BASE.STARTUP_TIMEOUT_S,
            "hardware transport attachment for safe egress",
        )
        self.wait_for(
            lambda: (
                self.feedback_connected is True
                and self.stm32_connected is True
                and self.voltage_v is not None
                and self.voltage_v >= WAYPOINT.BASE.MIN_BATTERY_VOLTAGE_V
                and self.latest_raw_pose is not None
            ),
            WAYPOINT.BASE.STARTUP_TIMEOUT_S,
            "fresh feedback and raw odometry for safe egress",
        )
        # /amcl_pose is volatile and AMCL may not publish while the chassis is
        # stationary.  A newly created egress client must therefore request a
        # no-motion update before requiring a pose from its own subscription.
        self._request_amcl_nomotion_update()
        self.wait_for(
            lambda: self.latest_amcl_pose is not None,
            WAYPOINT.BASE.SERVICE_TIMEOUT_S,
            "fresh AMCL pose for safe egress",
        )
        self._wait_for_fresh_scan()
        self.egress_gate = _evaluate_node_egress_gates(self)
        self._record("egress_preflight_gate", self.egress_gate)
        if self.egress_gate["status"] != "PASS":
            raise RuntimeError(
                "safe egress preflight gate failed: %s"
                % "; ".join(self.egress_gate["failures"])
            )
        self.egress_distance_m = float(
            self.egress_gate["selected_distance_m"]
        )
        self.egress_mode = self.egress_gate.get("mode", EGRESS_MODE_STANDARD)
        if self.egress_mode == EGRESS_MODE_SIDE_NEAR and not SIDE_NEAR_EGRESS_FIELD_CERTIFIED:
            raise RuntimeError("side-near escape is not field certified")
        if self.egress_mode == EGRESS_MODE_SIDE_NEAR:
            self._enter_side_near_profile()
        if not self.egress_action.wait_for_server(
            timeout_sec=WAYPOINT.BASE.STARTUP_TIMEOUT_S
        ):
            raise RuntimeError("DriveOnHeading action server is unavailable")
        if self.armed:
            raise RuntimeError("hardware controller unexpectedly started armed")

        self.publish_estop(False)
        self._call_reset()
        self.wait_for(
            lambda: self.state == "DISARMED",
            WAYPOINT.BASE.SERVICE_TIMEOUT_S,
            "DISARMED state after safe egress reset",
        )
        deadline = time.monotonic() + WAYPOINT.BASE.DEADMAN_PRIME_S
        while time.monotonic() < deadline:
            self.publish_deadman(True)
            rclpy.spin_once(self, timeout_sec=WAYPOINT.BASE.INTERVAL_S)
        self._start_deadman_keepalive()
        self._call_arm(True)
        self.wait_for(
            lambda: self.armed is True,
            WAYPOINT.BASE.SERVICE_TIMEOUT_S,
            "armed state for safe egress",
        )

        self.egress_start_raw_pose = dict(self.latest_raw_pose)
        self.egress_start_amcl_pose = dict(self.latest_amcl_pose)
        self.egress_feedback_invalid = False
        goal = DriveOnHeading.Goal()
        goal.target.x = self.egress_distance_m
        goal.target.y = 0.0
        goal.target.z = 0.0
        goal.speed = float(EGRESS_SPEED_M_S)
        goal.time_allowance.sec = int(EGRESS_ACTION_TIME_ALLOWANCE_S)
        goal.time_allowance.nanosec = 0
        send_future = self.egress_action.send_goal_async(
            goal,
            feedback_callback=self._on_egress_feedback,
        )
        self.goal_handle = self._wait_future_with_deadman(
            send_future,
            WAYPOINT.BASE.SERVICE_TIMEOUT_S,
            "DriveOnHeading goal acceptance",
        )
        if self.goal_handle is None or not self.goal_handle.accepted:
            raise RuntimeError("DriveOnHeading safe egress goal was rejected")
        self._record("egress_goal_accepted", True)
        result_response = self._wait_egress_result(
            self.goal_handle.get_result_async()
        )
        action_status = int(result_response.status)
        self._record("egress_action_status", action_status)
        if action_status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(
                "DriveOnHeading safe egress did not report success (status=%d)"
                % action_status
            )
        for _ in range(WAYPOINT.BASE.POST_ACTION_ZERO_FRAMES):
            self.publish_deadman(True)
            self.publish_zero_command()
            rclpy.spin_once(self, timeout_sec=WAYPOINT.BASE.INTERVAL_S)
        self._call_arm(False)
        self._stop_deadman_keepalive()
        self.publish_deadman(False)
        self.wait_for(
            lambda: self.armed is False
            and self.state in ("DISARMED", "WATCHDOG_LATCHED", "ESTOP_LATCHED"),
            WAYPOINT.BASE.SERVICE_TIMEOUT_S,
            "disarmed state after safe egress",
        )
        self._wait_for_stable_feedback_stop()
        self.egress_final_raw_pose = dict(self.latest_raw_pose)
        self.egress_final_amcl_pose = dict(self.latest_amcl_pose)

        # Motion invalidates the old localization cache.  Requiring fresh
        # AMCL samples here prevents the retry from planning from stale pose.
        self.localization_gate_scan_sample_count = -1
        self.localization_gate_result = None
        self.localization_gate_pose = None
        self._request_amcl_nomotion_update()
        self._wait_for_fresh_scan()
        self._check_static_map_localization()
        self._restore_collision_profile()
        return {
            "status": "PASS",
            "distance_m": self.egress_distance_m,
            "speed_m_s": EGRESS_SPEED_M_S,
            "runtime_scan_checks": self.egress_runtime_scan_checks,
            "feedback_samples": self.egress_feedback_samples,
            "feedback_distance_m": self.egress_feedback_distance_m,
            "preflight_gate": self.egress_gate,
            "start_raw_pose": self.egress_start_raw_pose,
            "final_raw_pose": self.egress_final_raw_pose,
            "start_amcl_pose": self.egress_start_amcl_pose,
            "final_amcl_pose": self.egress_final_amcl_pose,
            "relocalization_gate": self.localization_gate_result,
            "collision_profile_before": self.collision_profile_before,
            "collision_profile_after": self.collision_profile_after,
            "collision_profile_restored": self.collision_profile_restored,
            "mode": self.egress_mode,
            "final_armed": self.armed,
            "final_state": self.state,
        }


def exception_diagnostics(exc):
    """Return a non-empty, structured representation of any exception."""
    exception_type = type(exc).__name__
    message = str(exc)
    error = message if message else exception_type
    return {
        "error": error,
        "error_type": exception_type,
        "error_repr": repr(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _failure_result(node, exc):
    diagnostic = exception_diagnostics(exc)
    result = node.result("FAIL", diagnostic["error"])
    result.update(diagnostic)
    result["departure_diagnostics"] = _departure_diagnostics(node)
    return result


def _departure_diagnostics(node):
    """Keep the actual preflight reason even when egress creates a new node."""
    fields = (
        "side_turn_review", "initial_path_heading_delta_rad",
        "path_departure_poses", "path_check_status", "path_check_pose_count",
        "path_check_error", "start_cost", "start_clearance_mode",
        "pre_arm_front_clearance_m", "latest_amcl_pose", "rotation_clearance",
    )
    return {field: getattr(node, field, None) for field in fields}


def _finite_pose_copy(value):
    if not isinstance(value, dict):
        return None
    try:
        pose = dict(value)
        for key in ("x", "y", "yaw"):
            pose[key] = float(pose[key])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(pose[key]) for key in ("x", "y", "yaw")):
        return None
    return pose


def update_continuity_anchor_after_task(continuity_anchor, task_result):
    """Apply a verified stopped inspection pose to the route continuity anchor."""

    anchor = _finite_pose_copy(continuity_anchor)
    if anchor is None or not isinstance(task_result, dict):
        return anchor

    if task_result.get("final_armed") is False:
        task_pose = _finite_pose_copy(task_result.get("final_amcl_pose"))
        if task_pose is not None:
            return task_pose
        try:
            task_yaw = float(task_result.get("final_heading_rad"))
        except (TypeError, ValueError):
            task_yaw = math.nan
        if math.isfinite(task_yaw):
            anchor["yaw"] = WAYPOINT.BASE.normalize_angle(task_yaw)
            return anchor

    for attempt in reversed(task_result.get("heading_attempts") or []):
        if not isinstance(attempt, dict) or attempt.get("final_armed") is not False:
            continue
        attempt_pose = _finite_pose_copy(attempt.get("final_amcl_pose"))
        if attempt_pose is not None:
            return attempt_pose
        try:
            attempt_yaw = float(attempt.get("final_amcl_yaw_rad"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(attempt_yaw):
            anchor["yaw"] = WAYPOINT.BASE.normalize_angle(attempt_yaw)
            return anchor
    return anchor


def continuity_anchor_from_results(results):
    """Recover the last stopped pose when continuing a checkpointed route."""

    for result in reversed(results or []):
        if not isinstance(result, dict) or result.get("status") != "PASS":
            continue
        anchor = _finite_pose_copy(result.get("final_amcl_pose"))
        if anchor is None:
            continue
        return update_continuity_anchor_after_task(
            anchor,
            result.get("inspection_task"),
        )
    return None


def route_start_continuity_anchor(initial_results, resume_localization_verified):
    """Select the first segment anchor for a new or resumed route invocation."""

    if resume_localization_verified:
        return {VERIFIED_RESUME_REBASE_KEY: True}
    return continuity_anchor_from_results(initial_results)


def heading_correction_delta(remaining, response_gain=1.0):
    amount = abs(remaining)
    if amount <= ORIGIN_HEADING_SPIN_CHUNK_MAX_RAD:
        amount = .75 * amount / max(1.0, min(3.0, response_gain))
    return math.copysign(min(amount, ORIGIN_HEADING_SPIN_CHUNK_MAX_RAD), remaining)


class FinalHeadingClient(HEADING.Nav2HeadingClient):
    """Run one bounded relative Spin to an absolute map-frame heading."""

    def __init__(
        self,
        target_yaw: float,
        *,
        completion_tolerance_rad: float = ORIGIN_HEADING_SKIP_TOLERANCE_RAD,
        require_raw_delta_match: bool = True,
        accept_terminal_action_failure: bool = False,
        max_direction_reversals: int = 1,
    ):
        super().__init__()
        self.target_yaw = WAYPOINT.BASE.normalize_angle(target_yaw)
        self.completion_tolerance_rad = float(completion_tolerance_rad)
        if (
            not math.isfinite(self.completion_tolerance_rad)
            or self.completion_tolerance_rad <= 0.0
            or self.completion_tolerance_rad > ORIGIN_HEADING_TOLERANCE_RAD
        ):
            raise ValueError("invalid final-heading completion tolerance")
        self.require_raw_delta_match = bool(require_raw_delta_match)
        self.accept_terminal_action_failure = bool(accept_terminal_action_failure)
        self.max_direction_reversals = int(max_direction_reversals)
        if self.max_direction_reversals < 0:
            raise ValueError("max direction reversals must be non-negative")
        self.terminal_action_failure_accepted = False
        self.amcl_nomotion_client = self.create_client(
            Empty,
            "/request_nomotion_update",
        )
        self.latest_amcl_yaw = None
        self.amcl_samples = 0
        self.requested_delta_yaw = None
        self.skipped = False
        self.spin_chunks = []
        self.overall_start_raw_yaw = None
        self.final_amcl_yaw = None
        self.final_amcl_error_rad = None
        self.latest_heading_amcl_pose = None
        self.final_heading_amcl_pose = None
        self.monitor_output_samples = 0
        self.latest_monitor_command_monotonic = None
        self.spin_command_samples = 0
        self.latest_spin_command_monotonic = None
        self.expected_spin_delta = None
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_final_amcl,
            AMCL_QOS,
        )
        self.create_subscription(
            Twist,
            "/rvclaw/nav2_cmd_vel",
            self._on_monitor_command,
            10,
        )

    def _on_final_amcl(self, message):
        self.amcl_samples += 1
        yaw = WAYPOINT.BASE.quaternion_to_yaw(
            message.pose.pose.orientation
        )
        self.latest_amcl_yaw = yaw
        covariance = message.pose.covariance
        self.latest_heading_amcl_pose = {
            "x": float(message.pose.pose.position.x),
            "y": float(message.pose.pose.position.y),
            "yaw": float(yaw),
            "cov_x": float(covariance[0]),
            "cov_y": float(covariance[7]),
            "cov_yaw": float(covariance[35]),
        }

    def capture_final_amcl_observation(self):
        """Preserve the last trusted stopped pose, including failure exits."""

        if self.latest_amcl_yaw is None:
            return
        self.final_amcl_yaw = float(self.latest_amcl_yaw)
        self.final_amcl_error_rad = shortest_heading_delta(
            self.target_yaw,
            self.final_amcl_yaw,
        )
        if isinstance(self.latest_heading_amcl_pose, dict):
            self.final_heading_amcl_pose = dict(self.latest_heading_amcl_pose)

    def _on_monitor_command(self, message):
        self.monitor_output_samples += 1
        self.latest_monitor_command_monotonic = time.monotonic()
        angular = float(message.angular.z)
        linear = (float(message.linear.x), float(message.linear.y))
        if (
            self.expected_spin_delta is not None
            and all(math.isfinite(v) for v in (*linear, angular))
            and all(abs(v) <= WAYPOINT.BASE.ZERO_EPSILON for v in linear)
            and angular * self.expected_spin_delta > 0.0
            and abs(angular) <= ORIGIN_HEADING_MAX_GUARDED_RAD_S
        ):
            self.spin_command_samples += 1
            self.latest_spin_command_monotonic = self.latest_monitor_command_monotonic

    def _refresh_amcl_heading_once(self):
        # One update immediately after a changed scan view may be transient.
        # Require three fresh, mutually close stopped observations before
        # another chunk or a success decision. Never turn to chase that first
        # transient pose, and never reuse a pre-turn heading.
        stable = []
        for sample_index in range(12):
            # Several no-motion requests can otherwise update from the same
            # scan before the lidar has acquired another view. Three callbacks
            # on one frame are not three independent stopped observations.
            previous_scan_samples = self.scan_samples
            self.wait_for(lambda: self.scan_samples > previous_scan_samples,
                          HEADING.SERVICE_TIMEOUT_S,
                          "a new lidar scan for stopped heading verification")
            previous_samples = self.amcl_samples
            future = self.amcl_nomotion_client.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=HEADING.SERVICE_TIMEOUT_S)
            if not future.done() or future.exception() is not None:
                raise RuntimeError("AMCL no-motion heading refresh timed out")
            self.wait_for(lambda: self.amcl_samples > previous_samples,
                          HEADING.SERVICE_TIMEOUT_S, "fresh AMCL heading after Spin chunk")
            pose = _finite_pose_copy(self.latest_heading_amcl_pose)
            if pose is None:
                raise RuntimeError("fresh stopped AMCL pose is unavailable")
            if stable and any(
                math.hypot(pose["x"] - old["x"], pose["y"] - old["y"]) > 0.05
                or abs(shortest_heading_delta(pose["yaw"], old["yaw"])) > math.radians(3.0)
                for old in stable
            ):
                stable = []
            stable.append(pose)
            if len(stable) >= 3:
                self._record("stopped_heading_observations_stable", {
                    "fresh_updates": sample_index + 1, "observations": stable,
                })
                return
        raise RuntimeError("stopped AMCL heading did not stabilize after Spin")

    def _wait_for_post_goal_command_stream(
        self,
        baseline_samples: int,
        goal_dispatched_monotonic: float,
        result_future=None,
    ):
        def stream_is_fresh():
            if result_future is not None and result_future.done():
                raise RuntimeError("Spin ended before a valid command stream; refusing hardware arming")
            latest = self.latest_spin_command_monotonic
            if latest is None or latest < goal_dispatched_monotonic:
                return False
            age_s = time.monotonic() - latest
            return (
                self.spin_command_samples
                >= baseline_samples + WAYPOINT.BASE.PREARM_NAV2_COMMAND_MIN_SAMPLES
                and 0.0 <= age_s <= WAYPOINT.BASE.MAX_PREARM_NAV2_COMMAND_AGE_S
            )

        self.wait_for(
            stream_is_fresh,
            WAYPOINT.BASE.PREARM_NAV2_COMMAND_TIMEOUT_S,
            "fresh post-goal Spin command stream before hardware arming",
        )
        self._record(
            "prearm_spin_command_stream",
            {
                "baseline_samples": baseline_samples,
                "observed_samples": self.spin_command_samples,
                "new_samples": self.spin_command_samples - baseline_samples,
                "latest_age_s": (
                    None
                    if self.latest_spin_command_monotonic is None
                    else time.monotonic()
                    - self.latest_spin_command_monotonic
                ),
            },
        )

    def _wait_for_amcl_heading(self):
        if not self.amcl_nomotion_client.wait_for_service(
            timeout_sec=HEADING.STARTUP_TIMEOUT_S
        ):
            raise RuntimeError("AMCL no-motion update service is unavailable")
        self._refresh_amcl_heading_once()
        self._record("amcl_nomotion_updates_confirmed", 3)

    def _evaluate(self):
        failures = []
        if (
            self.action_status != GoalStatus.STATUS_SUCCEEDED
            and not self.terminal_action_failure_accepted
        ):
            failures.append("Nav2 Spin did not report success")
        if self.action_feedback_samples < 2:
            failures.append("Nav2 Spin emitted too few feedback samples")
        if self.requested_delta_yaw is None:
            failures.append("requested Spin delta is unavailable")
        elif abs(self.requested_delta_yaw) > ORIGIN_HEADING_SKIP_TOLERANCE_RAD:
            if self.guarded_angular_max is None:
                failures.append("hardware node emitted no guarded Spin samples")
            else:
                if (
                    abs(self.guarded_angular_max) > ORIGIN_HEADING_MAX_GUARDED_RAD_S
                    or self.guarded_angular_min is None
                    or abs(self.guarded_angular_min) > ORIGIN_HEADING_MAX_GUARDED_RAD_S
                ):
                    failures.append("guarded Spin exceeded the angular ceiling")
            if self.requested_delta_yaw > 0.0 and self.guarded_angular_min is None:
                failures.append("Spin direction samples are unavailable")
            elif self.requested_delta_yaw < 0.0 and self.guarded_angular_max is None:
                failures.append("Spin direction samples are unavailable")
            if self.max_guarded_linear_x > WAYPOINT.BASE.ZERO_EPSILON:
                failures.append("Spin contained guarded linear X velocity")
            if self.max_guarded_linear_y > WAYPOINT.BASE.ZERO_EPSILON:
                failures.append("Spin contained guarded lateral velocity")
            raw_delta = self.raw_yaw_delta()
            if raw_delta is None:
                failures.append("raw heading delta is unavailable")
            elif self.require_raw_delta_match and abs(
                WAYPOINT.BASE.normalize_angle(raw_delta - self.requested_delta_yaw)
            ) > ORIGIN_HEADING_TOLERANCE_RAD:
                failures.append("raw heading did not reach the requested shortest delta")
            if len(self.spin_chunks) > ORIGIN_HEADING_MAX_CHUNKS:
                failures.append("Spin exceeded the bounded chunk count")
            total_requested = sum(
                abs(float(item["requested_yaw_rad"])) for item in self.spin_chunks
            )
            if total_requested > ORIGIN_HEADING_MAX_TOTAL_RAD:
                failures.append("Spin exceeded the bounded total rotation")
            directions = [
                math.copysign(1.0, float(item["requested_yaw_rad"]))
                for item in self.spin_chunks
                if abs(float(item["requested_yaw_rad"]))
                > WAYPOINT.BASE.ZERO_EPSILON
            ]
            direction_reversals = sum(
                current != previous
                for previous, current in zip(directions, directions[1:])
            )
            if direction_reversals > self.max_direction_reversals:
                failures.append(
                    "Spin exceeded the bounded direction correction limit "
                    "(%d > %d)"
                    % (direction_reversals, self.max_direction_reversals)
                )
            if self.final_amcl_error_rad is None:
                failures.append("final AMCL heading error is unavailable")
            elif abs(self.final_amcl_error_rad) > ORIGIN_HEADING_TOLERANCE_RAD:
                failures.append("final AMCL heading did not reach the target tolerance")
        if abs(self.latest_feedback_x) > HEADING.MAX_POST_STOP_LINEAR_M_S:
            failures.append("post-Spin X feedback remained above the stop limit")
        if abs(self.latest_feedback_y) > HEADING.MAX_POST_STOP_LINEAR_M_S:
            failures.append("post-Spin Y feedback remained above the stop limit")
        if abs(self.latest_feedback_angular) > HEADING.MAX_POST_STOP_ANGULAR_RAD_S:
            failures.append("post-Spin angular feedback remained above the stop limit")
        if self.armed is not False:
            failures.append("hardware controller did not finish disarmed")
        return failures

    def run_to_absolute_heading(self):
        self._wait_for_amcl_heading()
        self.requested_delta_yaw = shortest_heading_delta(
            self.target_yaw,
            self.latest_amcl_yaw,
        )
        if abs(self.requested_delta_yaw) <= self.completion_tolerance_rad:
            self.skipped = True
            self.action_status = GoalStatus.STATUS_SUCCEEDED
            self.start_raw_yaw = self.latest_raw_yaw
            self.final_raw_yaw = self.latest_raw_yaw
            self.capture_final_amcl_observation()
            return self.result("PASS", None)

        self.wait_for(
            lambda: (
                self.transport_attached is True
                and self.feedback_connected is True
                and self.stm32_connected is True
                and self.voltage_v is not None
                and self.voltage_v >= HEADING.MIN_BATTERY_VOLTAGE_V
            ),
            HEADING.STARTUP_TIMEOUT_S,
            "fresh feedback and safe voltage",
        )
        self.wait_for(
            lambda: (
                self.scan_samples >= HEADING.MINIMUM_SCAN_SAMPLES
                and self.raw_odom_samples >= HEADING.MINIMUM_RAW_ODOM_SAMPLES
                and self.costmap_samples >= HEADING.MINIMUM_COSTMAP_SAMPLES
                and self.latest_raw_yaw is not None
            ),
            HEADING.STARTUP_TIMEOUT_S,
            "scan, raw odometry, and local costmap readiness",
        )
        if not self.spin_action.wait_for_server(timeout_sec=HEADING.STARTUP_TIMEOUT_S):
            raise RuntimeError("Nav2 Spin action server is unavailable")
        if self.armed:
            raise RuntimeError("hardware controller unexpectedly started armed")
        self.start_raw_yaw = self.latest_raw_yaw
        self.overall_start_raw_yaw = self.latest_raw_yaw

        self.publish_estop(False)
        self._call_reset()
        self.wait_for(
            lambda: self.state == "DISARMED",
            HEADING.SERVICE_TIMEOUT_S,
            "DISARMED state after reset",
        )
        action_started = time.monotonic()
        remaining = float(self.requested_delta_yaw)
        chunk_index = 0
        response_gain = 1.0
        while abs(remaining) > self.completion_tolerance_rad:
            chunk_index += 1
            if chunk_index > ORIGIN_HEADING_MAX_CHUNKS:
                raise RuntimeError("final heading exceeded the bounded Spin chunk count")
            # Taper the final correction and use the observed stopped map-yaw
            # response. Full residual commands caused repeated overshoot when
            # the map correction exceeded wheel yaw at a changed scan view.
            chunk = heading_correction_delta(remaining, response_gain)
            chunk_start_amcl_yaw = self.latest_amcl_yaw
            total_requested = sum(
                abs(float(item["requested_yaw_rad"])) for item in self.spin_chunks
            ) + abs(chunk)
            if total_requested > ORIGIN_HEADING_MAX_TOTAL_RAD:
                raise RuntimeError("final heading exceeded the bounded total rotation")
            directions = [math.copysign(1.0, item["requested_yaw_rad"])
                          for item in self.spin_chunks] + [math.copysign(1.0, chunk)]
            reversals = sum(a != b for a, b in zip(directions, directions[1:]))
            if reversals > self.max_direction_reversals:
                raise RuntimeError("Spin exceeded the bounded direction correction limit before dispatch")
            chunk_start_raw_yaw = self.latest_raw_yaw
            goal = Spin.Goal()
            goal.target_yaw = float(chunk)
            goal.time_allowance.sec = ORIGIN_HEADING_ACTION_TIME_ALLOWANCE_S
            goal.time_allowance.nanosec = 0
            self.expected_spin_delta = chunk
            self.latest_spin_command_monotonic = None
            self._start_deadman_keepalive()
            deadline = time.monotonic() + HEADING.DEADMAN_PRIME_S
            while time.monotonic() < deadline:
                self.publish_deadman(True)
                rclpy.spin_once(self, timeout_sec=HEADING.INTERVAL_S)
            command_stream_baseline = self.spin_command_samples
            goal_dispatched_monotonic = time.monotonic()
            send_future = self.spin_action.send_goal_async(
                goal,
                feedback_callback=self._on_action_feedback,
            )
            self.goal_handle = self._wait_future_with_deadman(
                send_future,
                HEADING.SERVICE_TIMEOUT_S,
                "final heading Spin goal acceptance",
            )
            if self.goal_handle is None or not self.goal_handle.accepted:
                raise RuntimeError("final heading Spin goal was rejected")
            self._record("action_goal_accepted", {"chunk_index": chunk_index, "target_yaw_rad": chunk})
            result_future = self.goal_handle.get_result_async()
            self._wait_for_post_goal_command_stream(
                command_stream_baseline,
                goal_dispatched_monotonic,
                result_future,
            )
            if result_future.done():
                raise RuntimeError("Spin ended before hardware arming")
            self._call_arm(True)
            self.wait_for(
                lambda: self.armed is True,
                HEADING.SERVICE_TIMEOUT_S,
                "armed state after fresh Spin command stream",
            )
            self.wait_for(
                lambda: self.state == "ACTIVE"
                or self.state in ("WATCHDOG_LATCHED", "ESTOP_LATCHED"),
                HEADING.SERVICE_TIMEOUT_S,
                "ACTIVE state from post-arm Spin command stream",
            )
            if self.state != "ACTIVE":
                raise RuntimeError(
                    "hardware did not enter ACTIVE after Spin arming "
                    "(state=%s reason=%s)"
                    % (self.state, self.transition_reason)
                )
            self._record("post_goal_hardware_active", {"chunk_index": chunk_index})
            result_response = self._wait_future_with_deadman(
                result_future,
                ORIGIN_HEADING_ACTION_RESULT_TIMEOUT_S,
                "final heading Spin result",
            )
            self.action_status = int(result_response.status)
            chunk_raw_delta = None
            if chunk_start_raw_yaw is not None and self.latest_raw_yaw is not None:
                chunk_raw_delta = WAYPOINT.BASE.normalize_angle(
                    self.latest_raw_yaw - chunk_start_raw_yaw
                )
            self.spin_chunks.append(
                {
                    "index": chunk_index,
                    "requested_yaw_rad": chunk,
                    "action_status": self.action_status,
                    "raw_yaw_delta_rad": chunk_raw_delta,
                    "feedback_samples": self.action_feedback_samples,
                }
            )
            for _ in range(HEADING.POST_ACTION_ZERO_FRAMES):
                self.publish_deadman(True)
                self.publish_zero_command()
                rclpy.spin_once(self, timeout_sec=HEADING.INTERVAL_S)
            self._call_arm(False)
            self._stop_deadman_keepalive()
            self.publish_deadman(False)
            self.wait_for(
                lambda: self.armed is False and self.state == "DISARMED",
                HEADING.SERVICE_TIMEOUT_S,
                "DISARMED state after Spin chunk",
            )
            self._wait_for_stable_feedback_stop()
            self._refresh_amcl_heading_once()
            self.capture_final_amcl_observation()
            remaining = shortest_heading_delta(
                self.target_yaw,
                self.latest_amcl_yaw,
            )
            map_progress = shortest_heading_delta(self.latest_amcl_yaw, chunk_start_amcl_yaw)
            if map_progress * chunk > 0.0:
                response_gain = max(1.0, min(3.0, abs(map_progress / chunk)))
            self._record(
                "spin_chunk_completed",
                {
                    "index": chunk_index,
                    "remaining_yaw_rad": remaining,
                    "amcl_yaw_rad": self.latest_amcl_yaw,
                    "response_gain": response_gain,
                },
            )
            within_tolerance = abs(remaining) <= self.completion_tolerance_rad
            if self.action_status != GoalStatus.STATUS_SUCCEEDED:
                if self.accept_terminal_action_failure and within_tolerance:
                    self.terminal_action_failure_accepted = True
                    self._record(
                        "terminal_spin_status_tolerated",
                        {
                            "chunk_index": chunk_index,
                            "action_status": self.action_status,
                            "remaining_yaw_rad": remaining,
                        },
                    )
                    break
                raise RuntimeError(
                    "Nav2 Spin chunk %d did not report success" % chunk_index
                )
            if (
                (chunk_raw_delta is None or abs(chunk_raw_delta) < min(0.10, .5*abs(chunk)))
                and not within_tolerance
            ):
                raise RuntimeError(
                    "Nav2 Spin chunk %d produced insufficient raw heading change"
                    % chunk_index
                )
            if chunk_raw_delta is not None and chunk * chunk_raw_delta < 0.0:
                raise RuntimeError(
                    "Nav2 Spin chunk %d produced opposite-direction raw heading"
                    % chunk_index
                )

        self.action_elapsed_s = time.monotonic() - action_started
        self.action_result_raw_yaw = self.latest_raw_yaw
        self._stop_deadman_keepalive()
        self.publish_deadman(False)
        self.final_raw_yaw = self.latest_raw_yaw
        self.capture_final_amcl_observation()
        failures = self._evaluate()
        if failures:
            raise RuntimeError("; ".join(failures))
        return self.result("PASS", None)

    def result(self, status, error):
        result = super().result(status, error)
        result.update(
            {
                "target_yaw_rad": self.target_yaw,
                "target_yaw_deg": math.degrees(self.target_yaw),
                "requested_delta_yaw_rad": self.requested_delta_yaw,
                "requested_delta_yaw_deg": (
                    None
                    if self.requested_delta_yaw is None
                    else math.degrees(self.requested_delta_yaw)
                ),
                "spin_chunk_max_rad": ORIGIN_HEADING_SPIN_CHUNK_MAX_RAD,
                "spin_max_chunks": ORIGIN_HEADING_MAX_CHUNKS,
                "spin_max_total_rad": ORIGIN_HEADING_MAX_TOTAL_RAD,
                "spin_max_direction_reversals": self.max_direction_reversals,
                "spin_chunks": self.spin_chunks,
                "origin_heading_skipped": self.skipped,
                "origin_heading_tolerance_rad": ORIGIN_HEADING_TOLERANCE_RAD,
                "completion_tolerance_rad": self.completion_tolerance_rad,
                "require_raw_delta_match": self.require_raw_delta_match,
                "terminal_action_failure_accepted": self.terminal_action_failure_accepted,
                "final_amcl_yaw_rad": self.final_amcl_yaw,
                "final_amcl_error_rad": self.final_amcl_error_rad,
                "final_amcl_pose": (
                    None
                    if self.final_heading_amcl_pose is None
                    else dict(self.final_heading_amcl_pose)
                ),
            }
        )
        return result


def origin_pose_within_goal_tolerance(current_pose, waypoint):
    """Return XY/yaw evidence when a stationary pose already satisfies ORIGIN."""

    current = _finite_pose_copy(current_pose)
    goal = _finite_pose_copy(waypoint)
    if current is None or goal is None:
        return None
    distance_m = math.hypot(current["x"] - goal["x"], current["y"] - goal["y"])
    yaw_error_rad = abs(shortest_heading_delta(goal["yaw"], current["yaw"]))
    if (
        distance_m > ORIGIN_GOAL_XY_TOLERANCE_M
        or yaw_error_rad > ORIGIN_GOAL_YAW_TOLERANCE_RAD
    ):
        return None
    return {
        "distance_m": distance_m,
        "yaw_error_rad": yaw_error_rad,
        "yaw_error_deg": math.degrees(yaw_error_rad),
    }


def _try_stationary_origin_completion(node, waypoint):
    """Complete ORIGIN without arming when fresh AMCL already satisfies it."""

    try:
        node._request_amcl_nomotion_update()
        if (
            node.amcl_samples < WAYPOINT.BASE.MINIMUM_AMCL_SAMPLES
            or node.armed is not False
            or str(node.state).upper() != "DISARMED"
        ):
            return None
        node._check_static_map_localization()
        current = _finite_pose_copy(node.latest_amcl_pose)
        evidence = origin_pose_within_goal_tolerance(current, waypoint)
        if evidence is None:
            return None
        node.goal_pose = dict(waypoint)
        node.start_amcl_pose = dict(current)
        node.final_amcl_pose = dict(current)
        if getattr(node, "latest_raw_pose", None) is not None:
            node.start_raw_pose = dict(node.latest_raw_pose)
            node.final_raw_pose = dict(node.latest_raw_pose)
        result = node.result("PASS", None)
        result.update(
            {
                "origin_navigation_skipped": True,
                "origin_completion_mode": "STATIONARY_WITHIN_TOLERANCE",
                "origin_stationary_evidence": evidence,
            }
        )
        return result
    except BaseException as exc:
        try:
            node._record("origin_stationary_completion_probe_failed", str(exc))
        except BaseException:
            pass
        return None


def _verify_stopped_return_pose():
    # This helper has no motion interface. A new full-view consensus and the
    # independent final gate must pass before any position correction is used.
    from nav2_fast_localization import FastLocalizationClient
    node = FastLocalizationClient(Path(os.environ['RVCLAW_MAP_FILE']),
                                  Path(os.environ['RVCLAW_LOCALIZATION_CACHE']))
    node.force_static_verification = True
    try:
        evidence = node.run(20.0)
        if evidence['status'] != 'PASS' or node.final_verified_pose is None:
            return None, evidence
        return dict(node.final_verified_pose), evidence
    except Exception as exc:
        return None, node.result('FAIL', 'return_stationary_verification', str(exc))
    finally:
        node.destroy_node()


def _run_origin(waypoint, continuity_anchor=None):
    first = _run_origin_once(waypoint, continuity_anchor)
    if first.get('error') != 'final AMCL position outside ORIGIN tolerance after stopped heading':
        return first
    pose, evidence = _verify_stopped_return_pose()
    first['origin_stationary_reverification'] = evidence
    if pose is None:
        return first
    distance = math.hypot(pose['x']-waypoint['x'],pose['y']-waypoint['y'])
    yaw_error = abs(shortest_heading_delta(waypoint['yaw'],pose['yaw']))
    if distance <= WAYPOINT.MAX_FINAL_POSITION_ERROR_M and yaw_error <= ORIGIN_HEADING_TOLERANCE_RAD:
        original = dict(first)
        first.update(status='PASS',error=None,final_amcl_pose=pose,
                     final_amcl_yaw_rad=pose['yaw'],final_amcl_error_rad=shortest_heading_delta(waypoint['yaw'],pose['yaw']),
                     origin_final_position_error_m=distance)
        first['origin_completion_attempts'] = [original]
        first['origin_heading_stage'] = dict(first.get('origin_heading_stage') or {},
            status='PASS',error=None,final_amcl_pose=pose,
            final_amcl_yaw_rad=pose['yaw'],final_amcl_error_rad=shortest_heading_delta(waypoint['yaw'],pose['yaw']))
        return first
    # One short correction through the usual costmap/controller safety gates.
    # Never recurse or chase a distant relocalization hypothesis.
    if distance > 1.0:
        first['error'] = 'verified return position is outside bounded correction distance'
        return first
    corrected = _run_origin_once(waypoint, continuity_anchor=pose)
    corrected['origin_completion_attempts'] = [first]
    corrected['origin_stationary_reverification'] = evidence
    corrected['origin_position_correction_count'] = 1
    return corrected


def _run_origin_navigation(node):
    """Give synchronous ORIGIN travel its own budget, then restore the default."""
    previous_timeout = WAYPOINT.BASE.ACTION_RESULT_TIMEOUT_S
    WAYPOINT.BASE.ACTION_RESULT_TIMEOUT_S = ORIGIN_TRANSIT_ACTION_RESULT_TIMEOUT_S
    try:
        result = node.run_navigation()
        result["origin_transit_timeout_s"] = ORIGIN_TRANSIT_ACTION_RESULT_TIMEOUT_S
        return result
    finally:
        WAYPOINT.BASE.ACTION_RESULT_TIMEOUT_S = previous_timeout


def _run_origin_once(waypoint, continuity_anchor=None):
    if ORIGIN_NAVIGATION_MODE == "absolute_pose":
        if continuity_anchor is None:
            node = OriginPoseWaypointClient(waypoint)
        else:
            node = OriginPoseWaypointClient(
                waypoint,
                continuity_anchor=continuity_anchor,
            )
        try:
            result = _try_stationary_origin_completion(node, waypoint)
            if result is None:
                result = _run_origin_navigation(node)
        except BaseException as exc:
            try:
                node.emergency_cleanup()
            except BaseException as cleanup_exc:
                node._record("cleanup_error", str(cleanup_exc))
            result = _failure_result(node, exc)
        finally:
            node.destroy_node()
        result.update(
            {
                "origin_navigation_mode": ORIGIN_NAVIGATION_MODE,
                "origin_pose_goal": {
                    "x": waypoint["x"],
                    "y": waypoint["y"],
                    "yaw": waypoint["yaw"],
                },
            }
        )
        return result

    # Position and heading are deliberately separated so the controller's
    # translation-only progress checker cannot abort valid in-place rotation.
    position_waypoint = dict(waypoint)
    if continuity_anchor is None:
        position_node = PositionOnlyWaypointClient(position_waypoint)
    else:
        position_node = PositionOnlyWaypointClient(
            position_waypoint,
            continuity_anchor=continuity_anchor,
        )
    near_pose = None
    near_candidate = False
    near_error = None
    try:
        position_node._request_amcl_nomotion_update()
        if position_node.latest_amcl_pose is not None:
            current = position_node.latest_amcl_pose
            if math.hypot(
                current["x"] - waypoint["x"],
                current["y"] - waypoint["y"],
            ) <= ORIGIN_GOAL_XY_TOLERANCE_M:
                near_candidate = True
                # A prior failure cleanup intentionally leaves the chassis
                # ESTOP_LATCHED. This is stopped, not a reason to dispatch a
                # zero-distance NavigateToPose and demand motion-only samples.
                if position_node.armed is not False or position_node.state not in {
                    "DISARMED", "ESTOP_LATCHED", "WATCHDOG_LATCHED",
                }:
                    raise RuntimeError("ORIGIN position skip requires a disarmed controller")
                position_node._wait_for_stable_feedback_stop()
                for _ in range(3):
                    position_node._wait_for_fresh_scan()
                    pose_count = position_node.amcl_samples
                    future = position_node.amcl_nomotion_client.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(
                        position_node, future,
                        timeout_sec=WAYPOINT.BASE.AMCL_NOMOTION_RESPONSE_TIMEOUT_S,
                    )
                    if not future.done() or future.exception() is not None:
                        raise RuntimeError("ORIGIN stopped pose refresh failed")
                    position_node.wait_for(
                        lambda: position_node.amcl_samples > pose_count,
                        WAYPOINT.BASE.SERVICE_TIMEOUT_S,
                        "new stopped AMCL observation at ORIGIN",
                    )
                position_node._check_static_map_localization()
                current = _finite_pose_copy(position_node.latest_amcl_pose)
                if current is not None and math.hypot(
                    current["x"] - waypoint["x"], current["y"] - waypoint["y"]
                ) <= ORIGIN_GOAL_XY_TOLERANCE_M:
                    near_pose = dict(current)
    except Exception as exc:
        near_error = exc if near_candidate else None
        position_node._record("stationary_origin_validation_error", str(exc))
        near_pose = None
    if near_error is not None:
        try:
            position_node.emergency_cleanup()
        except Exception as cleanup_exc:
            position_node._record("cleanup_error", str(cleanup_exc))
        position_result = _failure_result(position_node, near_error)
    elif near_pose is not None:
        position_node.goal_pose = dict(waypoint)
        position_node.start_amcl_pose = dict(near_pose)
        position_node.final_amcl_pose = dict(near_pose)
        position_result = position_node.result("PASS", None)
        position_result.update(
            {
                "position_stage": "SKIPPED_NEAR_GOAL",
                "deferred_reason": "ORIGIN position already within 0.05 m",
                "current_goal_distance_m": math.hypot(
                    near_pose["x"] - waypoint["x"],
                    near_pose["y"] - waypoint["y"],
                ),
            }
        )
    else:
        try:
            position_result = _run_origin_navigation(position_node)
        except BaseException as exc:
            try:
                position_node.emergency_cleanup()
            except BaseException as cleanup_exc:
                position_node._record("cleanup_error", str(cleanup_exc))
            position_result = _failure_result(position_node, exc)
    position_node.destroy_node()
    if position_result.get("status") != "PASS":
        return {
            **position_result,
            "waypoint": waypoint,
            "origin_navigation_mode": ORIGIN_NAVIGATION_MODE,
            "origin_position_stage": position_result,
            "origin_heading_stage": None,
        }

    heading_node = FinalHeadingClient(
        waypoint["yaw"],
        # Use the existing absolute-heading acceptance limit as the loop's
        # stopping limit as well. Chasing a stricter four-degree micro-turn
        # after meeting that limit caused a second reversal and a false FAIL.
        completion_tolerance_rad=ORIGIN_HEADING_TOLERANCE_RAD,
        # Raw wheel yaw and map-frame AMCL yaw can differ after a long turn.
        # Absolute stopped AMCL heading decides this absolute map-frame goal;
        # finite, correctly directed raw motion and bounded rotation are still
        # mandatory for every chunk.
        require_raw_delta_match=False,
    )
    try:
        heading_result = heading_node.run_to_absolute_heading()
    except BaseException as exc:
        try:
            heading_node.emergency_cleanup()
        except BaseException as cleanup_exc:
            heading_node._record("cleanup_error", str(cleanup_exc))
        heading_node.capture_final_amcl_observation()
        heading_result = _failure_result(heading_node, exc)
    finally:
        heading_node.destroy_node()
    final_pose = _finite_pose_copy(heading_result.get("final_amcl_pose"))
    final_position_error = None if final_pose is None else math.hypot(
        final_pose["x"] - waypoint["x"], final_pose["y"] - waypoint["y"])
    heading_result["origin_final_position_error_m"] = final_position_error
    heading_result["origin_final_position_tolerance_m"] = WAYPOINT.MAX_FINAL_POSITION_ERROR_M
    if heading_result.get("status") == "PASS" and (
        final_position_error is None
        or final_position_error > WAYPOINT.MAX_FINAL_POSITION_ERROR_M
    ):
        heading_result["status"] = "FAIL"
        heading_result["error"] = "final AMCL position outside ORIGIN tolerance after stopped heading"
    if heading_result.get("status") != "PASS":
        return {
            **heading_result,
            "waypoint": waypoint,
            "origin_navigation_mode": ORIGIN_NAVIGATION_MODE,
            "origin_position_stage": position_result,
            "origin_heading_stage": heading_result,
        }
    return {
        **heading_result,
        "status": "PASS",
        "error": None,
        "waypoint": waypoint,
        "origin_navigation_mode": ORIGIN_NAVIGATION_MODE,
        "origin_position_stage": position_result,
        "origin_heading_stage": heading_result,
        "final_armed": heading_result.get("final_armed"),
        "final_state": heading_result.get("final_state"),
    }


def _run_inspection_heading(target_yaw):
    """Align the chassis through the existing guarded Nav2 Spin path."""

    node = FinalHeadingClient(
        target_yaw,
        completion_tolerance_rad=INSPECTION_HEADING_COMPLETION_TOLERANCE_RAD,
        require_raw_delta_match=False,
        accept_terminal_action_failure=True,
        max_direction_reversals=2,
    )
    try:
        result = node.run_to_absolute_heading()
    except BaseException as exc:
        try:
            node.emergency_cleanup()
        except BaseException as cleanup_exc:
            node._record("cleanup_error", str(cleanup_exc))
        node.capture_final_amcl_observation()
        result = _failure_result(node, exc)
    finally:
        node.destroy_node()
    result["inspection_heading_stage"] = True
    return result


def _build_inspection_task_runner(plan, run_id):
    executor = InspectionExecutor(plan)

    def run_task(waypoint, route_index, navigation_result):
        try:
            return executor.execute(
                waypoint,
                route_index,
                navigation_result,
                _run_inspection_heading,
                run_id=run_id,
            )
        except BaseException as exc:
            point = plan.point(str(waypoint.get("name", "")))
            return {
                "status": "FAIL",
                "task_succeeded": False,
                "failure_policy": (
                    "abort" if point is None else point.failure_policy
                ),
                "motion_authorized": False,
                "task_id": None if point is None else point.task_id,
                "meter_id": None if point is None else point.meter_id,
                "kind": None if point is None else point.kind,
                **exception_diagnostics(exc),
            }

    return run_task


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--extract-global-pose",
        metavar="TRACE",
        help="print a converged global-localization pose as shell assignments",
    )
    parser.add_argument("--trace-file")
    parser.add_argument(
        "--checkpoint-file",
        default=os.getenv("RVCLAW_ROUTE_CHECKPOINT_FILE", ""),
        help="durable paused-route checkpoint; cleared after a complete route",
    )
    parser.add_argument(
        "--resume-context",
        default=os.getenv("RVCLAW_ROUTE_RESUME_CONTEXT", ""),
        help="checkpoint whose completed inspection results precede this route",
    )
    parser.add_argument(
        "--resume-localization-verified",
        action="store_true",
        help=(
            "rebase the first resumed segment on the current AMCL pose after an "
            "independent fast-localization PASS"
        ),
    )
    parser.add_argument("--arm-token")
    parser.add_argument("--waypoints-file", default=WAYPOINT.DEFAULT_WAYPOINTS_FILE)
    parser.add_argument(
        "--route",
        default=",".join(DEFAULT_ROUTE),
        help="comma-separated waypoint names, in execution order",
    )
    parser.add_argument(
        "--inspection-config",
        default=os.getenv("RVCLAW_INSPECTION_CONFIG", ""),
        help=(
            "optional waypoint inspection YAML; omitted keeps the validated "
            "navigation-only route unchanged"
        ),
    )
    parser.add_argument(
        "--inspection-readiness-only",
        action="store_true",
        help="perform read-only Studio API and meter calibration checks",
    )
    return parser.parse_args()


def _load_route(path: str, route_text: str):
    waypoints = load_waypoints(path)
    names = tuple(part.strip() for part in route_text.split(",") if part.strip())
    if not names:
        raise ValueError("route must contain at least one waypoint")
    if len(set(names)) != len(names):
        raise ValueError("route must not contain duplicate waypoint names")
    selected = []
    for name in names:
        if name not in waypoints:
            available = ", ".join(sorted(waypoints))
            raise ValueError(
                "unknown waypoint %s; available: %s" % (name, available)
            )
        waypoint = waypoints[name]
        if math.hypot(waypoint["x"], waypoint["y"]) > WAYPOINT.MAX_WAYPOINT_DISTANCE_M:
            raise ValueError(
                "waypoint %s is outside the reviewed map distance limit" % name
            )
        selected.append(waypoint)
    return names, selected


def _validate_trace(path_text: str) -> Path:
    if not path_text:
        raise ValueError("trace-file is required")
    path = Path(path_text).resolve()
    try:
        path.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("trace file must remain under the workspace root") from exc
    if path.exists() or not path.parent.is_dir():
        raise ValueError("trace file must be new and its parent must exist")
    return path


def load_route_resume_context(path_text, route_names):
    """Load only completed waypoint records from the matching paused route."""

    if not path_text:
        return []
    path = Path(path_text).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported route resume checkpoint schema")
    requested = tuple(
        str(name).strip().upper() for name in document.get("route_requested", ())
    )
    completed = tuple(
        str(name).strip().upper() for name in document.get("completed_waypoints", ())
    )
    remaining = tuple(
        str(name).strip().upper() for name in document.get("remaining_waypoints", ())
    )
    expected_remaining = tuple(str(name).strip().upper() for name in route_names)
    if not requested or not remaining:
        raise ValueError("resume checkpoint route is incomplete")
    if any(not name for name in requested + completed + remaining):
        raise ValueError("resume checkpoint contains an empty waypoint")
    if len(set(requested)) != len(requested):
        raise ValueError("resume checkpoint route contains duplicates")
    if completed + remaining != requested:
        raise ValueError("resume checkpoint progress is not a contiguous route prefix")
    if remaining != expected_remaining:
        raise ValueError("resume checkpoint does not match the requested remaining route")
    failed_waypoint = str(document.get("failed_waypoint", "")).strip().upper()
    if failed_waypoint and failed_waypoint != remaining[0]:
        raise ValueError("resume checkpoint failed waypoint does not match route head")
    prior = []
    prior_names = []
    for item in document.get("route_results", ()):
        if not isinstance(item, dict) or item.get("status") != "PASS":
            continue
        waypoint = item.get("waypoint") or {}
        name = str(waypoint.get("name", "")).upper()
        if not name or name in remaining:
            continue
        prior.append(dict(item))
        prior_names.append(name)
    if tuple(prior_names) != completed:
        raise ValueError("resume checkpoint completed results do not match progress")
    return prior


def write_route_checkpoint(path_text, route_names, result):
    """Atomically save a resumable fail-stop checkpoint, or clear it on success."""

    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    if result.get("status") == "PASS":
        path.unlink(missing_ok=True)
        return None
    resume_plan = result.get("resume_plan")
    if not isinstance(resume_plan, dict):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    completed_waypoints = [
        str(name).strip().upper()
        for name in result.get("completed_waypoints", ())
    ]
    remaining_waypoints = [
        str(name).strip().upper()
        for name in resume_plan.get("remaining_waypoints", ())
    ]
    requested_route = completed_waypoints + remaining_waypoints
    if not remaining_waypoints or any(not name for name in requested_route):
        raise ValueError("cannot write an incomplete route checkpoint")
    if len(set(requested_route)) != len(requested_route):
        raise ValueError("cannot write a checkpoint with duplicate waypoints")
    failed_waypoint = str(resume_plan.get("failed_waypoint", "")).strip().upper()
    if failed_waypoint and failed_waypoint != remaining_waypoints[0]:
        raise ValueError("checkpoint failed waypoint does not match remaining route")
    payload = {
        "schema_version": 1,
        "timestamp": time.time(),
        "route_requested": requested_route,
        "completed_waypoints": completed_waypoints,
        "remaining_waypoints": remaining_waypoints,
        "failed_waypoint": failed_waypoint or remaining_waypoints[0],
        "failure_code": resume_plan.get("failure_code"),
        "operator_action": resume_plan.get("operator_action"),
        "resume_command": resume_plan.get("resume_command"),
        "requires_global_localization": bool(
            resume_plan.get("requires_global_localization")
        ),
        "route_results": list(result.get("route", ())),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=".inspection-route-checkpoint-", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        temporary = Path(stream.name)
    os.replace(temporary, path)
    return path


def describe_plan(route_names):
    return {
        "hardware_io": False,
        "goal_mode": "absolute_waypoint_route",
        "frame_id": "map",
        "route": list(route_names),
        "required_arm_token": ROUTE_ARM_TOKEN,
        "single_navigation_stack": True,
        "initial_pose_required_once": True,
        "initial_pose_input": "RVCLAW_NAV2_INITIAL_POSE=x,y,yaw_deg or three RVCLAW_NAV2_INITIAL_* variables",
        "initial_heading_periodic": True,
        "global_localization_pose_import": "--extract-global-pose TRACE",
        "fresh_scan_gate_per_waypoint": True,
        "independent_static_map_localization_gate_at_route_start": True,
        "continuous_localization_gate_between_waypoints": True,
        "continuous_localization_max_position_jump_m": (
            CONTINUOUS_ROUTE_MAX_POSITION_JUMP_M
        ),
        "continuous_localization_max_yaw_jump_rad": (
            CONTINUOUS_ROUTE_MAX_YAW_JUMP_RAD
        ),
        "path_preflight_before_operator_gate": True,
        "all_route_segments_preflighted": True,
        "near_start_egress_supported": True,
        "near_start_egress_state": "NEAR_START_EGRESS_REQUIRED",
        "near_start_egress_distance_m": EGRESS_DISTANCE_M,
        "near_start_egress_minimum_distance_m": EGRESS_MIN_DISTANCE_M,
        "near_start_egress_adaptive_distance": True,
        "near_start_egress_speed_m_s": EGRESS_SPEED_M_S,
        "near_start_egress_max_checked_steps": MAX_AUTONOMOUS_EGRESS_STEPS,
        "near_start_egress_review_scope": "every_non_origin_segment",
        "near_start_egress_requires_relocalization": True,
        "near_start_egress_runtime_scan_checks": EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
        "near_start_egress_modes": [EGRESS_MODE_STANDARD],
        "side_near_egress_field_certified": SIDE_NEAR_EGRESS_FIELD_CERTIFIED,
        "side_near_egress_policy": "STOP_UNTIL_FULL_ENVELOPE_CERTIFICATION",
        "near_start_egress_side_near_margin_m": EGRESS_SIDE_NEAR_SAFETY_MARGIN_M,
        "near_start_egress_profile_restored_before_success": True,
        "post_stop_check_per_waypoint": True,
        "fail_stop": True,
        "dynamic_obstacle_recovery": {
            "policy_version": DYNAMIC.POLICY_VERSION,
            "default_action_on_uncertainty": DYNAMIC.ACTION_HOLD_STOP,
            "flow": [
                DYNAMIC.ACTION_SLOW_AND_REPLAN,
                DYNAMIC.ACTION_REPLAN_ALTERNATE,
                DYNAMIC.ACTION_STOP_WAIT_REPLAN,
                DYNAMIC.ACTION_SPIN_SEARCH,
                DYNAMIC.ACTION_ABORT_GOAL,
            ],
            "max_replan_attempts": DYNAMIC.MAX_REPLAN_ATTEMPTS,
            "max_wait_attempts": DYNAMIC.MAX_WAIT_ATTEMPTS,
            "max_spin_attempts": DYNAMIC.MAX_SPIN_ATTEMPTS,
            "spin_requires_full_swept_clearance": True,
            "requires_fresh_motion_feedback": True,
            "progress_stall_policy": "stop_then_bounded_replan_then_abort",
            "reverse_allowed": False,
            "lateral_allowed": False,
        },
        "automatic_reverse": False,
        "origin_navigation_mode": ORIGIN_NAVIGATION_MODE,
        "intermediate_navigation_mode": INTERMEDIATE_NAVIGATION_MODE,
        "intermediate_heading_policy": "path_bearing_hint_with_relaxed_yaw_goal",
        "intermediate_behavior_tree": POSITION_ONLY_BT_NAME,
        "origin_pose_goal_includes_heading": ORIGIN_NAVIGATION_MODE == "absolute_pose",
        "origin_heading_mode": (
            "dwb_full_pose_with_origin_goal_checker_and_rotation_guard"
            if ORIGIN_NAVIGATION_MODE == "absolute_pose"
            else "position_stop_then_bounded_spin_and_final_xy_validation"
        ),
        "origin_behavior_tree": (ORIGIN_FULL_POSE_BT_NAME
                                 if ORIGIN_NAVIGATION_MODE == "absolute_pose" else POSITION_ONLY_BT_NAME),
        "origin_replan_policy": ("distance_triggered" if ORIGIN_NAVIGATION_MODE == "absolute_pose"
                                 else "position_only_then_no_replan_spin"),
        "origin_replan_distance_m": (ORIGIN_REPLAN_DISTANCE_M
                                     if ORIGIN_NAVIGATION_MODE == "absolute_pose" else None),
        "origin_goal_checker": ("origin_goal_checker" if ORIGIN_NAVIGATION_MODE == "absolute_pose"
                                else "position_goal_checker"),
        "origin_goal_xy_tolerance_m": ORIGIN_GOAL_XY_TOLERANCE_M,
        "origin_goal_yaw_tolerance_rad": ORIGIN_GOAL_YAW_TOLERANCE_RAD,
        "origin_post_entry_max_rotation_rad": ORIGIN_POST_ENTRY_MAX_ROTATION_RAD,
        "arrival_event_waypoints": ["A", "B", "C"],
        "arrival_event_directory_default": DEFAULT_ARRIVAL_EVENT_DIR,
        "origin_heading_skip_tolerance_deg": math.degrees(
            ORIGIN_HEADING_SKIP_TOLERANCE_RAD
        ),
        "origin_heading_tolerance_deg": math.degrees(
            ORIGIN_HEADING_TOLERANCE_RAD
        ),
        "origin_heading_spin_chunk_max_rad": ORIGIN_HEADING_SPIN_CHUNK_MAX_RAD,
        "origin_heading_max_chunks": ORIGIN_HEADING_MAX_CHUNKS,
        "origin_heading_max_total_rad": ORIGIN_HEADING_MAX_TOTAL_RAD,
        "origin_transit_timeout_s": ORIGIN_TRANSIT_ACTION_RESULT_TIMEOUT_S,
        "max_route_duration_s": ROUTE_MAX_DURATION_S,
    }


def preflight_exit_status(result):
    """Return a shell status for a route preflight result.

    A near-start lethal cell is an expected, reviewable state when the
    independent live-scan and static-map egress gates pass.  It must reach the
    operator gate so the runtime client can perform the bounded egress.  Any
    other preflight failure remains a hard failure.
    """
    if result.get("status") == "PASS":
        return 0
    egress_gate = result.get("safe_egress_gate") or {}
    if (
        result.get("status") == "REQUIRE_SAFE_EGRESS"
        and result.get("egress_required") is True
        and result.get("egress_state") == "NEAR_START_EGRESS_REQUIRED"
        and isinstance(egress_gate, dict)
        and egress_gate.get("status") == "PASS"
    ):
        return 0
    return 1


def _run_one(
    waypoint,
    allow_near_start_egress=False,
    continuity_anchor=None,
):
    if waypoint.get("name") == "ORIGIN":
        return _run_origin(waypoint, continuity_anchor=continuity_anchor)
    if continuity_anchor is None:
        node = PositionOnlyWaypointClient(dict(waypoint))
    else:
        node = PositionOnlyWaypointClient(
            dict(waypoint),
            continuity_anchor=continuity_anchor,
        )
    node_destroyed = False
    try:
        result = node.run_navigation()
        result.update(
            {
                "intermediate_navigation_mode": INTERMEDIATE_NAVIGATION_MODE,
                "catalog_waypoint_yaw": waypoint["yaw"],
            }
        )
        return result
    except BaseException as exc:
        if allow_near_start_egress and (
            _is_near_start_path_failure(node, exc)
            or isinstance(exc, SideNearTurnEgressRequired)
        ):
            initial_failure = exception_diagnostics(exc)
            initial_failure["departure_diagnostics"] = _departure_diagnostics(node)
            try:
                node.emergency_cleanup()
            except BaseException as cleanup_exc:
                node._record("cleanup_error", str(cleanup_exc))
            node.destroy_node()
            node_destroyed = True
            return _run_staged_egress_then_retry(waypoint, initial_failure)
        try:
            node.emergency_cleanup()
        except BaseException as cleanup_exc:
            node._record("cleanup_error", str(cleanup_exc))
        failed = _failure_result(node, exc)
        if isinstance(exc, SideNearTurnUnsafe):
            failed["automatic_recovery_forbidden"] = True
            failed["safety_pause_reason"] = "turn clearance cannot be certified"
        return failed
    finally:
        if not node_destroyed:
            node.destroy_node()


def _run_staged_egress_then_retry(waypoint, initial_failure):
    """Replan after *every* bounded step; never spin or blindly chain drives."""
    steps = []
    last_state = None
    while True:
        budget = step_budget(item["distance_m"] for item in steps)
        if budget["status"] != "ALLOW_CHECKED_STEP":
            return {
                "status": "FAIL", "error": "navigation obstacle: %s" % budget["reason"],
                "initial_path_failure": initial_failure,
                "safe_egress_steps": steps,
                "safe_egress": steps[-1] if steps else None,
                "waypoint": waypoint, "final_armed": False,
                "final_state": last_state,
                "automatic_recovery_forbidden": True,
                "safety_pause_reason": "autonomous escape budget exhausted",
            }
        egress = SafeEgressClient()
        try:
            egress_result = egress.run_egress()
            step_evidence = validate_completed_step(egress_result)
            if step_evidence["status"] == "PASS" and steps:
                step_evidence = validate_chain_continuity(steps[-1], egress_result)
            if step_evidence["status"] != "PASS":
                egress.emergency_cleanup()
        except BaseException as egress_exc:
            try:
                egress.emergency_cleanup()
            except BaseException as cleanup_exc:
                egress._record("cleanup_error", str(cleanup_exc))
            egress_result = egress.result("FAIL", str(egress_exc))
            egress_result.update(exception_diagnostics(egress_exc))
            step_evidence = {"status": "STOP", "reason": str(egress_exc)}
        finally:
            egress.destroy_node()
        if step_evidence["status"] != "PASS":
            return {
                "status": "FAIL",
                "error": "safe egress stopped: %s" % step_evidence["reason"],
                "initial_path_failure": initial_failure,
                "safe_egress_steps": steps,
                "safe_egress": egress_result,
                "step_evidence": step_evidence,
                "waypoint": waypoint,
                "final_armed": getattr(egress, "armed", None),
                "final_state": getattr(egress, "state", None),
                "automatic_recovery_forbidden": True,
                "safety_pause_reason": step_evidence["reason"],
            }
        steps.append(dict(egress_result))
        retry_node = PositionOnlyWaypointClient(
            dict(waypoint), continuing_egress=True,
        )
        try:
            result = retry_node.run_navigation()
            result.update({
                "intermediate_navigation_mode": INTERMEDIATE_NAVIGATION_MODE,
                "catalog_waypoint_yaw": waypoint["yaw"],
                "near_start_egress_state": "STAGED_EGRESS_RELOCALIZED_AND_REPLANNED",
                "safe_egress": steps[-1],
                "safe_egress_steps": steps,
                "initial_path_failure": initial_failure,
            })
            return result
        except BaseException as retry_exc:
            try:
                retry_node.emergency_cleanup()
            except BaseException as cleanup_exc:
                retry_node._record("cleanup_error", str(cleanup_exc))
            last_state = getattr(retry_node, "state", None)
            can_stage_again = (
                getattr(retry_node, "armed", None) is False
                and (isinstance(retry_exc, SideNearTurnEgressRequired)
                     or _is_near_start_path_failure(retry_node, retry_exc))
            )
            if can_stage_again:
                # The next iteration must independently read a new scan,
                # current map and localized pose before moving another step.
                continue
            failed = _failure_result(retry_node, retry_exc)
            failed.update({
                "safe_egress": steps[-1],
                "safe_egress_steps": steps,
                "initial_path_failure": initial_failure,
            })
            if isinstance(retry_exc, SideNearTurnUnsafe):
                failed["automatic_recovery_forbidden"] = True
                failed["safety_pause_reason"] = "turn clearance cannot be certified"
            return failed
        finally:
            retry_node.destroy_node()


def _preflight_one(waypoint, start_pose, defer_if_current_goal=False):
    if (
        waypoint.get("name") == "ORIGIN"
        and ORIGIN_NAVIGATION_MODE == "absolute_pose"
    ):
        node = WAYPOINT.WaypointClient(waypoint)
    else:
        node = PositionOnlyWaypointClient(
            dict(waypoint),
            start_pose_hint=start_pose,
        )
    try:
        result = node.run_path_preflight(start_pose=start_pose)
        if waypoint.get("name") != "ORIGIN":
            result.update(
                {
                    "intermediate_navigation_mode": INTERMEDIATE_NAVIGATION_MODE,
                    "intermediate_behavior_tree": POSITION_ONLY_BT_NAME,
                }
            )
        elif ORIGIN_NAVIGATION_MODE == "split_spin":
            result.update(
                {
                    "origin_position_navigation_mode": "position_only",
                    "origin_heading_navigation_mode": "bounded_shortest_relative_spin",
                    "intermediate_behavior_tree": POSITION_ONLY_BT_NAME,
                }
            )
        return result
    except BaseException as exc:
        current_pose = node.latest_amcl_pose
        current_goal_distance = None
        egress_gate = None
        if current_pose is not None:
            current_goal_distance = math.hypot(
                current_pose["x"] - waypoint["x"],
                current_pose["y"] - waypoint["y"],
            )
        if start_pose is None and (
            _is_near_start_path_failure(node, exc)
            or isinstance(exc, SideNearTurnEgressRequired)
        ):
            try:
                egress_gate = _evaluate_node_egress_gates(node)
            except BaseException as gate_exc:
                egress_gate = {
                    "status": "FAIL",
                    "failures": ["safe egress gate error: %s" % gate_exc],
                }
            if egress_gate.get("status") == "PASS":
                return {
                    "status": "REQUIRE_SAFE_EGRESS",
                    "error": None,
                    "egress_required": True,
                    "egress_state": "NEAR_START_EGRESS_REQUIRED",
                    "egress_reason": (
                        "side-near sharp turn is not sweep-clear; only a "
                        "bounded straight step has a separately reviewed gate"
                        if isinstance(exc, SideNearTurnEgressRequired)
                        else "lethal start cell; forward egress corridor is clear"
                    ),
                    "side_turn_review": getattr(node, "side_turn_review", None),
                    "safe_egress_gate": egress_gate,
                    "path_departure_poses": getattr(node, "path_departure_poses", None),
                    "current_goal_distance_m": current_goal_distance,
                    "goal_cost": node.goal_cost,
                    "goal_pose": node.goal_pose,
                    "start_pose": start_pose,
                    "start_source": "current_amcl",
                    "path_check_status": node.path_check_status,
                    "path_check_pose_count": node.path_check_pose_count,
                    "path_check_elapsed_s": node.path_check_elapsed_s,
                    "path_check_error": node.path_check_error,
                    "pre_arm_front_clearance_m": node.pre_arm_front_clearance_m,
                    "start_cost": node.start_cost,
                    "initial_path_heading_delta_rad": node.initial_path_heading_delta_rad,
                    "start_clearance_mode": node.start_clearance_mode,
                    "final_armed": node.armed,
                }
        if (
            ORIGIN_NAVIGATION_MODE == "split_spin"
            and waypoint.get("name") == "ORIGIN"
            and start_pose is None
            and current_goal_distance is not None
            and current_goal_distance <= ORIGIN_GOAL_XY_TOLERANCE_M
            and node.armed is False
        ):
            return {
                "status": "DEFERRED",
                "error": None,
                "deferred_reason": "ORIGIN position is already reached; heading-only stage",
                "current_goal_distance_m": current_goal_distance,
                "goal_cost": node.goal_cost,
                "goal_pose": waypoint,
                "start_pose": start_pose,
                "start_source": "current_amcl",
                "path_check_status": node.path_check_status,
                "path_check_pose_count": node.path_check_pose_count,
                "path_check_elapsed_s": node.path_check_elapsed_s,
                "path_check_error": node.path_check_error,
                "pre_arm_front_clearance_m": getattr(
                    node, "pre_arm_front_clearance_m", None
                ),
                "start_cost": getattr(node, "start_cost", None),
                "initial_path_heading_delta_rad": getattr(
                    node, "initial_path_heading_delta_rad", None
                ),
                "start_clearance_mode": getattr(
                    node, "start_clearance_mode", None
                ),
                "final_armed": node.armed,
            }
        if (
            defer_if_current_goal
            and str(exc) == "relative goal cell is not admitted free space"
            and current_goal_distance is not None
            and current_goal_distance <= MAX_DEFERRED_RETURN_DISTANCE_M
            and node.armed is False
        ):
            return {
                "status": "DEFERRED",
                "error": None,
                "deferred_reason": "final return goal overlaps current robot pose",
                "current_goal_distance_m": current_goal_distance,
                "goal_cost": node.goal_cost,
                "goal_pose": waypoint,
                "start_pose": start_pose,
                "start_source": "explicit",
                "path_check_status": node.path_check_status,
                "path_check_pose_count": node.path_check_pose_count,
                "path_check_elapsed_s": node.path_check_elapsed_s,
                "path_check_error": node.path_check_error,
                "pre_arm_front_clearance_m": getattr(
                    node, "pre_arm_front_clearance_m", None
                ),
                "start_cost": getattr(node, "start_cost", None),
                "initial_path_heading_delta_rad": getattr(
                    node, "initial_path_heading_delta_rad", None
                ),
                "start_clearance_mode": getattr(
                    node, "start_clearance_mode", None
                ),
                "final_armed": node.armed,
            }
        diagnostic = exception_diagnostics(exc)
        return {
            "status": "FAIL",
            **diagnostic,
            "static_map_localization_gate": getattr(node, "localization_gate_result", None),
            "side_turn_review": getattr(node, "side_turn_review", None),
            "safe_egress_gate": egress_gate,
            "path_departure_poses": getattr(node, "path_departure_poses", None),
            "current_goal_distance_m": current_goal_distance,
            "goal_cost": node.goal_cost,
            "goal_pose": waypoint,
            "start_pose": start_pose,
            "start_source": "explicit" if start_pose is not None else "current_amcl",
            "path_check_status": node.path_check_status,
            "path_check_pose_count": node.path_check_pose_count,
            "path_check_elapsed_s": node.path_check_elapsed_s,
            "path_check_error": node.path_check_error,
            "pre_arm_front_clearance_m": getattr(
                node, "pre_arm_front_clearance_m", None
            ),
            "start_cost": getattr(node, "start_cost", None),
            "initial_path_heading_delta_rad": getattr(
                node, "initial_path_heading_delta_rad", None
            ),
            "start_clearance_mode": getattr(node, "start_clearance_mode", None),
            "final_armed": node.armed,
        }
    finally:
        node.destroy_node()


def run_route_preflight(waypoints, preflight_one=_preflight_one):
    results = []
    failed_waypoints = []
    deferred_segments = []
    egress_segments = []
    start_pose = None
    start_name = "CURRENT"
    for index, waypoint in enumerate(waypoints, start=1):
        allow_deferred_current_goal = (
            waypoint.get("name") == "ORIGIN" and index == len(waypoints)
        )
        result = dict(
            preflight_one(
                waypoint,
                start_pose,
                allow_deferred_current_goal,
            )
        )
        result["route_index"] = index
        result["segment"] = "%s->%s" % (start_name, waypoint["name"])
        result["waypoint"] = waypoint
        results.append(result)
        if result.get("status") == "REQUIRE_SAFE_EGRESS" and index == 1:
            egress_segments.append(result["segment"])
        elif result.get("status") != "PASS":
            if result.get("status") == "DEFERRED":
                deferred_segments.append(result["segment"])
            else:
                failed_waypoints.append(waypoint["name"])
        goal_pose = result.get("goal_pose")
        start_pose = {
            "x": waypoint["x"],
            "y": waypoint["y"],
            "yaw": (
                goal_pose["yaw"]
                if isinstance(goal_pose, dict) and "yaw" in goal_pose
                else waypoint["yaw"]
            ),
        }
        start_name = waypoint["name"]
    if failed_waypoints:
        return {
            "status": "FAIL",
            "error": "path preflight failed for segment(s) ending at %s"
            % ",".join(failed_waypoints),
            "segments": results,
            "deferred_segments": deferred_segments,
        }
    return {
        "status": "PASS",
        "error": None,
        "segments": results,
        "deferred_segments": deferred_segments,
        "egress_required": bool(egress_segments),
        "egress_segments": egress_segments,
    }


def publish_arrival_event(waypoint, event_dir):
    """Atomically publish a verified A/B/C arrival for the voice service."""
    name = str(waypoint.get("name", "")).upper()
    sequence = {"A": "100", "B": "200", "C": "300"}
    if name not in sequence:
        return None
    target_dir = Path(event_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(uuid4()),
        "event": "waypoint_arrived",
        "waypoint": name,
        "status": "PASS",
        "timestamp": time.time(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        prefix=".arrival-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        temp_path = Path(stream.name)
    final_path = target_dir / (
        "%s-waypoint-%s-%s.json" % (sequence[name], name, payload["event_id"])
    )
    os.replace(temp_path, final_path)
    return final_path


def publish_inspection_result_event(waypoint, task_result, event_dir):
    """Atomically publish a meter result for the independent voice service."""

    name = str(waypoint.get("name", "")).upper()
    sequence = {"A": "150", "B": "250", "C": "350"}
    spoken_name = {
        "A": "第一个巡检点",
        "B": "第二个巡检点",
        "C": "第三个巡检点",
    }
    if name not in sequence or task_result.get("status") == "SKIPPED":
        return None

    succeeded = (
        str(task_result.get("status", "")).upper() == "PASS"
        and task_result.get("task_succeeded") is True
    )
    failure_stage = None
    if succeeded:
        abnormal = task_result.get("is_anomaly") is True
        event_name = "meter_inspection_result"
        condition = "ABNORMAL" if abnormal else "NORMAL"
        speech_key = "meter_abnormal" if abnormal else "meter_normal"
        result_text = "异常" if abnormal else "正常"
        speech_text = "%s。仪表检测完成。结果%s。" % (
            spoken_name[name],
            result_text,
        )
        status = "PASS"
    else:
        failure_stage = str(task_result.get("failure_stage") or "vision").lower()
        condition = "UNKNOWN"
        if failure_stage == "heading_alignment":
            event_name = "meter_inspection_heading_failed"
            speech_key = "meter_heading_failed"
            speech_text = (
                "%s。观察方向调整未完成，未执行仪表识别。将继续后续任务。"
                % spoken_name[name]
            )
        elif failure_stage == "runtime":
            event_name = "meter_inspection_runtime_failed"
            speech_key = "meter_runtime_failed"
            speech_text = "%s。仪表巡检系统异常。将继续后续任务。" % spoken_name[name]
        else:
            failure_stage = "vision"
            event_name = "meter_inspection_failed"
            speech_key = "meter_unreadable"
            speech_text = "%s。仪表检测失败。将继续后续任务。" % spoken_name[name]
        status = "FAIL"

    target_dir = Path(event_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(uuid4()),
        "event": event_name,
        "waypoint": name,
        "status": status,
        "condition": condition,
        "speech_key": speech_key,
        "speech_text": speech_text,
        "timestamp": time.time(),
        "task_id": task_result.get("task_id"),
        "meter_id": task_result.get("meter_id"),
        "meter_kind": task_result.get("kind"),
        "value": task_result.get("value"),
        "unit": task_result.get("unit"),
        "confidence": task_result.get("confidence"),
        "failure_stage": failure_stage,
        "trial_count": task_result.get("trial_count", len(task_result.get("trials") or [])),
        "result_path": task_result.get("result_path"),
        "error": task_result.get("error"),
        "requires_ack": True,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        prefix=".meter-result-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        temp_path = Path(stream.name)
    final_path = target_dir / (
        "%s-meter-%s-%s.json" % (sequence[name], name, payload["event_id"])
    )
    os.replace(temp_path, final_path)
    return final_path


def _inspection_count_word(value):
    words = {0: "零", 1: "一", 2: "两", 3: "三"}
    number = int(value)
    return words.get(number, str(number))


def _inspection_count_clause(value, nonzero_prefix, zero_text):
    number = int(value)
    if number == 0:
        return zero_text
    return "%s%s项" % (nonzero_prefix, _inspection_count_word(number))


def inspection_summary_text(summary):
    """Build the fixed Chinese summary announced after a safe ORIGIN return."""

    clauses = [
        _inspection_count_clause(
            summary["attempted"], "共检查", "没有执行巡检点检查"
        ),
        _inspection_count_clause(
            summary["completed"], "成功完成", "没有成功完成的点位"
        ),
    ]
    for key, prefix in (
        ("heading_failed", "观察方向失败"),
        ("vision_failed", "仪表检测失败"),
        ("runtime_failed", "系统异常"),
    ):
        number = int(summary[key])
        if number > 0:
            clauses.append("%s%s项" % (prefix, _inspection_count_word(number)))
    clauses.extend(
        [
            _inspection_count_clause(
                summary["normal"], "正常结果", "没有正常结果"
            ),
            _inspection_count_clause(
                summary["abnormal"], "异常结果", "没有异常结果"
            ),
        ]
    )
    return "巡检结束。%s。" % "。".join(clauses)


def summarize_inspection_results(route_results):
    """Count readable, failed, normal and abnormal point inspections once."""

    attempted = normal = abnormal = failed = skipped = 0
    heading_failed = vision_failed = runtime_failed = 0
    point_results = []
    for route_result in route_results:
        task = route_result.get("inspection_task")
        if not isinstance(task, dict):
            continue
        waypoint = str((route_result.get("waypoint") or {}).get("name", "")).upper()
        status = str(task.get("status", "")).upper()
        if status == "SKIPPED":
            skipped += 1
            continue
        attempted += 1
        if status == "PASS" and task.get("task_succeeded") is True:
            condition = "ABNORMAL" if task.get("is_anomaly") is True else "NORMAL"
            if condition == "ABNORMAL":
                abnormal += 1
            else:
                normal += 1
        else:
            condition = "FAILED"
            failed += 1
            failure_stage = str(task.get("failure_stage") or "vision").lower()
            if failure_stage == "heading_alignment":
                heading_failed += 1
            elif failure_stage == "runtime":
                runtime_failed += 1
            else:
                failure_stage = "vision"
                vision_failed += 1
        point_results.append(
            {
                "waypoint": waypoint,
                "condition": condition,
                "failure_stage": None if condition != "FAILED" else failure_stage,
                "result_path": task.get("result_path"),
                "error": task.get("error"),
            }
        )
    completed = normal + abnormal
    if attempted != completed + failed:
        raise RuntimeError("inspection summary accounting invariant failed")
    if failed != heading_failed + vision_failed + runtime_failed:
        raise RuntimeError("inspection failure-stage accounting invariant failed")
    summary = {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "heading_failed": heading_failed,
        "vision_failed": vision_failed,
        "runtime_failed": runtime_failed,
        "normal": normal,
        "abnormal": abnormal,
        "skipped": skipped,
        "outcome": "PASS" if failed == 0 else "COMPLETED_WITH_FAILURES",
        "point_results": point_results,
    }
    summary["speech_text"] = inspection_summary_text(summary)
    summary["speech_key"] = "inspection_summary"
    return summary


def publish_inspection_summary_event(summary, event_dir):
    """Atomically publish the final count summary after ORIGIN is reached."""

    if not isinstance(summary, dict) or int(summary.get("attempted", 0)) <= 0:
        return None
    target_dir = Path(event_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(uuid4()),
        "event": "inspection_summary",
        "status": "PASS",
        "timestamp": time.time(),
        "requires_ack": True,
        **summary,
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        prefix=".inspection-summary-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        temp_path = Path(stream.name)
    final_path = target_dir / (
        "850-inspection-summary-%s.json" % payload["event_id"]
    )
    os.replace(temp_path, final_path)
    return final_path


def publish_navigation_recovery_event(
    event_name,
    waypoint,
    decision,
    event_dir,
    *,
    attempt=None,
):
    """Publish one acknowledged progress or pause prompt for route recovery."""

    event_profiles = {
        "navigation_recovery_started": (
            "STARTED",
            "400",
            decision.get("started_speech_key"),
        ),
        "navigation_recovery_succeeded": (
            "PASS",
            "450",
            "navigation_recovery_succeeded",
        ),
        "navigation_paused": (
            "FAIL",
            "500",
            decision.get("paused_speech_key"),
        ),
    }
    if event_name not in event_profiles:
        raise ValueError("unsupported navigation recovery event: %s" % event_name)
    status, sequence, speech_key = event_profiles[event_name]
    prompt = NAVIGATION_RECOVERY_PROMPTS.get(speech_key)
    if prompt is None:
        raise ValueError("navigation recovery prompt is unavailable: %s" % speech_key)
    target_dir = Path(event_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_id": str(uuid4()),
        "event": event_name,
        "waypoint": str(waypoint.get("name", "")).upper(),
        "status": status,
        "failure_code": decision.get("code"),
        "attempt": attempt,
        "max_attempts": decision.get("max_attempts"),
        "speech_key": speech_key,
        "speech_text": prompt[0],
        "operator_action": decision.get("operator_action"),
        "resume_command": decision.get("resume_command"),
        "requires_global_localization": bool(
            decision.get("requires_global_localization")
        ),
        "requires_ack": True,
        "timestamp": time.time(),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target_dir,
        prefix=".navigation-recovery-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        temp_path = Path(stream.name)
    final_path = target_dir / (
        "%s-%s-%s.json" % (sequence, event_name, payload["event_id"])
    )
    os.replace(temp_path, final_path)
    return final_path


def speech_ack_path(event_path):
    return Path(str(event_path) + ".ack")


def wait_for_speech_ack(
    event_path,
    timeout_s,
    *,
    clock=time.monotonic,
    sleeper=time.sleep,
):
    """Wait a bounded time for the voice owner to finish a prompt attempt."""

    timeout = max(0.0, float(timeout_s))
    if timeout <= 0.0:
        return {"status": "DISABLED", "played": None, "waited_s": 0.0}
    event_path = Path(event_path)
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event_id = str(event.get("event_id", ""))
    except (OSError, ValueError, AttributeError):
        event_id = ""
    ack_path = speech_ack_path(event_path)
    started = clock()
    deadline = started + timeout
    while clock() < deadline:
        if ack_path.is_file():
            try:
                ack = json.loads(ack_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ack = None
            if isinstance(ack, dict) and (
                not event_id or str(ack.get("event_id", "")) == event_id
            ):
                ack_path.unlink(missing_ok=True)
                return {
                    **ack,
                    "waited_s": round(max(0.0, clock() - started), 3),
                }
        sleeper(min(SPEECH_ACK_POLL_INTERVAL_S, max(0.0, deadline - clock())))
    return {
        "status": "TIMEOUT",
        "played": False,
        "event_id": event_id or None,
        "waited_s": round(max(0.0, clock() - started), 3),
        "error": "voice playback acknowledgement timed out",
    }


def run_route(
    waypoints,
    run_one=_run_one,
    clock=time.monotonic,
    arrival_event_dir=None,
    task_runner=None,
    speech_ack_timeout_s=0.0,
    navigation_speech_ack_timeout_s=0.0,
    sleeper=time.sleep,
    initial_results=None,
    resume_localization_verified=False,
):
    if arrival_event_dir is not None:
        Path(arrival_event_dir).mkdir(parents=True, exist_ok=True)
    results = [dict(item) for item in (initial_results or ())]
    route_index_offset = len(results)
    started = clock()
    continuity_anchor = route_start_continuity_anchor(
        results,
        resume_localization_verified,
    )
    warnings = []
    for local_index, waypoint in enumerate(waypoints, start=1):
        index = route_index_offset + local_index
        recovery_attempts = 0
        recovery_decision = None
        attempt_history = []
        retry_anchor = continuity_anchor
        while True:
            if run_one is _run_one:
                result = dict(
                    run_one(
                        waypoint,
                        allow_near_start_egress=True,
                        continuity_anchor=retry_anchor,
                    )
                )
            else:
                result = dict(run_one(waypoint))
            result["route_index"] = index
            result["navigation_attempt"] = recovery_attempts + 1
            if result.get("status") == "PASS":
                attempt_history.append(dict(result))
                if recovery_attempts:
                    result["navigation_recovered"] = True
                    result["navigation_recovery_attempts"] = recovery_attempts
                    result["navigation_failure_code"] = recovery_decision.get("code")
                    if arrival_event_dir is not None:
                        event_path = publish_navigation_recovery_event(
                            "navigation_recovery_succeeded",
                            waypoint,
                            recovery_decision,
                            arrival_event_dir,
                            attempt=recovery_attempts,
                        )
                        result["navigation_recovery_succeeded_event_file"] = str(
                            event_path
                        )
                        ack = wait_for_speech_ack(
                            event_path,
                            navigation_speech_ack_timeout_s,
                            clock=clock,
                            sleeper=sleeper,
                        )
                        result["navigation_recovery_succeeded_speech_ack"] = ack
                        if ack.get("status") not in {"PASS", "DISABLED"}:
                            warnings.append(
                                "navigation recovery speech was not confirmed at waypoint %s: %s"
                                % (waypoint["name"], ack.get("status"))
                            )
                result["navigation_attempts"] = attempt_history
                break

            decision = classify_navigation_failure(result)
            result["navigation_failure"] = decision
            can_retry = (
                retry_is_safe(result, decision)
                and result.get("automatic_recovery_forbidden") is not True
                and recovery_attempts < int(decision.get("max_attempts", 0))
                and clock() - started <= ROUTE_MAX_DURATION_S
            )
            result["automatic_recovery_eligible"] = can_retry
            attempt_history.append(dict(result))
            if not can_retry:
                result["navigation_attempts"] = attempt_history
                results.append(result)
                completed = [
                    item["waypoint"]["name"]
                    for item in results
                    if item.get("status") == "PASS"
                ]
                pause_event = None
                pause_ack = None
                if arrival_event_dir is not None:
                    pause_event = publish_navigation_recovery_event(
                        "navigation_paused",
                        waypoint,
                        decision,
                        arrival_event_dir,
                        attempt=recovery_attempts,
                    )
                    pause_ack = wait_for_speech_ack(
                        pause_event,
                        navigation_speech_ack_timeout_s,
                        clock=clock,
                        sleeper=sleeper,
                    )
                    result["navigation_paused_event_file"] = str(pause_event)
                    result["navigation_paused_speech_ack"] = pause_ack
                    if pause_ack.get("status") not in {"PASS", "DISABLED"}:
                        warnings.append(
                            "navigation pause speech was not confirmed at waypoint %s: %s"
                            % (waypoint["name"], pause_ack.get("status"))
                        )
                remaining = [
                    str(item.get("name", "")).upper()
                    for item in waypoints[local_index - 1 :]
                ]
                return {
                    "status": "FAIL",
                    "route_state": "PAUSED",
                    "error": "route paused after waypoint %s failed" % waypoint["name"],
                    "duration_s": clock() - started,
                    "route": results,
                    "completed_waypoints": completed,
                    "resume_plan": {
                        "failed_waypoint": str(waypoint.get("name", "")).upper(),
                        "remaining_waypoints": remaining,
                        "failure_code": decision.get("code"),
                        "operator_action": decision.get("operator_action"),
                        "resume_command": decision.get("resume_command"),
                        "requires_global_localization": bool(
                            decision.get("requires_global_localization")
                        ),
                    },
                    "warnings": warnings,
                }

            recovery_attempts += 1
            recovery_decision = decision
            if (arrival_event_dir is not None
                    and decision.get("started_speech_key") is not None):
                event_path = publish_navigation_recovery_event(
                    "navigation_recovery_started",
                    waypoint,
                    decision,
                    arrival_event_dir,
                    attempt=recovery_attempts,
                )
                attempt_history[-1]["navigation_recovery_started_event_file"] = str(
                    event_path
                )
                ack = wait_for_speech_ack(
                    event_path,
                    navigation_speech_ack_timeout_s,
                    clock=clock,
                    sleeper=sleeper,
                )
                attempt_history[-1]["navigation_recovery_started_speech_ack"] = ack
                if ack.get("status") not in {"PASS", "DISABLED"}:
                    warnings.append(
                        "navigation recovery start speech was not confirmed at waypoint %s: %s"
                        % (waypoint["name"], ack.get("status"))
                    )
            failed_pose = result.get("final_amcl_pose")
            if isinstance(failed_pose, dict):
                retry_anchor = dict(failed_pose)
            sleeper(float(decision.get("retry_delay_s", 0.0)))

        results.append(result)
        completed = [
            item["waypoint"]["name"]
            for item in results
            if item.get("status") == "PASS"
        ]
        final_amcl_pose = result.get("final_amcl_pose")
        if isinstance(final_amcl_pose, dict):
            continuity_anchor = dict(final_amcl_pose)
        elif run_one is _run_one and index < len(waypoints):
            return {
                "status": "FAIL",
                "error": (
                    "route stopped after waypoint %s returned no final AMCL pose"
                    % waypoint["name"]
                ),
                "duration_s": clock() - started,
                "route": results,
                "completed_waypoints": completed,
            }
        if arrival_event_dir is not None:
            event_path = publish_arrival_event(
                result.get("waypoint") or waypoint,
                arrival_event_dir,
            )
            if event_path is not None:
                result["arrival_event_file"] = str(event_path)
        if task_runner is not None:
            try:
                task_result = dict(task_runner(waypoint, index, result))
            except BaseException as exc:
                task_result = {
                    "status": "FAIL",
                    "task_succeeded": False,
                    "failure_policy": "abort",
                    "motion_authorized": False,
                    **exception_diagnostics(exc),
                }
            result["inspection_task"] = task_result
            if arrival_event_dir is not None:
                event_path = publish_inspection_result_event(
                    result.get("waypoint") or waypoint,
                    task_result,
                    arrival_event_dir,
                )
                if event_path is not None:
                    task_result["speech_event_file"] = str(event_path)
                    speech_ack = wait_for_speech_ack(
                        event_path,
                        speech_ack_timeout_s,
                        clock=clock,
                        sleeper=sleeper,
                    )
                    task_result["speech_ack"] = speech_ack
                    if speech_ack.get("status") not in {"PASS", "DISABLED"}:
                        warnings.append(
                            "inspection speech was not confirmed at waypoint %s: %s"
                            % (waypoint["name"], speech_ack.get("status"))
                        )
            continuity_anchor = update_continuity_anchor_after_task(
                continuity_anchor,
                task_result,
            )
            if task_result.get("status") not in {"PASS", "SKIPPED"}:
                if task_result.get("failure_policy", "abort") == "abort":
                    completed_tasks = [
                        item["waypoint"]["name"]
                        for item in results[:-1]
                        if item.get("status") == "PASS"
                    ]
                    return {
                        "status": "FAIL",
                        "error": "inspection task failed at waypoint %s" % waypoint["name"],
                        "duration_s": clock() - started,
                        "route": results,
                        "completed_waypoints": completed_tasks,
                    }
        if clock() - started > ROUTE_MAX_DURATION_S:
            return {
                "status": "FAIL",
                "error": "route duration limit exceeded",
                "duration_s": clock() - started,
                "route": results,
                "completed_waypoints": completed,
            }
    summary = summarize_inspection_results(results)
    if (
        arrival_event_dir is not None
        and summary["attempted"] > 0
        and results
        and str(results[-1].get("waypoint", {}).get("name", "")).upper()
        == "ORIGIN"
    ):
        summary_event = publish_inspection_summary_event(summary, arrival_event_dir)
        if summary_event is not None:
            summary["speech_event_file"] = str(summary_event)
            summary_ack = wait_for_speech_ack(
                summary_event,
                speech_ack_timeout_s,
                clock=clock,
                sleeper=sleeper,
            )
            summary["speech_ack"] = summary_ack
            if summary_ack.get("status") not in {"PASS", "DISABLED"}:
                warnings.append(
                    "inspection summary speech was not confirmed: %s"
                    % summary_ack.get("status")
                )
    return {
        "status": "PASS",
        "error": None,
        "duration_s": clock() - started,
        "route": results,
        "completed_waypoints": [item["waypoint"]["name"] for item in results],
        "inspection_summary": summary,
        "warnings": warnings,
    }


def main() -> int:
    args = _parse_args()
    try:
        # Configuration-only gate: the catalog and static safety grid must
        # belong to the same selected site before localization or any motion.
        validate_map_waypoint_binding(_active_map_file(), args.waypoints_file)
        route_names, waypoints = _load_route(args.waypoints_file, args.route)
        resume_results = load_route_resume_context(args.resume_context, route_names)
        inspection_plan = (
            load_inspection_plan(args.inspection_config)
            if args.inspection_config
            else None
        )
        if args.extract_global_pose:
            pose = load_global_localization_pose(args.extract_global_pose)
            print("export RVCLAW_NAV2_INITIAL_X=%.9f" % pose["x"])
            print("export RVCLAW_NAV2_INITIAL_Y=%.9f" % pose["y"])
            print("export RVCLAW_NAV2_INITIAL_YAW_DEG=%.9f" % pose["yaw_deg"])
            return 0
        if args.describe:
            description = describe_plan(route_names)
            description["inspection"] = (
                None
                if inspection_plan is None
                else inspection_plan_summary(inspection_plan)
            )
            print(json.dumps(description, indent=2, sort_keys=True))
            return 0
        if args.inspection_readiness_only:
            if inspection_plan is None:
                raise ValueError("--inspection-readiness-only requires --inspection-config")
            readiness = InspectionExecutor(inspection_plan).readiness()
            print(json.dumps(readiness, indent=2, ensure_ascii=False, sort_keys=True))
            return 0 if readiness["status"] == "PASS" else 1
        if args.resume_localization_verified and not args.resume_context:
            raise ValueError(
                "--resume-localization-verified requires --resume-context"
            )
        if not args.preflight_only and args.arm_token != ROUTE_ARM_TOKEN:
            raise ValueError("exact waypoint-route arm token is required")
        trace_path = _validate_trace(args.trace_file)
    except ValueError as exc:
        print("ARGUMENT_ERROR: %s" % exc, file=sys.stderr)
        return 2

    WAYPOINT.BASE.ACTION_RESULT_TIMEOUT_S = WAYPOINT.MAX_ACTION_RESULT_TIMEOUT_S
    rclpy.init()
    try:
        if args.preflight_only:
            result = run_route_preflight(waypoints)
        else:
            task_runner = (
                None
                if inspection_plan is None
                else _build_inspection_task_runner(inspection_plan, trace_path.stem)
            )
            result = run_route(
                waypoints,
                arrival_event_dir=os.getenv(
                    "RVCLAW_ARRIVAL_EVENT_DIR",
                    DEFAULT_ARRIVAL_EVENT_DIR,
                ),
                task_runner=task_runner,
                speech_ack_timeout_s=(
                    0.0
                    if inspection_plan is None
                    else inspection_plan.speech_ack_timeout_s
                ),
                navigation_speech_ack_timeout_s=RECOVERY_SPEECH_ACK_TIMEOUT_S,
                initial_results=resume_results,
                resume_localization_verified=args.resume_localization_verified,
            )
    except BaseException as exc:
        diagnostic = exception_diagnostics(exc)
        result = {
            "status": "FAIL",
            **diagnostic,
            "duration_s": 0.0,
            "route": [],
            "completed_waypoints": [],
        }
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    if not args.preflight_only:
        try:
            checkpoint = write_route_checkpoint(
                args.checkpoint_file,
                route_names,
                result,
            )
            result["checkpoint_file"] = None if checkpoint is None else str(checkpoint)
        except (OSError, ValueError) as exc:
            result["checkpoint_error"] = str(exc)
    trace_path.write_text(
        json.dumps(
            {
                "goal_mode": "absolute_waypoint_route",
                "preflight_only": args.preflight_only,
                "route_requested": list(route_names),
                "inspection": (
                    None
                    if inspection_plan is None
                    else inspection_plan_summary(inspection_plan)
                ),
                **result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.preflight_only:
        return preflight_exit_status(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
