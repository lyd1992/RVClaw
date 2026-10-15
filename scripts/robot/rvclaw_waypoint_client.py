#!/usr/bin/env python3
"""Navigate to one named absolute pose from the reviewed map catalog."""

import argparse
import json
import math
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory
import rclpy
from action_msgs.msg import GoalStatus


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(
    0,
    str(WORKSPACE_ROOT / "ros2_ws" / "src" / "rvclaw_nav2_bringup"),
)

import nav2_forward_goal_030 as BASE  # noqa: E402
from rvclaw_nav2_bringup.waypoint_catalog import select_waypoint  # noqa: E402


ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_NAV2_WAYPOINT"
DEFAULT_WAYPOINTS_FILE = (
    "/data/rvclaw-k3-lab/maps/slam-20260827-105711/waypoints.yaml"
)
MAX_WAYPOINT_DISTANCE_M = 12.0
MAX_ACTION_RESULT_TIMEOUT_S = 120.0
MAX_FINAL_POSITION_ERROR_M = 0.15
MAX_FINAL_YAW_ERROR_RAD = 0.30


class WaypointClient(BASE.Nav2ForwardGoalClient):
    def __init__(self, waypoint):
        super().__init__()
        self.waypoint = waypoint

    def _compute_goal_pose(self):
        return {
            "x": self.waypoint["x"],
            "y": self.waypoint["y"],
            "yaw": self.waypoint["yaw"],
        }

    def _configure_navigation_goal(self, goal):
        path = (
            Path(get_package_share_directory("rvclaw_nav2_bringup"))
            / "behavior_trees" / "navigate_to_pose_heading_forward_only.xml"
        )
        if not path.is_file():
            raise RuntimeError("full-pose behavior tree is unavailable: %s" % path)
        goal.behavior_tree = str(path)

    def _pre_arm_gate(self, check_localization: bool = True):
        """Validate the absolute catalog pose before admitting the goal."""
        super()._pre_arm_gate(check_localization=check_localization)
        if any(
            not math.isclose(
                self.goal_pose[key], self.waypoint[key], rel_tol=0.0, abs_tol=1e-9
            )
            for key in ("x", "y", "yaw")
        ):
            raise RuntimeError("absolute waypoint goal was not preserved")

    def evaluate_navigation_evidence(self, metrics):
        failures = []
        if metrics.get("action_status") != GoalStatus.STATUS_SUCCEEDED:
            failures.append("NavigateToPose did not report success")
        if int(metrics.get("action_feedback_samples", 0)) < 2:
            failures.append("NavigateToPose emitted too few feedback samples")
        if float(metrics.get("action_feedback_final_distance_m", math.inf)) > 0.15:
            failures.append("NavigateToPose final feedback remained too far from goal")
        for key, label in (
            ("motion_raw_odom_samples", "raw odometry"),
            ("motion_amcl_samples", "AMCL"),
            ("monitor_output_samples", "Collision Monitor output"),
        ):
            if int(metrics.get(key, 0)) < (5 if key == "motion_raw_odom_samples" else 1):
                failures.append("too few %s samples during navigation" % label)
        if metrics.get("guarded_linear_min_m_s") is None:
            failures.append("no guarded command was observed")
        if float(metrics.get("guarded_linear_max_m_s", 0.0)) > (
            BASE.MAX_GUARDED_LINEAR_X_M_S + BASE.ZERO_EPSILON
        ):
            failures.append("guarded command exceeded forward velocity ceiling")
        if float(metrics.get("max_guarded_linear_y_m_s", math.inf)) > BASE.ZERO_EPSILON:
            failures.append("guarded command contained lateral velocity")
        if float(metrics.get("max_guarded_angular_z_rad_s", math.inf)) > (
            BASE.MAX_GUARDED_ANGULAR_Z_RAD_S + BASE.ZERO_EPSILON
        ):
            failures.append("guarded command exceeded angular velocity ceiling")
        if float(metrics.get("pre_arm_scan_age_s", math.inf)) > BASE.MAXIMUM_SCAN_AGE_S:
            failures.append("pre-arm scan was stale")
        failures.extend(BASE.evaluate_start_clearance_evidence(metrics))
        if int(metrics.get("goal_cost", 255)) > BASE.MAXIMUM_GOAL_COST:
            failures.append("goal cell cost exceeded the admitted limit")
        if int(metrics.get("post_stop_stable_samples", 0)) < BASE.POST_ACTION_STABLE_SAMPLES:
            failures.append("post-action feedback did not remain stably stopped")
        if abs(float(metrics.get("final_feedback_x_m_s", math.inf))) > BASE.MAX_POST_STOP_LINEAR_M_S:
            failures.append("post-action X feedback remained above stop limit")
        if abs(float(metrics.get("final_feedback_angular_rad_s", math.inf))) > BASE.MAX_POST_STOP_ANGULAR_RAD_S:
            failures.append("post-action angular feedback remained above stop limit")
        if metrics.get("final_armed") is not False:
            failures.append("hardware controller did not finish disarmed")

        final_pose = self.final_amcl_pose
        if final_pose is None:
            failures.append("final AMCL pose is unavailable")
        else:
            position_error = math.hypot(
                final_pose["x"] - self.waypoint["x"],
                final_pose["y"] - self.waypoint["y"],
            )
            yaw_error = abs(BASE.normalize_angle(final_pose["yaw"] - self.waypoint["yaw"]))
            if position_error > MAX_FINAL_POSITION_ERROR_M:
                failures.append("final AMCL position error exceeded %.2f m" % MAX_FINAL_POSITION_ERROR_M)
            if yaw_error > MAX_FINAL_YAW_ERROR_RAD:
                failures.append("final AMCL yaw error exceeded %.2f rad" % MAX_FINAL_YAW_ERROR_RAD)
        return failures

    def result(self, status, error):
        result = super().result(status, error)
        result.update({
            "goal_mode": "absolute_waypoint",
            "waypoint": self.waypoint,
            "waypoint_arm_token": ARM_TOKEN,
        })
        return result


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--trace-file")
    parser.add_argument("--arm-token")
    parser.add_argument("--waypoint", default="A")
    parser.add_argument("--waypoints-file", default=DEFAULT_WAYPOINTS_FILE)
    return parser.parse_args()


def describe_plan(waypoint):
    return {
        "hardware_io": False,
        "goal_mode": "absolute_waypoint",
        "frame_id": "map",
        "waypoint": waypoint,
        "required_arm_token": ARM_TOKEN,
        "action_name": BASE.ACTION_NAME,
        "max_action_result_timeout_s": MAX_ACTION_RESULT_TIMEOUT_S,
        "max_waypoint_distance_m": MAX_WAYPOINT_DISTANCE_M,
        "collision_monitor_required": True,
        "fresh_scan_gate_required": True,
        "automatic_reverse": False,
        "initial_pose_required": True,
    }


def main() -> int:
    args = _parse_args()
    try:
        waypoint = select_waypoint(args.waypoints_file, args.waypoint)
    except ValueError as exc:
        print("WAYPOINT_ERROR: %s" % exc, file=sys.stderr)
        return 2
    if math.hypot(waypoint["x"], waypoint["y"]) > MAX_WAYPOINT_DISTANCE_M:
        print("WAYPOINT_ERROR: waypoint is outside the reviewed map distance limit", file=sys.stderr)
        return 2
    if args.describe:
        print(json.dumps(describe_plan(waypoint), indent=2, sort_keys=True))
        return 0
    if args.arm_token != ARM_TOKEN:
        print("ARGUMENT_ERROR: exact waypoint arm token is required", file=sys.stderr)
        return 2
    if not args.trace_file:
        print("ARGUMENT_ERROR: trace-file is required", file=sys.stderr)
        return 2
    trace_path = Path(args.trace_file).resolve()
    try:
        trace_path.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        print("ARGUMENT_ERROR: trace file must remain under the workspace root", file=sys.stderr)
        return 2
    if trace_path.exists() or not trace_path.parent.is_dir():
        print("ARGUMENT_ERROR: trace file must be new and its parent must exist", file=sys.stderr)
        return 2

    BASE.ACTION_RESULT_TIMEOUT_S = MAX_ACTION_RESULT_TIMEOUT_S
    rclpy.init()
    node = WaypointClient(waypoint)
    try:
        result = node.run_navigation()
    except BaseException as exc:
        try:
            node.emergency_cleanup()
        except BaseException as cleanup_exc:
            node._record("cleanup_error", str(cleanup_exc))
        result = node.result("FAIL", str(exc))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    trace_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
