#!/usr/bin/env python3
"""Localize from a stationary scan, with a safe bounded-spin fallback."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
import rclpy
from nav2_msgs.action import DriveOnHeading
from nav2_msgs.srv import SetInitialPose
from rclpy.action import ActionClient
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from nav2_heading_spin_left_0350 import Nav2HeadingClient  # noqa: E402
from rvclaw_nav2_bringup.amcl_localization_observer import (  # noqa: E402
    quaternion_to_yaw,
    summarize_global_pose_samples,
)
from stm32_control_client import (  # noqa: E402
    MIN_BATTERY_VOLTAGE_V,
    STARTUP_TIMEOUT_S,
)
from rvclaw_nav2_bringup.safe_egress_policy import (  # noqa: E402
    EGRESS_ACTION_TIME_ALLOWANCE_S,
    EGRESS_DISTANCE_M,
    EGRESS_MIN_DISTANCE_M,
    EGRESS_RESULT_TIMEOUT_S,
    EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
    EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
    EGRESS_SPEED_M_S,
    evaluate_front_egress_scan,
    evaluate_runtime_egress_monitor,
    evaluate_safe_egress,
    evaluate_static_map_egress,
    remaining_egress_distance,
    select_safe_egress,
)


ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_AMCL_GLOBAL_SCAN"
STATIONARY_ONLY_ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_AMCL_STATIC_LOCALIZATION"
SEGMENT_COUNT = 4
SEGMENT_YAW_RAD = math.pi / 2.0
SEGMENT_TIME_ALLOWANCE_S = 10
SEGMENT_RESULT_TIMEOUT_S = 14.0
MIN_SEGMENT_RAW_YAW_RAD = 1.0
MAX_SEGMENT_RAW_YAW_RAD = 2.2
MAX_TOTAL_RAW_SCAN_YAW_RAD = 2.0 * math.pi + math.radians(5.0)
SCAN_STOP_RESERVE_RAD = math.radians(5.0)
OBSERVATION_UPDATES = 8
OBSERVATION_INTERVAL_S = 0.75
MAX_POSITION_DRIFT_M = 0.10
MAX_YAW_DRIFT_DEG = 10.0
MAX_POSITION_VARIANCE = 0.10
MAX_YAW_VARIANCE = math.radians(10.0) ** 2
# AMCL startup and global particle reinitialization can take several seconds
# on the K3 under concurrent lidar/costmap load. Keep a generous bounded
# timeout so a slow response is distinguished from a real service failure.
SERVICE_TIMEOUT_S = 30.0
STATIC_MATCH_COARSE_POSITION_STEP_M = 0.40
STATIC_MATCH_COARSE_YAW_STEP_DEG = 20.0
STATIC_MATCH_REFINE_POSITION_RADIUS_M = 0.40
STATIC_MATCH_REFINE_YAW_RADIUS_DEG = 20.0
STATIC_MATCH_REFINE_POSITION_STEP_M = 0.05
STATIC_MATCH_REFINE_YAW_STEP_DEG = 2.5
STATIC_MATCH_MAXIMUM_POINTS = 120
STATIC_MATCH_MINIMUM_SCORE = 0.80
STATIC_MATCH_MINIMUM_KNOWN_RATIO = 0.85
STATIC_MATCH_MINIMUM_SCORE_MARGIN = 0.05
STATIC_MATCH_MAXIMUM_SECOND_RATIO = 0.95
STATIC_MATCH_MAXIMUM_ATTEMPTS = 3
ROTATION_SWEEP_RADIUS_M = 0.205
ROTATION_CLEARANCE_MARGIN_M = 0.045
ROTATION_CLEARANCE_MIN_M = ROTATION_SWEEP_RADIUS_M + ROTATION_CLEARANCE_MARGIN_M
ROTATION_CLEARANCE_MIN_COVERAGE_DEG = 330.0
EGRESS_MIN_FEEDBACK_DISTANCE_M = 0.10
EGRESS_MAX_FEEDBACK_DISTANCE_M = EGRESS_DISTANCE_M + 0.10
EGRESS_MAX_GUARDED_SPEED_M_S = EGRESS_SPEED_M_S + 0.02


def describe_plan(stationary_only: bool = False) -> Dict[str, object]:
    return {
        "hardware_io": False,
        "motion_on_describe": False,
        "required_arm_token": (
            STATIONARY_ONLY_ARM_TOKEN if stationary_only else ARM_TOKEN
        ),
        "localization_mode": (
            "stationary_static_scan_only"
            if stationary_only
            else "static_scan_first_safe_spin_fallback"
        ),
        "stationary_only": stationary_only,
        "motion_fallback_allowed": not stationary_only,
        "static_match_service": "/set_initial_pose",
        "static_match_search_mode": "coarse_to_fine",
        "static_match_coarse_position_step_m": STATIC_MATCH_COARSE_POSITION_STEP_M,
        "static_match_coarse_yaw_step_deg": STATIC_MATCH_COARSE_YAW_STEP_DEG,
        "static_match_refine_position_radius_m": STATIC_MATCH_REFINE_POSITION_RADIUS_M,
        "static_match_refine_yaw_radius_deg": STATIC_MATCH_REFINE_YAW_RADIUS_DEG,
        "static_match_refine_position_step_m": STATIC_MATCH_REFINE_POSITION_STEP_M,
        "static_match_refine_yaw_step_deg": STATIC_MATCH_REFINE_YAW_STEP_DEG,
        "static_match_maximum_points": STATIC_MATCH_MAXIMUM_POINTS,
        "static_match_minimum_score": STATIC_MATCH_MINIMUM_SCORE,
        "static_match_minimum_known_ratio": STATIC_MATCH_MINIMUM_KNOWN_RATIO,
        "static_match_minimum_score_margin": STATIC_MATCH_MINIMUM_SCORE_MARGIN,
        "static_match_maximum_second_ratio": STATIC_MATCH_MAXIMUM_SECOND_RATIO,
        "static_match_does_not_move": True,
        "static_match_maximum_attempts": STATIC_MATCH_MAXIMUM_ATTEMPTS,
        "static_match_required_consecutive_passes": 2,
        "global_localization_service": "/reinitialize_global_localization",
        "spin_action": "/spin",
        "segment_count": SEGMENT_COUNT,
        "segment_yaw_rad": SEGMENT_YAW_RAD,
        "segment_yaw_deg": math.degrees(SEGMENT_YAW_RAD),
        "total_scan_yaw_rad": SEGMENT_COUNT * SEGMENT_YAW_RAD,
        "total_scan_yaw_deg": SEGMENT_COUNT * math.degrees(SEGMENT_YAW_RAD),
        "maximum_measured_scan_yaw_deg": math.degrees(MAX_TOTAL_RAW_SCAN_YAW_RAD),
        "adjusts_segments_using_stopped_raw_odometry": True,
        "separate_arm_cycle_per_segment": True,
        "linear_motion_allowed": False,
        "collision_checks_required": True,
        "final_disarm_required": True,
        "final_localization_updates": OBSERVATION_UPDATES,
        "maximum_position_drift_m": MAX_POSITION_DRIFT_M,
        "maximum_yaw_drift_deg": MAX_YAW_DRIFT_DEG,
        "maximum_position_variance": MAX_POSITION_VARIANCE,
        "maximum_yaw_variance": MAX_YAW_VARIANCE,
        "navigation_goal_on_success": False,
        "spin_fallback_requires_clearance": True,
        "rotation_clearance_min_m": ROTATION_CLEARANCE_MIN_M,
        "rotation_clearance_min_coverage_deg": ROTATION_CLEARANCE_MIN_COVERAGE_DEG,
        "safe_egress_enabled": True,
        "safe_egress_action": "/drive_on_heading",
        "safe_egress_distance_m": EGRESS_DISTANCE_M,
        "safe_egress_minimum_distance_m": EGRESS_MIN_DISTANCE_M,
        "safe_egress_adaptive_distance": True,
        "safe_egress_speed_m_s": EGRESS_SPEED_M_S,
        "safe_egress_action_time_allowance_s": EGRESS_ACTION_TIME_ALLOWANCE_S,
        "safe_egress_requires_static_map_gate": True,
        "safe_egress_requires_live_scan_gate": True,
        "safe_egress_runtime_scan_monitor_required": True,
        "safe_egress_runtime_maximum_scan_age_s": EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
        "safe_egress_runtime_minimum_scan_checks": EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
        "safe_egress_runtime_failure_policy": "cancel_estop_zero_disarm",
        "safe_egress_relocalizes_after_motion": True,
        "safe_egress_allows_reverse": False,
        "safe_egress_allows_lateral": False,
        "safe_egress_allows_spin": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--stationary-only", action="store_true")
    parser.add_argument("--trace-file")
    parser.add_argument("--arm-token")
    return parser.parse_args()


def _validate_run_args(args: argparse.Namespace) -> Path:
    required_token = (
        STATIONARY_ONLY_ARM_TOKEN if args.stationary_only else ARM_TOKEN
    )
    if args.arm_token != required_token:
        raise ValueError("exact AMCL localization arm token is required")
    if not args.trace_file:
        raise ValueError("trace-file is required")
    candidate = Path(args.trace_file)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    trace_path = candidate.resolve()
    try:
        trace_path.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("trace file must remain under the workspace root") from exc
    if not trace_path.parent.is_dir():
        raise ValueError("trace file parent directory does not exist")
    if trace_path.exists():
        raise ValueError("trace file already exists")
    return trace_path


class GlobalLocalizationScanClient(Nav2HeadingClient):
    def __init__(self, stationary_only: bool = False) -> None:
        super().__init__(
            target_yaw_rad=SEGMENT_YAW_RAD,
            action_time_allowance_s=SEGMENT_TIME_ALLOWANCE_S,
            action_result_timeout_s=SEGMENT_RESULT_TIMEOUT_S,
            minimum_raw_yaw_delta_rad=MIN_SEGMENT_RAW_YAW_RAD,
            maximum_raw_yaw_delta_rad=MAX_SEGMENT_RAW_YAW_RAD,
        )
        self.global_localization_client = self.create_client(
            Empty,
            "/reinitialize_global_localization",
        )
        self.initial_pose_client = self.create_client(
            SetInitialPose,
            "/set_initial_pose",
        )
        self.nomotion_client = self.create_client(
            Empty,
            "/request_nomotion_update",
        )
        self.amcl_samples = []
        self.segment_results = []
        self.localization_summary: Optional[Dict[str, object]] = None
        self.latest_match_scan = None
        self.latest_localization_scan = None
        self.latest_localization_scan_monotonic = None
        self.latest_match_scan_monotonic: Optional[float] = None
        self.static_map_localization_gate: Optional[Dict[str, object]] = None
        self.static_match_summary: Optional[Dict[str, object]] = None
        self.static_match_search: Optional[Dict[str, object]] = None
        self.static_match_used = False
        self.rotation_clearance: Optional[Dict[str, object]] = None
        self.safe_egress_summary: Optional[Dict[str, object]] = None
        self.stationary_only = bool(stationary_only)
        self.egress_action = ActionClient(self, DriveOnHeading, "/drive_on_heading")
        self.egress_active = False
        self.egress_feedback_samples = 0
        self.egress_feedback_distance_m = 0.0
        self.egress_action_status: Optional[int] = None
        self.egress_guarded_linear_min: Optional[float] = None
        self.egress_guarded_linear_max: Optional[float] = None
        self.egress_guarded_lateral_max = 0.0
        self.egress_guarded_angular_max = 0.0
        self.egress_runtime_scan_checks = 0
        self.egress_runtime_latest_gate: Optional[Dict[str, object]] = None
        self.egress_runtime_failure: Optional[Dict[str, object]] = None
        self.egress_runtime_cancelled = False
        qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_amcl_pose,
            qos,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_match_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan, "/scan_localization", self._on_localization_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            "/rvclaw/control/guarded_cmd_vel",
            self._on_egress_guarded_command,
            qos,
        )

    def _on_localization_scan(self, message: LaserScan) -> None:
        self.latest_localization_scan = message
        self.latest_localization_scan_monotonic = time.monotonic()

    def _on_match_scan(self, message: LaserScan) -> None:
        self.latest_match_scan = message
        self.latest_match_scan_monotonic = time.monotonic()
        if not self.egress_active:
            return
        selected_distance_m = (
            self.safe_egress_summary.get("selected_distance_m", EGRESS_DISTANCE_M)
            if self.safe_egress_summary is not None
            else EGRESS_DISTANCE_M
        )
        remaining_distance_m = remaining_egress_distance(
            selected_distance_m,
            self.egress_feedback_distance_m,
        )
        gate = evaluate_front_egress_scan(
            message.ranges,
            message.angle_min,
            message.angle_increment,
            message.range_min,
            message.range_max,
            distance_m=remaining_distance_m,
        )
        self.egress_runtime_scan_checks += 1
        self.egress_runtime_latest_gate = gate
        monitor = evaluate_runtime_egress_monitor(gate, self._match_scan_age_s())
        monitor["selected_distance_m"] = selected_distance_m
        monitor["distance_traveled_m"] = self.egress_feedback_distance_m
        monitor["remaining_distance_m"] = remaining_distance_m
        if monitor["status"] != "PASS" and self.egress_runtime_failure is None:
            self.egress_runtime_failure = monitor
            self._record("safe_egress_runtime_scan_failure", monitor)

    def _match_scan_stamp_ns(self) -> int:
        if self.latest_match_scan is None:
            return 0
        stamp = self.latest_match_scan.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _match_scan_age_s(self) -> float:
        if self.latest_match_scan_monotonic is None:
            return math.inf
        stamp_ns = self._match_scan_stamp_ns()
        source_age_s = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if stamp_ns <= 0 or source_age_s < -0.05:
            return math.inf
        return max(
            0.0, source_age_s,
            time.monotonic() - self.latest_match_scan_monotonic,
        )

    def _refresh_match_scan(self, reason: str) -> None:
        # Global matching blocks this executor. Require a newly acquired scan,
        # not merely delivery of a queued message after computation finishes.
        started_ns = self.get_clock().now().nanoseconds
        started_monotonic = time.monotonic()
        self.wait_for(
            lambda: (
                self.latest_match_scan_monotonic is not None
                and self.latest_match_scan_monotonic >= started_monotonic
                and self._match_scan_stamp_ns() > started_ns
                and self._match_scan_age_s() <= EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S
                and self.latest_localization_scan is not None
                and self.latest_localization_scan_monotonic is not None
                and self.latest_localization_scan_monotonic >= started_monotonic
                and (self.latest_localization_scan.header.stamp.sec * 1_000_000_000
                     + self.latest_localization_scan.header.stamp.nanosec) > started_ns
            ),
            STARTUP_TIMEOUT_S,
            "fresh acquired scan for " + reason,
        )
        self._record("fresh_match_scan", {
            "reason": reason,
            "scan_age_s": self._match_scan_age_s(),
            "wait_s": time.monotonic() - started_monotonic,
        })

    def _on_egress_guarded_command(self, message: Twist) -> None:
        if not self.egress_active:
            return
        linear_x = float(message.linear.x)
        if self.egress_guarded_linear_min is None:
            self.egress_guarded_linear_min = linear_x
            self.egress_guarded_linear_max = linear_x
        else:
            self.egress_guarded_linear_min = min(
                self.egress_guarded_linear_min, linear_x
            )
            self.egress_guarded_linear_max = max(
                self.egress_guarded_linear_max, linear_x
            )
        self.egress_guarded_lateral_max = max(
            self.egress_guarded_lateral_max, abs(float(message.linear.y))
        )
        self.egress_guarded_angular_max = max(
            self.egress_guarded_angular_max, abs(float(message.angular.z))
        )

    def _on_egress_feedback(self, message) -> None:
        value = float(message.feedback.distance_traveled)
        if math.isfinite(value) and value >= 0.0:
            self.egress_feedback_samples += 1
            self.egress_feedback_distance_m = value
            return
        self.egress_runtime_failure = {
            "status": "FAIL",
            "failures": [
                "safe egress received invalid DriveOnHeading feedback"
            ],
            "distance_traveled_m": repr(value),
        }
        self._record("safe_egress_feedback_invalid", self.egress_runtime_failure)

    def _check_static_map_localization(
        self,
        search_override: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        if self.latest_localization_scan is None or not self.amcl_samples:
            raise RuntimeError("scan or AMCL pose is unavailable for static-map gate")
        from rvclaw_nav2_bringup.scan_map_consistency import (
            compare_candidate_to_search,
            evaluate_localization_gate,
            evaluate_pose_ray_consistency,
            filter_ray_inconsistent_matches,
            load_static_map,
            scan_endpoints,
            score_pose,
            search_global_matches_coarse_to_fine,
        )

        map_file = os.environ.get(
            "RVCLAW_MAP_FILE",
            "/data/rvclaw-k3-lab/maps/slam-20260827-105711/map.yaml",
        )
        pose = self.amcl_samples[-1]
        scan = self.latest_localization_scan
        grid = load_static_map(map_file)
        endpoints = scan_endpoints(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            maximum_points=120,
        )
        candidate = score_pose(grid, endpoints, pose["x"], pose["y"], pose["yaw"])
        ray_endpoints = scan_endpoints(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max, maximum_points=480,
        )
        search = search_override or search_global_matches_coarse_to_fine(
            grid,
            endpoints,
            coarse_position_step_m=STATIC_MATCH_COARSE_POSITION_STEP_M,
            coarse_yaw_step_deg=STATIC_MATCH_COARSE_YAW_STEP_DEG,
            refine_position_radius_m=STATIC_MATCH_REFINE_POSITION_RADIUS_M,
            refine_yaw_radius_deg=STATIC_MATCH_REFINE_YAW_RADIUS_DEG,
            refine_position_step_m=STATIC_MATCH_REFINE_POSITION_STEP_M,
            refine_yaw_step_deg=STATIC_MATCH_REFINE_YAW_STEP_DEG,
            maximum_seeds=24, maximum_matches=24, ray_endpoints=ray_endpoints,
        )
        search = filter_ray_inconsistent_matches(grid, ray_endpoints, search)
        comparison = compare_candidate_to_search(candidate, search)
        gate = evaluate_localization_gate(candidate, comparison)
        ray_gate = evaluate_pose_ray_consistency(grid, ray_endpoints, pose)
        if ray_gate["status"] != "PASS":
            gate = dict(gate, status="FAIL", failures=gate["failures"] + ray_gate["failures"])
        return {
            "status": gate["status"],
            "failures": gate["failures"],
            "candidate": candidate,
            "comparison": comparison,
            "ray_consistency": ray_gate,
            "best_matches": search.get("matches", [])[:3],
            "gate": gate,
            "map_file": map_file,
        }

    def _run_static_match(self) -> Dict[str, object]:
        """Require agreement before accepting a pose or considering motion fallback."""
        from static_match_consensus import evaluate_static_match_consensus

        attempts = []
        for attempt in range(STATIC_MATCH_MAXIMUM_ATTEMPTS):
            self._refresh_match_scan("independent static match %d" % (attempt + 1))
            summary = self._run_static_match_once()
            attempts.append(summary)
            consensus = evaluate_static_match_consensus(attempts)
            self._record("static_match_consensus", consensus)
            if consensus["status"] == "PASS":
                return dict(summary, consensus=consensus, attempts=attempts)
        return dict(summary, status="FAIL", failures=consensus["failures"],
                    consensus=consensus, attempts=attempts)

    def _run_static_match_once(self) -> Dict[str, object]:
        """Search the reviewed map using one stationary scan, without motion."""
        if self.latest_localization_scan is None:
            raise RuntimeError("stationary scan is unavailable for global matching")
        from rvclaw_nav2_bringup.scan_map_consistency import (
            evaluate_static_match_gate,
            filter_ray_inconsistent_matches,
            load_static_map,
            scan_endpoints,
            search_global_matches_coarse_to_fine,
        )

        map_file = os.environ.get(
            "RVCLAW_MAP_FILE",
            "/data/rvclaw-k3-lab/maps/slam-20260827-105711/map.yaml",
        )
        scan = self.latest_localization_scan
        scan_stamp_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
        grid = load_static_map(map_file)
        endpoints = scan_endpoints(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            maximum_points=STATIC_MATCH_MAXIMUM_POINTS,
        )
        ray_endpoints = scan_endpoints(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max, maximum_points=480,
        )
        search = search_global_matches_coarse_to_fine(
            grid,
            endpoints,
            coarse_position_step_m=STATIC_MATCH_COARSE_POSITION_STEP_M,
            coarse_yaw_step_deg=STATIC_MATCH_COARSE_YAW_STEP_DEG,
            refine_position_radius_m=STATIC_MATCH_REFINE_POSITION_RADIUS_M,
            refine_yaw_radius_deg=STATIC_MATCH_REFINE_YAW_RADIUS_DEG,
            refine_position_step_m=STATIC_MATCH_REFINE_POSITION_STEP_M,
            refine_yaw_step_deg=STATIC_MATCH_REFINE_YAW_STEP_DEG,
            maximum_seeds=24, maximum_matches=24, ray_endpoints=ray_endpoints,
        )
        search = filter_ray_inconsistent_matches(grid, ray_endpoints, search)
        gate = evaluate_static_match_gate(
            search,
            minimum_best_score=STATIC_MATCH_MINIMUM_SCORE,
            minimum_known_endpoint_ratio=STATIC_MATCH_MINIMUM_KNOWN_RATIO,
            minimum_score_margin=STATIC_MATCH_MINIMUM_SCORE_MARGIN,
            maximum_second_to_best_ratio=STATIC_MATCH_MAXIMUM_SECOND_RATIO,
        )
        summary = {
            "status": gate["status"],
            "failures": gate["failures"],
            "scan_stamp_ns": scan_stamp_ns,
            "map_file": map_file,
            "endpoint_count": len(endpoints),
            "search_mode": search.get("search_mode"),
            "evaluated_position_count": search.get(
                "coarse_evaluated_position_count",
                search.get("evaluated_position_count"),
            ),
            "evaluated_pose_count": search.get("evaluated_pose_count"),
            "best_matches": search.get("matches", [])[:3],
            "ray_consistency_required": True,
            "ray_rejected_matches": search.get("ray_rejected_matches", []),
            "gate": gate,
        }
        self.static_match_search = search
        self._record("static_match_gate", summary)
        return summary

    def _set_initial_pose_from_static_match(self, summary: Dict[str, object]) -> None:
        best = summary.get("gate", {}).get("best")
        if not isinstance(best, dict):
            raise RuntimeError("static match did not provide an initial pose")
        if not self.initial_pose_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError("AMCL set_initial_pose service is unavailable")
        yaw = float(best["yaw"])
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.pose.position.x = float(best["x"])
        pose.pose.pose.position.y = float(best["y"])
        pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
        pose.pose.covariance[0] = 0.20 ** 2
        pose.pose.covariance[7] = 0.20 ** 2
        pose.pose.covariance[35] = math.radians(15.0) ** 2
        request = SetInitialPose.Request()
        request.pose = pose
        future = self.initial_pose_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=SERVICE_TIMEOUT_S,
        )
        if not future.done() or future.exception() is not None:
            raise RuntimeError("AMCL set_initial_pose service failed")
        self.static_match_used = True
        self._record(
            "static_match_initial_pose_set",
            {
                "x": float(best["x"]),
                "y": float(best["y"]),
                "yaw_deg": float(best.get("yaw_deg", math.degrees(yaw))),
            },
        )

    def _check_rotation_clearance(self) -> Dict[str, object]:
        """Reject a spin fallback when the measured swept envelope is occupied."""
        self._refresh_match_scan("rotation clearance after global matching")
        scan = self.latest_match_scan
        if scan is None:
            return {
                "status": "FAIL",
                "failures": ["stationary scan is unavailable for spin clearance"],
            }
        from rvclaw_nav2_bringup.scan_trial_policy import (
            evaluate_rotation_clearance,
        )

        return evaluate_rotation_clearance(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            minimum_radial_m=ROTATION_CLEARANCE_MIN_M,
            minimum_coverage_deg=ROTATION_CLEARANCE_MIN_COVERAGE_DEG,
        )

    def _evaluate_safe_egress(self) -> Dict[str, object]:
        """Evaluate a short forward escape using live scan and map evidence."""
        self._refresh_match_scan("safe egress selection")
        if self.latest_match_scan is None:
            raise RuntimeError("stationary scan is unavailable for safe egress")
        if not self.static_match_search:
            raise RuntimeError("static search is unavailable for safe egress")
        from rvclaw_nav2_bringup.scan_map_consistency import load_static_map

        map_file = os.environ.get(
            "RVCLAW_MAP_FILE",
            "/data/rvclaw-k3-lab/maps/slam-20260827-105711/map.yaml",
        )
        scan = self.latest_match_scan
        summary = select_safe_egress(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            scan.range_min,
            scan.range_max,
            load_static_map(map_file),
            self.static_match_search.get("matches", []),
        )
        if summary["status"] == "PASS":
            self._refresh_match_scan("safe egress after map corridor computation")
            scan = self.latest_match_scan
            summary["scan_gate"] = evaluate_front_egress_scan(
                scan.ranges, scan.angle_min, scan.angle_increment,
                scan.range_min, scan.range_max,
                distance_m=float(summary["selected_distance_m"]),
            )
            runtime_gate = evaluate_runtime_egress_monitor(
                summary["scan_gate"],
                self._match_scan_age_s(),
            )
            if runtime_gate["status"] != "PASS":
                summary = dict(summary)
                summary["status"] = "FAIL"
                summary["decision"] = "REQUIRE_OPERATOR_REPOSITION"
                summary["failures"] = list(runtime_gate["failures"])
        summary["map_file"] = map_file
        summary["prearm_scan_age_s"] = self._match_scan_age_s()
        summary["runtime_monitor"] = {
            "status": "NOT_STARTED",
            "scan_checks": 0,
            "minimum_scan_checks": EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
            "maximum_scan_age_s": EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
            "cancelled": False,
        }
        self._record("safe_egress_gate", summary)
        return summary

    def _runtime_egress_monitor(self) -> Dict[str, object]:
        if self.egress_runtime_failure is not None:
            return self.egress_runtime_failure
        scan_age_s = self._match_scan_age_s()
        scan_gate = self.egress_runtime_latest_gate or {
            "status": "FAIL",
            "failures": ["runtime forward scan has not been evaluated"],
        }
        monitor = evaluate_runtime_egress_monitor(scan_gate, scan_age_s)
        if monitor["status"] != "PASS":
            self.egress_runtime_failure = monitor
            self._record("safe_egress_runtime_scan_failure", monitor)
        return monitor

    def _store_runtime_egress_summary(
        self,
        monitor: Dict[str, object],
        cancelled: bool,
    ) -> None:
        if self.safe_egress_summary is None:
            return
        self.safe_egress_summary["runtime_monitor"] = {
            **monitor,
            "scan_checks": self.egress_runtime_scan_checks,
            "minimum_scan_checks": EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
            "cancelled": cancelled,
        }

    def _abort_safe_egress(self, goal_handle, monitor: Dict[str, object]) -> None:
        """Block output first, then cancel the Nav2 action and disarm."""
        self.egress_active = False
        self.egress_runtime_cancelled = True
        self._store_runtime_egress_summary(monitor, cancelled=True)
        self.publish_estop(True)
        self.publish_deadman(False)
        self.publish_zero_command()
        cancel_future = None
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except BaseException as exc:
            self._record("safe_egress_cancel_error", str(exc))
        for _ in range(6):
            self.publish_estop(True)
            self.publish_deadman(False)
            self.publish_zero_command()
            rclpy.spin_once(self, timeout_sec=0.05)
        try:
            self._call_arm(False)
        except BaseException as exc:
            self._record("safe_egress_disarm_error", str(exc))
        if cancel_future is not None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not cancel_future.done():
                self.publish_estop(True)
                self.publish_deadman(False)
                self.publish_zero_command()
                rclpy.spin_once(self, timeout_sec=0.05)
        self._record("safe_egress_runtime_cancelled", monitor)

    def _wait_egress_result_with_runtime_guard(self, goal_handle, result_future):
        deadline = time.monotonic() + EGRESS_RESULT_TIMEOUT_S
        while time.monotonic() < deadline:
            self.publish_deadman(True)
            rclpy.spin_once(self, timeout_sec=0.05)
            monitor = self._runtime_egress_monitor()
            if monitor["status"] != "PASS":
                self._abort_safe_egress(goal_handle, monitor)
                raise RuntimeError(
                    "safe egress runtime scan guard stopped motion: %s"
                    % "; ".join(monitor["failures"])
                )
            if result_future.done():
                return result_future.result()
        monitor = {
            "status": "FAIL",
            "failures": ["timed out waiting for DriveOnHeading result"],
            "scan_age_s": (
                math.inf
                if self.latest_match_scan_monotonic is None
                else time.monotonic() - self.latest_match_scan_monotonic
            ),
            "maximum_scan_age_s": EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
            "scan_gate": self.egress_runtime_latest_gate,
        }
        self._abort_safe_egress(goal_handle, monitor)
        raise RuntimeError("timed out waiting for DriveOnHeading result")

    def _run_safe_egress(self) -> Dict[str, object]:
        """Run one bounded positive DriveOnHeading, then stop explicitly."""
        if self.safe_egress_summary is None:
            raise RuntimeError("safe egress gate was not evaluated")
        if self.safe_egress_summary["status"] != "PASS":
            raise RuntimeError(
                "safe egress was refused: %s"
                % "; ".join(self.safe_egress_summary["failures"])
            )
        egress_distance_m = float(
            self.safe_egress_summary["selected_distance_m"]
        )
        if not self.egress_action.wait_for_server(timeout_sec=STARTUP_TIMEOUT_S):
            raise RuntimeError("Nav2 DriveOnHeading action server is unavailable")
        self.egress_feedback_samples = 0
        self.egress_feedback_distance_m = 0.0
        self.egress_action_status = None
        self.egress_guarded_linear_min = None
        self.egress_guarded_linear_max = None
        self.egress_guarded_lateral_max = 0.0
        self.egress_guarded_angular_max = 0.0
        self.egress_runtime_scan_checks = 0
        self.egress_runtime_latest_gate = self.safe_egress_summary["scan_gate"]
        self.egress_runtime_failure = None
        self.egress_runtime_cancelled = False

        self.publish_estop(False)
        self._call_reset()
        self.wait_for(
            lambda: self.state == "DISARMED",
            SERVICE_TIMEOUT_S,
            "DISARMED state before safe egress",
        )
        deadline = time.monotonic() + 0.30
        while time.monotonic() < deadline:
            self.publish_deadman(True)
            rclpy.spin_once(self, timeout_sec=0.05)
        self._refresh_match_scan("safe egress before arming")
        scan = self.latest_match_scan
        self.egress_runtime_latest_gate = evaluate_front_egress_scan(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max, distance_m=egress_distance_m,
        )
        prearm_monitor = self._runtime_egress_monitor()
        self._record("safe_egress_fresh_prearm_gate", prearm_monitor)
        if prearm_monitor["status"] != "PASS":
            raise RuntimeError("fresh safe egress prearm scan refused motion: %s"
                               % "; ".join(prearm_monitor["failures"]))
        self.publish_deadman(True)
        self._call_arm(True)
        self.wait_for(lambda: self.armed is True, SERVICE_TIMEOUT_S, "armed state")

        goal = DriveOnHeading.Goal()
        goal.target.x = egress_distance_m
        goal.target.y = 0.0
        goal.target.z = 0.0
        goal.speed = float(EGRESS_SPEED_M_S)
        goal.time_allowance.sec = int(EGRESS_ACTION_TIME_ALLOWANCE_S)
        goal.time_allowance.nanosec = 0
        self.egress_active = True
        try:
            send_future = self.egress_action.send_goal_async(
                goal,
                feedback_callback=self._on_egress_feedback,
            )
            goal_handle = self._wait_future_with_deadman(
                send_future,
                SERVICE_TIMEOUT_S,
                "DriveOnHeading goal acceptance",
            )
            if goal_handle is None or not goal_handle.accepted:
                raise RuntimeError("Nav2 DriveOnHeading goal was rejected")
            result_future = goal_handle.get_result_async()
            result_response = self._wait_egress_result_with_runtime_guard(
                goal_handle,
                result_future,
            )
            self.egress_action_status = int(result_response.status)
            for _ in range(6):
                self.publish_deadman(True)
                self.publish_zero_command()
                rclpy.spin_once(self, timeout_sec=0.05)
            self._call_arm(False)
            self.publish_deadman(False)
            self.wait_for(
                lambda: self.armed is False and self.state == "DISARMED",
                SERVICE_TIMEOUT_S,
                "explicit disarm after safe egress",
            )
            self._wait_for_stable_feedback_stop()
        finally:
            self.egress_active = False

        if self.egress_action_status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError("Nav2 DriveOnHeading did not report success")
        monitor = self._runtime_egress_monitor()
        if self.egress_runtime_scan_checks < EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS:
            monitor = {
                **monitor,
                "status": "FAIL",
                "failures": list(monitor.get("failures", []))
                + [
                    "safe egress received too few runtime scans (%d < %d)"
                    % (
                        self.egress_runtime_scan_checks,
                        EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
                    )
                ],
            }
            self._store_runtime_egress_summary(monitor, cancelled=False)
            raise RuntimeError(monitor["failures"][-1])
        self._store_runtime_egress_summary(monitor, cancelled=False)
        if self.egress_feedback_samples < 2:
            raise RuntimeError("DriveOnHeading emitted too few feedback samples")
        if not (
            max(0.05, egress_distance_m - 0.05)
            <= self.egress_feedback_distance_m
            <= egress_distance_m + 0.10
        ):
            raise RuntimeError(
                "safe egress distance feedback is outside bounds (%.3f m)"
                % self.egress_feedback_distance_m
            )
        if self.egress_guarded_linear_min is None:
            raise RuntimeError("safe egress emitted no guarded linear samples")
        if self.egress_guarded_linear_min < -1e-6:
            raise RuntimeError("safe egress contained reverse guarded output")
        if self.egress_guarded_linear_max > EGRESS_MAX_GUARDED_SPEED_M_S:
            raise RuntimeError("safe egress exceeded the linear speed ceiling")
        if self.egress_guarded_lateral_max > 1e-6:
            raise RuntimeError("safe egress contained lateral guarded output")
        if self.egress_guarded_angular_max > 1e-6:
            raise RuntimeError("safe egress contained angular guarded output")
        result = {
            "status": "PASS",
            "action_name": "/drive_on_heading",
            "action_status": self.egress_action_status,
            "feedback_samples": self.egress_feedback_samples,
            "feedback_distance_m": self.egress_feedback_distance_m,
            "guarded_linear_min_m_s": self.egress_guarded_linear_min,
            "guarded_linear_max_m_s": self.egress_guarded_linear_max,
            "guarded_lateral_max_m_s": self.egress_guarded_lateral_max,
            "guarded_angular_max_rad_s": self.egress_guarded_angular_max,
            "distance_m": egress_distance_m,
            "speed_m_s": EGRESS_SPEED_M_S,
            "requires_relocalization": True,
            "runtime_scan_checks": self.egress_runtime_scan_checks,
            "runtime_scan_monitor": monitor,
        }
        self._record("safe_egress_completed", result)
        return result

    def _finish_static_match_localization(self) -> Dict[str, object]:
        """Inject and independently validate a static-match pose."""
        self._set_initial_pose_from_static_match(self.static_match_summary)
        self.localization_summary = self._observe_convergence()
        if self.localization_summary["status"] != "PASS":
            raise RuntimeError("AMCL static-match localization did not converge")
        self.static_map_localization_gate = self._check_static_map_localization(
            search_override=self.static_match_search,
        )
        if self.static_map_localization_gate["status"] != "PASS":
            raise RuntimeError(
                "AMCL static-match localization failed independent map validation: %s"
                % "; ".join(self.static_map_localization_gate["failures"])
            )
        return self.global_result("PASS", None)

    def _on_amcl_pose(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        covariance = message.pose.covariance
        try:
            yaw = quaternion_to_yaw(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        except ValueError:
            yaw = math.nan
        self.amcl_samples.append(
            {
                "stamp_s": float(message.header.stamp.sec)
                + float(message.header.stamp.nanosec) / 1e9,
                "frame_id": message.header.frame_id,
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "yaw": yaw,
                "cov_x": float(covariance[0]),
                "cov_y": float(covariance[7]),
                "cov_yaw": float(covariance[35]),
            }
        )

    def _call_empty_service(self, client, name: str) -> None:
        if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError("%s service is unavailable" % name)
        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=SERVICE_TIMEOUT_S,
        )
        if not future.done() or future.result() is None or future.exception() is not None:
            raise RuntimeError("%s service failed" % name)

    def _wait_for_runtime(self) -> None:
        self.wait_for(
            lambda: self.transport_attached is True,
            STARTUP_TIMEOUT_S,
            "hardware transport attachment",
        )
        self.wait_for(
            lambda: (
                self.feedback_connected is True
                and self.stm32_connected is True
                and self.voltage_v is not None
                and self.voltage_v >= MIN_BATTERY_VOLTAGE_V
                and self.scan_samples >= 5
                and self.raw_odom_samples >= 5
                and self.latest_raw_yaw is not None
            ),
            STARTUP_TIMEOUT_S,
            "fresh hardware feedback, scan, and odometry",
        )
        if self.armed:
            raise RuntimeError("hardware controller unexpectedly started armed")
        # A stationary scan match is what establishes map->odom. Requiring a
        # map-frame costmap or Spin server before that pose exists creates a
        # startup cycle: Nav2 waits for map->odom while localization waits for
        # Nav2. The route wrapper checks every navigation lifecycle node after
        # this client has injected and independently validated the pose.
        if self.stationary_only:
            self._record("stationary_runtime_ready", True)
            return
        if not self.spin_action.wait_for_server(timeout_sec=STARTUP_TIMEOUT_S):
            raise RuntimeError("Nav2 Spin action server is unavailable")
        if not self.global_localization_client.wait_for_service(
            timeout_sec=SERVICE_TIMEOUT_S
        ):
            raise RuntimeError(
                "AMCL reinitialize_global_localization service is unavailable"
            )
        self._record("global_localization_service_ready", True)

    def _observe_convergence(self) -> Dict[str, object]:
        self.amcl_samples.clear()
        for _ in range(OBSERVATION_UPDATES):
            self._call_empty_service(
                self.nomotion_client,
                "AMCL request_nomotion_update",
            )
            deadline = time.monotonic() + OBSERVATION_INTERVAL_S
            while time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
        return summarize_global_pose_samples(
            self.amcl_samples,
            minimum_samples=5,
            maximum_position_drift_m=MAX_POSITION_DRIFT_M,
            maximum_yaw_drift_deg=MAX_YAW_DRIFT_DEG,
            maximum_position_variance=MAX_POSITION_VARIANCE,
            maximum_yaw_variance=MAX_YAW_VARIANCE,
        )

    def run_global_scan(self) -> Dict[str, object]:
        self._wait_for_runtime()
        self.static_match_summary = self._run_static_match()
        if self.static_match_summary["status"] == "PASS":
            return self._finish_static_match_localization()

        if self.stationary_only:
            raise RuntimeError(
                "stationary scan did not produce a unique map pose; "
                "motion fallback is disabled"
            )

        # A stationary match can be ambiguous in a repetitive or mostly empty
        # area. If an in-place sweep is blocked, a short, straight, supervised
        # egress may create clearance. It is admitted only from independent
        # live-scan and static-map gates, then localization must be repeated.
        self.rotation_clearance = self._check_rotation_clearance()
        if self.rotation_clearance["status"] != "PASS":
            self.safe_egress_summary = self._evaluate_safe_egress()
            if self.safe_egress_summary["status"] != "PASS":
                raise RuntimeError(
                    "stationary match and spin fallback were unsafe; safe egress was refused: %s"
                    % "; ".join(self.safe_egress_summary["failures"])
                )
            self._run_safe_egress()
            scan_count_before = self.scan_samples
            self.wait_for(
                lambda: self.scan_samples >= scan_count_before + 5,
                STARTUP_TIMEOUT_S,
                "fresh scan after safe egress",
            )
            self.static_match_summary = self._run_static_match()
            if self.static_match_summary["status"] == "PASS":
                return self._finish_static_match_localization()
            self.rotation_clearance = self._check_rotation_clearance()
            if self.rotation_clearance["status"] != "PASS":
                raise RuntimeError(
                    "safe egress did not produce a unique match and spin remains unsafe: %s"
                    % "; ".join(self.rotation_clearance["failures"])
                )
        self._call_empty_service(
            self.global_localization_client,
            "AMCL reinitialize_global_localization",
        )
        self._record("global_localization_requested", True)
        self.amcl_samples.clear()

        measured_total = 0.0
        for index in range(SEGMENT_COUNT):
            remaining = MAX_TOTAL_RAW_SCAN_YAW_RAD - measured_total
            if remaining <= SCAN_STOP_RESERVE_RAD:
                break
            self.target_yaw_rad = min(SEGMENT_YAW_RAD, remaining - SCAN_STOP_RESERVE_RAD)
            self.minimum_raw_yaw_delta_rad = max(0.05, self.target_yaw_rad - 0.15)
            self.maximum_raw_yaw_delta_rad = min(remaining, self.target_yaw_rad + 0.15)
            self.maximum_runtime_yaw_delta_rad = min(
                self.maximum_raw_yaw_delta_rad, remaining - SCAN_STOP_RESERVE_RAD,
            )
            self.raw_stop_target_rad = self.target_yaw_rad
            segment = super().run_heading()
            if segment["status"] != "PASS":
                raise RuntimeError("scan segment %d failed" % (index + 1))
            self.segment_results.append(
                {
                    "index": index + 1,
                    "status": segment["status"],
                    "target_yaw_rad": segment["target_yaw_rad"],
                    "action_elapsed_s": segment["action_elapsed_s"],
                    "action_feedback_final_rad": segment[
                        "action_feedback_final_rad"
                    ],
                    "raw_yaw_delta_rad": segment["raw_yaw_delta_rad"],
                    "start_raw_yaw_rad": segment["start_raw_yaw_rad"],
                    "action_result_raw_yaw_rad": segment["action_result_raw_yaw_rad"],
                    "final_raw_yaw_rad": segment["final_raw_yaw_rad"],
                    "post_stop_settle_s": segment["post_stop_settle_s"],
                    "final_armed": segment["final_armed"],
                }
            )
            measured_total += segment["raw_yaw_delta_rad"]
            self._record("measured_scan_rotation_budget", {
                "total_raw_yaw_rad": measured_total,
                "remaining_raw_yaw_rad": MAX_TOTAL_RAW_SCAN_YAW_RAD - measured_total,
            })
            if measured_total > MAX_TOTAL_RAW_SCAN_YAW_RAD:
                raise RuntimeError("global scan exceeded its measured total rotation budget")
            # Stop as soon as newly observed geometry uniquely identifies the
            # map pose. Never require a whole revolution after it is proven.
            self.static_match_summary = self._run_static_match()
            self._record("post_segment_static_match", {
                "segment": index + 1, "status": self.static_match_summary["status"],
            })
            if self.static_match_summary["status"] == "PASS":
                return self._finish_static_match_localization()

        if self.armed is not False or self.state != "DISARMED":
            raise RuntimeError("global scan did not finish disarmed")
        self.localization_summary = self._observe_convergence()
        if self.localization_summary["status"] != "PASS":
            raise RuntimeError("AMCL global localization did not converge")
        self.static_map_localization_gate = self._check_static_map_localization()
        if self.static_map_localization_gate["status"] != "PASS":
            raise RuntimeError(
                "AMCL global localization failed independent static-map validation: %s"
                % "; ".join(self.static_map_localization_gate["failures"])
            )
        return self.global_result("PASS", None)

    def global_result(
        self,
        status: str,
        error: Optional[str],
    ) -> Dict[str, object]:
        result = super().result(status, error)
        result.update(
            {
                "required_arm_token": (
                    STATIONARY_ONLY_ARM_TOKEN
                    if self.stationary_only
                    else ARM_TOKEN
                ),
                "stationary_only": self.stationary_only,
                "motion_fallback_allowed": not self.stationary_only,
                "localization_mode": (
                    "safe_egress_then_static_scan_match"
                    if self.safe_egress_summary is not None
                    and self.static_match_used
                    else "static_scan_match"
                    if self.static_match_used
                    else "amcl_global_with_controlled_scan"
                ),
                "segment_count_required": SEGMENT_COUNT,
                "segment_count_completed": len(self.segment_results),
                "segments": self.segment_results,
                "total_target_yaw_rad": SEGMENT_COUNT * SEGMENT_YAW_RAD,
                "amcl_sample_count": len(self.amcl_samples),
                "localization": self.localization_summary,
                "static_map_localization_gate": self.static_map_localization_gate,
                "static_match": self.static_match_summary,
                "static_match_used": self.static_match_used,
                "rotation_clearance": self.rotation_clearance,
                "safe_egress": self.safe_egress_summary,
                "navigation_goal_sent": False,
            }
        )
        return result


def main() -> int:
    args = _parse_args()
    if args.describe:
        print(
            json.dumps(
                describe_plan(stationary_only=args.stationary_only),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        trace_path = _validate_run_args(args)
    except ValueError as exc:
        print("ARGUMENT_ERROR: %s" % exc, file=sys.stderr)
        return 2

    rclpy.init()
    node = GlobalLocalizationScanClient(stationary_only=args.stationary_only)
    try:
        result = node.run_global_scan()
    except BaseException as exc:
        try:
            node.emergency_cleanup()
        except BaseException as cleanup_exc:
            node._record("cleanup_error", str(cleanup_exc))
        result = node.global_result("FAIL", str(exc))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    trace_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
