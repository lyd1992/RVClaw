#!/usr/bin/env python3
"""Run one bounded Nav2 Spin action through the RVClaw hardware safety path."""

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Dict, Optional

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import Spin
from nav2_msgs.msg import Costmap
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from stm32_control_client import (  # noqa: E402
    DEADMAN_PRIME_S,
    INTERVAL_S,
    MIN_BATTERY_VOLTAGE_V,
    SERVICE_TIMEOUT_S,
    STARTUP_TIMEOUT_S,
    ZeroSmokeClient,
)


ARM_TOKEN = "WHEELS_GROUND_PS2_OFF_NAV2_SPIN_LEFT_0350"
FIELD_SITE = "wheels_ground"
ACTION_NAME = "/spin"
TARGET_YAW_RAD = 0.35
ACTION_TIME_ALLOWANCE_S = 4
ACTION_RESULT_TIMEOUT_S = 6.0
MINIMUM_SCAN_SAMPLES = 5
MINIMUM_RAW_ODOM_SAMPLES = 5
MINIMUM_COSTMAP_SAMPLES = 1
MIN_GUARDED_ANGULAR_Z_RAD_S = 0.20
MAX_GUARDED_ANGULAR_Z_RAD_S = 0.60
MIN_RAW_YAW_DELTA_RAD = 0.20
MAX_RAW_YAW_DELTA_RAD = 0.60
MAX_POST_STOP_LINEAR_M_S = 0.03
MAX_POST_STOP_ANGULAR_RAD_S = 0.10
POST_ACTION_ZERO_FRAMES = 6
POST_ACTION_STOP_TIMEOUT_S = 2.0
POST_ACTION_STABLE_SAMPLES = 3
ZERO_EPSILON = 1e-9
COSTMAP_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
COSTMAP_TOPICS = ("/costmap/costmap_raw", "/global_costmap/costmap_raw")


def describe_plan() -> Dict[str, object]:
    return {
        "hardware_io": False,
        "field_site": FIELD_SITE,
        "required_arm_token": ARM_TOKEN,
        "action_name": ACTION_NAME,
        "target_yaw_rad": TARGET_YAW_RAD,
        "target_yaw_deg": math.degrees(TARGET_YAW_RAD),
        "action_time_allowance_s": ACTION_TIME_ALLOWANCE_S,
        "action_result_timeout_s": ACTION_RESULT_TIMEOUT_S,
        "collision_checks_required": True,
        "minimum_scan_samples": MINIMUM_SCAN_SAMPLES,
        "minimum_raw_odom_samples": MINIMUM_RAW_ODOM_SAMPLES,
        "minimum_costmap_samples": MINIMUM_COSTMAP_SAMPLES,
        "minimum_guarded_angular_z_rad_s": MIN_GUARDED_ANGULAR_Z_RAD_S,
        "maximum_guarded_angular_z_rad_s": MAX_GUARDED_ANGULAR_Z_RAD_S,
        "minimum_battery_voltage_v": MIN_BATTERY_VOLTAGE_V,
        "post_action_stop_timeout_s": POST_ACTION_STOP_TIMEOUT_S,
        "post_action_stable_samples": POST_ACTION_STABLE_SAMPLES,
        "direct_nonzero_velocity_publish": False,
        "launches_hardware_node": False,
        "transport_write_bypass": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--trace-file")
    parser.add_argument("--arm-token")
    return parser.parse_args()


def _validate_run_args(args: argparse.Namespace) -> Path:
    if args.arm_token != ARM_TOKEN:
        raise ValueError("exact Nav2 ground Spin arm token is required")
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


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_to_yaw(message: Odometry) -> float:
    orientation = message.pose.pose.orientation
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


class Nav2HeadingClient(ZeroSmokeClient):
    def __init__(
        self,
        target_yaw_rad: float = TARGET_YAW_RAD,
        action_time_allowance_s: int = ACTION_TIME_ALLOWANCE_S,
        action_result_timeout_s: float = ACTION_RESULT_TIMEOUT_S,
        minimum_raw_yaw_delta_rad: float = MIN_RAW_YAW_DELTA_RAD,
        maximum_raw_yaw_delta_rad: float = MAX_RAW_YAW_DELTA_RAD,
    ) -> None:
        super().__init__()
        self.target_yaw_rad = target_yaw_rad
        self.action_time_allowance_s = action_time_allowance_s
        self.action_result_timeout_s = action_result_timeout_s
        self.minimum_raw_yaw_delta_rad = minimum_raw_yaw_delta_rad
        self.maximum_raw_yaw_delta_rad = maximum_raw_yaw_delta_rad
        self.spin_action = ActionClient(self, Spin, ACTION_NAME)
        self.scan_samples = 0
        self.raw_odom_samples = 0
        self.costmap_samples = 0
        self.latest_raw_yaw: Optional[float] = None
        self.start_raw_yaw: Optional[float] = None
        self.action_result_raw_yaw: Optional[float] = None
        self.final_raw_yaw: Optional[float] = None
        self.guarded_angular_min: Optional[float] = None
        self.guarded_angular_max: Optional[float] = None
        self.latest_feedback_x = 0.0
        self.latest_feedback_y = 0.0
        self.latest_feedback_angular = 0.0
        self.feedback_samples = 0
        self.post_stop_stable_samples = 0
        self.post_stop_settle_s: Optional[float] = None
        self.action_feedback_samples = 0
        self.action_feedback_final_rad = 0.0
        self.action_feedback_max_abs_rad = 0.0
        self.action_status: Optional[int] = None
        self.action_elapsed_s: Optional[float] = None
        self.goal_handle = None
        self._deadman_keepalive_stop = None
        self._deadman_keepalive_thread = None

        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_raw_odometry,
            qos_profile_sensor_data,
        )
        # The standalone heading gate publishes /costmap/costmap_raw, while
        # full Nav2 may expose /global_costmap/costmap_raw. Subscribe to both
        # so readiness reflects the active launch profile.
        for topic in COSTMAP_TOPICS:
            self.create_subscription(
                Costmap,
                topic,
                self._on_costmap,
                COSTMAP_QOS,
            )

    def _reset_action_measurements(self) -> None:
        self.start_raw_yaw = None
        self.action_result_raw_yaw = None
        self.final_raw_yaw = None
        self.guarded_angular_min = None
        self.guarded_angular_max = None
        self.post_stop_stable_samples = 0
        self.post_stop_settle_s = None
        self.action_feedback_samples = 0
        self.action_feedback_final_rad = 0.0
        self.action_feedback_max_abs_rad = 0.0
        self.action_status = None
        self.action_elapsed_s = None
        self.goal_handle = None

    def _on_scan(self, _message: LaserScan) -> None:
        self.scan_samples += 1

    def _on_raw_odometry(self, message: Odometry) -> None:
        self.raw_odom_samples += 1
        self.latest_raw_yaw = quaternion_to_yaw(message)

    def _on_costmap(self, _message: Costmap) -> None:
        self.costmap_samples += 1

    def _on_guarded_command(self, message) -> None:
        super()._on_guarded_command(message)
        value = float(message.angular.z)
        if self.guarded_angular_min is None:
            self.guarded_angular_min = value
            self.guarded_angular_max = value
        else:
            self.guarded_angular_min = min(self.guarded_angular_min, value)
            self.guarded_angular_max = max(self.guarded_angular_max, value)

    def _on_odometry(self, message: Odometry) -> None:
        super()._on_odometry(message)
        twist = message.twist.twist
        self.feedback_samples += 1
        self.latest_feedback_x = float(twist.linear.x)
        self.latest_feedback_y = float(twist.linear.y)
        self.latest_feedback_angular = float(twist.angular.z)

    def _on_action_feedback(self, message) -> None:
        value = float(message.feedback.angular_distance_traveled)
        self.action_feedback_samples += 1
        self.action_feedback_final_rad = value
        self.action_feedback_max_abs_rad = max(
            self.action_feedback_max_abs_rad,
            abs(value),
        )

    def _wait_future_with_deadman(self, future, timeout_s: float, description: str):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.publish_deadman(True)
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if future.done():
                return future.result()
        raise RuntimeError("timed out waiting for %s" % description)

    def _start_deadman_keepalive(self) -> None:
        """Keep the short hardware deadman window alive during Nav2 waits."""
        if self._deadman_keepalive_thread is not None:
            return
        stop_event = threading.Event()
        self._deadman_keepalive_stop = stop_event

        def keepalive() -> None:
            while not stop_event.is_set():
                try:
                    self.publish_deadman(True)
                except BaseException:
                    break
                stop_event.wait(0.03)

        thread = threading.Thread(
            target=keepalive,
            name="rvclaw-heading-deadman-keepalive",
            daemon=True,
        )
        self._deadman_keepalive_thread = thread
        thread.start()

    def _stop_deadman_keepalive(self) -> None:
        stop_event = self._deadman_keepalive_stop
        thread = self._deadman_keepalive_thread
        self._deadman_keepalive_stop = None
        self._deadman_keepalive_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _cancel_goal(self) -> None:
        if self.goal_handle is None:
            return
        try:
            future = self.goal_handle.cancel_goal_async()
            self._wait_future_with_deadman(future, 2.0, "Spin cancellation")
            self._record("action_cancel_requested", True)
        except BaseException as exc:
            self._record("action_cancel_error", str(exc))

    def _feedback_is_stopped(self) -> bool:
        return (
            abs(self.latest_feedback_x) <= MAX_POST_STOP_LINEAR_M_S
            and abs(self.latest_feedback_y) <= MAX_POST_STOP_LINEAR_M_S
            and abs(self.latest_feedback_angular) <= MAX_POST_STOP_ANGULAR_RAD_S
        )

    def _wait_for_stable_feedback_stop(self) -> None:
        started = time.monotonic()
        deadline = started + POST_ACTION_STOP_TIMEOUT_S
        last_sample_count = self.feedback_samples
        stable_samples = 0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if self.feedback_samples == last_sample_count:
                continue
            last_sample_count = self.feedback_samples
            if self._feedback_is_stopped():
                stable_samples += 1
                if stable_samples >= POST_ACTION_STABLE_SAMPLES:
                    self.post_stop_stable_samples = stable_samples
                    self.post_stop_settle_s = time.monotonic() - started
                    self._record("post_stop_stable_samples", stable_samples)
                    self._record("post_stop_settle_s", self.post_stop_settle_s)
                    return
            else:
                stable_samples = 0
        raise RuntimeError("timed out waiting for stable post-Spin feedback stop")

    def run_heading(self) -> Dict[str, object]:
        self._reset_action_measurements()
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
            ),
            STARTUP_TIMEOUT_S,
            "fresh feedback and safe voltage",
        )
        self.wait_for(
            lambda: (
                self.scan_samples >= MINIMUM_SCAN_SAMPLES
                and self.raw_odom_samples >= MINIMUM_RAW_ODOM_SAMPLES
                and self.costmap_samples >= MINIMUM_COSTMAP_SAMPLES
                and self.latest_raw_yaw is not None
            ),
            STARTUP_TIMEOUT_S,
            "scan, raw odometry, and local costmap readiness",
        )
        if not self.spin_action.wait_for_server(timeout_sec=STARTUP_TIMEOUT_S):
            raise RuntimeError("Nav2 Spin action server is unavailable")
        if self.armed:
            raise RuntimeError("hardware controller unexpectedly started armed")

        self.publish_estop(False)
        self._call_reset()
        self.wait_for(
            lambda: self.state == "DISARMED",
            SERVICE_TIMEOUT_S,
            "DISARMED state after reset",
        )

        deadline = time.monotonic() + DEADMAN_PRIME_S
        while time.monotonic() < deadline:
            self.publish_deadman(True)
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)

        self._call_arm(True)
        self.wait_for(
            lambda: self.armed is True,
            SERVICE_TIMEOUT_S,
            "armed state",
        )

        self.start_raw_yaw = self.latest_raw_yaw
        goal = Spin.Goal()
        goal.target_yaw = self.target_yaw_rad
        goal.time_allowance.sec = self.action_time_allowance_s
        goal.time_allowance.nanosec = 0
        action_started = time.monotonic()
        send_future = self.spin_action.send_goal_async(
            goal,
            feedback_callback=self._on_action_feedback,
        )
        self.goal_handle = self._wait_future_with_deadman(
            send_future,
            SERVICE_TIMEOUT_S,
            "Spin goal acceptance",
        )
        if self.goal_handle is None or not self.goal_handle.accepted:
            raise RuntimeError("Nav2 Spin goal was rejected")
        self._record("action_goal_accepted", True)

        result_future = self.goal_handle.get_result_async()
        result_response = self._wait_future_with_deadman(
            result_future,
            self.action_result_timeout_s,
            "Spin result",
        )
        self.action_elapsed_s = time.monotonic() - action_started
        self.action_status = int(result_response.status)
        self._record("action_status", self.action_status)
        self.action_result_raw_yaw = self.latest_raw_yaw

        for _ in range(POST_ACTION_ZERO_FRAMES):
            self.publish_deadman(True)
            self.publish_zero_command()
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)

        self._call_arm(False)
        self.publish_deadman(False)
        self.wait_for(
            lambda: self.armed is False and self.state == "DISARMED",
            SERVICE_TIMEOUT_S,
            "explicit disarm",
        )
        self._wait_for_stable_feedback_stop()
        self.final_raw_yaw = self.latest_raw_yaw

        if self.action_status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError("Nav2 Spin did not report success")
        if self.action_feedback_samples < 2:
            raise RuntimeError("Nav2 Spin emitted too few feedback samples")
        if self.guarded_angular_max is None:
            raise RuntimeError("hardware node emitted no guarded Spin samples")
        if self.guarded_angular_max < MIN_GUARDED_ANGULAR_Z_RAD_S:
            raise RuntimeError("guarded Spin did not overcome the turn threshold")
        if self.guarded_angular_max > MAX_GUARDED_ANGULAR_Z_RAD_S + ZERO_EPSILON:
            raise RuntimeError("guarded Spin exceeded the angular ceiling")
        if self.guarded_angular_min is None or self.guarded_angular_min < -ZERO_EPSILON:
            raise RuntimeError("left Spin contained right-turn guarded output")
        if self.max_guarded_linear_x > ZERO_EPSILON:
            raise RuntimeError("Spin contained guarded linear X velocity")
        if self.max_guarded_linear_y > ZERO_EPSILON:
            raise RuntimeError("Spin contained guarded lateral velocity")

        yaw_delta = self.raw_yaw_delta()
        if yaw_delta is None or yaw_delta < self.minimum_raw_yaw_delta_rad:
            raise RuntimeError("raw heading did not reach the minimum left delta")
        if yaw_delta > self.maximum_raw_yaw_delta_rad:
            raise RuntimeError("raw heading exceeded the admitted left delta")
        if abs(self.latest_feedback_x) > MAX_POST_STOP_LINEAR_M_S:
            raise RuntimeError("post-Spin X feedback remained above the stop limit")
        if abs(self.latest_feedback_y) > MAX_POST_STOP_LINEAR_M_S:
            raise RuntimeError("post-Spin Y feedback remained above the stop limit")
        if abs(self.latest_feedback_angular) > MAX_POST_STOP_ANGULAR_RAD_S:
            raise RuntimeError("post-Spin angular feedback remained above the stop limit")

        return self.result("PASS", None)

    def raw_yaw_delta(self) -> Optional[float]:
        if self.start_raw_yaw is None or self.final_raw_yaw is None:
            return None
        return normalize_angle(self.final_raw_yaw - self.start_raw_yaw)

    def emergency_cleanup(self) -> None:
        self._stop_deadman_keepalive()
        self.publish_estop(True)
        self.publish_deadman(False)
        try:
            self._cancel_goal()
        finally:
            super().emergency_cleanup()

    def result(self, status: str, error: Optional[str]) -> Dict[str, object]:
        result = super().result(status, error)
        result.update(
            {
                "field_site": FIELD_SITE,
                "action_name": ACTION_NAME,
                "target_yaw_rad": self.target_yaw_rad,
                "target_yaw_deg": math.degrees(self.target_yaw_rad),
                "action_time_allowance_s": self.action_time_allowance_s,
                "action_elapsed_s": self.action_elapsed_s,
                "action_status": self.action_status,
                "action_feedback_samples": self.action_feedback_samples,
                "action_feedback_final_rad": self.action_feedback_final_rad,
                "action_feedback_max_abs_rad": self.action_feedback_max_abs_rad,
                "scan_samples": self.scan_samples,
                "raw_odom_samples": self.raw_odom_samples,
                "costmap_samples": self.costmap_samples,
                "start_raw_yaw_rad": self.start_raw_yaw,
                "action_result_raw_yaw_rad": self.action_result_raw_yaw,
                "final_raw_yaw_rad": self.final_raw_yaw,
                "raw_yaw_delta_rad": self.raw_yaw_delta(),
                "post_stop_timeout_s": POST_ACTION_STOP_TIMEOUT_S,
                "post_stop_required_stable_samples": POST_ACTION_STABLE_SAMPLES,
                "post_stop_stable_samples": self.post_stop_stable_samples,
                "post_stop_settle_s": self.post_stop_settle_s,
                "guarded_angular_min_rad_s": self.guarded_angular_min,
                "guarded_angular_max_rad_s": self.guarded_angular_max,
                "final_feedback_x_m_s": self.latest_feedback_x,
                "final_feedback_y_m_s": self.latest_feedback_y,
                "final_feedback_angular_rad_s": self.latest_feedback_angular,
            }
        )
        return result


def main() -> int:
    args = _parse_args()
    if args.describe:
        print(json.dumps(describe_plan(), indent=2, sort_keys=True))
        return 0
    try:
        trace_path = _validate_run_args(args)
    except ValueError as exc:
        print("ARGUMENT_ERROR: %s" % exc, file=sys.stderr)
        return 2

    rclpy.init()
    node = Nav2HeadingClient()
    try:
        result = node.run_heading()
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

    trace_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
