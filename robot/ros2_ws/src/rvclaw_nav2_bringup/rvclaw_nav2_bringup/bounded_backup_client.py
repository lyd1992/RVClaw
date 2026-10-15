"""Run one supervised, bounded Nav2 BackUp action through the safety gate."""

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import BackUp
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger

from rvclaw_nav2_bringup.bounded_backup_policy import ACTION_TIME_ALLOWANCE_S
from rvclaw_nav2_bringup.bounded_backup_policy import ARM_TOKEN
from rvclaw_nav2_bringup.bounded_backup_policy import BACKUP_DISTANCE_M
from rvclaw_nav2_bringup.bounded_backup_policy import BACKUP_SPEED_M_S
from rvclaw_nav2_bringup.bounded_backup_policy import CLIENT_RESULT_TIMEOUT_S
from rvclaw_nav2_bringup.bounded_backup_policy import CORRIDOR_SETTLE_S
from rvclaw_nav2_bringup.bounded_backup_policy import PREARM_COLLISION_PROBE_S
from rvclaw_nav2_bringup.bounded_backup_policy import SERVER_WAIT_TIMEOUT_S
from rvclaw_nav2_bringup.bounded_backup_policy import describe_plan
from rvclaw_nav2_bringup.bounded_backup_policy import require_arm_token


RATE_HZ = 20.0
INTERVAL_S = 1.0 / RATE_HZ
STARTUP_TIMEOUT_S = 8.0
SERVICE_TIMEOUT_S = 3.0
DEADMAN_PRIME_S = 0.30
MIN_BATTERY_VOLTAGE_V = 9.0
MINIMUM_SCAN_SAMPLES = 5
MAXIMUM_SCAN_AGE_S = 0.30
MAX_POST_STOP_LINEAR_M_S = 0.03
MAX_POST_STOP_ANGULAR_RAD_S = 0.10
POST_ACTION_ZERO_FRAMES = 6
POST_ACTION_STOP_TIMEOUT_S = 2.0
POST_ACTION_STABLE_SAMPLES = 3
MIN_BACKUP_DISPLACEMENT_M = 0.05
MAX_BACKUP_DISPLACEMENT_M = 0.24
MAX_BACKUP_LATERAL_M = 0.10
MAX_BACKUP_YAW_DELTA_RAD = 0.20
MAX_PREARM_PROBE_ODOM_DRIFT_M = 0.01

STATE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _quaternion_to_yaw(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _body_delta(first: Dict[str, float], second: Dict[str, float]):
    dx = second["x"] - first["x"]
    dy = second["y"] - first["y"]
    cosine = math.cos(first["yaw"])
    sine = math.sin(first["yaw"])
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


class BoundedBackupClient(Node):
    """Own the complete arm/action/stop sequence for one bounded backup."""

    def __init__(self):
        super().__init__("rvclaw_bounded_backup_client")
        self.client = ActionClient(self, BackUp, "/backup")
        self.deadman_publisher = self.create_publisher(
            Bool, "/rvclaw/control/deadman", COMMAND_QOS
        )
        self.estop_publisher = self.create_publisher(
            Bool, "/rvclaw/control/emergency_stop", COMMAND_QOS
        )
        # A zero command keeps the autonomy stream fresh while disarming.
        self.command_publisher = self.create_publisher(
            TwistStamped, "/rvclaw/cmd_vel/autonomy", COMMAND_QOS
        )
        self.reset_client = self.create_client(Trigger, "/rvclaw/control/reset_latches")
        self.arm_client = self.create_client(SetBool, "/rvclaw/control/set_armed")
        self.corridor_client = self.create_client(
            SetBool, "/rvclaw/backup_corridor/set_active"
        )
        self.transport_attached: Optional[bool] = None
        self.feedback_connected: Optional[bool] = None
        self.stm32_connected: Optional[bool] = None
        self.armed: Optional[bool] = None
        self.state: Optional[str] = None
        self.voltage_v: Optional[float] = None
        self.latest_scan_monotonic: Optional[float] = None
        self.scan_samples = 0
        self.feedback_samples = 0
        self.action_feedback_samples = 0
        self.action_feedback_distance_m: Optional[float] = None
        self.latest_feedback_x = 0.0
        self.latest_feedback_y = 0.0
        self.latest_feedback_angular = 0.0
        self.max_guarded_linear_x = 0.0
        self.max_guarded_angular_z = 0.0
        self.latest_pose: Optional[Dict[str, float]] = None
        self.start_pose: Optional[Dict[str, float]] = None
        self.final_pose: Optional[Dict[str, float]] = None
        self.goal_handle = None
        self.action_status: Optional[int] = None
        self.post_stop_stable_samples = 0
        self.corridor_active: Optional[bool] = None
        self.corridor_authorized = False
        self.prearm_collision_probe_passed = False
        self.nav2_reverse_samples = 0

        self.create_subscription(
            Bool,
            "/rvclaw/control/hardware_transport_attached",
            lambda message: setattr(self, "transport_attached", bool(message.data)),
            STATE_QOS,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/control/feedback_connected",
            lambda message: setattr(self, "feedback_connected", bool(message.data)),
            STATE_QOS,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/stm32/connected",
            lambda message: setattr(self, "stm32_connected", bool(message.data)),
            STATE_QOS,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/control/armed",
            lambda message: setattr(self, "armed", bool(message.data)),
            STATE_QOS,
        )
        self.create_subscription(
            String,
            "/rvclaw/control/state",
            lambda message: setattr(self, "state", message.data),
            STATE_QOS,
        )
        self.create_subscription(Float32, "/PowerVoltage", self._on_voltage, STATE_QOS)
        self.create_subscription(
            Bool,
            "/rvclaw/backup_corridor/active",
            lambda message: setattr(self, "corridor_active", bool(message.data)),
            STATE_QOS,
        )
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(
            Twist,
            "/rvclaw/control/guarded_cmd_vel",
            self._on_guarded,
            COMMAND_QOS,
        )
        self.create_subscription(
            Twist,
            "/rvclaw/nav2_cmd_vel",
            self._on_nav2_command,
            COMMAND_QOS,
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)

    def _on_voltage(self, message: Float32) -> None:
        self.voltage_v = float(message.data)

    def _on_scan(self, message: LaserScan) -> None:
        self.scan_samples += 1
        self.latest_scan_monotonic = time.monotonic()

    def _on_guarded(self, message: Twist) -> None:
        self.max_guarded_linear_x = max(self.max_guarded_linear_x, abs(float(message.linear.x)))
        self.max_guarded_angular_z = max(self.max_guarded_angular_z, abs(float(message.angular.z)))

    def _on_nav2_command(self, message: Twist) -> None:
        if float(message.linear.x) < 0.0:
            self.nav2_reverse_samples += 1

    def _on_odom(self, message) -> None:
        self.feedback_samples += 1
        twist = message.twist.twist
        self.latest_feedback_x = float(twist.linear.x)
        self.latest_feedback_y = float(twist.linear.y)
        self.latest_feedback_angular = float(twist.angular.z)
        pose = message.pose.pose
        self.latest_pose = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": _quaternion_to_yaw(pose.orientation),
        }

    def wait_for(self, predicate, timeout_s: float, description: str) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise RuntimeError("timed out waiting for %s" % description)

    def _call_reset(self) -> None:
        if not self.reset_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError("reset_latches service is unavailable")
        future = self.reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_S)
        if not future.done() or future.result() is None or not future.result().success:
            message = (
                "timeout"
                if not future.done() or future.result() is None
                else future.result().message
            )
            raise RuntimeError("reset_latches refused: %s" % message)

    def _call_arm(self, armed: bool) -> None:
        if not self.arm_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError("set_armed service is unavailable")
        request = SetBool.Request()
        request.data = armed
        future = self.arm_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_S)
        if not future.done() or future.result() is None or not future.result().success:
            message = (
                "timeout"
                if not future.done() or future.result() is None
                else future.result().message
            )
            raise RuntimeError("set_armed refused: %s" % message)

    def _call_corridor(self, active: bool) -> None:
        if not self.corridor_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError("backup corridor service is unavailable")
        request = SetBool.Request()
        request.data = active
        future = self.corridor_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_TIMEOUT_S)
        if not future.done() or future.result() is None or not future.result().success:
            message = (
                "timeout"
                if not future.done() or future.result() is None
                else future.result().message
            )
            raise RuntimeError("backup corridor refused: %s" % message)

    def publish_deadman(self, asserted: bool) -> None:
        message = Bool()
        message.data = asserted
        self.deadman_publisher.publish(message)

    def publish_estop(self, asserted: bool) -> None:
        message = Bool()
        message.data = asserted
        self.estop_publisher.publish(message)

    def publish_zero(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        self.command_publisher.publish(message)

    def _wait_future_with_deadman(self, future, timeout_s: float, description: str):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.corridor_active is not True:
                raise RuntimeError("backup corridor authorization was lost")
            self.publish_deadman(True)
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if future.done():
                return future.result()
        raise RuntimeError("timed out waiting for %s" % description)

    def _wait_future_with_corridor(self, future, timeout_s: float, description: str):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if self.corridor_active is not True:
                raise RuntimeError("backup corridor authorization was lost")
            if future.done():
                return future.result()
        raise RuntimeError("timed out waiting for %s" % description)

    @staticmethod
    def _goal():
        goal = BackUp.Goal()
        goal.target.x = -BACKUP_DISTANCE_M
        goal.speed = BACKUP_SPEED_M_S
        goal.time_allowance.sec = ACTION_TIME_ALLOWANCE_S
        return goal

    def _run_prearm_collision_probe(self) -> None:
        if self.armed is not False or self.state != "ESTOP_LATCHED":
            raise RuntimeError("prearm probe requires the startup estop latch")
        probe_start = dict(self.latest_pose)
        feedback_before = self.action_feedback_samples
        reverse_before = self.nav2_reverse_samples
        send_future = self.client.send_goal_async(
            self._goal(), feedback_callback=self._on_feedback
        )
        self.goal_handle = self._wait_future_with_corridor(
            send_future,
            SERVER_WAIT_TIMEOUT_S,
            "prearm BackUp goal acceptance",
        )
        if self.goal_handle is None or not self.goal_handle.accepted:
            raise RuntimeError("prearm BackUp goal was rejected")
        result_future = self.goal_handle.get_result_async()
        deadline = time.monotonic() + PREARM_COLLISION_PROBE_S
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if self.corridor_active is not True:
                raise RuntimeError("backup corridor expired during prearm probe")
            if result_future.done():
                wrapped = result_future.result()
                status = None if wrapped is None else int(wrapped.status)
                raise RuntimeError(
                    "prearm BackUp terminated before probe window; status=%s" % status
                )
        cancel_future = self.goal_handle.cancel_goal_async()
        self._wait_future_with_corridor(
            cancel_future,
            SERVER_WAIT_TIMEOUT_S,
            "prearm BackUp cancellation",
        )
        wrapped = self._wait_future_with_corridor(
            result_future,
            SERVER_WAIT_TIMEOUT_S,
            "prearm BackUp canceled result",
        )
        status = None if wrapped is None else int(wrapped.status)
        if status != GoalStatus.STATUS_CANCELED:
            raise RuntimeError("prearm BackUp did not finish canceled")
        if self.action_feedback_samples - feedback_before < 2:
            raise RuntimeError("prearm BackUp emitted too few feedback samples")
        if self.nav2_reverse_samples - reverse_before < 2:
            raise RuntimeError("prearm BackUp emitted too few reverse commands")
        if self.armed is not False or self.state != "ESTOP_LATCHED":
            raise RuntimeError("hardware safety state changed during prearm probe")
        probe_forward, probe_lateral = _body_delta(probe_start, self.latest_pose)
        if math.hypot(probe_forward, probe_lateral) > MAX_PREARM_PROBE_ODOM_DRIFT_M:
            raise RuntimeError("odometry moved during disarmed prearm probe")
        self.goal_handle = None
        self.action_feedback_samples = 0
        self.action_feedback_distance_m = None
        self.prearm_collision_probe_passed = True

    def _wait_for_stable_stop(self) -> None:
        deadline = time.monotonic() + POST_ACTION_STOP_TIMEOUT_S
        stable = 0
        previous_samples = self.feedback_samples
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if self.feedback_samples == previous_samples:
                continue
            previous_samples = self.feedback_samples
            stopped = (
                abs(self.latest_feedback_x) <= MAX_POST_STOP_LINEAR_M_S
                and abs(self.latest_feedback_y) <= MAX_POST_STOP_LINEAR_M_S
                and abs(self.latest_feedback_angular) <= MAX_POST_STOP_ANGULAR_RAD_S
            )
            stable = stable + 1 if stopped else 0
            if stable >= POST_ACTION_STABLE_SAMPLES:
                self.post_stop_stable_samples = stable
                return
        raise RuntimeError("timed out waiting for stable post-backup stop")

    def execute(self) -> Dict[str, object]:
        self.wait_for(
            lambda: self.transport_attached is True,
            STARTUP_TIMEOUT_S,
            "hardware transport attachment",
        )
        self.wait_for(
            lambda: self.feedback_connected is True
            and self.stm32_connected is True
            and self.voltage_v is not None
            and self.voltage_v >= MIN_BATTERY_VOLTAGE_V,
            STARTUP_TIMEOUT_S,
            "fresh feedback and safe voltage",
        )
        self.wait_for(
            lambda: self.scan_samples >= MINIMUM_SCAN_SAMPLES,
            STARTUP_TIMEOUT_S,
            "fresh LiDAR scans",
        )
        self.wait_for(
            lambda: self.latest_pose is not None and self.feedback_samples >= 5,
            STARTUP_TIMEOUT_S,
            "odometry",
        )
        if (
            self.latest_scan_monotonic is None
            or time.monotonic() - self.latest_scan_monotonic > MAXIMUM_SCAN_AGE_S
        ):
            raise RuntimeError("LiDAR scan is stale before arming")
        if self.armed:
            raise RuntimeError("hardware controller unexpectedly started armed")
        if not self.client.wait_for_server(timeout_sec=STARTUP_TIMEOUT_S):
            raise RuntimeError("/backup action server is unavailable")

        self._call_corridor(True)
        self.corridor_authorized = True
        self.wait_for(
            lambda: self.corridor_active is True,
            SERVICE_TIMEOUT_S,
            "active backup corridor",
        )
        settle_deadline = time.monotonic() + CORRIDOR_SETTLE_S
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
            if self.corridor_active is not True:
                raise RuntimeError("backup corridor expired while costmap was settling")

        self._run_prearm_collision_probe()

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
        self.wait_for(lambda: self.armed is True, SERVICE_TIMEOUT_S, "armed state")
        self.start_pose = dict(self.latest_pose)

        send_future = self.client.send_goal_async(
            self._goal(), feedback_callback=self._on_feedback
        )
        self.goal_handle = self._wait_future_with_deadman(
            send_future,
            SERVER_WAIT_TIMEOUT_S,
            "bounded BackUp goal acceptance",
        )
        if self.goal_handle is None or not self.goal_handle.accepted:
            raise RuntimeError("bounded BackUp goal was rejected")
        result_future = self.goal_handle.get_result_async()
        wrapped = self._wait_future_with_deadman(
            result_future,
            CLIENT_RESULT_TIMEOUT_S,
            "bounded BackUp result",
        )
        self.action_status = None if wrapped is None else int(wrapped.status)

        for _ in range(POST_ACTION_ZERO_FRAMES):
            self.publish_deadman(True)
            self.publish_zero()
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)
        self._call_arm(False)
        self.publish_deadman(False)
        self.wait_for(
            lambda: self.armed is False and self.state == "DISARMED",
            SERVICE_TIMEOUT_S,
            "explicit disarm",
        )
        self._call_corridor(False)
        self.wait_for(
            lambda: self.corridor_active is False,
            SERVICE_TIMEOUT_S,
            "inactive backup corridor after action",
        )
        self._wait_for_stable_stop()
        self.final_pose = dict(self.latest_pose)
        metrics = self._metrics()
        failures = []
        if self.action_status != GoalStatus.STATUS_SUCCEEDED:
            failures.append("bounded BackUp did not report success")
        if self.action_feedback_samples < 2:
            failures.append("bounded BackUp emitted too few feedback samples")
        if (
            metrics["raw_backward_m"] < MIN_BACKUP_DISPLACEMENT_M
            or metrics["raw_backward_m"] > MAX_BACKUP_DISPLACEMENT_M
        ):
            failures.append("raw odometry backward displacement is outside the bounded window")
        if abs(metrics["raw_lateral_m"]) > MAX_BACKUP_LATERAL_M:
            failures.append("raw odometry lateral drift exceeded the limit")
        if abs(metrics["raw_yaw_delta_rad"]) > MAX_BACKUP_YAW_DELTA_RAD:
            failures.append("raw odometry yaw change exceeded the limit")
        if self.armed is not False:
            failures.append("hardware controller did not finish disarmed")
        result = {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "action_status": self.action_status,
            "action_feedback_samples": self.action_feedback_samples,
            "action_feedback_distance_m": self.action_feedback_distance_m,
            "latest_voltage_v": self.voltage_v,
            "max_guarded_linear_x_m_s": self.max_guarded_linear_x,
            "max_guarded_angular_z_rad_s": self.max_guarded_angular_z,
            "post_stop_stable_samples": self.post_stop_stable_samples,
            "final_armed": self.armed,
            "corridor_authorized": self.corridor_authorized,
            "corridor_active_final": self.corridor_active,
            "prearm_collision_probe_passed": self.prearm_collision_probe_passed,
            **metrics,
        }
        if failures:
            raise RuntimeError("; ".join(failures))
        return result

    def _on_feedback(self, feedback_message) -> None:
        value = float(feedback_message.feedback.distance_traveled)
        if math.isfinite(value):
            self.action_feedback_samples += 1
            self.action_feedback_distance_m = value

    def _metrics(self) -> Dict[str, float]:
        if self.start_pose is None or self.final_pose is None:
            raise RuntimeError("backup odometry endpoints are unavailable")
        forward, lateral = _body_delta(self.start_pose, self.final_pose)
        return {
            "raw_backward_m": -forward,
            "raw_lateral_m": lateral,
            "raw_yaw_delta_rad": _normalize_angle(self.final_pose["yaw"] - self.start_pose["yaw"]),
        }

    def emergency_cleanup(self) -> None:
        self.publish_estop(True)
        self.publish_deadman(False)
        if self.goal_handle is not None:
            try:
                cancel_future = self.goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=1.0)
            except BaseException:
                pass
        try:
            self._call_arm(False)
        except BaseException:
            pass
        try:
            self._call_corridor(False)
        except BaseException:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect or execute the fixed supervised BackUp action",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--arm-token")
    parser.add_argument("--trace-file", type=Path)
    return parser.parse_args(argv)


def main(args=None):
    parsed = parse_args(args)
    if parsed.describe:
        print(json.dumps(describe_plan(), indent=2, sort_keys=True))
        return

    try:
        require_arm_token(parsed.arm_token)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc

    rclpy.init(args=None)
    node = BoundedBackupClient()
    result = None
    try:
        result = node.execute()
        print(json.dumps(result, indent=2, sort_keys=True))
        exit_code = 0
    except (KeyboardInterrupt, ExternalShutdownException):
        exit_code = 130
    except BaseException as exc:
        result = {"status": "FAIL", "error": str(exc)}
        print(json.dumps(result, indent=2), file=sys.stderr)
        try:
            node.emergency_cleanup()
        except BaseException:
            pass
        exit_code = 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if parsed.trace_file is not None and result is not None:
        parsed.trace_file.parent.mkdir(parents=True, exist_ok=True)
        parsed.trace_file.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
