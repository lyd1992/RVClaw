#!/usr/bin/env python3
"""Exercise the single-owner ROS hardware path using zero commands only."""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Optional

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ARM_TOKEN = "WHEELS_UP_PS2_OFF_ROS_HARDWARE_ZERO_ONLY"
RATE_HZ = 20.0
INTERVAL_S = 1.0 / RATE_HZ
DEADMAN_PRIME_S = 0.30
ZERO_COMMAND_DURATION_S = 1.0
STARTUP_TIMEOUT_S = 8.0
SERVICE_TIMEOUT_S = 3.0
MIN_BATTERY_VOLTAGE_V = 9.0
MAX_ZERO_FEEDBACK_LINEAR_M_S = 0.03
MAX_ZERO_FEEDBACK_ANGULAR_RAD_S = 0.10
ZERO_EPSILON = 1e-9


def describe_plan() -> Dict[str, object]:
    return {
        "hardware_io": False,
        "required_arm_token": ARM_TOKEN,
        "rate_hz": RATE_HZ,
        "deadman_prime_s": DEADMAN_PRIME_S,
        "zero_command_duration_s": ZERO_COMMAND_DURATION_S,
        "command_velocity_x_m_s": 0.0,
        "command_velocity_y_m_s": 0.0,
        "command_angular_z_rad_s": 0.0,
        "minimum_battery_voltage_v": MIN_BATTERY_VOLTAGE_V,
        "launches_hardware_node": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--trace-file")
    parser.add_argument("--arm-token")
    return parser.parse_args()


def _validate_run_args(args: argparse.Namespace) -> Path:
    if args.arm_token != ARM_TOKEN:
        raise ValueError("exact wheels-up zero-only arm token is required")
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


class ZeroSmokeClient(Node):
    def __init__(self) -> None:
        super().__init__("rvclaw_hardware_zero_smoke_client")
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.started_monotonic = time.monotonic()
        self.events: List[Dict[str, object]] = []
        self.transport_attached: Optional[bool] = None
        self.feedback_connected: Optional[bool] = None
        self.stm32_connected: Optional[bool] = None
        self.armed: Optional[bool] = None
        self.state: Optional[str] = None
        self.transition_reason: Optional[str] = None
        self.voltage_v: Optional[float] = None
        self.max_guarded_linear_x = 0.0
        self.max_guarded_linear_y = 0.0
        self.max_guarded_angular_z = 0.0
        self.max_feedback_linear_x = 0.0
        self.max_feedback_linear_y = 0.0
        self.max_feedback_angular_z = 0.0

        self.deadman_publisher = self.create_publisher(
            Bool,
            "/rvclaw/control/deadman",
            command_qos,
        )
        self.estop_publisher = self.create_publisher(
            Bool,
            "/rvclaw/control/emergency_stop",
            command_qos,
        )
        self.command_publisher = self.create_publisher(
            TwistStamped,
            "/rvclaw/cmd_vel/autonomy",
            command_qos,
        )
        self.reset_client = self.create_client(
            Trigger,
            "/rvclaw/control/reset_latches",
        )
        self.arm_client = self.create_client(
            SetBool,
            "/rvclaw/control/set_armed",
        )

        self.create_subscription(
            Bool,
            "/rvclaw/control/hardware_transport_attached",
            lambda message: self._set_state("transport_attached", message.data),
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/control/feedback_connected",
            lambda message: self._set_state("feedback_connected", message.data),
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/stm32/connected",
            lambda message: self._set_state("stm32_connected", message.data),
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/control/armed",
            lambda message: self._set_state("armed", message.data),
            state_qos,
        )
        self.create_subscription(
            String,
            "/rvclaw/control/state",
            lambda message: self._set_state("state", message.data),
            state_qos,
        )
        self.create_subscription(
            String,
            "/rvclaw/control/transition_reason",
            lambda message: self._set_state("transition_reason", message.data),
            state_qos,
        )
        self.create_subscription(
            Float32,
            "/PowerVoltage",
            self._on_voltage,
            state_qos,
        )
        self.create_subscription(
            Twist,
            "/rvclaw/control/guarded_cmd_vel",
            self._on_guarded_command,
            command_qos,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odometry,
            sensor_qos,
        )

    def _record(self, name: str, value: object) -> None:
        self.events.append(
            {
                "elapsed_s": time.monotonic() - self.started_monotonic,
                "name": name,
                "value": value,
            }
        )

    def _set_state(self, name: str, value: object) -> None:
        if getattr(self, name) == value:
            return
        setattr(self, name, value)
        self._record(name, value)

    def _on_voltage(self, message: Float32) -> None:
        value = float(message.data)
        if self.voltage_v is None or abs(value - self.voltage_v) >= 0.001:
            self.voltage_v = value
            self._record("voltage_v", value)

    def _on_guarded_command(self, message: Twist) -> None:
        self.max_guarded_linear_x = max(
            self.max_guarded_linear_x,
            abs(message.linear.x),
        )
        self.max_guarded_linear_y = max(
            self.max_guarded_linear_y,
            abs(message.linear.y),
        )
        self.max_guarded_angular_z = max(
            self.max_guarded_angular_z,
            abs(message.angular.z),
        )

    def _on_odometry(self, message: Odometry) -> None:
        twist = message.twist.twist
        self.max_feedback_linear_x = max(
            self.max_feedback_linear_x,
            abs(twist.linear.x),
        )
        self.max_feedback_linear_y = max(
            self.max_feedback_linear_y,
            abs(twist.linear.y),
        )
        self.max_feedback_angular_z = max(
            self.max_feedback_angular_z,
            abs(twist.angular.z),
        )

    def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout_s: float,
        description: str,
    ) -> None:
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
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=SERVICE_TIMEOUT_S,
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("reset_latches service timed out")
        response = future.result()
        self._record("reset_response", response.message)
        if not response.success:
            raise RuntimeError("reset_latches refused: %s" % response.message)

    def _call_arm(self, armed: bool) -> None:
        if not self.arm_client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            raise RuntimeError("set_armed service is unavailable")
        request = SetBool.Request()
        request.data = armed
        future = self.arm_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=SERVICE_TIMEOUT_S,
        )
        if not future.done() or future.result() is None:
            raise RuntimeError("set_armed service timed out")
        response = future.result()
        self._record("arm_response", response.message)
        if not response.success:
            raise RuntimeError("set_armed refused: %s" % response.message)

    def publish_deadman(self, asserted: bool) -> None:
        message = Bool()
        message.data = asserted
        self.deadman_publisher.publish(message)

    def publish_estop(self, asserted: bool) -> None:
        message = Bool()
        message.data = asserted
        self.estop_publisher.publish(message)

    def publish_zero_command(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        self.command_publisher.publish(message)

    def run_zero_sequence(self) -> Dict[str, object]:
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

        zero_started = time.monotonic()
        while time.monotonic() - zero_started < ZERO_COMMAND_DURATION_S:
            self.publish_deadman(True)
            self.publish_zero_command()
            rclpy.spin_once(self, timeout_sec=INTERVAL_S)

        self.wait_for(
            lambda: self.state == "ACTIVE",
            SERVICE_TIMEOUT_S,
            "ACTIVE state from fresh zero commands",
        )
        self._call_arm(False)
        self.publish_deadman(False)
        self.wait_for(
            lambda: self.armed is False and self.state == "DISARMED",
            SERVICE_TIMEOUT_S,
            "explicit disarm",
        )

        guarded_maximum = max(
            self.max_guarded_linear_x,
            self.max_guarded_linear_y,
            self.max_guarded_angular_z,
        )
        if guarded_maximum > ZERO_EPSILON:
            raise RuntimeError("zero-only client observed a nonzero guarded command")
        if self.max_feedback_linear_x > MAX_ZERO_FEEDBACK_LINEAR_M_S:
            raise RuntimeError("zero-only linear X feedback exceeded the limit")
        if self.max_feedback_linear_y > MAX_ZERO_FEEDBACK_LINEAR_M_S:
            raise RuntimeError("zero-only linear Y feedback exceeded the limit")
        if self.max_feedback_angular_z > MAX_ZERO_FEEDBACK_ANGULAR_RAD_S:
            raise RuntimeError("zero-only angular feedback exceeded the limit")

        return self.result("PASS", None)

    def emergency_cleanup(self) -> None:
        self.publish_estop(True)
        self.publish_deadman(False)
        try:
            self._call_arm(False)
        except Exception as exc:
            self._record("cleanup_disarm_error", str(exc))
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.05)

    def result(self, status: str, error: Optional[str]) -> Dict[str, object]:
        return {
            "status": status,
            "error": error,
            "duration_s": time.monotonic() - self.started_monotonic,
            "transport_attached": self.transport_attached,
            "feedback_connected": self.feedback_connected,
            "stm32_connected": self.stm32_connected,
            "final_armed": self.armed,
            "final_state": self.state,
            "final_transition_reason": self.transition_reason,
            "latest_voltage_v": self.voltage_v,
            "max_guarded_linear_x_m_s": self.max_guarded_linear_x,
            "max_guarded_linear_y_m_s": self.max_guarded_linear_y,
            "max_guarded_angular_z_rad_s": self.max_guarded_angular_z,
            "max_feedback_linear_x_m_s": self.max_feedback_linear_x,
            "max_feedback_linear_y_m_s": self.max_feedback_linear_y,
            "max_feedback_angular_z_rad_s": self.max_feedback_angular_z,
            "events": self.events,
        }


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
    node = ZeroSmokeClient()
    result: Dict[str, object]
    try:
        result = node.run_zero_sequence()
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
