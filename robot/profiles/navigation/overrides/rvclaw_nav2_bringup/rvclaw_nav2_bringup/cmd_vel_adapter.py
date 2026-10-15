"""Stamp Nav2 planar velocity samples for the RVClaw safety supervisor."""

import math
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


DEFAULT_SCAN_TIMEOUT_SEC = 0.30
DEFAULT_MAX_REVERSE_SPEED_M_S = 0.03
DEFAULT_MAX_FORWARD_SPEED_M_S = 0.20
DEFAULT_UPSTREAM_SILENCE_ZERO_SEC = 0.10
DEFAULT_ZERO_KEEPALIVE_PERIOD_SEC = 0.04
# Nav2 recovery actions can intentionally pause the upstream cmd_vel stream for
# about one second.  Keep publishing stamped zero during that bounded pause so
# the hardware command watchdog sees a safe command rather than a false loss of
# control.  The independent 0.15 s deadman still disarms immediately if the
# route owner disappears.
DEFAULT_ZERO_KEEPALIVE_MAX_SILENCE_SEC = 2.00


def planar_components(linear_x: float, angular_z: float) -> Optional[Tuple[float, float]]:
    """Return finite planar components or reject the entire sample."""
    if not math.isfinite(linear_x) or not math.isfinite(angular_z):
        return None
    return linear_x, angular_z


def scan_is_fresh(scan_age_sec: Optional[float], timeout_sec: float) -> bool:
    """Return whether a timestamped scan is recent enough for nonzero motion."""
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        return False
    if scan_age_sec is None or not math.isfinite(scan_age_sec):
        return False
    return 0.0 <= scan_age_sec <= timeout_sec


def limit_reverse_speed(linear_x: float, max_reverse_speed_m_s: float) -> float:
    """Clamp reverse velocity while leaving zero and forward commands unchanged."""
    if not math.isfinite(max_reverse_speed_m_s) or max_reverse_speed_m_s <= 0.0:
        raise ValueError("max_reverse_speed_m_s must be finite and positive")
    return max(linear_x, -max_reverse_speed_m_s)


def limit_forward_speed(linear_x: float, max_forward_speed_m_s: float) -> float:
    """Clamp positive velocity for a reviewed low-speed operating profile."""
    if not math.isfinite(max_forward_speed_m_s) or max_forward_speed_m_s <= 0.0:
        raise ValueError("max_forward_speed_m_s must be finite and positive")
    return min(linear_x, max_forward_speed_m_s)


def zero_keepalive_required(
    command_age_sec: Optional[float],
    start_after_sec: float,
    stop_after_sec: float,
) -> bool:
    """Return whether an upstream gap should be bridged with a stamped zero."""
    if command_age_sec is None or not math.isfinite(command_age_sec):
        return False
    if (
        not math.isfinite(start_after_sec)
        or not math.isfinite(stop_after_sec)
        or start_after_sec <= 0.0
        or stop_after_sec <= start_after_sec
    ):
        return False
    return start_after_sec <= command_age_sec <= stop_after_sec


class CmdVelAdapter(Node):
    """Convert unstamped Nav2 output into the supervisor's stamped contract."""

    def __init__(self) -> None:
        super().__init__("rvclaw_nav2_cmd_adapter")
        self.declare_parameter("input_topic", "rvclaw/nav2_cmd_vel")
        self.declare_parameter("output_topic", "rvclaw/cmd_vel/autonomy")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("scan_timeout_sec", DEFAULT_SCAN_TIMEOUT_SEC)
        self.declare_parameter(
            "max_reverse_speed_m_s",
            DEFAULT_MAX_REVERSE_SPEED_M_S,
        )
        self.declare_parameter(
            "max_forward_speed_m_s",
            DEFAULT_MAX_FORWARD_SPEED_M_S,
        )
        self.declare_parameter(
            "upstream_silence_zero_sec",
            DEFAULT_UPSTREAM_SILENCE_ZERO_SEC,
        )
        self.declare_parameter(
            "zero_keepalive_period_sec",
            DEFAULT_ZERO_KEEPALIVE_PERIOD_SEC,
        )
        self.declare_parameter(
            "zero_keepalive_max_silence_sec",
            DEFAULT_ZERO_KEEPALIVE_MAX_SILENCE_SEC,
        )

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.max_reverse_speed_m_s = float(
            self.get_parameter("max_reverse_speed_m_s").value
        )
        self.max_forward_speed_m_s = float(
            self.get_parameter("max_forward_speed_m_s").value
        )
        self.upstream_silence_zero_sec = float(
            self.get_parameter("upstream_silence_zero_sec").value
        )
        self.zero_keepalive_period_sec = float(
            self.get_parameter("zero_keepalive_period_sec").value
        )
        self.zero_keepalive_max_silence_sec = float(
            self.get_parameter("zero_keepalive_max_silence_sec").value
        )
        if not math.isfinite(self.scan_timeout_sec) or self.scan_timeout_sec <= 0.0:
            raise ValueError("scan_timeout_sec must be finite and positive")
        if (
            not math.isfinite(self.max_reverse_speed_m_s)
            or self.max_reverse_speed_m_s <= 0.0
        ):
            raise ValueError("max_reverse_speed_m_s must be finite and positive")
        if (
            not math.isfinite(self.max_forward_speed_m_s)
            or self.max_forward_speed_m_s <= 0.0
        ):
            raise ValueError("max_forward_speed_m_s must be finite and positive")
        if (
            not math.isfinite(self.upstream_silence_zero_sec)
            or self.upstream_silence_zero_sec <= 0.0
        ):
            raise ValueError("upstream_silence_zero_sec must be finite and positive")
        if (
            not math.isfinite(self.zero_keepalive_period_sec)
            or self.zero_keepalive_period_sec <= 0.0
            or self.zero_keepalive_period_sec >= self.upstream_silence_zero_sec
        ):
            raise ValueError(
                "zero_keepalive_period_sec must be positive and shorter than "
                "upstream_silence_zero_sec"
            )
        if (
            not math.isfinite(self.zero_keepalive_max_silence_sec)
            or self.zero_keepalive_max_silence_sec
            <= self.upstream_silence_zero_sec
        ):
            raise ValueError(
                "zero_keepalive_max_silence_sec must exceed "
                "upstream_silence_zero_sec"
            )

        self.publisher = self.create_publisher(
            TwistStamped, output_topic,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.subscription = self.create_subscription(
            Twist,
            input_topic,
            self._on_command,
            10,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            scan_topic,
            self._on_scan,
            10,
        )
        self.latest_scan_stamp_ns = None
        self.latest_input_monotonic = None
        self.scan_blocked = False
        self.zero_keepalive_active = False
        self.zero_keepalive_expired = False
        self.invalid_samples = 0
        self.zero_keepalive_timer = self.create_timer(
            self.zero_keepalive_period_sec,
            self._on_zero_keepalive_timer,
        )
        self.get_logger().info(
            "stamping guarded planar velocity commands: "
            f"{input_topic} -> {output_topic}; scan={scan_topic}; "
            f"scan_timeout={self.scan_timeout_sec:.2f}s; "
            f"max_forward={self.max_forward_speed_m_s:.2f}m/s; "
            f"max_reverse={self.max_reverse_speed_m_s:.2f}m/s; "
            f"zero_keepalive_after={self.upstream_silence_zero_sec:.2f}s; "
            f"zero_keepalive_until={self.zero_keepalive_max_silence_sec:.2f}s"
        )

    def _on_scan(self, message: LaserScan) -> None:
        self.latest_scan_stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _scan_age_sec(self) -> Optional[float]:
        if self.latest_scan_stamp_ns is None:
            return None
        return (self.get_clock().now().nanoseconds - self.latest_scan_stamp_ns) / 1e9

    def _publish_command(self, linear_x: float, angular_z: float) -> None:
        output = TwistStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.twist.linear.x = linear_x
        output.twist.angular.z = angular_z
        self.publisher.publish(output)

    def _on_zero_keepalive_timer(self) -> None:
        if self.latest_input_monotonic is None:
            return
        command_age_sec = time.monotonic() - self.latest_input_monotonic
        if zero_keepalive_required(
            command_age_sec,
            self.upstream_silence_zero_sec,
            self.zero_keepalive_max_silence_sec,
        ):
            if not self.zero_keepalive_active:
                self.get_logger().warning(
                    "Nav2 command stream paused; publishing bounded stamped zero "
                    f"after {command_age_sec:.3f}s"
                )
            self.zero_keepalive_active = True
            self.zero_keepalive_expired = False
            self._publish_command(0.0, 0.0)
            return

        if (
            command_age_sec > self.zero_keepalive_max_silence_sec
            and not self.zero_keepalive_expired
        ):
            self.get_logger().error(
                "Nav2 command stream remained silent; bounded zero keepalive "
                f"expired after {command_age_sec:.3f}s"
            )
            self.zero_keepalive_expired = True
        self.zero_keepalive_active = False

    def _on_command(self, message: Twist) -> None:
        now_monotonic = time.monotonic()
        components = planar_components(message.linear.x, message.angular.z)
        if components is None:
            self.invalid_samples += 1
            if self.invalid_samples == 1 or self.invalid_samples % 100 == 0:
                self.get_logger().error(
                    f"rejected non-finite Nav2 command; count={self.invalid_samples}"
                )
            return

        if self.zero_keepalive_active or self.zero_keepalive_expired:
            command_gap_sec = (
                None
                if self.latest_input_monotonic is None
                else now_monotonic - self.latest_input_monotonic
            )
            self.get_logger().info(
                "Nav2 command stream resumed"
                + (
                    ""
                    if command_gap_sec is None
                    else f" after {command_gap_sec:.3f}s"
                )
            )
        self.latest_input_monotonic = now_monotonic
        self.zero_keepalive_active = False
        self.zero_keepalive_expired = False

        linear_x = limit_forward_speed(
            limit_reverse_speed(components[0], self.max_reverse_speed_m_s),
            self.max_forward_speed_m_s,
        )
        angular_z = components[1]
        nonzero = linear_x != 0.0 or angular_z != 0.0
        scan_fresh = scan_is_fresh(self._scan_age_sec(), self.scan_timeout_sec)
        if nonzero and not scan_fresh:
            linear_x = 0.0
            angular_z = 0.0
            if not self.scan_blocked:
                self.get_logger().error(
                    "blocked nonzero Nav2 command because /scan is missing or stale"
                )
            self.scan_blocked = True
        elif nonzero and self.scan_blocked:
            self.get_logger().info("fresh /scan restored; Nav2 command gate reopened")
            self.scan_blocked = False

        self._publish_command(linear_x, angular_z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
