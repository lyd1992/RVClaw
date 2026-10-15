"""ROS 2 command safety supervisor with no hardware transport."""

import math
import time
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String, UInt64, UInt8MultiArray
from std_srvs.srv import SetBool, Trigger

from .command_gate import NANOSECONDS_PER_SECOND, StampedCommandGate
from .protocol import encode_velocity_command
from .safety import ControlDecision, SafetySupervisor


class SafetySupervisorNode(Node):
    """Publish guarded commands while leaving serial ownership to a future transport."""

    def __init__(
        self,
        node_name: str = "rvclaw_safety_supervisor",
        preview_only: bool = True,
        subscribe_feedback_connected: bool = True,
    ) -> None:
        super().__init__(node_name)

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("command_timeout_s", 0.25)
        self.declare_parameter("max_command_stamp_age_s", 0.10)
        self.declare_parameter("max_future_command_stamp_skew_s", 0.02)
        self.declare_parameter("deadman_timeout_s", 0.15)
        self.declare_parameter("max_linear_x_m_s", 0.18)
        self.declare_parameter("max_angular_z_rad_s", 0.30)
        self.declare_parameter("max_linear_accel_m_s2", 0.20)
        self.declare_parameter("max_angular_accel_rad_s2", 0.60)
        self.declare_parameter("require_feedback_connected", False)

        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        if not math.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be finite and positive")

        self.safety = SafetySupervisor(
            command_timeout_s=float(
                self.get_parameter("command_timeout_s").value
            ),
            deadman_timeout_s=float(
                self.get_parameter("deadman_timeout_s").value
            ),
            max_linear_x_m_s=float(
                self.get_parameter("max_linear_x_m_s").value
            ),
            max_angular_z_rad_s=float(
                self.get_parameter("max_angular_z_rad_s").value
            ),
            max_linear_accel_m_s2=float(
                self.get_parameter("max_linear_accel_m_s2").value
            ),
            max_angular_accel_rad_s2=float(
                self.get_parameter("max_angular_accel_rad_s2").value
            ),
            require_feedback_connected=bool(
                self.get_parameter("require_feedback_connected").value
            ),
        )
        self.command_gate = StampedCommandGate(
            max_age_s=float(
                self.get_parameter("max_command_stamp_age_s").value
            ),
            max_future_skew_s=float(
                self.get_parameter(
                    "max_future_command_stamp_skew_s"
                ).value
            ),
        )
        self.last_command_rejection = ""
        self.latched_command_fault_rejection = ""
        self.last_rejection_log_signature: Optional[tuple] = None
        self.last_rejection_log_monotonic = 0.0

        self.command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Velocity is replaceable state: retransmitted old samples must never
        # queue ahead of the newest command at an arm boundary. Loss remains
        # fail-closed through the independent timestamp and watchdog gates.
        self.autonomy_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            TwistStamped,
            "rvclaw/cmd_vel/teleop",
            lambda message: self._on_command("teleop", message),
            self.command_qos,
        )
        self.create_subscription(
            TwistStamped,
            "rvclaw/cmd_vel/autonomy",
            lambda message: self._on_command("autonomy", message),
            self.autonomy_qos,
        )
        self.create_subscription(
            Bool,
            "rvclaw/control/deadman",
            self._on_deadman,
            self.command_qos,
        )
        self.create_subscription(
            Bool,
            "rvclaw/control/emergency_stop",
            self._on_emergency_stop,
            self.command_qos,
        )
        if subscribe_feedback_connected:
            self.create_subscription(
                Bool,
                "rvclaw/stm32/connected",
                self._on_feedback_connected,
                self.state_qos,
            )

        self.create_service(
            SetBool,
            "rvclaw/control/set_armed",
            self._on_set_armed,
        )
        self.create_service(
            Trigger,
            "rvclaw/control/reset_latches",
            self._on_reset_latches,
        )

        self.output_publisher = self.create_publisher(
            Twist,
            "rvclaw/control/guarded_cmd_vel",
            self.command_qos,
        )
        self.frame_preview_publisher = self.create_publisher(
            UInt8MultiArray,
            "rvclaw/control/encoded_frame_preview",
            self.command_qos,
        )

        self.state_publishers: Dict[str, object] = {
            "armed": self.create_publisher(
                Bool, "rvclaw/control/armed", self.state_qos
            ),
            "estop_latched": self.create_publisher(
                Bool, "rvclaw/control/estop_latched", self.state_qos
            ),
            "watchdog_latched": self.create_publisher(
                Bool, "rvclaw/control/watchdog_latched", self.state_qos
            ),
            "deadman_fresh": self.create_publisher(
                Bool, "rvclaw/control/deadman_fresh", self.state_qos
            ),
            "feedback_connected": self.create_publisher(
                Bool, "rvclaw/control/feedback_connected", self.state_qos
            ),
            "watchdog_active": self.create_publisher(
                Bool, "rvclaw/control/watchdog_active", self.state_qos
            ),
            "hardware_transport_attached": self.create_publisher(
                Bool,
                "rvclaw/control/hardware_transport_attached",
                self.state_qos,
            ),
            "state": self.create_publisher(
                String, "rvclaw/control/state", self.state_qos
            ),
            "command_source": self.create_publisher(
                String, "rvclaw/control/command_source", self.state_qos
            ),
            "transition_reason": self.create_publisher(
                String, "rvclaw/control/transition_reason", self.state_qos
            ),
            "command_rejection_reason": self.create_publisher(
                String,
                "rvclaw/control/command_rejection_reason",
                self.state_qos,
            ),
            "arm_epoch_ns": self.create_publisher(
                UInt64, "rvclaw/control/arm_epoch_ns", self.state_qos
            ),
        }
        self.last_log_signature: Optional[tuple] = None
        self.timer = self.create_timer(1.0 / control_rate_hz, self._on_timer)
        self._publish_decision(self.safety.evaluate(time.monotonic()))
        if preview_only:
            self.get_logger().warning(
                "safety supervisor is preview-only: no hardware transport is attached"
            )

    @staticmethod
    def _message_stamp_ns(message: TwistStamped) -> int:
        seconds = int(message.header.stamp.sec)
        nanoseconds = int(message.header.stamp.nanosec)
        if seconds < 0 or not 0 <= nanoseconds < NANOSECONDS_PER_SECOND:
            return -1
        return seconds * NANOSECONDS_PER_SECOND + nanoseconds

    def _log_command_rejection(
        self,
        source: str,
        reason: str,
        fault: bool,
    ) -> None:
        now = time.monotonic()
        signature = (source, reason)
        if (
            signature == self.last_rejection_log_signature
            and now - self.last_rejection_log_monotonic < 1.0
        ):
            return
        message = "rejected %s command: %s" % (source, reason)
        if fault:
            self.get_logger().error(message)
        else:
            self.get_logger().warning(message)
        self.last_rejection_log_signature = signature
        self.last_rejection_log_monotonic = now

    def _sync_command_gate(self) -> None:
        if not self.safety.armed and self.command_gate.arm_epoch_ns is not None:
            self.command_gate.disarm()

    def _record_command_rejection(self, reason: str, fault: bool) -> None:
        if fault:
            if not self.latched_command_fault_rejection:
                self.latched_command_fault_rejection = reason
            self.last_command_rejection = self.latched_command_fault_rejection
        elif not self.latched_command_fault_rejection:
            self.last_command_rejection = reason

    def _clear_command_rejection(self) -> None:
        self.last_command_rejection = ""
        self.latched_command_fault_rejection = ""

    def _on_command(self, source: str, message: TwistStamped) -> None:
        twist = message.twist
        values = (twist.linear.x, twist.linear.y, twist.angular.z)
        now_monotonic = time.monotonic()
        if any(not math.isfinite(value) for value in values):
            self.safety.submit_command(
                source=source,
                velocity_x_m_s=twist.linear.x,
                velocity_y_m_s=twist.linear.y,
                angular_z_rad_s=twist.angular.z,
                now_monotonic=now_monotonic,
            )
            self.command_gate.disarm()
            self._record_command_rejection(
                self.safety.transition_reason,
                fault=True,
            )
            self._log_command_rejection(
                source,
                self.safety.transition_reason,
                fault=True,
            )
            return

        stamp_ns = self._message_stamp_ns(message)
        now_ns = self.get_clock().now().nanoseconds
        stamp_decision = self.command_gate.validate(
            source=source,
            stamp_ns=stamp_ns,
            now_ns=now_ns,
        )
        if not stamp_decision.accepted:
            if self.safety.armed:
                self.get_logger().warning(
                    "armed command timestamp diagnostic: source=%s reason=%s "
                    "stamp_ns=%s now_ns=%s arm_epoch_ns=%s age_s=%.6f"
                    % (source, stamp_decision.reason, stamp_ns, now_ns,
                       self.command_gate.arm_epoch_ns, (now_ns - stamp_ns) / 1e9)
                )
            self._record_command_rejection(
                stamp_decision.reason,
                stamp_decision.fault,
            )
            if stamp_decision.fault:
                self.safety.latch_estop(stamp_decision.reason)
                self.command_gate.disarm()
            self._log_command_rejection(
                source,
                stamp_decision.reason,
                stamp_decision.fault,
            )
            return

        accepted = self.safety.submit_command(
            source=source,
            velocity_x_m_s=twist.linear.x,
            velocity_y_m_s=twist.linear.y,
            angular_z_rad_s=twist.angular.z,
            now_monotonic=now_monotonic,
        )
        if accepted:
            self._clear_command_rejection()
            return

        self._sync_command_gate()
        if self.safety.estop_latched:
            self._record_command_rejection(
                self.safety.transition_reason,
                fault=True,
            )
            self._log_command_rejection(
                source,
                self.safety.transition_reason,
                fault=True,
            )
        else:
            self._record_command_rejection(
                "controller_disarmed",
                fault=False,
            )

    def _on_deadman(self, message: Bool) -> None:
        self.safety.update_deadman(message.data, time.monotonic())
        self._sync_command_gate()

    def _on_emergency_stop(self, message: Bool) -> None:
        self.safety.update_emergency_stop(message.data)
        self._sync_command_gate()

    def _on_feedback_connected(self, message: Bool) -> None:
        self.safety.update_feedback_connected(message.data)
        self._sync_command_gate()

    def _on_set_armed(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        success, explanation = self.safety.set_armed(
            request.data,
            time.monotonic(),
        )
        if success:
            if request.data:
                self.command_gate.arm(self.get_clock().now().nanoseconds)
                self._clear_command_rejection()
            else:
                self.command_gate.disarm()
        response.success = success
        response.message = explanation + self._hardware_status_suffix()
        return response

    def _on_reset_latches(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        success, explanation = self.safety.reset_latches()
        if success:
            self.command_gate.disarm()
            self._clear_command_rejection()
        response.success = success
        response.message = explanation + self._hardware_status_suffix()
        return response

    @staticmethod
    def _bool_message(value: bool) -> Bool:
        message = Bool()
        message.data = value
        return message

    def _hardware_transport_attached(self) -> bool:
        return False

    def _hardware_status_suffix(self) -> str:
        if self._hardware_transport_attached():
            return "; hardware transport attached"
        return "; hardware transport remains detached"

    @staticmethod
    def _string_message(value: str) -> String:
        message = String()
        message.data = value
        return message

    @staticmethod
    def _uint64_message(value: int) -> UInt64:
        message = UInt64()
        message.data = value
        return message

    def _publish_decision(self, decision: ControlDecision) -> None:
        output = Twist()
        output.linear.x = decision.velocity_x_m_s
        output.linear.y = decision.velocity_y_m_s
        output.angular.z = decision.angular_z_rad_s
        self.output_publisher.publish(output)

        frame_preview = UInt8MultiArray()
        frame_preview.data = list(
            encode_velocity_command(
                decision.velocity_x_m_s,
                decision.velocity_y_m_s,
                decision.angular_z_rad_s,
            )
        )
        self.frame_preview_publisher.publish(frame_preview)

        now = time.monotonic()
        bool_states = {
            "armed": self.safety.armed,
            "estop_latched": self.safety.estop_latched,
            "watchdog_latched": self.safety.watchdog_latched,
            "deadman_fresh": self.safety.deadman_fresh(now),
            "feedback_connected": self.safety.feedback_connected,
            "watchdog_active": decision.watchdog_active,
            "hardware_transport_attached": self._hardware_transport_attached(),
        }
        for name, value in bool_states.items():
            self.state_publishers[name].publish(self._bool_message(value))
        self.state_publishers["state"].publish(
            self._string_message(decision.state)
        )
        self.state_publishers["command_source"].publish(
            self._string_message(decision.source)
        )
        self.state_publishers["transition_reason"].publish(
            self._string_message(decision.transition_reason)
        )
        self.state_publishers["command_rejection_reason"].publish(
            self._string_message(self.last_command_rejection)
        )
        self.state_publishers["arm_epoch_ns"].publish(
            self._uint64_message(self.command_gate.arm_epoch_ns or 0)
        )

        signature = (
            decision.state,
            decision.source,
            decision.transition_reason,
        )
        if signature != self.last_log_signature:
            self.get_logger().info(
                "state=%s source=%s reason=%s"
                % signature
            )
            self.last_log_signature = signature

    def _on_timer(self) -> None:
        decision = self.safety.evaluate(time.monotonic())
        self._sync_command_gate()
        self._publish_decision(decision)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[SafetySupervisorNode] = None
    try:
        node = SafetySupervisorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
