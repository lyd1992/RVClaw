"""Single-owner ROS 2 safety, motion, telemetry, and odometry bridge."""

import math
import time
from typing import Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
try:
    from rclpy.exceptions import RCLError
except ImportError:
    from rclpy._rclpy_pybind11 import RCLError
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, UInt8
from std_srvs.srv import SetBool

from .protocol import FeedbackFrame
from .safety import ControlDecision
from .safety_supervisor_node import SafetySupervisorNode
from .serial_transport import (
    GuardedMotionLimits,
    GuardedSerialTransport,
    TransportError,
    ZeroOnlyTransportConfig,
)


ODOM_POSE_COVARIANCE = [
    1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e3,
]
ODOM_TWIST_COVARIANCE = list(ODOM_POSE_COVARIANCE)


class Stm32HardwareNode(SafetySupervisorNode):
    """Own the STM32 endpoint and keep safety state atomic with serial output."""

    def __init__(self) -> None:
        self.transport: Optional[GuardedSerialTransport] = None
        self.transport_attached = False
        self.feedback_connected: Optional[bool] = None
        self.latest_voltage_v: Optional[float] = None
        self.last_feedback_monotonic: Optional[float] = None
        self.position_x = 0.0
        self.position_y = 0.0
        self.yaw = 0.0

        super().__init__(
            "rvclaw_stm32_hardware",
            preview_only=False,
            subscribe_feedback_connected=False,
        )

        self.declare_parameter(
            "device",
            "/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00",
        )
        self.declare_parameter("baud", 115200)
        self.declare_parameter("hardware_output_enabled", False)
        self.declare_parameter("connection_timeout_s", 0.5)
        self.declare_parameter("min_battery_voltage_v", 9.0)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("imu_frame_id", "imu_link")
        self.declare_parameter("odom_x_scale", 1.0)
        self.declare_parameter("odom_y_scale", 1.0)
        self.declare_parameter("odom_z_scale_positive", 1.0)
        self.declare_parameter("odom_z_scale_negative", 1.0)

        self.device = str(self.get_parameter("device").value)
        self.baud = int(self.get_parameter("baud").value)
        self.hardware_output_enabled = bool(
            self.get_parameter("hardware_output_enabled").value
        )
        self.connection_timeout_s = float(
            self.get_parameter("connection_timeout_s").value
        )
        self.min_battery_voltage_v = float(
            self.get_parameter("min_battery_voltage_v").value
        )
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.imu_frame_id = str(self.get_parameter("imu_frame_id").value)
        self.odom_x_scale = float(self.get_parameter("odom_x_scale").value)
        self.odom_y_scale = float(self.get_parameter("odom_y_scale").value)
        self.odom_z_scale_positive = float(
            self.get_parameter("odom_z_scale_positive").value
        )
        self.odom_z_scale_negative = float(
            self.get_parameter("odom_z_scale_negative").value
        )

        if (
            not math.isfinite(self.connection_timeout_s)
            or self.connection_timeout_s <= 0
        ):
            raise ValueError("connection_timeout_s must be finite and positive")
        if (
            not math.isfinite(self.min_battery_voltage_v)
            or self.min_battery_voltage_v <= 0
        ):
            raise ValueError("min_battery_voltage_v must be finite and positive")
        scales = (
            self.odom_x_scale,
            self.odom_y_scale,
            self.odom_z_scale_positive,
            self.odom_z_scale_negative,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("odometry scales must be finite and positive")
        if self.hardware_output_enabled and not self.safety.require_feedback_connected:
            raise ValueError(
                "hardware mode requires require_feedback_connected=true"
            )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odom_publisher = self.create_publisher(Odometry, "odom", sensor_qos)
        self.imu_publisher = self.create_publisher(Imu, "imu/data_raw", sensor_qos)
        self.voltage_publisher = self.create_publisher(
            Float32, "PowerVoltage", self.state_qos
        )
        self.connected_publisher = self.create_publisher(
            Bool, "rvclaw/stm32/connected", self.state_qos
        )
        self.disable_flag_publisher = self.create_publisher(
            UInt8, "rvclaw/stm32/software_disable_flag_raw", self.state_qos
        )
        self._set_feedback_connected(False)

        if not self.hardware_output_enabled:
            self.get_logger().warning(
                "hardware node is disabled: set hardware_output_enabled=true "
                "only during an approved hardware test"
            )
            return

        self.transport = GuardedSerialTransport(
            ZeroOnlyTransportConfig(
                device=self.device,
                baud=self.baud,
                tx_enabled=True,
                zero_only=True,
                zero_rate_hz=20.0,
            ),
            GuardedMotionLimits(
                max_linear_x_m_s=self.safety.max_linear_x_m_s,
                max_angular_z_rad_s=self.safety.max_angular_z_rad_s,
            ),
        )
        try:
            self.transport.__enter__()
        except BaseException:
            self.transport = None
            raise
        self.transport_attached = True
        self.get_logger().warning(
            "hardware transport attached to %s; controller remains estop-latched"
            % self.device
        )

    def _hardware_transport_attached(self) -> bool:
        return bool(getattr(self, "transport_attached", False))

    def _voltage_safe(self) -> bool:
        return (
            self.latest_voltage_v is not None
            and self.latest_voltage_v >= self.min_battery_voltage_v
        )

    def _set_feedback_connected(self, connected: bool) -> None:
        self.safety.update_feedback_connected(connected)
        if self.feedback_connected is connected:
            return
        self.feedback_connected = connected
        context = getattr(self, "context", None)
        if context is not None and not context.ok():
            return
        message = Bool()
        message.data = connected
        self.connected_publisher.publish(message)
        if connected:
            self.get_logger().info("STM32 feedback connection established")
        else:
            self.get_logger().warning("STM32 feedback connection unavailable")

    def _publish_feedback_frame(
        self,
        frame: FeedbackFrame,
        now_monotonic: float,
    ) -> None:
        velocity_x = frame.velocity_x_m_s * self.odom_x_scale
        velocity_y = frame.velocity_y_m_s * self.odom_y_scale
        if frame.angular_z_rad_s >= 0.0:
            angular_z = frame.angular_z_rad_s * self.odom_z_scale_positive
        else:
            angular_z = frame.angular_z_rad_s * self.odom_z_scale_negative

        dt = 0.0
        if self.last_feedback_monotonic is not None:
            candidate_dt = now_monotonic - self.last_feedback_monotonic
            if 0.0 < candidate_dt <= self.connection_timeout_s:
                dt = candidate_dt
        previous_yaw = self.yaw
        self.position_x += (
            velocity_x * math.cos(previous_yaw)
            - velocity_y * math.sin(previous_yaw)
        ) * dt
        self.position_y += (
            velocity_x * math.sin(previous_yaw)
            + velocity_y * math.cos(previous_yaw)
        ) * dt
        self.yaw = math.atan2(
            math.sin(self.yaw + angular_z * dt),
            math.cos(self.yaw + angular_z * dt),
        )

        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        odom.pose.pose.position.x = self.position_x
        odom.pose.pose.position.y = self.position_y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.pose.covariance = ODOM_POSE_COVARIANCE
        odom.twist.twist.linear.x = velocity_x
        odom.twist.twist.linear.y = velocity_y
        odom.twist.twist.angular.z = angular_z
        odom.twist.covariance = ODOM_TWIST_COVARIANCE
        self.odom_publisher.publish(odom)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = self.imu_frame_id
        imu.orientation_covariance[0] = -1.0
        imu.angular_velocity.x = frame.angular_velocity_x_rad_s
        imu.angular_velocity.y = frame.angular_velocity_y_rad_s
        imu.angular_velocity.z = frame.angular_velocity_z_rad_s
        imu.linear_acceleration.x = frame.acceleration_x_m_s2
        imu.linear_acceleration.y = frame.acceleration_y_m_s2
        imu.linear_acceleration.z = frame.acceleration_z_m_s2
        self.imu_publisher.publish(imu)

        voltage = Float32()
        voltage.data = frame.battery_voltage_v
        self.voltage_publisher.publish(voltage)
        disable_flag = UInt8()
        disable_flag.data = frame.software_disable_flag_raw
        self.disable_flag_publisher.publish(disable_flag)

        self.latest_voltage_v = frame.battery_voltage_v
        self.last_feedback_monotonic = now_monotonic
        if not self._voltage_safe():
            self.safety.latch_estop("low_battery_voltage")
            self.command_gate.disarm()

    def _poll_feedback(self) -> None:
        if self.transport is None or not self.transport_attached:
            self._set_feedback_connected(False)
            return
        frames = self.transport.poll_feedback()
        for frame in frames:
            self._publish_feedback_frame(frame, time.monotonic())
        connection_now = time.monotonic()
        connected = (
            self.last_feedback_monotonic is not None
            and 0.0 <= connection_now - self.last_feedback_monotonic
            <= self.connection_timeout_s
        )
        self._set_feedback_connected(connected)

    def _force_zero(self) -> None:
        if self.transport is None or not self.transport_attached:
            return
        self.transport.revoke_auto_session()
        self.transport.send_zero()

    def _handle_transport_failure(self, exc: BaseException) -> None:
        self.get_logger().error("hardware transport failed: %s" % exc)
        self.safety.latch_estop("hardware_transport_failure")
        self.command_gate.disarm()
        transport = self.transport
        self.transport = None
        self.transport_attached = False
        self._set_feedback_connected(False)
        if transport is not None:
            transport.revoke_auto_session()
            if not transport.close():
                self.get_logger().error(
                    "hardware transport cleanup did not confirm final zero/drain: %s"
                    % (transport.last_close_error or "unknown close failure")
                )

    def _apply_decision(self, decision: ControlDecision) -> None:
        if self.transport is None or not self.transport_attached:
            return
        if not self.safety.armed:
            self._force_zero()
            return
        if not self.transport.auto_session_admitted:
            self.transport.send_auto_enable_zero_sequence()
        self.transport.send_guarded_velocity(
            decision.velocity_x_m_s,
            decision.velocity_y_m_s,
            decision.angular_z_rad_s,
        )

    def _on_command(self, source, message) -> None:
        super()._on_command(source, message)
        if not self.safety.armed:
            try:
                self._force_zero()
            except TransportError as exc:
                self._handle_transport_failure(exc)

    def _on_deadman(self, message: Bool) -> None:
        super()._on_deadman(message)
        if not self.safety.armed:
            try:
                self._force_zero()
            except TransportError as exc:
                self._handle_transport_failure(exc)

    def _on_emergency_stop(self, message: Bool) -> None:
        super()._on_emergency_stop(message)
        if not self.safety.armed:
            try:
                self._force_zero()
            except TransportError as exc:
                self._handle_transport_failure(exc)

    def _on_set_armed(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        if request.data and not self._voltage_safe():
            response.success = False
            response.message = (
                "safe, fresh battery voltage is required; hardware transport attached"
            )
            return response
        result = super()._on_set_armed(request, response)
        if not self.safety.armed:
            try:
                self._force_zero()
            except TransportError as exc:
                self._handle_transport_failure(exc)
        return result

    def _on_timer(self) -> None:
        try:
            self._poll_feedback()
        except TransportError as exc:
            self._handle_transport_failure(exc)

        now = time.monotonic()
        decision = self.safety.evaluate(now)
        self._sync_command_gate()
        try:
            self._apply_decision(decision)
        except TransportError as exc:
            self._handle_transport_failure(exc)
            decision = self.safety.evaluate(time.monotonic())
        self._publish_decision(decision)

    def close(self) -> None:
        self.safety.latch_estop("hardware_node_shutdown")
        self.command_gate.disarm()
        transport = self.transport
        self.transport = None
        self.transport_attached = False
        if transport is not None:
            transport.revoke_auto_session()
            if not transport.close():
                self.get_logger().error(
                    "shutdown did not confirm final zero/drain: %s"
                    % (transport.last_close_error or "unknown close failure")
                )
        self._set_feedback_connected(False)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[Stm32HardwareNode] = None
    try:
        node = Stm32HardwareNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
