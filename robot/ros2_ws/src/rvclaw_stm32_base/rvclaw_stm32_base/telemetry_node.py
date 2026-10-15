"""Read-only ROS 2 bridge for V550 STM32 chassis telemetry."""

import errno
import fcntl
import math
import os
import stat
import termios
import time
from typing import List, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, UInt8

from .protocol import FeedbackFrame, FeedbackParser


BAUD_CONSTANTS = {115200: termios.B115200}
ODOM_POSE_COVARIANCE = [
    1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e3,
]
ODOM_TWIST_COVARIANCE = list(ODOM_POSE_COVARIANCE)


class Stm32TelemetryNode(Node):
    """Publish validated feedback while keeping the serial descriptor read-only."""

    def __init__(self) -> None:
        super().__init__("rvclaw_stm32_telemetry")

        self.declare_parameter(
            "device",
            "/dev/serial/by-id/usb-WCH.CN_USB_Single_Serial_0002-if00",
        )
        self.declare_parameter("baud", 115200)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("imu_frame_id", "imu_link")
        self.declare_parameter("connection_timeout_s", 0.5)
        self.declare_parameter("reconnect_interval_s", 1.0)
        self.declare_parameter("odom_x_scale", 1.0)
        self.declare_parameter("odom_y_scale", 1.0)
        self.declare_parameter("odom_z_scale_positive", 1.0)
        self.declare_parameter("odom_z_scale_negative", 1.0)

        self.device = str(self.get_parameter("device").value)
        self.baud = int(self.get_parameter("baud").value)
        self.odom_frame_id = str(self.get_parameter("odom_frame_id").value)
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.imu_frame_id = str(self.get_parameter("imu_frame_id").value)
        self.connection_timeout_s = float(
            self.get_parameter("connection_timeout_s").value
        )
        self.reconnect_interval_s = float(
            self.get_parameter("reconnect_interval_s").value
        )
        self.odom_x_scale = float(self.get_parameter("odom_x_scale").value)
        self.odom_y_scale = float(self.get_parameter("odom_y_scale").value)
        self.odom_z_scale_positive = float(
            self.get_parameter("odom_z_scale_positive").value
        )
        self.odom_z_scale_negative = float(
            self.get_parameter("odom_z_scale_negative").value
        )

        if self.baud not in BAUD_CONSTANTS:
            raise ValueError("only the confirmed 115200 baud rate is supported")
        if self.connection_timeout_s <= 0.0 or self.reconnect_interval_s <= 0.0:
            raise ValueError("connection and reconnect timeouts must be positive")

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.odom_publisher = self.create_publisher(Odometry, "odom", sensor_qos)
        self.imu_publisher = self.create_publisher(Imu, "imu/data_raw", sensor_qos)
        self.voltage_publisher = self.create_publisher(
            Float32, "PowerVoltage", state_qos
        )
        self.connected_publisher = self.create_publisher(
            Bool, "rvclaw/stm32/connected", state_qos
        )
        self.disable_flag_publisher = self.create_publisher(
            UInt8, "rvclaw/stm32/software_disable_flag_raw", state_qos
        )

        self.parser = FeedbackParser()
        self.fd: Optional[int] = None
        self.original_termios: Optional[List[object]] = None
        self.connected: Optional[bool] = None
        self.last_valid_monotonic: Optional[float] = None
        self.next_open_monotonic = 0.0
        self.last_open_error = ""
        self.last_open_error_monotonic = 0.0
        self.position_x = 0.0
        self.position_y = 0.0
        self.yaw = 0.0

        self._set_connected(False)
        self.poll_timer = self.create_timer(0.01, self._poll_serial)
        self.get_logger().info(
            "telemetry-only mode: serial is opened O_RDONLY and no command topic exists"
        )

    def _configure_serial(self, fd: int) -> List[object]:
        original = termios.tcgetattr(fd)
        configured = list(original)
        configured[0] = 0
        configured[1] = 0
        configured[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        configured[3] = 0
        configured[4] = BAUD_CONSTANTS[self.baud]
        configured[5] = BAUD_CONSTANTS[self.baud]
        configured[6] = list(original[6])
        configured[6][termios.VMIN] = 0
        configured[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, configured)
        termios.tcflush(fd, termios.TCIFLUSH)
        return original

    def _open_serial(self) -> None:
        now = time.monotonic()
        self.next_open_monotonic = now + self.reconnect_interval_s
        try:
            mode = os.stat(self.device).st_mode
            if not stat.S_ISCHR(mode):
                raise OSError(errno.ENODEV, "configured path is not a character device")
            fd = os.open(
                self.device,
                os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if hasattr(termios, "TIOCEXCL"):
                    fcntl.ioctl(fd, termios.TIOCEXCL)
                original = self._configure_serial(fd)
            except Exception:
                if hasattr(termios, "TIOCNXCL"):
                    try:
                        fcntl.ioctl(fd, termios.TIOCNXCL)
                    except OSError:
                        pass
                os.close(fd)
                raise
        except (OSError, termios.error) as exc:
            message = str(exc)
            if (
                message != self.last_open_error
                or now - self.last_open_error_monotonic >= 5.0
            ):
                self.get_logger().error(
                    "cannot open telemetry device %s: %s" % (self.device, message)
                )
                self.last_open_error = message
                self.last_open_error_monotonic = now
            return

        self.fd = fd
        self.original_termios = original
        self.parser.buffer.clear()
        self.last_open_error = ""
        self.get_logger().info(
            "opened read-only telemetry device %s -> %s at %d baud"
            % (self.device, os.path.realpath(self.device), self.baud)
        )

    def _close_serial(self) -> None:
        if self.fd is None:
            return
        fd = self.fd
        self.fd = None
        if self.original_termios is not None:
            try:
                termios.tcsetattr(fd, termios.TCSANOW, self.original_termios)
            except (OSError, termios.error):
                pass
        self.original_termios = None
        if hasattr(termios, "TIOCNXCL"):
            try:
                fcntl.ioctl(fd, termios.TIOCNXCL)
            except OSError:
                pass
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def _set_connected(self, connected: bool) -> None:
        if self.connected is connected:
            return
        self.connected = connected
        message = Bool()
        message.data = connected
        self.connected_publisher.publish(message)
        if connected:
            self.get_logger().info("STM32 feedback connection established")
        elif self.last_valid_monotonic is not None:
            self.get_logger().warning("STM32 feedback connection timed out")

    def _poll_serial(self) -> None:
        now = time.monotonic()
        if self.fd is None:
            if now >= self.next_open_monotonic:
                self._open_serial()
            return

        try:
            for _ in range(8):
                try:
                    chunk = os.read(self.fd, 4096)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                for frame in self.parser.feed(chunk):
                    self._publish_frame(frame)
                if len(chunk) < 4096:
                    break
        except OSError as exc:
            self.get_logger().error("telemetry read failed: %s" % exc)
            self._close_serial()
            self._set_connected(False)
            self.next_open_monotonic = now + self.reconnect_interval_s
            return

        if (
            self.last_valid_monotonic is not None
            and now - self.last_valid_monotonic > self.connection_timeout_s
        ):
            self._set_connected(False)

    def _publish_frame(self, frame: FeedbackFrame) -> None:
        now_monotonic = time.monotonic()
        now_ros = self.get_clock().now().to_msg()

        velocity_x = frame.velocity_x_m_s * self.odom_x_scale
        velocity_y = frame.velocity_y_m_s * self.odom_y_scale
        if frame.angular_z_rad_s >= 0.0:
            angular_z = frame.angular_z_rad_s * self.odom_z_scale_positive
        else:
            angular_z = frame.angular_z_rad_s * self.odom_z_scale_negative

        dt = 0.0
        if self.last_valid_monotonic is not None:
            candidate_dt = now_monotonic - self.last_valid_monotonic
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

        odom = Odometry()
        odom.header.stamp = now_ros
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
        imu.header.stamp = now_ros
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

        self.last_valid_monotonic = now_monotonic
        self._set_connected(True)
        if self.parser.valid_frames % 200 == 0:
            self.get_logger().info(
                "rx frames=%d checksum_failures=%d discarded_bytes=%d"
                % (
                    self.parser.valid_frames,
                    self.parser.checksum_failures,
                    self.parser.discarded_bytes,
                )
            )

    def close(self) -> None:
        self._close_serial()
        if self.context.ok():
            self._set_connected(False)
        else:
            self.connected = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[Stm32TelemetryNode] = None
    try:
        node = Stm32TelemetryNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
