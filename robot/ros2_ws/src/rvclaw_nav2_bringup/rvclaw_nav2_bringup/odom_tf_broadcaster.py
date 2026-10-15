"""Broadcast a validated odom-to-base transform from integrated STM32 odometry."""

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import TransformBroadcaster


def normalized_quaternion(
    x: float,
    y: float,
    z: float,
    w: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Return a finite unit quaternion or reject the sample."""
    values = (x, y, z, w)
    if not all(math.isfinite(value) for value in values):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        return None
    return tuple(value / norm for value in values)


def transform_sample_is_valid(
    parent_frame: str,
    child_frame: str,
    expected_parent_frame: str,
    expected_child_frame: str,
    stamp_ns: int,
    previous_stamp_ns: Optional[int],
    position: Tuple[float, float, float],
    quaternion: Optional[Tuple[float, float, float, float]],
) -> bool:
    """Validate the frame contract, timestamp, and transform components."""
    if parent_frame != expected_parent_frame or child_frame != expected_child_frame:
        return False
    if stamp_ns <= 0 or (previous_stamp_ns is not None and stamp_ns <= previous_stamp_ns):
        return False
    if not all(math.isfinite(value) for value in position):
        return False
    return quaternion is not None


class OdomTfBroadcaster(Node):
    """Republish the integrated odometry pose as odom-to-base TF only."""

    def __init__(self) -> None:
        super().__init__("rvclaw_odom_tf")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_footprint")

        odom_topic = str(self.get_parameter("odom_topic").value)
        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.last_stamp_ns = None
        self.rejected_samples = 0
        self.broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            Odometry,
            odom_topic,
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"broadcasting validated {self.parent_frame}->{self.child_frame} "
            f"from {odom_topic}"
        )

    def _reject(self) -> None:
        self.rejected_samples += 1
        if self.rejected_samples == 1 or self.rejected_samples % 100 == 0:
            self.get_logger().error(
                f"rejected invalid odometry transform; count={self.rejected_samples}"
            )

    def _on_odometry(self, message: Odometry) -> None:
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        pose = message.pose.pose
        quaternion = normalized_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        position = (pose.position.x, pose.position.y, pose.position.z)
        if not transform_sample_is_valid(
            message.header.frame_id,
            message.child_frame_id,
            self.parent_frame,
            self.child_frame,
            stamp_ns,
            self.last_stamp_ns,
            position,
            quaternion,
        ):
            self._reject()
            return

        transform = TransformStamped()
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = float(position[0])
        transform.transform.translation.y = float(position[1])
        transform.transform.translation.z = float(position[2])
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.broadcaster.sendTransform(transform)
        self.last_stamp_ns = stamp_ns


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
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
