"""One-shot scan filter for the supervised, odom-fixed backup corridor."""

import copy
import math
import time
from typing import Optional, Tuple

from nav_msgs.msg import Odometry
import rclpy
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
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

from rvclaw_nav2_bringup.bounded_backup_policy import (
    CORRIDOR_AUTHORIZATION_TTL_S,
    filter_authorized_scan_ranges,
)


LASER_X_M = 0.07725
LASER_Y_M = 0.0
LASER_YAW_RAD = 0.0
MAX_INPUT_AGE_S = 0.30
MAX_STATIONARY_LINEAR_M_S = 0.03
MAX_STATIONARY_ANGULAR_RAD_S = 0.08

STATE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def quaternion_to_yaw(orientation) -> float:
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


class BoundedBackupCorridorFilter(Node):
    """Pass raw scans until one short, supervised corridor is authorized."""

    def __init__(self):
        super().__init__("rvclaw_backup_corridor_filter")
        self.publisher = self.create_publisher(
            LaserScan, "/rvclaw/backup_scan", qos_profile_sensor_data
        )
        self.active_publisher = self.create_publisher(
            Bool, "/rvclaw/backup_corridor/active", STATE_QOS
        )
        self.create_subscription(
            LaserScan, "/scan", self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, "/odom", self._on_odom, qos_profile_sensor_data
        )
        self.create_service(
            SetBool,
            "/rvclaw/backup_corridor/set_active",
            self._on_set_active,
        )
        self.create_timer(0.05, self._on_timer)
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.authorized_start_pose: Optional[Tuple[float, float, float]] = None
        self.latest_scan_monotonic: Optional[float] = None
        self.latest_odom_monotonic: Optional[float] = None
        self.latest_linear_x = 0.0
        self.latest_linear_y = 0.0
        self.latest_angular_z = 0.0
        self.authorization_deadline: Optional[float] = None
        self.ever_authorized = False
        self.active = False
        self.masked_returns = 0
        self._publish_active()
        self.get_logger().info(
            "backup filter ready; inactive raw-scan passthrough; "
            "active mode clears current-footprint returns only"
        )

    def _publish_active(self) -> None:
        message = Bool()
        message.data = self.active
        self.active_publisher.publish(message)

    def _deactivate(self) -> None:
        self.active = False
        self.authorization_deadline = None
        self.authorized_start_pose = None
        self._publish_active()

    def _on_timer(self) -> None:
        if (
            self.active
            and self.authorization_deadline is not None
            and time.monotonic() >= self.authorization_deadline
        ):
            self.get_logger().warning("backup corridor authorization expired")
            self._deactivate()

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        self.current_pose = (
            float(pose.position.x),
            float(pose.position.y),
            quaternion_to_yaw(pose.orientation),
        )
        twist = message.twist.twist
        self.latest_linear_x = float(twist.linear.x)
        self.latest_linear_y = float(twist.linear.y)
        self.latest_angular_z = float(twist.angular.z)
        self.latest_odom_monotonic = time.monotonic()

    def _on_scan(self, message: LaserScan) -> None:
        self.latest_scan_monotonic = time.monotonic()
        output = copy.deepcopy(message)
        if (
            self.active
            and self.current_pose is not None
            and self.authorized_start_pose is not None
        ):
            output.ranges, masked = filter_authorized_scan_ranges(
                message.ranges,
                float(message.angle_min),
                float(message.angle_increment),
                float(message.range_min),
                float(message.range_max),
                LASER_X_M,
                LASER_Y_M,
                LASER_YAW_RAD,
                self.current_pose,
                self.authorized_start_pose,
            )
            self.masked_returns += masked
        self.publisher.publish(output)

    def _on_set_active(self, request, response):
        if not request.data:
            self._deactivate()
            response.success = True
            response.message = "backup corridor inactive"
            return response

        now = time.monotonic()
        if self.ever_authorized:
            response.success = False
            response.message = "backup corridor is one-shot"
            return response
        if (
            self.current_pose is None
            or self.latest_scan_monotonic is None
            or self.latest_odom_monotonic is None
            or now - self.latest_scan_monotonic > MAX_INPUT_AGE_S
            or now - self.latest_odom_monotonic > MAX_INPUT_AGE_S
        ):
            response.success = False
            response.message = "fresh scan and odometry are required"
            return response
        if (
            abs(self.latest_linear_x) > MAX_STATIONARY_LINEAR_M_S
            or abs(self.latest_linear_y) > MAX_STATIONARY_LINEAR_M_S
            or abs(self.latest_angular_z) > MAX_STATIONARY_ANGULAR_RAD_S
        ):
            response.success = False
            response.message = "vehicle must be stationary before authorization"
            return response

        self.authorized_start_pose = tuple(self.current_pose)
        self.authorization_deadline = now + CORRIDOR_AUTHORIZATION_TTL_S
        self.ever_authorized = True
        self.active = True
        self._publish_active()
        response.success = True
        response.message = "fixed odom swept corridor authorized once"
        self.get_logger().warning(
            "authorized one supervised backup corridor for "
            f"{CORRIDOR_AUTHORIZATION_TTL_S:.1f} seconds; "
            "only current-footprint self-returns will be cleared"
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoundedBackupCorridorFilter()
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
