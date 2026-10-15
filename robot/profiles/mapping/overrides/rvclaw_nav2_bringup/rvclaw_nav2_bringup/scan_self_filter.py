#!/usr/bin/env python3
"""Remove laser returns that geometrically fall inside the V550 chassis."""

import math

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


FILTERED_SCAN_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


class ScanSelfFilter(Node):
    """Filter self returns while preserving real obstacles outside the chassis."""

    def __init__(self) -> None:
        super().__init__("rvclaw_scan_self_filter")
        self.declare_parameter("input_topic", "/scan_raw")
        self.declare_parameter("output_topic", "/scan")
        self.declare_parameter("laser_x_m", 0.07725)
        self.declare_parameter("laser_y_m", 0.0)
        self.declare_parameter("laser_yaw_deg", 0.0)
        self.declare_parameter("body_front_m", 0.13275)
        self.declare_parameter("body_rear_m", 0.13440)
        self.declare_parameter("body_left_m", 0.10725)
        self.declare_parameter("body_right_m", 0.10725)
        self.declare_parameter("body_padding_m", 0.005)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.laser_x_m = float(self.get_parameter("laser_x_m").value)
        self.laser_y_m = float(self.get_parameter("laser_y_m").value)
        self.laser_yaw_rad = math.radians(
            float(self.get_parameter("laser_yaw_deg").value)
        )
        self.body_front_m = float(self.get_parameter("body_front_m").value)
        self.body_rear_m = float(self.get_parameter("body_rear_m").value)
        self.body_left_m = float(self.get_parameter("body_left_m").value)
        self.body_right_m = float(self.get_parameter("body_right_m").value)
        self.body_padding_m = float(self.get_parameter("body_padding_m").value)
        if any(
            value < 0.0
            for value in (
                self.body_front_m,
                self.body_rear_m,
                self.body_left_m,
                self.body_right_m,
                self.body_padding_m,
            )
        ):
            raise ValueError("chassis dimensions and padding must be non-negative")
        # Scan geometry is stable for a running lidar. Cache direction vectors
        # so the hot path does not repeat trigonometry for every scan sample.
        self._mask_geometry = None
        self._ray_directions = ()
        self.publisher = self.create_publisher(
            LaserScan,
            output_topic,
            FILTERED_SCAN_QOS,
        )
        self.subscription = self.create_subscription(
            LaserScan,
            input_topic,
            self._on_scan,
            qos_profile_sensor_data,
        )

    def _ray_directions_for_scan(self, message: LaserScan):
        geometry = (
            float(message.angle_min),
            float(message.angle_increment),
            len(message.ranges),
        )
        if geometry != self._mask_geometry:
            self._mask_geometry = geometry
            self._ray_directions = tuple(
                (
                    math.cos(
                        geometry[0] + index * geometry[1] + self.laser_yaw_rad
                    ),
                    math.sin(
                        geometry[0] + index * geometry[1] + self.laser_yaw_rad
                    ),
                )
                for index in range(geometry[2])
            )
        return self._ray_directions

    def _is_self_return(self, distance_m: float, direction) -> bool:
        x_m = self.laser_x_m + distance_m * direction[0]
        y_m = self.laser_y_m + distance_m * direction[1]
        padding = self.body_padding_m
        return (
            -self.body_rear_m - padding <= x_m <= self.body_front_m + padding
            and -self.body_right_m - padding <= y_m <= self.body_left_m + padding
        )

    def _on_scan(self, message: LaserScan) -> None:
        filtered = LaserScan()
        filtered.header = message.header
        filtered.angle_min = message.angle_min
        filtered.angle_max = message.angle_max
        filtered.angle_increment = message.angle_increment
        filtered.time_increment = message.time_increment
        filtered.scan_time = message.scan_time
        filtered.range_min = message.range_min
        filtered.range_max = message.range_max
        filtered.ranges = list(message.ranges)
        filtered.intensities = list(message.intensities)
        directions = self._ray_directions_for_scan(message)
        for index, value in enumerate(filtered.ranges):
            if (
                math.isfinite(value)
                and message.range_min <= value <= message.range_max
                and self._is_self_return(value, directions[index])
            ):
                filtered.ranges[index] = float("inf")
        self.publisher.publish(filtered)


def main() -> None:
    rclpy.init()
    node = ScanSelfFilter()
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
