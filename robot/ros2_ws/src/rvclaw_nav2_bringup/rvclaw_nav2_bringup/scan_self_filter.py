#!/usr/bin/env python3
"""Remove laser returns that geometrically fall inside the V550 chassis."""

import math
from copy import deepcopy

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


def crop_forward_scan(message, half_angle_deg, laser_yaw_rad=0.0):
    """Publish only observed forward rays; excluded bearings are not free rays."""
    if not math.isfinite(half_angle_deg) or not 0 < half_angle_deg <= 180:
        raise ValueError("forward half angle must be in (0, 180] degrees")
    if half_angle_deg == 180:
        return message
    if not message.ranges or not math.isfinite(message.angle_increment) or message.angle_increment == 0:
        raise ValueError("cannot crop a scan without valid angular geometry")
    limit = math.radians(half_angle_deg)
    selected = []
    for index in range(len(message.ranges)):
        angle = message.angle_min + index * message.angle_increment + laser_yaw_rad
        angle = math.atan2(math.sin(angle), math.cos(angle))
        if abs(angle) <= limit + 1e-6:
            selected.append(index)
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("forward scan must form a contiguous observed sector")
    first, last = selected[0], selected[-1]
    cropped = deepcopy(message)
    cropped.angle_min = message.angle_min + first * message.angle_increment
    cropped.angle_max = message.angle_min + last * message.angle_increment
    cropped.ranges = list(message.ranges[first:last + 1])
    if message.intensities:
        if len(message.intensities) != len(message.ranges):
            raise ValueError("scan intensity/range lengths disagree")
        cropped.intensities = list(message.intensities[first:last + 1])
    offset_ns = round(first * message.time_increment * 1e9)
    stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec + offset_ns
    cropped.header.stamp.sec, cropped.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    return cropped


def localization_range_scan(message, minimum_range_m):
    if not math.isfinite(minimum_range_m) or not 0 <= minimum_range_m <= .5:
        raise ValueError("invalid localization minimum range")
    if minimum_range_m <= message.range_min:
        return message
    result = deepcopy(message)
    result.range_min = max(float(message.range_min), minimum_range_m)
    result.ranges = [float('nan') if math.isfinite(v) and v < result.range_min else v
                     for v in message.ranges]
    return result


class ScanSelfFilter(Node):
    """Filter self returns while preserving real obstacles outside the chassis."""

    def __init__(self) -> None:
        super().__init__("rvclaw_scan_self_filter")
        self.declare_parameter("input_topic", "/scan_raw")
        self.declare_parameter("output_topic", "/scan")
        self.declare_parameter("localization_output_topic", "/scan_localization")
        self.declare_parameter("full_output_topic", "/scan_full")
        self.declare_parameter("forward_half_angle_deg", 180.0)
        self.declare_parameter("localization_half_angle_deg", 180.0)
        self.declare_parameter("localization_min_range_m", 0.0)
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
        localization_topic = str(self.get_parameter("localization_output_topic").value)
        self.forward_half_angle_deg = float(self.get_parameter("forward_half_angle_deg").value)
        self.localization_half_angle_deg = float(self.get_parameter("localization_half_angle_deg").value)
        self.localization_min_range_m = float(self.get_parameter("localization_min_range_m").value)
        if not math.isfinite(self.localization_min_range_m) or not 0 <= self.localization_min_range_m <= .5:
            raise ValueError("localization minimum range must be in [0, 0.5] m")
        if not math.isfinite(self.forward_half_angle_deg) or not 0 < self.forward_half_angle_deg <= 180:
            raise ValueError("forward half angle must be in (0, 180] degrees")
        if localization_topic == output_topic:
            raise ValueError("navigation and localization scans require distinct topics")
        if not math.isfinite(self.localization_half_angle_deg) or not 0 < self.localization_half_angle_deg <= 180:
            raise ValueError("invalid localization half angle")
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
        self.localization_publisher = self.create_publisher(
            LaserScan, localization_topic, FILTERED_SCAN_QOS,
        )
        self.full_publisher = self.create_publisher(
            LaserScan, str(self.get_parameter("full_output_topic").value), FILTERED_SCAN_QOS,
        )
        self.get_logger().info(
            "Navigation: +/-%.1f deg; localization: +/-%.1f deg; raw/full retained"
            % (self.forward_half_angle_deg, self.localization_half_angle_deg)
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
        # Close returns remain available to the navigation obstacle monitor.
        # A localization-only range floor prevents chassis/multipath echoes
        # from creating false nearby walls in the likelihood-field matcher.
        full_localization = localization_range_scan(filtered, self.localization_min_range_m)
        self.full_publisher.publish(full_localization)
        try:
            navigation = crop_forward_scan(filtered, self.forward_half_angle_deg, self.laser_yaw_rad)
            localization = crop_forward_scan(full_localization, self.localization_half_angle_deg, self.laser_yaw_rad)
        except ValueError as exc:
            self.get_logger().error("Navigation scan rejected: %s" % exc)
            return
        self.localization_publisher.publish(localization)
        self.publisher.publish(navigation)


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
