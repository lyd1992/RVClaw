"""Observe scan, odometry, and TF contracts without starting or controlling hardware."""

import argparse
import json
from pathlib import Path
import time
from typing import Optional

from .readiness_contract import build_readiness_result
from .readiness_contract import inspect_odom
from .readiness_contract import inspect_scan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe an already-running navigation sensor chain; no command is published."
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--minimum-messages", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--raw-odom-topic", default="/odom")
    return parser


def _odom_values(message):
    pose = message.pose.pose
    twist = message.twist.twist
    return (
        pose.position.x,
        pose.position.y,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
        twist.linear.x,
        twist.linear.y,
        twist.angular.z,
    )


def main(args=None) -> None:
    parser = _parser()
    options, ros_args = parser.parse_known_args(args)
    if options.timeout <= 0.0 or options.timeout > 120.0:
        parser.error("--timeout must be in (0, 120]")
    if options.minimum_messages < 1:
        parser.error("--minimum-messages must be positive")

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener

    class Observer(Node):
        def __init__(self) -> None:
            super().__init__("rvclaw_navigation_readiness_observer")
            self.scan_count = 0
            self.raw_count = 0
            self.scan_report = {"issues": ["no scan received"]}
            self.raw_report = {"issues": ["no raw odometry received"]}
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.message_subscriptions = [
                self.create_subscription(
                    LaserScan,
                    options.scan_topic,
                    self._on_scan,
                    qos_profile_sensor_data,
                ),
                self.create_subscription(
                    Odometry,
                    options.raw_odom_topic,
                    self._on_raw,
                    qos_profile_sensor_data,
                ),
            ]

        def _on_scan(self, message: LaserScan) -> None:
            self.scan_count += 1
            self.scan_report = inspect_scan(
                message.header.frame_id,
                message.angle_min,
                message.angle_max,
                message.angle_increment,
                message.range_min,
                message.range_max,
                message.ranges,
            )

        def _on_raw(self, message: Odometry) -> None:
            self.raw_count += 1
            self.raw_report = inspect_odom(
                message.header.frame_id,
                message.child_frame_id,
                _odom_values(message),
            )

        def transform_available(self, target: str, source: str) -> bool:
            try:
                self.buffer.lookup_transform(
                    target,
                    source,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                return True
            except TransformException:
                return False

    print("SUBSCRIPTION_ONLY=true")
    print("COMMAND_PUBLISHERS=0")
    print("This tool does not launch sensors, Nav2, or the STM32 hardware node.")
    rclpy.init(args=ros_args)
    node: Optional[Observer] = Observer()
    deadline = time.monotonic() + options.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    transforms = {
        "base_footprint->laser": node.transform_available(
            "base_footprint", "laser"
        ),
        "odom->base_footprint": node.transform_available(
            "odom", "base_footprint"
        ),
    }
    report = build_readiness_result(
        node.scan_count,
        node.raw_count,
        options.minimum_messages,
        node.scan_report,
        node.raw_report,
        transforms,
    )
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    print(f"output={output}")
    if report["status"] != "PASS":
        for failure in report["failures"]:
            print(f"failure={failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
