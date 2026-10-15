#!/usr/bin/env python3
"""Verify that raw odometry remains stationary without publishing."""

import argparse
import json
import math
import sys
import time
from pathlib import Path


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def quaternion_to_yaw(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid orientation quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def summarize(samples, minimum_messages, limits):
    finite = all(
        math.isfinite(value)
        for sample in samples
        for value in (
            sample["stamp_s"],
            sample["x_m"],
            sample["y_m"],
            sample["yaw_rad"],
            sample["linear_x_m_s"],
            sample["linear_y_m_s"],
            sample["angular_z_rad_s"],
        )
    )
    stamps_increasing = all(
        first["stamp_s"] < second["stamp_s"]
        for first, second in zip(samples, samples[1:])
    )
    frames_valid = bool(samples) and all(
        sample["frame_id"] == "odom"
        and sample["child_frame_id"] == "base_footprint"
        for sample in samples
    )

    if samples:
        start = samples[0]
        max_position_drift = max(
            math.hypot(sample["x_m"] - start["x_m"], sample["y_m"] - start["y_m"])
            for sample in samples
        )
        max_yaw_drift_deg = max(
            abs(math.degrees(normalize_angle(sample["yaw_rad"] - start["yaw_rad"])))
            for sample in samples
        )
        peak_linear_x = max(abs(sample["linear_x_m_s"]) for sample in samples)
        peak_linear_y = max(abs(sample["linear_y_m_s"]) for sample in samples)
        peak_angular_z = max(abs(sample["angular_z_rad_s"]) for sample in samples)
        duration = samples[-1]["stamp_s"] - start["stamp_s"]
    else:
        max_position_drift = math.inf
        max_yaw_drift_deg = math.inf
        peak_linear_x = math.inf
        peak_linear_y = math.inf
        peak_angular_z = math.inf
        duration = 0.0

    checks = {
        "message_count": len(samples) >= minimum_messages,
        "frames": frames_valid,
        "finite_values": finite,
        "timestamps_increasing": len(samples) >= 2 and stamps_increasing,
        "linear_x_stationary": peak_linear_x <= limits["max_linear_m_s"],
        "linear_y_stationary": peak_linear_y <= limits["max_linear_m_s"],
        "angular_z_stationary": peak_angular_z <= limits["max_angular_rad_s"],
        "position_drift": max_position_drift <= limits["max_position_drift_m"],
        "yaw_drift": max_yaw_drift_deg <= limits["max_yaw_drift_deg"],
    }
    return {
        "sample_count": len(samples),
        "duration_s": round(duration, 6),
        "peak_abs_linear_x_m_s": peak_linear_x,
        "peak_abs_linear_y_m_s": peak_linear_y,
        "peak_abs_angular_z_rad_s": peak_angular_z,
        "max_position_drift_m": max_position_drift,
        "max_yaw_drift_deg": max_yaw_drift_deg,
        "checks": checks,
    }


def capture(duration, raw_topic):
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    class Observer(Node):
        def __init__(self):
            super().__init__("rvclaw_navigation_stationary_observer")
            self.raw = []
            self.message_subscriptions = [
                self.create_subscription(
                    Odometry,
                    raw_topic,
                    lambda message: self._store(message, self.raw),
                    qos_profile_sensor_data,
                ),
            ]

        @staticmethod
        def _store(message, destination):
            pose = message.pose.pose
            twist = message.twist.twist
            stamp = message.header.stamp
            values = (
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
            try:
                yaw = quaternion_to_yaw(*values[2:6])
            except ValueError:
                yaw = math.nan
            destination.append(
                {
                    "stamp_s": float(stamp.sec) + float(stamp.nanosec) / 1e9,
                    "frame_id": message.header.frame_id,
                    "child_frame_id": message.child_frame_id,
                    "x_m": float(values[0]),
                    "y_m": float(values[1]),
                    "yaw_rad": yaw,
                    "linear_x_m_s": float(values[6]),
                    "linear_y_m_s": float(values[7]),
                    "angular_z_rad_s": float(values[8]),
                }
            )

    rclpy.init()
    node = Observer()
    deadline = time.monotonic() + duration
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        return list(node.raw)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Observe stationary odometry; this tool has no publishers"
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--minimum-messages", type=int, default=50)
    parser.add_argument("--raw-topic", default="/odom")
    parser.add_argument("--max-linear-m-s", type=float, default=0.03)
    parser.add_argument("--max-angular-rad-s", type=float, default=0.08)
    parser.add_argument("--max-position-drift-m", type=float, default=0.02)
    parser.add_argument("--max-yaw-drift-deg", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test(options):
    samples = []
    for index in range(100):
        samples.append(
            {
                "stamp_s": index * 0.05,
                "frame_id": "odom",
                "child_frame_id": "base_footprint",
                "x_m": index * 1e-6,
                "y_m": 0.0,
                "yaw_rad": 0.0,
                "linear_x_m_s": 0.0,
                "linear_y_m_s": 0.0,
                "angular_z_rad_s": 0.0,
            }
        )
    limits = limits_from_options(options)
    result = summarize(samples, options.minimum_messages, limits)
    if not all(result["checks"].values()):
        raise AssertionError(result)
    print("NAVIGATION_STATIONARY_SELF_TEST_PASS")
    return 0


def limits_from_options(options):
    return {
        "max_linear_m_s": options.max_linear_m_s,
        "max_angular_rad_s": options.max_angular_rad_s,
        "max_position_drift_m": options.max_position_drift_m,
        "max_yaw_drift_deg": options.max_yaw_drift_deg,
    }


def main():
    options = parse_args()
    if options.self_test:
        return run_self_test(options)
    if options.duration <= 0.0 or options.duration > 120.0:
        print("duration must be in (0, 120]", file=sys.stderr)
        return 2
    if options.minimum_messages < 2:
        print("minimum-messages must be at least 2", file=sys.stderr)
        return 2

    limits = limits_from_options(options)
    if any(not math.isfinite(value) or value <= 0.0 for value in limits.values()):
        print("all limits must be finite and positive", file=sys.stderr)
        return 2

    print("SUBSCRIPTION_ONLY=true")
    print("COMMAND_PUBLISHERS=0")
    raw = capture(options.duration, options.raw_topic)
    result = {
        "status": "PASS",
        "subscription_only": True,
        "command_publishers": 0,
        "duration_seconds": options.duration,
        "limits": limits,
        "raw_odom": summarize(raw, options.minimum_messages, limits),
    }
    if not all(result["raw_odom"]["checks"].values()):
        result["status"] = "FAIL"

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output={options.output}")
    if result["status"] == "PASS":
        print("NAVIGATION_STATIONARY_PASS")
        return 0
    print("NAVIGATION_STATIONARY_FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
