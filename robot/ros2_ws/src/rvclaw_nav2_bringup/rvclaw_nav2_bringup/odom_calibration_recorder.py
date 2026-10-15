"""Subscribe to odometry and write a calibration trace without commanding motion."""

import argparse
import json
from pathlib import Path
import time
from typing import List, Optional

from .calibration_math import CalibrationError
from .calibration_math import OdomSample
from .calibration_math import calibration_recommendation
from .calibration_math import quaternion_to_yaw
from .calibration_math import summarize_dominant_turn
from .calibration_math import summarize_samples


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record an existing ROS odometry topic; no command is published."
    )
    parser.add_argument("--motion", choices=("straight", "turn"), required=True)
    parser.add_argument("--topic", default="/odom")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--actual-distance-m", type=float)
    parser.add_argument("--actual-angle-deg", type=float)
    return parser


def main(args=None) -> None:
    parser = _parser()
    options, ros_args = parser.parse_known_args(args)
    if options.duration <= 0.0 or options.duration > 600.0:
        parser.error("--duration must be in (0, 600]")

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("rvclaw_odom_calibration_recorder")
            self.samples: List[OdomSample] = []
            self.subscription = self.create_subscription(
                Odometry,
                options.topic,
                self._on_odom,
                qos_profile_sensor_data,
            )

        def _on_odom(self, message: Odometry) -> None:
            stamp = message.header.stamp
            time_s = float(stamp.sec) + float(stamp.nanosec) / 1e9
            pose = message.pose.pose
            twist = message.twist.twist
            try:
                yaw = quaternion_to_yaw(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
            except CalibrationError as error:
                self.get_logger().error(f"rejected odometry sample: {error}")
                return
            self.samples.append(
                OdomSample(
                    time_s=time_s,
                    x_m=pose.position.x,
                    y_m=pose.position.y,
                    yaw_rad=yaw,
                    linear_x_m_s=twist.linear.x,
                    angular_z_rad_s=twist.angular.z,
                )
            )

    print("SUBSCRIPTION_ONLY=true")
    print("COMMAND_PUBLISHERS=0")
    print(f"recording {options.topic} for {options.duration:g} seconds")
    rclpy.init(args=ros_args)
    node: Optional[Recorder] = Recorder()
    deadline = time.monotonic() + options.duration
    interrupted = False
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        samples = list(node.samples)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    try:
        summary = summarize_samples(samples, options.motion)
        motion_segment_summary = None
        recommendation_summary = summary
        if options.motion == "turn":
            motion_segment_summary = summarize_dominant_turn(samples)
            recommendation_summary = motion_segment_summary
        recommendation = None
        if options.actual_distance_m is not None or options.actual_angle_deg is not None:
            recommendation = calibration_recommendation(
                recommendation_summary,
                actual_distance_m=options.actual_distance_m,
                actual_angle_deg=options.actual_angle_deg,
            )
    except CalibrationError as error:
        parser.exit(2, f"calibration trace rejected: {error}\n")

    artifact = {
        "schema_version": 2,
        "subscription_only": True,
        "command_publishers": 0,
        "topic": options.topic,
        "interrupted": interrupted,
        "summary": summary,
        "motion_segment_summary": motion_segment_summary,
        "recommendation": recommendation,
        "samples": [sample.__dict__ for sample in samples],
    }
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"samples={len(samples)}")
    print(f"output={output}")
    if motion_segment_summary is not None:
        print(
            "motion_segment_yaw_change_deg="
            f"{motion_segment_summary['yaw_change_deg']:.6f}"
        )
    print("review_required=true")


if __name__ == "__main__":
    main()
