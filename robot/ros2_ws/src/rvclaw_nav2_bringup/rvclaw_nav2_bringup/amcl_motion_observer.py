"""Observe a short PS2 motion while AMCL remains subscription-only."""

import argparse
import json
import math
from pathlib import Path
import time

from rvclaw_nav2_bringup.amcl_localization_observer import normalize_angle
from rvclaw_nav2_bringup.amcl_localization_observer import quaternion_to_yaw


def _distance(first, second):
    return math.hypot(second["x"] - first["x"], second["y"] - first["y"])


def _yaw_delta_deg(first, second):
    return math.degrees(normalize_angle(second["yaw"] - first["yaw"]))


def _body_delta(first, second):
    dx = second["x"] - first["x"]
    dy = second["y"] - first["y"]
    cosine = math.cos(first["yaw"])
    sine = math.sin(first["yaw"])
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def summarize_motion(
    raw_samples,
    amcl_samples,
    raw_start_index,
    raw_stop_index,
    amcl_baseline_end_index,
    amcl_stop_index,
    transforms,
):
    failures = []
    raw_motion = raw_samples[max(0, raw_start_index - 1):raw_stop_index]
    baseline_amcl = amcl_samples[:amcl_baseline_end_index]
    motion_amcl = amcl_samples[amcl_baseline_end_index:amcl_stop_index]
    post_amcl = amcl_samples[amcl_stop_index:]

    if len(raw_motion) < 20:
        failures.append("too few raw odometry samples during the motion window")
    if len(baseline_amcl) < 3:
        failures.append("too few baseline AMCL samples")
    if len(motion_amcl) < 2:
        failures.append("too few AMCL updates during motion")
    if len(post_amcl) < 3:
        failures.append("too few post-stop AMCL samples")

    raw_fields = (
        "stamp_s",
        "x",
        "y",
        "yaw",
        "linear_x",
        "linear_y",
        "angular_z",
    )
    amcl_fields = ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
    if not all(
        math.isfinite(sample[field])
        for sample in raw_samples
        for field in raw_fields
    ):
        failures.append("raw odometry contains a non-finite value")
    if not all(
        math.isfinite(sample[field])
        for sample in amcl_samples
        for field in amcl_fields
    ):
        failures.append("AMCL contains a non-finite value")
    if raw_samples and not all(
        sample["frame_id"] == "odom"
        and sample["child_frame_id"] == "base_footprint"
        for sample in raw_samples
    ):
        failures.append("raw odometry frame contract failed")
    if amcl_samples and not all(sample["frame_id"] == "map" for sample in amcl_samples):
        failures.append("AMCL frame contract failed")
    if any(
        second["stamp_s"] <= first["stamp_s"]
        for first, second in zip(raw_samples, raw_samples[1:])
    ):
        failures.append("raw odometry timestamps did not increase")
    if any(
        second["stamp_s"] < first["stamp_s"]
        for first, second in zip(amcl_samples, amcl_samples[1:])
    ):
        failures.append("AMCL timestamps decreased")
    if any(
        sample[field] < 0.0
        for sample in amcl_samples
        for field in ("cov_x", "cov_y", "cov_yaw")
    ):
        failures.append("AMCL covariance contains a negative diagonal")

    raw_metrics = {}
    if len(raw_motion) >= 2:
        raw_start = raw_motion[0]
        raw_final = raw_motion[-1]
        forward, lateral = _body_delta(raw_start, raw_final)
        raw_path = sum(
            _distance(first, second)
            for first, second in zip(raw_motion, raw_motion[1:])
        )
        raw_yaw = _yaw_delta_deg(raw_start, raw_final)
        raw_metrics = {
            "sample_count": len(raw_motion),
            "forward_delta_m": forward,
            "lateral_delta_m": lateral,
            "net_distance_m": _distance(raw_start, raw_final),
            "path_length_m": raw_path,
            "yaw_delta_deg": raw_yaw,
            "peak_abs_linear_x_m_s": max(
                abs(sample["linear_x"]) for sample in raw_motion
            ),
            "peak_abs_angular_z_rad_s": max(
                abs(sample["angular_z"]) for sample in raw_motion
            ),
        }
        if not 0.10 <= forward <= 0.80:
            failures.append("raw forward displacement is outside the admitted window")
        if abs(lateral) > 0.25:
            failures.append("raw lateral displacement exceeded the admitted window")
        if abs(raw_yaw) > 30.0:
            failures.append("raw yaw change exceeded the admitted window")

    amcl_metrics = {
        "sample_count": len(amcl_samples),
        "baseline_sample_count": len(baseline_amcl),
        "motion_sample_count": len(motion_amcl),
        "post_stop_sample_count": len(post_amcl),
    }
    if baseline_amcl and post_amcl:
        amcl_start = baseline_amcl[-1]
        amcl_final = post_amcl[-1]
        forward, lateral = _body_delta(amcl_start, amcl_final)
        displacement = _distance(amcl_start, amcl_final)
        max_jump = max(
            (
                _distance(first, second)
                for first, second in zip(amcl_samples, amcl_samples[1:])
            ),
            default=0.0,
        )
        max_yaw_jump = max(
            (
                abs(_yaw_delta_deg(first, second))
                for first, second in zip(amcl_samples, amcl_samples[1:])
            ),
            default=0.0,
        )
        post_position_drift = max(
            (_distance(post_amcl[0], sample) for sample in post_amcl),
            default=0.0,
        )
        post_yaw_drift = max(
            (abs(_yaw_delta_deg(post_amcl[0], sample)) for sample in post_amcl),
            default=0.0,
        )
        amcl_metrics.update(
            {
                "forward_delta_m": forward,
                "lateral_delta_m": lateral,
                "net_distance_m": displacement,
                "yaw_delta_deg": _yaw_delta_deg(amcl_start, amcl_final),
                "max_consecutive_position_jump_m": max_jump,
                "max_consecutive_yaw_jump_deg": max_yaw_jump,
                "post_stop_position_drift_m": post_position_drift,
                "post_stop_yaw_drift_deg": post_yaw_drift,
                "final_pose": {
                    key: amcl_final[key]
                    for key in ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
                },
            }
        )
        if not 0.08 <= displacement <= 1.00:
            failures.append("AMCL displacement is outside the admitted window")
        if max_jump > 0.50:
            failures.append("AMCL consecutive position jump exceeded the limit")
        if max_yaw_jump > 30.0:
            failures.append("AMCL consecutive yaw jump exceeded the limit")
        if post_position_drift > 0.15:
            failures.append("post-stop AMCL position drift exceeded the limit")
        if post_yaw_drift > 10.0:
            failures.append("post-stop AMCL yaw drift exceeded the limit")
        if raw_metrics and abs(displacement - raw_metrics["net_distance_m"]) > 0.35:
            failures.append("AMCL and raw odometry displacement disagree excessively")

    failures.extend(
        f"missing transform: {name}"
        for name, available in transforms.items()
        if not available
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "raw_odom": raw_metrics,
        "amcl": amcl_metrics,
        "transforms": transforms,
        "failures": failures,
    }


def parser():
    result = argparse.ArgumentParser(
        description="Observe one short PS2 forward motion without publishing commands"
    )
    result.add_argument("--ready-file", type=Path, required=True)
    result.add_argument("--stop-file", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--baseline-updates", type=int, default=3)
    result.add_argument("--post-stop-updates", type=int, default=5)
    result.add_argument("--update-interval", type=float, default=1.0)
    return result


def main(args=None):
    options, ros_args = parser().parse_known_args(args)
    if options.timeout < 30.0 or options.timeout > 300.0:
        raise SystemExit("timeout must be in [30, 300]")
    if not 3 <= options.baseline_updates <= 10:
        raise SystemExit("baseline-updates must be in [3, 10]")
    if not 3 <= options.post_stop_updates <= 10:
        raise SystemExit("post-stop-updates must be in [3, 10]")
    if not 0.2 <= options.update_interval <= 5.0:
        raise SystemExit("update-interval must be in [0.2, 5.0]")

    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformException, TransformListener

    class Observer(Node):
        def __init__(self):
            super().__init__("rvclaw_amcl_motion_observer")
            self.raw_samples = []
            self.amcl_samples = []
            amcl_qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                Odometry, "/odom", self._on_odom, qos_profile_sensor_data
            )
            self.create_subscription(
                PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl, amcl_qos
            )
            self.nomotion_client = self.create_client(Empty, "/request_nomotion_update")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        def _on_odom(self, message):
            pose = message.pose.pose
            twist = message.twist.twist
            try:
                yaw = quaternion_to_yaw(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
            except ValueError:
                yaw = math.nan
            self.raw_samples.append(
                {
                    "stamp_s": float(message.header.stamp.sec)
                    + float(message.header.stamp.nanosec) / 1e9,
                    "frame_id": message.header.frame_id,
                    "child_frame_id": message.child_frame_id,
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "yaw": yaw,
                    "linear_x": float(twist.linear.x),
                    "linear_y": float(twist.linear.y),
                    "angular_z": float(twist.angular.z),
                }
            )

        def _on_amcl(self, message):
            pose = message.pose.pose
            covariance = message.pose.covariance
            try:
                yaw = quaternion_to_yaw(
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
            except ValueError:
                yaw = math.nan
            self.amcl_samples.append(
                {
                    "stamp_s": float(message.header.stamp.sec)
                    + float(message.header.stamp.nanosec) / 1e9,
                    "frame_id": message.header.frame_id,
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "yaw": yaw,
                    "cov_x": float(covariance[0]),
                    "cov_y": float(covariance[7]),
                    "cov_yaw": float(covariance[35]),
                }
            )

        def transform_available(self, target, source):
            try:
                self.tf_buffer.lookup_transform(
                    target, source, Time(), timeout=Duration(seconds=1.0)
                )
                return True
            except TransformException:
                return False

    def spin_for(node, duration, deadline):
        interval_deadline = min(deadline, time.monotonic() + duration)
        while rclpy.ok() and time.monotonic() < interval_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

    def request_updates(node, count, deadline):
        for _ in range(count):
            future = node.nomotion_client.call_async(Empty.Request())
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("AMCL motion observation timed out")
            rclpy.spin_until_future_complete(
                node, future, timeout_sec=min(3.0, remaining)
            )
            if not future.done() or future.exception() is not None:
                raise RuntimeError("AMCL no-motion update failed")
            spin_for(node, options.update_interval, deadline)

    print("SUBSCRIPTION_ONLY=true")
    print("COMMAND_PUBLISHERS=0")
    print("HARDWARE_INTERFACES=0")
    rclpy.init(args=ros_args)
    node = Observer()
    deadline = time.monotonic() + options.timeout
    report = None
    try:
        while (
            rclpy.ok()
            and (not node.raw_samples or not node.nomotion_client.service_is_ready())
            and time.monotonic() < deadline
        ):
            node.nomotion_client.wait_for_service(timeout_sec=0.1)
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.raw_samples:
            raise RuntimeError("raw odometry was not received")
        if not node.nomotion_client.service_is_ready():
            raise RuntimeError("AMCL no-motion update service is unavailable")

        node.amcl_samples.clear()
        request_updates(node, options.baseline_updates, deadline)
        if len(node.amcl_samples) < 3:
            raise RuntimeError("AMCL baseline did not produce three samples")
        amcl_baseline_end_index = len(node.amcl_samples)
        raw_start_index = len(node.raw_samples)
        options.ready_file.parent.mkdir(parents=True, exist_ok=True)
        options.ready_file.write_text("AMCL_MOTION_OBSERVER_READY\n", encoding="utf-8")
        print("AMCL_MOTION_OBSERVER_READY", flush=True)

        while (
            rclpy.ok()
            and not options.stop_file.exists()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if not options.stop_file.exists():
            raise RuntimeError("operator stop marker was not received before timeout")
        raw_stop_index = len(node.raw_samples)
        amcl_stop_index = len(node.amcl_samples)
        request_updates(node, options.post_stop_updates, deadline)

        transforms = {
            "map->odom": node.transform_available("map", "odom"),
            "map->base_footprint": node.transform_available("map", "base_footprint"),
        }
        report = summarize_motion(
            node.raw_samples,
            node.amcl_samples,
            raw_start_index,
            raw_stop_index,
            amcl_baseline_end_index,
            amcl_stop_index,
            transforms,
        )
        report.update(
            {
                "subscription_only": True,
                "command_publishers": 0,
                "hardware_interfaces": 0,
            }
        )
    except Exception as error:
        report = {
            "status": "FAIL",
            "subscription_only": True,
            "command_publishers": 0,
            "hardware_interfaces": 0,
            "failures": [str(error)],
        }
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output={options.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)
    print("AMCL_MOTION_OBSERVER_PASS")


if __name__ == "__main__":
    main()
