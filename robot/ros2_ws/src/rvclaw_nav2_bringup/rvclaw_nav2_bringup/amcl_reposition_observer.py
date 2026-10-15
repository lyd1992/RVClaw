"""Observe a supervised PS2 reposition and export the final AMCL pose."""

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


def summarize_reposition(
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
        failures.append("too few raw odometry samples during reposition")
    if len(baseline_amcl) < 3:
        failures.append("too few baseline AMCL samples")
    if len(motion_amcl) < 2:
        failures.append("too few AMCL updates during reposition")
    if len(post_amcl) < 5:
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

    raw_metrics = {"sample_count": len(raw_motion)}
    if len(raw_motion) >= 2:
        raw_start = raw_motion[0]
        raw_final = raw_motion[-1]
        raw_path = sum(
            _distance(first, second)
            for first, second in zip(raw_motion, raw_motion[1:])
        )
        raw_net = _distance(raw_start, raw_final)
        raw_metrics.update(
            {
                "net_distance_m": raw_net,
                "path_length_m": raw_path,
                "yaw_delta_deg": _yaw_delta_deg(raw_start, raw_final),
                "peak_abs_linear_x_m_s": max(
                    abs(sample["linear_x"]) for sample in raw_motion
                ),
                "peak_abs_angular_z_rad_s": max(
                    abs(sample["angular_z"]) for sample in raw_motion
                ),
            }
        )
        if not 0.20 <= raw_net <= 3.00:
            failures.append("raw net displacement is outside the admitted window")
        if not 0.20 <= raw_path <= 5.00:
            failures.append("raw path length is outside the admitted window")
        if raw_metrics["peak_abs_linear_x_m_s"] > 0.40:
            failures.append("manual linear speed exceeded the admitted limit")
        if raw_metrics["peak_abs_angular_z_rad_s"] > 1.50:
            failures.append("manual angular speed exceeded the admitted limit")

    amcl_metrics = {
        "sample_count": len(amcl_samples),
        "baseline_sample_count": len(baseline_amcl),
        "motion_sample_count": len(motion_amcl),
        "post_stop_sample_count": len(post_amcl),
    }
    recommended_pose = {}
    if baseline_amcl and post_amcl:
        amcl_start = baseline_amcl[-1]
        amcl_final = post_amcl[-1]
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
        yaw_deg = math.degrees(normalize_angle(amcl_final["yaw"]))
        final_pose = {
            key: amcl_final[key]
            for key in ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
        }
        final_pose["yaw_deg"] = yaw_deg
        amcl_metrics.update(
            {
                "net_distance_m": displacement,
                "yaw_delta_deg": _yaw_delta_deg(amcl_start, amcl_final),
                "max_consecutive_position_jump_m": max_jump,
                "max_consecutive_yaw_jump_deg": max_yaw_jump,
                "post_stop_position_drift_m": post_position_drift,
                "post_stop_yaw_drift_deg": post_yaw_drift,
                "final_pose": final_pose,
            }
        )
        recommended_pose = {
            "RVCLAW_NAV2_INITIAL_X": amcl_final["x"],
            "RVCLAW_NAV2_INITIAL_Y": amcl_final["y"],
            "RVCLAW_NAV2_INITIAL_YAW_DEG": yaw_deg,
            "RVCLAW_NAV2_INITIAL_POSITION_STD_M": 0.15,
            "RVCLAW_NAV2_INITIAL_YAW_STD_DEG": 10.0,
        }
        if not 0.15 <= displacement <= 3.00:
            failures.append("AMCL displacement is outside the admitted window")
        if max_jump > 0.75:
            failures.append("AMCL consecutive position jump exceeded the limit")
        if max_yaw_jump > 60.0:
            failures.append("AMCL consecutive yaw jump exceeded the limit")
        if post_position_drift > 0.12:
            failures.append("post-stop AMCL position drift exceeded the limit")
        if post_yaw_drift > 12.0:
            failures.append("post-stop AMCL yaw drift exceeded the limit")
        if raw_metrics.get("net_distance_m") is not None and abs(
            displacement - raw_metrics["net_distance_m"]
        ) > 0.75:
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
        "recommended_nav2_initial_pose": recommended_pose,
        "transforms": transforms,
        "failures": failures,
    }


def write_pose_env(path, values):
    keys = (
        "RVCLAW_NAV2_INITIAL_X",
        "RVCLAW_NAV2_INITIAL_Y",
        "RVCLAW_NAV2_INITIAL_YAW_DEG",
        "RVCLAW_NAV2_INITIAL_POSITION_STD_M",
        "RVCLAW_NAV2_INITIAL_YAW_STD_DEG",
    )
    if any(key not in values for key in keys):
        raise ValueError("recommended Nav2 initial pose is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"export {key}={values[key]:.9f}\n" for key in keys),
        encoding="ascii",
    )


def parser():
    result = argparse.ArgumentParser(
        description="Observe one supervised PS2 map reposition without commands"
    )
    result.add_argument("--ready-file", type=Path, required=True)
    result.add_argument("--stop-file", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pose-env-file", type=Path, required=True)
    result.add_argument("--timeout", type=float, default=300.0)
    result.add_argument("--baseline-updates", type=int, default=5)
    result.add_argument("--post-stop-updates", type=int, default=8)
    result.add_argument("--update-interval", type=float, default=1.0)
    return result


def main(args=None):
    options, ros_args = parser().parse_known_args(args)
    if options.timeout < 60.0 or options.timeout > 300.0:
        raise SystemExit("timeout must be in [60, 300]")
    if not 3 <= options.baseline_updates <= 10:
        raise SystemExit("baseline-updates must be in [3, 10]")
    if not 5 <= options.post_stop_updates <= 12:
        raise SystemExit("post-stop-updates must be in [5, 12]")
    if not 0.2 <= options.update_interval <= 5.0:
        raise SystemExit("update-interval must be in [0.2, 5.0]")

    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformException, TransformListener

    class Observer(Node):
        def __init__(self):
            super().__init__("rvclaw_amcl_reposition_observer")
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
                raise RuntimeError("AMCL reposition observation timed out")
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
        options.ready_file.write_text(
            "AMCL_REPOSITION_OBSERVER_READY\n", encoding="ascii"
        )
        print("AMCL_REPOSITION_OBSERVER_READY", flush=True)

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
        report = summarize_reposition(
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
    write_pose_env(options.pose_env_file, report["recommended_nav2_initial_pose"])
    print(f"pose_env_file={options.pose_env_file}")
    print("AMCL_REPOSITION_OBSERVER_PASS")


if __name__ == "__main__":
    main()
