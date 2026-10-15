"""Initialize and observe AMCL without publishing a velocity command."""

import argparse
import json
import math
from pathlib import Path
import time


POSITION_VARIANCE = 0.25
YAW_VARIANCE = math.radians(15.0) ** 2
DEFAULT_POSITION_STD_M = math.sqrt(POSITION_VARIANCE)
DEFAULT_YAW_STD_DEG = 15.0
DEFAULT_GLOBAL_POSITION_DRIFT_M = 0.10
DEFAULT_GLOBAL_YAW_DRIFT_DEG = 10.0
DEFAULT_GLOBAL_POSITION_VARIANCE = 0.10
DEFAULT_GLOBAL_YAW_VARIANCE = math.radians(10.0) ** 2
MAX_INITIAL_YAW_INPUT_DEG = 1000000.0


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def normalize_initial_yaw_deg(value):
    """Accept periodic degree inputs and return ROS-normalized degrees."""
    value = float(value)
    if not math.isfinite(value) or abs(value) > MAX_INITIAL_YAW_INPUT_DEG:
        raise ValueError(
            "initial yaw must be finite and within +/-%.0f degrees"
            % MAX_INITIAL_YAW_INPUT_DEG
        )
    return math.degrees(normalize_angle(math.radians(value)))


def quaternion_to_yaw(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid orientation quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def summarize_pose_samples(
    samples,
    initial_x,
    initial_y,
    initial_yaw,
    minimum_samples=3,
    maximum_initial_position_offset_m=1.0,
    maximum_initial_yaw_offset_deg=45.0,
    maximum_stationary_position_drift_m=0.10,
    maximum_stationary_yaw_drift_deg=10.0,
):
    failures = []
    if len(samples) < minimum_samples:
        failures.append(
            f"amcl_pose received {len(samples)}, requires {minimum_samples}"
        )
    if not samples:
        return {
            "status": "FAIL",
            "sample_count": 0,
            "failures": failures or ["no amcl_pose received"],
        }

    numeric_keys = ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
    if not all(
        math.isfinite(sample[key]) for sample in samples for key in numeric_keys
    ):
        failures.append("amcl_pose contains a non-finite value")
    if not all(sample["frame_id"] == "map" for sample in samples):
        failures.append("amcl_pose frame is not map")
    if not all(
        sample[key] >= 0.0
        for sample in samples
        for key in ("cov_x", "cov_y", "cov_yaw")
    ):
        failures.append("amcl_pose covariance contains a negative diagonal")
    if any(
        second["stamp_s"] < first["stamp_s"]
        for first, second in zip(samples, samples[1:])
    ):
        failures.append("amcl_pose timestamps decreased")

    first = samples[0]
    final = samples[-1]
    stationary_position_drift = max(
        math.hypot(sample["x"] - first["x"], sample["y"] - first["y"])
        for sample in samples
    )
    stationary_yaw_drift_deg = max(
        abs(math.degrees(normalize_angle(sample["yaw"] - first["yaw"])))
        for sample in samples
    )
    initial_position_offset = math.hypot(
        final["x"] - initial_x, final["y"] - initial_y
    )
    initial_yaw_offset_deg = abs(
        math.degrees(normalize_angle(final["yaw"] - initial_yaw))
    )
    if initial_position_offset > maximum_initial_position_offset_m:
        failures.append("localized position is too far from the admitted initial pose")
    if initial_yaw_offset_deg > maximum_initial_yaw_offset_deg:
        failures.append("localized yaw is too far from the admitted initial pose")
    if stationary_position_drift > maximum_stationary_position_drift_m:
        failures.append("stationary localized position drift exceeded the limit")
    if stationary_yaw_drift_deg > maximum_stationary_yaw_drift_deg:
        failures.append("stationary localized yaw drift exceeded the limit")

    return {
        "status": "PASS" if not failures else "FAIL",
        "sample_count": len(samples),
        "frame_ids": sorted({sample["frame_id"] for sample in samples}),
        "initial_position_offset_m": initial_position_offset,
        "initial_yaw_offset_deg": initial_yaw_offset_deg,
        "stationary_position_drift_m": stationary_position_drift,
        "stationary_yaw_drift_deg": stationary_yaw_drift_deg,
        "final_pose": {
            key: final[key]
            for key in ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
        },
        "failures": failures,
    }


def summarize_global_pose_samples(
    samples,
    minimum_samples=5,
    maximum_position_drift_m=DEFAULT_GLOBAL_POSITION_DRIFT_M,
    maximum_yaw_drift_deg=DEFAULT_GLOBAL_YAW_DRIFT_DEG,
    maximum_position_variance=DEFAULT_GLOBAL_POSITION_VARIANCE,
    maximum_yaw_variance=DEFAULT_GLOBAL_YAW_VARIANCE,
):
    """Validate convergence after AMCL global localization without a seed."""
    failures = []
    if len(samples) < minimum_samples:
        failures.append(
            f"amcl_pose received {len(samples)}, requires {minimum_samples}"
        )
    if not samples:
        return {
            "status": "FAIL",
            "sample_count": 0,
            "failures": failures or ["no amcl_pose received"],
        }

    numeric_keys = ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
    if not all(
        math.isfinite(sample[key]) for sample in samples for key in numeric_keys
    ):
        failures.append("amcl_pose contains a non-finite value")
    if not all(sample["frame_id"] == "map" for sample in samples):
        failures.append("amcl_pose frame is not map")
    if any(
        second["stamp_s"] < first["stamp_s"]
        for first, second in zip(samples, samples[1:])
    ):
        failures.append("amcl_pose timestamps decreased")

    first = samples[0]
    final = samples[-1]
    position_drift = max(
        math.hypot(sample["x"] - first["x"], sample["y"] - first["y"])
        for sample in samples
    )
    yaw_drift_deg = max(
        abs(math.degrees(normalize_angle(sample["yaw"] - first["yaw"])))
        for sample in samples
    )
    if position_drift > maximum_position_drift_m:
        failures.append("globally localized position did not settle")
    if yaw_drift_deg > maximum_yaw_drift_deg:
        failures.append("globally localized yaw did not settle")
    if final["cov_x"] > maximum_position_variance or final["cov_y"] > maximum_position_variance:
        failures.append("globally localized position covariance remains high")
    if final["cov_yaw"] > maximum_yaw_variance:
        failures.append("globally localized yaw covariance remains high")

    return {
        "status": "PASS" if not failures else "FAIL",
        "sample_count": len(samples),
        "position_drift_m": position_drift,
        "yaw_drift_deg": yaw_drift_deg,
        "final_pose": {
            key: final[key]
            for key in ("stamp_s", "x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
        },
        "failures": failures,
    }


def parser():
    result = argparse.ArgumentParser(
        description="Publish one initial pose and observe stationary AMCL updates"
    )
    result.add_argument("--initial-x", type=float, default=0.0)
    result.add_argument("--initial-y", type=float, default=0.0)
    result.add_argument("--initial-yaw-deg", type=float, default=0.0)
    result.add_argument(
        "--initial-position-std-m",
        type=float,
        default=DEFAULT_POSITION_STD_M,
    )
    result.add_argument(
        "--initial-yaw-std-deg",
        type=float,
        default=DEFAULT_YAW_STD_DEG,
    )
    result.add_argument("--updates", type=int, default=5)
    result.add_argument("--update-interval", type=float, default=1.0)
    result.add_argument("--timeout", type=float, default=20.0)
    result.add_argument(
        "--max-stationary-position-drift-m",
        type=float,
        default=0.10,
    )
    result.add_argument(
        "--max-stationary-yaw-drift-deg",
        type=float,
        default=10.0,
    )
    result.add_argument(
        "--global-localization",
        action="store_true",
        help="request AMCL global localization instead of publishing an initial pose",
    )
    result.add_argument(
        "--max-global-position-drift-m",
        type=float,
        default=DEFAULT_GLOBAL_POSITION_DRIFT_M,
    )
    result.add_argument(
        "--max-global-yaw-drift-deg",
        type=float,
        default=DEFAULT_GLOBAL_YAW_DRIFT_DEG,
    )
    result.add_argument(
        "--max-global-position-variance",
        type=float,
        default=DEFAULT_GLOBAL_POSITION_VARIANCE,
    )
    result.add_argument(
        "--max-global-yaw-variance",
        type=float,
        default=DEFAULT_GLOBAL_YAW_VARIANCE,
    )
    result.add_argument("--output", type=Path, required=True)
    return result


def main(args=None):
    options, ros_args = parser().parse_known_args(args)
    values = (options.initial_x, options.initial_y, options.initial_yaw_deg)
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("initial pose values must be finite")
    if abs(options.initial_x) > 20.0 or abs(options.initial_y) > 20.0:
        raise SystemExit("initial position must remain within 20 m of map origin")
    try:
        normalized_initial_yaw_deg = normalize_initial_yaw_deg(
            options.initial_yaw_deg
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if (
        not math.isfinite(options.initial_position_std_m)
        or options.initial_position_std_m <= 0.0
        or options.initial_position_std_m > 20.0
    ):
        raise SystemExit("initial position std must be in (0, 20] m")
    if (
        not math.isfinite(options.initial_yaw_std_deg)
        or options.initial_yaw_std_deg <= 0.0
        or options.initial_yaw_std_deg > 180.0
    ):
        raise SystemExit("initial yaw std must be in (0, 180] degrees")
    if options.updates < 3 or options.updates > 20:
        raise SystemExit("updates must be in [3, 20]")
    if options.update_interval <= 0.0 or options.update_interval > 10.0:
        raise SystemExit("update interval must be in (0, 10]")
    if options.timeout <= 0.0 or options.timeout > 120.0:
        raise SystemExit("timeout must be in (0, 120]")
    if (
        not math.isfinite(options.max_stationary_position_drift_m)
        or options.max_stationary_position_drift_m <= 0.0
    ):
        raise SystemExit("maximum stationary position drift must be positive")
    if (
        not math.isfinite(options.max_stationary_yaw_drift_deg)
        or options.max_stationary_yaw_drift_deg <= 0.0
    ):
        raise SystemExit("maximum stationary yaw drift must be positive")
    if (
        not math.isfinite(options.max_global_position_drift_m)
        or options.max_global_position_drift_m <= 0.0
        or not math.isfinite(options.max_global_yaw_drift_deg)
        or options.max_global_yaw_drift_deg <= 0.0
        or not math.isfinite(options.max_global_position_variance)
        or options.max_global_position_variance <= 0.0
        or not math.isfinite(options.max_global_yaw_variance)
        or options.max_global_yaw_variance <= 0.0
    ):
        raise SystemExit("global localization limits must be finite and positive")

    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from nav2_msgs.srv import SetInitialPose
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_srvs.srv import Empty
    from tf2_ros import Buffer, TransformException, TransformListener

    class Observer(Node):
        def __init__(self):
            super().__init__("rvclaw_amcl_localization_observer")
            qos = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.samples = []
            self.initial_pose_client = self.create_client(
                SetInitialPose, "/set_initial_pose"
            )
            self.global_localization_client = self.create_client(
                Empty, "/reinitialize_global_localization"
            )
            self.create_subscription(
                PoseWithCovarianceStamped,
                "/amcl_pose",
                self._on_pose,
                qos,
            )
            self.nomotion_client = self.create_client(Empty, "/request_nomotion_update")
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        def _on_pose(self, message):
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
            self.samples.append(
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
                    target,
                    source,
                    Time(),
                    timeout=Duration(seconds=1.0),
                )
                return True
            except TransformException:
                return False

    print("INITIAL_POSE_SERVICE_ONLY=%s" % (not options.global_localization))
    print("VELOCITY_PUBLISHERS=0")
    print("HARDWARE_INTERFACES=0")
    rclpy.init(args=ros_args)
    node = Observer()
    deadline = time.monotonic() + options.timeout
    try:
        if not node.nomotion_client.wait_for_service(
            timeout_sec=max(0.1, deadline - time.monotonic())
        ):
            raise RuntimeError("AMCL request_nomotion_update service is unavailable")

        node.samples.clear()
        initial_pose_service_response = None
        localization_mode = "global"
        if options.global_localization:
            if not node.global_localization_client.wait_for_service(
                timeout_sec=max(0.1, deadline - time.monotonic())
            ):
                raise RuntimeError(
                    "AMCL reinitialize_global_localization service is unavailable"
                )
            global_future = node.global_localization_client.call_async(Empty.Request())
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("AMCL global localization service timed out")
            rclpy.spin_until_future_complete(
                node,
                global_future,
                timeout_sec=min(5.0, remaining),
            )
            if not global_future.done() or global_future.exception() is not None:
                raise RuntimeError("AMCL global localization service failed")
            initial_pose_service_response = "global_localization_requested"
        else:
            localization_mode = "seeded"
            if not node.initial_pose_client.wait_for_service(
                timeout_sec=max(0.1, deadline - time.monotonic())
            ):
                raise RuntimeError("AMCL set_initial_pose service is unavailable")
            yaw = math.radians(normalized_initial_yaw_deg)
            initial_pose = PoseWithCovarianceStamped()
            initial_pose.header.frame_id = "map"
            initial_pose.pose.pose.position.x = options.initial_x
            initial_pose.pose.pose.position.y = options.initial_y
            initial_pose.pose.pose.orientation.z = math.sin(yaw / 2.0)
            initial_pose.pose.pose.orientation.w = math.cos(yaw / 2.0)
            initial_pose.pose.covariance[0] = options.initial_position_std_m ** 2
            initial_pose.pose.covariance[7] = options.initial_position_std_m ** 2
            initial_pose.pose.covariance[35] = math.radians(
                options.initial_yaw_std_deg
            ) ** 2
            initial_pose.header.stamp = node.get_clock().now().to_msg()
            initial_pose_stamp_s = (
                float(initial_pose.header.stamp.sec)
                + float(initial_pose.header.stamp.nanosec) / 1e9
            )
            initial_pose_request = SetInitialPose.Request()
            initial_pose_request.pose = initial_pose
            initial_pose_future = node.initial_pose_client.call_async(
                initial_pose_request
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("AMCL initial pose service timed out")
            rclpy.spin_until_future_complete(
                node,
                initial_pose_future,
                timeout_sec=min(3.0, remaining),
            )
            initial_pose_service_response = "ok"
            if initial_pose_future.done():
                if initial_pose_future.exception() is not None:
                    raise RuntimeError("AMCL initial pose service failed")
            else:
            # Some Nav2/RMW combinations apply the pose but fail to deliver
            # the empty service response while AMCL is updating map->odom.
            # Accept that case only when fresh pose and TF evidence prove that
            # the requested initialization actually took effect.
                initial_pose_service_response = "timeout_pending_evidence"
                evidence_deadline = min(deadline, time.monotonic() + 5.0)
                evidence_confirmed = False
                while rclpy.ok() and time.monotonic() < evidence_deadline:
                    rclpy.spin_once(node, timeout_sec=0.1)
                    recent_samples = [
                        sample
                        for sample in node.samples
                        if sample["stamp_s"] >= initial_pose_stamp_s - 0.5
                    ]
                    if recent_samples:
                        latest = recent_samples[-1]
                        position_error = math.hypot(
                            latest["x"] - options.initial_x,
                            latest["y"] - options.initial_y,
                        )
                        yaw_error = abs(
                            math.degrees(
                                normalize_angle(latest["yaw"] - yaw)
                            )
                        )
                        if (
                            position_error <= 0.30
                            and yaw_error <= 25.0
                            and node.transform_available(
                                "map", "base_footprint"
                            )
                        ):
                            evidence_confirmed = True
                            initial_pose_service_response = (
                                "timeout_recovered_by_pose_tf_evidence"
                            )
                            break
                if not evidence_confirmed:
                    raise RuntimeError("AMCL initial pose service failed")

        for _ in range(options.updates):
            future = node.nomotion_client.call_async(Empty.Request())
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("AMCL observation timed out")
            rclpy.spin_until_future_complete(
                node,
                future,
                timeout_sec=min(3.0, remaining),
            )
            if not future.done() or future.exception() is not None:
                raise RuntimeError("AMCL no-motion update failed")
            interval_deadline = min(
                deadline, time.monotonic() + options.update_interval
            )
            while rclpy.ok() and time.monotonic() < interval_deadline:
                rclpy.spin_once(node, timeout_sec=0.1)

        transforms = {
            "map->odom": node.transform_available("map", "odom"),
            "map->base_footprint": node.transform_available(
                "map", "base_footprint"
            ),
        }
        if options.global_localization:
            localization = summarize_global_pose_samples(
                node.samples,
                maximum_position_drift_m=options.max_global_position_drift_m,
                maximum_yaw_drift_deg=options.max_global_yaw_drift_deg,
                maximum_position_variance=options.max_global_position_variance,
                maximum_yaw_variance=options.max_global_yaw_variance,
            )
        else:
            localization = summarize_pose_samples(
                node.samples,
                options.initial_x,
                options.initial_y,
                yaw,
                maximum_stationary_position_drift_m=(
                    options.max_stationary_position_drift_m
                ),
                maximum_stationary_yaw_drift_deg=(
                    options.max_stationary_yaw_drift_deg
                ),
            )
        failures = list(localization["failures"])
        failures.extend(
            f"missing transform: {name}"
            for name, available in transforms.items()
            if not available
        )
        report = {
            "status": "PASS" if not failures else "FAIL",
            "localization_mode": localization_mode,
            "initial_pose_service_only": not options.global_localization,
            "velocity_publishers": 0,
            "hardware_interfaces": 0,
            "stationary_drift_limits": {
                "position_m": options.max_stationary_position_drift_m,
                "yaw_deg": options.max_stationary_yaw_drift_deg,
            },
            "requested_initial_pose": {
                "x": None if options.global_localization else options.initial_x,
                "y": None if options.global_localization else options.initial_y,
                "yaw_deg": None
                if options.global_localization
                else options.initial_yaw_deg,
                "normalized_yaw_deg": None
                if options.global_localization
                else normalized_initial_yaw_deg,
                "position_std_m": None
                if options.global_localization
                else options.initial_position_std_m,
                "yaw_std_deg": None
                if options.global_localization
                else options.initial_yaw_std_deg,
                "position_variance": None
                if options.global_localization
                else options.initial_position_std_m ** 2,
                "yaw_variance": None
                if options.global_localization
                else math.radians(options.initial_yaw_std_deg) ** 2,
            },
            "no_motion_update_requests": options.updates,
            "initial_pose_service_calls": 0 if options.global_localization else 1,
            "initial_pose_service_response": initial_pose_service_response,
            "localization": localization,
            "transforms": transforms,
            "failures": failures,
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
    print("AMCL_STATIONARY_LOCALIZATION_PASS")


if __name__ == "__main__":
    main()
