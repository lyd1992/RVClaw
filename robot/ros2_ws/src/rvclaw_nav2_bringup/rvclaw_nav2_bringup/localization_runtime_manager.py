"""Continuously guard whether the current map pose may be reused.

This node never publishes velocity and never authorizes a motor controller.  It
keeps the navigation stack warm while independently invalidating localization
after sensor gaps, explicit charging/reposition events, map changes, or
repeated scan/map disagreement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Dict, Optional

from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from rvclaw_nav2_bringup.amcl_localization_observer import quaternion_to_yaw
from rvclaw_nav2_bringup.fast_start_policy import BLOCKED_CHARGING
from rvclaw_nav2_bringup.fast_start_policy import HOT_READY
from rvclaw_nav2_bringup.fast_start_policy import REUSE_LIVE_POSE
from rvclaw_nav2_bringup.fast_start_policy import TEMPORARY_SCAN_OBSTRUCTION
from rvclaw_nav2_bringup.fast_start_policy import StartupEvidence
from rvclaw_nav2_bringup.fast_start_policy import build_pose_cache
from rvclaw_nav2_bringup.fast_start_policy import build_scan_signature
from rvclaw_nav2_bringup.fast_start_policy import compare_scan_signatures
from rvclaw_nav2_bringup.fast_start_policy import continuous_pose_reusable
from rvclaw_nav2_bringup.fast_start_policy import controlled_motion_validation_deferred
from rvclaw_nav2_bringup.fast_start_policy import decide_fast_start
from rvclaw_nav2_bringup.fast_start_policy import invalidate_pose_cache
from rvclaw_nav2_bringup.fast_start_policy import load_pose_cache
from rvclaw_nav2_bringup.fast_start_policy import map_identity
from rvclaw_nav2_bringup.fast_start_policy import save_pose_cache
from rvclaw_nav2_bringup.fast_start_policy import temporary_obstruction_grace_active
from rvclaw_nav2_bringup.fast_start_policy import validate_pose_cache
from rvclaw_nav2_bringup.scan_map_consistency import load_static_map
from rvclaw_nav2_bringup.scan_map_consistency import scan_endpoints
from rvclaw_nav2_bringup.scan_map_consistency import score_pose


STATUS_TOPIC = "/rvclaw/localization/status"
VALID_TOPIC = "/rvclaw/localization/valid"
CONTROLLED_MOTION_STATE = "CONTROLLED_MOTION"


class LocalizationRuntimeManager(Node):
    def __init__(self) -> None:
        super().__init__("rvclaw_localization_runtime")
        self.declare_parameter(
            "map_file",
            "/data/rvclaw-k3-lab/maps/slam-20260827-105711/map.yaml",
        )
        self.declare_parameter(
            "cache_file",
            "/data/rvclaw-k3-lab/state/localization-pose.json",
        )
        self.declare_parameter("evaluation_period_s", 0.50)
        self.declare_parameter("maximum_scan_age_s", 0.75)
        self.declare_parameter("maximum_pose_age_s", 1.00)
        self.declare_parameter("minimum_map_score", 0.62)
        self.declare_parameter("minimum_known_ratio", 0.75)
        self.declare_parameter("maximum_position_variance", 0.10)
        self.declare_parameter(
            "maximum_yaw_variance",
            math.radians(10.0) ** 2,
        )
        self.declare_parameter("stable_sample_count", 3)
        self.declare_parameter("mismatch_sample_count", 3)
        self.declare_parameter("cache_save_interval_s", 2.0)
        self.declare_parameter("post_motion_validation_delay_s", 2.0)
        self.declare_parameter("safety_pause_scan_grace_s", 300.0)
        self.declare_parameter("safety_pause_max_position_jump_m", 0.25)
        self.declare_parameter(
            "safety_pause_max_yaw_jump_rad",
            math.radians(30.0),
        )

        self.map_file = Path(
            self.get_parameter("map_file").get_parameter_value().string_value
        ).resolve()
        self.cache_file = Path(
            self.get_parameter("cache_file").get_parameter_value().string_value
        ).resolve()
        self.evaluation_period_s = self._double_parameter("evaluation_period_s")
        self.maximum_scan_age_s = self._double_parameter("maximum_scan_age_s")
        self.maximum_pose_age_s = self._double_parameter("maximum_pose_age_s")
        self.minimum_map_score = self._double_parameter("minimum_map_score")
        self.minimum_known_ratio = self._double_parameter("minimum_known_ratio")
        self.maximum_position_variance = self._double_parameter(
            "maximum_position_variance"
        )
        self.maximum_yaw_variance = self._double_parameter(
            "maximum_yaw_variance"
        )
        self.stable_sample_count = self._integer_parameter("stable_sample_count")
        self.mismatch_sample_count = self._integer_parameter(
            "mismatch_sample_count"
        )
        self.cache_save_interval_s = self._double_parameter(
            "cache_save_interval_s"
        )
        self.post_motion_validation_delay_s = self._double_parameter(
            "post_motion_validation_delay_s"
        )
        self.safety_pause_scan_grace_s = self._double_parameter(
            "safety_pause_scan_grace_s"
        )
        self.safety_pause_max_position_jump_m = self._double_parameter(
            "safety_pause_max_position_jump_m"
        )
        self.safety_pause_max_yaw_jump_rad = self._double_parameter(
            "safety_pause_max_yaw_jump_rad"
        )

        self.map_info = map_identity(self.map_file)
        self.grid = load_static_map(self.map_file)
        self.cached_document: Optional[Dict[str, object]] = None
        self.cached_validation: Dict[str, object] = {
            "status": "FAIL",
            "failures": ["pose cache has not been loaded"],
        }
        self._load_cache()

        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_monotonic: Optional[float] = None
        self.latest_pose: Optional[Dict[str, float]] = None
        self.latest_pose_monotonic: Optional[float] = None
        self.hardware_transport_attached: Optional[bool] = None
        self.stm32_connected: Optional[bool] = None
        self.armed: Optional[bool] = None
        self.controller_state: Optional[str] = None
        self.last_controlled_motion_monotonic: Optional[float] = None
        self.safety_pause_started_monotonic: Optional[float] = None
        self.safety_pause_anchor_pose: Optional[Dict[str, float]] = None
        self.runtime_gap_observed = True
        self.explicit_reposition = False
        self.charging = False
        self.external_verification_epoch = 0
        self.stable_samples = 0
        self.consecutive_scan_mismatches = 0
        self.last_hard_invalidation_reason: Optional[str] = None
        self.last_scan_score: Optional[Dict[str, object]] = None
        self.last_signature_comparison: Optional[Dict[str, object]] = None
        self.last_cache_save_monotonic: Optional[float] = None
        self.last_status: Dict[str, object] = {}

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        amcl_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, state_qos)
        self.valid_publisher = self.create_publisher(Bool, VALID_TOPIC, state_qos)
        self.create_subscription(
            Bool,
            "/rvclaw/control/hardware_transport_attached",
            self._on_transport,
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/stm32/connected",
            self._on_stm32,
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/rvclaw/control/armed",
            self._on_armed,
            state_qos,
        )
        self.create_subscription(
            String,
            "/rvclaw/control/state",
            self._on_controller_state,
            state_qos,
        )
        self.create_subscription(LaserScan, "/scan", self._on_scan, sensor_qos)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_pose,
            amcl_qos,
        )
        self.create_service(
            Trigger,
            "/rvclaw/localization/prepare_charge",
            self._prepare_charge,
        )
        self.create_service(
            Trigger,
            "/rvclaw/localization/resume_from_charge",
            self._resume_from_charge,
        )
        self.create_service(
            Trigger,
            "/rvclaw/localization/mark_repositioned",
            self._mark_repositioned,
        )
        self.create_service(
            Trigger,
            "/rvclaw/localization/accept_verified_pose",
            self._accept_verified_pose,
        )
        self.create_timer(self.evaluation_period_s, self._evaluate)
        self._publish_status(
            "startup requires validation",
            valid=False,
            state=(
                "WARM_VALIDATION_REQUIRED"
                if self.cached_validation.get("status") == "PASS"
                else "GLOBAL_LOCALIZATION_REQUIRED"
            ),
        )

    def _double_parameter(self, name: str) -> float:
        return float(self.get_parameter(name).get_parameter_value().double_value)

    def _integer_parameter(self, name: str) -> int:
        return int(self.get_parameter(name).get_parameter_value().integer_value)

    def _load_cache(self) -> None:
        try:
            self.cached_document = load_pose_cache(self.cache_file)
            self.cached_validation = validate_pose_cache(
                self.cached_document,
                self.map_info,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.cached_document = None
            self.cached_validation = {
                "status": "FAIL",
                "failures": ["pose cache unavailable: %s" % error],
            }

    def _sensor_age(self, received_at: Optional[float]) -> float:
        if received_at is None:
            return math.inf
        return max(0.0, time.monotonic() - received_at)

    def _on_transport(self, message: Bool) -> None:
        attached = bool(message.data)
        if self.hardware_transport_attached is True and not attached:
            self.runtime_gap_observed = True
            self.stable_samples = 0
        self.hardware_transport_attached = attached

    def _on_stm32(self, message: Bool) -> None:
        connected = bool(message.data)
        if self.stm32_connected is True and not connected:
            self.runtime_gap_observed = True
            self.stable_samples = 0
        self.stm32_connected = connected

    def _note_controlled_motion(self) -> None:
        self.last_controlled_motion_monotonic = time.monotonic()
        self.consecutive_scan_mismatches = 0
        self.stable_samples = 0

    def _on_armed(self, message: Bool) -> None:
        self.armed = bool(message.data)
        if self.armed:
            self._note_controlled_motion()

    def _on_controller_state(self, message: String) -> None:
        previous_state = str(self.controller_state or "").upper()
        new_state = str(message.data).upper()
        now = time.monotonic()
        self.controller_state = new_state
        if new_state == "ACTIVE":
            self.safety_pause_started_monotonic = None
            self.safety_pause_anchor_pose = None
            self._note_controlled_motion()
            return
        if new_state in {"WATCHDOG_LATCHED", "ESTOP_LATCHED", "STOP_LATCHED"}:
            recent_controlled_motion = (
                self.last_controlled_motion_monotonic is not None
                and 0.0
                <= now - self.last_controlled_motion_monotonic
                <= max(5.0, self.post_motion_validation_delay_s)
            )
            if previous_state == "ACTIVE" or recent_controlled_motion:
                if self.safety_pause_started_monotonic is None:
                    self.safety_pause_started_monotonic = now
                    self.safety_pause_anchor_pose = (
                        None if self.latest_pose is None else dict(self.latest_pose)
                    )
                self.consecutive_scan_mismatches = 0
                self.stable_samples = 0

    def _on_scan(self, message: LaserScan) -> None:
        self.latest_scan = message
        self.latest_scan_monotonic = time.monotonic()

    def _on_pose(self, message: PoseWithCovarianceStamped) -> None:
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
        self.latest_pose = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "yaw": float(yaw),
            "cov_x": float(covariance[0]),
            "cov_y": float(covariance[7]),
            "cov_yaw": float(covariance[35]),
        }
        self.latest_pose_monotonic = time.monotonic()

    def _hard_invalidate(self, reason: str) -> None:
        self.explicit_reposition = True
        self.runtime_gap_observed = True
        self.stable_samples = 0
        self.safety_pause_started_monotonic = None
        self.safety_pause_anchor_pose = None
        if self.last_hard_invalidation_reason == reason:
            return
        self.last_hard_invalidation_reason = reason
        if self.cache_file.exists():
            try:
                self.cached_document = invalidate_pose_cache(
                    self.cache_file,
                    reason,
                )
                self.cached_validation = validate_pose_cache(
                    self.cached_document,
                    self.map_info,
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.get_logger().error("failed to invalidate pose cache: %s" % error)

    def _prepare_charge(self, _request, response):
        self.charging = True
        self._hard_invalidate("charging mode entered")
        response.success = True
        response.message = "navigation blocked; global localization required after charging"
        self._publish_status(response.message, valid=False, state=BLOCKED_CHARGING)
        return response

    def _resume_from_charge(self, _request, response):
        self.charging = False
        self._hard_invalidate("charging mode exited after possible reposition")
        response.success = True
        response.message = "charging ended; full localization remains required"
        return response

    def _mark_repositioned(self, _request, response):
        self._hard_invalidate("manual reposition reported")
        response.success = True
        response.message = "manual reposition recorded; full localization required"
        return response

    def _accept_verified_pose(self, _request, response):
        scan_age = self._sensor_age(self.latest_scan_monotonic)
        pose_age = self._sensor_age(self.latest_pose_monotonic)
        if (
            self.hardware_transport_attached is not True
            or self.stm32_connected is not True
            or self.armed is not False
            or self.latest_scan is None
            or self.latest_pose is None
            or scan_age > self.maximum_scan_age_s
            or pose_age > self.maximum_pose_age_s
        ):
            response.success = False
            response.message = (
                "disarmed hardware with fresh lidar and AMCL pose is required"
            )
            return response
        if not self._covariance_ok():
            response.success = False
            response.message = "AMCL covariance is not acceptable"
            return response
        try:
            live_score = self._live_score()
        except (ValueError, IndexError) as error:
            response.success = False
            response.message = "live scan/map verification failed: %s" % error
            return response
        if (
            live_score is None
            or float(live_score["mean_likelihood"]) < self.minimum_map_score
            or float(live_score["known_endpoint_ratio"]) < self.minimum_known_ratio
        ):
            response.success = False
            response.message = "live scan/map score is not acceptable"
            return response
        self.charging = False
        self.explicit_reposition = False
        self.runtime_gap_observed = False
        self.consecutive_scan_mismatches = 0
        self.last_hard_invalidation_reason = None
        self.stable_samples = 0
        self.safety_pause_started_monotonic = None
        self.safety_pause_anchor_pose = None
        self.external_verification_epoch += 1
        response.success = True
        response.message = "verified pose accepted; waiting for stable live checks"
        return response

    def _live_score(self) -> Optional[Dict[str, object]]:
        if self.latest_scan is None or self.latest_pose is None:
            return None
        endpoints = scan_endpoints(
            self.latest_scan.ranges,
            self.latest_scan.angle_min,
            self.latest_scan.angle_increment,
            self.latest_scan.range_min,
            self.latest_scan.range_max,
            maximum_points=60,
        )
        return score_pose(
            self.grid,
            endpoints,
            self.latest_pose["x"],
            self.latest_pose["y"],
            self.latest_pose["yaw"],
        )

    def _current_signature(self) -> Optional[Dict[str, object]]:
        if self.latest_scan is None:
            return None
        return build_scan_signature(
            self.latest_scan.ranges,
            self.latest_scan.angle_min,
            self.latest_scan.angle_increment,
            self.latest_scan.range_min,
            self.latest_scan.range_max,
        )

    def _cache_signature_ok(self) -> bool:
        if self.cached_validation.get("status") != "PASS":
            self.last_signature_comparison = None
            return False
        current = self._current_signature()
        if current is None:
            self.last_signature_comparison = None
            return False
        self.last_signature_comparison = compare_scan_signatures(
            self.cached_validation["scan_signature"],
            current,
        )
        return self.last_signature_comparison.get("status") == "PASS"

    def _covariance_ok(self) -> bool:
        pose = self.latest_pose
        if pose is None:
            return False
        values = [pose[key] for key in ("x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")]
        if not all(math.isfinite(value) for value in values):
            return False
        return (
            pose["cov_x"] <= self.maximum_position_variance
            and pose["cov_y"] <= self.maximum_position_variance
            and pose["cov_yaw"] <= self.maximum_yaw_variance
        )

    def _safety_pause_pose_jump(self):
        anchor = self.safety_pause_anchor_pose
        current = self.latest_pose
        if anchor is None or current is None:
            return None, None
        position_jump = math.hypot(
            float(current["x"]) - float(anchor["x"]),
            float(current["y"]) - float(anchor["y"]),
        )
        yaw_delta = float(current["yaw"]) - float(anchor["yaw"])
        yaw_jump = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
        return position_jump, yaw_jump

    def _temporary_obstruction_grace_active(
        self,
        *,
        now: float,
        hardware_connected: bool,
        pose_message_fresh: bool,
    ) -> bool:
        age = (
            None
            if self.safety_pause_started_monotonic is None
            else now - self.safety_pause_started_monotonic
        )
        position_jump, yaw_jump = self._safety_pause_pose_jump()
        return temporary_obstruction_grace_active(
            safety_pause_age_s=age,
            grace_s=self.safety_pause_scan_grace_s,
            explicit_reposition=self.explicit_reposition,
            runtime_gap_observed=self.runtime_gap_observed,
            hardware_connected=hardware_connected,
            armed=self.armed,
            pose_message_fresh=pose_message_fresh,
            covariance_ok=self._covariance_ok(),
            position_jump_m=position_jump,
            yaw_jump_rad=yaw_jump,
            maximum_position_jump_m=self.safety_pause_max_position_jump_m,
            maximum_yaw_jump_rad=self.safety_pause_max_yaw_jump_rad,
        )

    def _save_verified_cache(self, signature: Dict[str, object]) -> None:
        now = time.monotonic()
        if (
            self.last_cache_save_monotonic is not None
            and now - self.last_cache_save_monotonic < self.cache_save_interval_s
        ):
            return
        if self.latest_pose is None or self.armed is not False:
            return
        document = build_pose_cache(
            self.map_info,
            self.latest_pose,
            signature,
            source="persistent_runtime",
            validation_epoch=self.external_verification_epoch,
        )
        save_pose_cache(self.cache_file, document)
        self.cached_document = document
        self.cached_validation = validate_pose_cache(document, self.map_info)
        self.last_cache_save_monotonic = now

    def _publish_status(self, reason: str, *, valid: bool, state: str) -> None:
        pose_age = self._sensor_age(self.latest_pose_monotonic)
        pose_message_fresh = (
            self.latest_pose is not None and pose_age <= self.maximum_pose_age_s
        )
        pose_reusable = continuous_pose_reusable(
            pose_available=self.latest_pose is not None,
            pose_age_s=pose_age,
            maximum_pose_age_s=self.maximum_pose_age_s,
            external_verification_epoch=self.external_verification_epoch,
            runtime_gap_observed=self.runtime_gap_observed,
            explicit_reposition=self.explicit_reposition,
        )
        report = {
            "schema_version": 1,
            "state": state,
            "valid": bool(valid),
            "motion_authorized": False,
            "reason": str(reason),
            "map_sha256": self.map_info["sha256"],
            "hardware_transport_attached": self.hardware_transport_attached,
            "stm32_connected": self.stm32_connected,
            "armed": self.armed,
            "controller_state": self.controller_state,
            "charging": self.charging,
            "runtime_gap_observed": self.runtime_gap_observed,
            "explicit_reposition": self.explicit_reposition,
            "external_verification_epoch": self.external_verification_epoch,
            "stable_samples": self.stable_samples,
            "consecutive_scan_mismatches": self.consecutive_scan_mismatches,
            "safety_pause_age_s": (
                None
                if self.safety_pause_started_monotonic is None
                else max(0.0, time.monotonic() - self.safety_pause_started_monotonic)
            ),
            "safety_pause_anchor_pose": self.safety_pause_anchor_pose,
            "scan_age_s": self._sensor_age(self.latest_scan_monotonic),
            "pose_age_s": pose_age,
            "pose_message_fresh": pose_message_fresh,
            "pose_reusable": pose_reusable,
            "hardware_connected": (
                self.hardware_transport_attached is True
                and self.stm32_connected is True
            ),
            "lidar_fresh": (
                self._sensor_age(self.latest_scan_monotonic)
                <= self.maximum_scan_age_s
            ),
            "scan_map_score": self.last_scan_score,
            "cache_validation": self.cached_validation,
            "cache_signature_comparison": self.last_signature_comparison,
            "published_at_s": time.time(),
        }
        self.last_status = report
        status_message = String()
        status_message.data = json.dumps(report, sort_keys=True)
        valid_message = Bool()
        valid_message.data = bool(valid)
        self.status_publisher.publish(status_message)
        self.valid_publisher.publish(valid_message)

    def _evaluate(self) -> None:
        now = time.monotonic()
        scan_age = self._sensor_age(self.latest_scan_monotonic)
        pose_age = self._sensor_age(self.latest_pose_monotonic)
        hardware_connected = (
            self.hardware_transport_attached is True
            and self.stm32_connected is True
        )
        lidar_fresh = scan_age <= self.maximum_scan_age_s
        sensors_ready = hardware_connected and lidar_fresh
        last_motion_age_s = (
            None
            if self.last_controlled_motion_monotonic is None
            else now - self.last_controlled_motion_monotonic
        )
        if controlled_motion_validation_deferred(
            armed=self.armed,
            controller_state=self.controller_state,
            last_motion_age_s=last_motion_age_s,
            settle_s=self.post_motion_validation_delay_s,
        ):
            self.consecutive_scan_mismatches = 0
            self.stable_samples = 0
            self.last_scan_score = None
            self.last_signature_comparison = None
            self._publish_status(
                "controlled motion or post-motion settling; localization validation deferred",
                valid=False,
                state=CONTROLLED_MOTION_STATE,
            )
            return
        pose_message_fresh = (
            self.latest_pose is not None and pose_age <= self.maximum_pose_age_s
        )
        pose_reusable = continuous_pose_reusable(
            pose_available=self.latest_pose is not None,
            pose_age_s=pose_age,
            maximum_pose_age_s=self.maximum_pose_age_s,
            external_verification_epoch=self.external_verification_epoch,
            runtime_gap_observed=self.runtime_gap_observed,
            explicit_reposition=self.explicit_reposition,
        )

        self.last_scan_score = None
        map_ok = False
        if sensors_ready and pose_reusable:
            try:
                self.last_scan_score = self._live_score()
            except (ValueError, IndexError) as error:
                self.last_scan_score = {"error": str(error)}
            if self.last_scan_score is not None and "error" not in self.last_scan_score:
                map_ok = (
                    float(self.last_scan_score["mean_likelihood"])
                    >= self.minimum_map_score
                    and float(self.last_scan_score["known_endpoint_ratio"])
                    >= self.minimum_known_ratio
                )
                if map_ok:
                    self.consecutive_scan_mismatches = 0
                else:
                    self.consecutive_scan_mismatches += 1
            else:
                self.consecutive_scan_mismatches += 1

        if (
            not map_ok
            and self.consecutive_scan_mismatches > 0
            and self._temporary_obstruction_grace_active(
                now=now,
                hardware_connected=hardware_connected,
                pose_message_fresh=pose_message_fresh,
            )
        ):
            self.consecutive_scan_mismatches = 0
            self.stable_samples = 0
            self.last_signature_comparison = None
            self._publish_status(
                "temporary scan obstruction after a safety stop; waiting for the hazard to clear",
                valid=False,
                state=TEMPORARY_SCAN_OBSTRUCTION,
            )
            return

        if (
            self.consecutive_scan_mismatches >= self.mismatch_sample_count
            and self.last_hard_invalidation_reason
            != "repeated live scan/map mismatch"
        ):
            self._hard_invalidate("repeated live scan/map mismatch")

        cache_signature_ok = False
        if sensors_ready and self.cached_validation.get("status") == "PASS":
            try:
                cache_signature_ok = self._cache_signature_ok()
            except (ValueError, IndexError):
                cache_signature_ok = False

        decision = decide_fast_start(
            StartupEvidence(
                charging=self.charging,
                explicit_reposition=self.explicit_reposition,
                hardware_connected=hardware_connected,
                lidar_fresh=lidar_fresh,
                runtime_gap_observed=self.runtime_gap_observed,
                live_pose_fresh=pose_reusable,
                live_covariance_ok=self._covariance_ok(),
                live_scan_map_ok=map_ok,
                cache_valid=self.cached_validation.get("status") == "PASS",
                cache_signature_ok=cache_signature_ok,
                consecutive_scan_mismatches=self.consecutive_scan_mismatches,
            )
        )

        if decision["decision"] == REUSE_LIVE_POSE:
            self.stable_samples += 1
            if self.stable_samples >= self.stable_sample_count:
                self.safety_pause_started_monotonic = None
                self.safety_pause_anchor_pose = None
                signature = self._current_signature()
                if signature is not None:
                    self._save_verified_cache(signature)
                self._publish_status(
                    "continuous pose is verified",
                    valid=True,
                    state=HOT_READY,
                )
                return
            self._publish_status(
                "collecting stable live localization samples",
                valid=False,
                state="WARM_VALIDATION_REQUIRED",
            )
            return

        self.stable_samples = 0
        self._publish_status(
            "; ".join(decision["reasons"]),
            valid=False,
            state=decision["state"],
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalizationRuntimeManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
