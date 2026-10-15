"""ROS-free waypoint inspection orchestration for the local RVClaw vision API.

The navigation process owns every chassis movement. This module can request a
bounded absolute heading through a callback supplied by the route runner, but
it never publishes velocity, touches the STM32 serial device, or opens a camera
device directly. The existing Studio service remains the single camera owner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import tempfile
import time
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

import yaml


SCHEMA_VERSION = 2
METER_TYPE_BY_KIND = {
    "digital_meter": "digital",
    "pointer_meter": "pointer",
}
SUPPORTED_TASK_KINDS = tuple(METER_TYPE_BY_KIND)
ALLOWED_FAILURE_POLICIES = ("abort", "continue")
ACCEPTED_METER_STATUSES = ("recognized", "stable")
MAX_SEARCH_OFFSET_DEG = 35.0
MAX_SEARCH_HEADINGS = 7
SAFE_CAPTURE_TERMINAL_ACTION_STATUSES = (5, 6)
SAFE_CAPTURE_SUCCESSFUL_CHUNK_WRAPPER_ERRORS = (
    "bounded direction correction limit",
)
SAFE_CAPTURE_MAX_LINEAR_M_S = 0.03
SAFE_CAPTURE_MAX_ANGULAR_RAD_S = 0.10
INSPECTION_CAMERA_MODE = "astra_raw"
INSPECTION_CAMERA_SWITCH_TIMEOUT_S = 12.0
# Nav2's guarded Spin path has a small-angle dead zone on the physical chassis.
# Treat a safely stopped heading within ten degrees as camera-aligned instead of
# repeatedly requesting a correction that cannot generate a fresh cmd_vel stream.
INSPECTION_BASE_HEADING_TOLERANCE_RAD = math.radians(10.0)


class InspectionConfigError(ValueError):
    """Raised when an inspection plan cannot be executed safely."""


class VisionApiError(RuntimeError):
    """Raised when the local vision service rejects or cannot serve a request."""


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InspectionConfigError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise InspectionConfigError(f"{label} must be finite")
    return number


def _positive_number(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number <= 0.0:
        raise InspectionConfigError(f"{label} must be positive")
    return number


def _bounded_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise InspectionConfigError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise InspectionConfigError(f"{label} must be an integer") from exc
    if number != value or number < minimum or number > maximum:
        raise InspectionConfigError(
            f"{label} must be an integer in [{minimum}, {maximum}]"
        )
    return number


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InspectionConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise InspectionConfigError(f"{label} must be true or false")
    return value


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def trusted_stopped_heading_observation(
    heading_result: Mapping[str, Any],
) -> tuple[Optional[dict[str, float]], Optional[float]]:
    """Return a stopped AMCL observation even when the heading action failed.

    A bounded heading action can end with an aborted Nav2 result after the
    chassis has already reached a usable pose.  Keeping that measured pose is
    essential for the next route segment's continuity gate.  An armed result
    is never trusted.
    """

    if heading_result.get("final_armed") is not False:
        return None, None
    pose_value = heading_result.get("final_amcl_pose")
    if isinstance(pose_value, Mapping):
        try:
            pose = {
                key: float(pose_value[key])
                for key in ("x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
            }
        except (KeyError, TypeError, ValueError):
            pose = None
        if pose is not None and all(math.isfinite(value) for value in pose.values()):
            return pose, pose["yaw"]
    yaw_value = heading_result.get("final_amcl_yaw_rad")
    try:
        yaw = float(yaw_value)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(yaw):
        return None, None
    return None, normalize_angle(yaw)


def safe_stationary_capture_after_heading_abort(
    heading_result: Mapping[str, Any],
    observed_pose: Optional[Mapping[str, float]],
    observed_heading: Optional[float],
) -> bool:
    """Allow camera-only inspection after a safely stopped terminal Spin abort.

    This is deliberately narrower than accepting the heading action itself.  It
    never authorizes more motion: the controller must be disarmed, the stopped
    feedback must be within the existing limits, and a finite AMCL pose must be
    available.  The current camera frame can then still contain the meter even
    when Nav2 refused the last in-place correction because of its collision
    projection.
    """

    if heading_result.get("final_armed") is not False:
        return False
    if str(heading_result.get("final_state", "")).upper() != "DISARMED":
        return False
    if observed_pose is None or observed_heading is None:
        return False
    try:
        action_status = int(heading_result.get("action_status"))
    except (TypeError, ValueError):
        return False
    if action_status == 4:
        # The bounded heading wrapper can reject the overall correction after
        # its final Nav2 Spin chunk succeeded, most notably when its reversal
        # guard trips.  Admit only that known wrapper-level outcome; a generic
        # failure with status 4 must not bypass the heading gate.
        error = str(heading_result.get("error", "")).lower()
        if not any(
            marker in error
            for marker in SAFE_CAPTURE_SUCCESSFUL_CHUNK_WRAPPER_ERRORS
        ):
            return False
    elif action_status not in SAFE_CAPTURE_TERMINAL_ACTION_STATUSES:
        return False
    try:
        stable_samples = int(heading_result.get("post_stop_stable_samples", 0))
        required_samples = int(
            heading_result.get("post_stop_required_stable_samples", 1)
        )
        final_x = float(heading_result.get("final_feedback_x_m_s", math.inf))
        final_y = float(heading_result.get("final_feedback_y_m_s", math.inf))
        final_angular = float(
            heading_result.get("final_feedback_angular_rad_s", math.inf)
        )
    except (TypeError, ValueError):
        return False
    return (
        required_samples > 0
        and stable_samples >= required_samples
        and math.isfinite(final_x)
        and math.isfinite(final_y)
        and math.isfinite(final_angular)
        and abs(final_x) <= SAFE_CAPTURE_MAX_LINEAR_M_S
        and abs(final_y) <= SAFE_CAPTURE_MAX_LINEAR_M_S
        and abs(final_angular) <= SAFE_CAPTURE_MAX_ANGULAR_RAD_S
    )


@dataclass(frozen=True)
class InspectionPoint:
    waypoint: str
    enabled: bool
    task_id: str
    kind: str
    meter_id: Optional[str]
    meter_type: str
    calibration: Mapping[str, Any]
    normal_min: Optional[float]
    normal_max: Optional[float]
    normal_min_inclusive: bool
    normal_max_inclusive: bool
    inspection_yaw_deg: Optional[float]
    target_x_m: Optional[float]
    target_y_m: Optional[float]
    search_offsets_deg: tuple[float, ...]
    reads_per_heading: int
    min_consistent_reads: int
    min_confidence: float
    numeric_tolerance: float
    timeout_s: float
    failure_policy: str
    raw_text_pattern: Optional[str] = None

    def classify_value(self, value: Any) -> tuple[str, bool]:
        number = _finite_number(value, f"{self.waypoint} reading")
        normal = True
        if self.normal_min is not None:
            normal = normal and (
                number >= self.normal_min
                if self.normal_min_inclusive
                else number > self.normal_min
            )
        if self.normal_max is not None:
            normal = normal and (
                number <= self.normal_max
                if self.normal_max_inclusive
                else number < self.normal_max
            )
        return ("NORMAL", False) if normal else ("ABNORMAL", True)

    def normal_rule(self) -> dict[str, Any]:
        return {
            "min": self.normal_min,
            "min_inclusive": self.normal_min_inclusive,
            "max": self.normal_max,
            "max_inclusive": self.normal_max_inclusive,
        }

    def base_yaw_rad(self, waypoint_pose: Mapping[str, Any]) -> float:
        """Resolve the deterministic camera-facing heading for this point."""

        if self.inspection_yaw_deg is not None:
            return normalize_angle(math.radians(self.inspection_yaw_deg))
        if self.target_x_m is None or self.target_y_m is None:
            raise InspectionConfigError(
                f"inspection point {self.waypoint} has no inspection heading or target"
            )
        try:
            x = float(waypoint_pose["x"])
            y = float(waypoint_pose["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InspectionConfigError("waypoint pose is missing finite x/y") from exc
        if not all(math.isfinite(value) for value in (x, y)):
            raise InspectionConfigError("waypoint pose is missing finite x/y")
        dx = self.target_x_m - x
        dy = self.target_y_m - y
        if math.hypot(dx, dy) < 0.05:
            raise InspectionConfigError(
                f"inspection target for {self.waypoint} is too close to its observation pose"
            )
        return math.atan2(dy, dx)

    def heading_candidates_rad(self, waypoint_pose: Mapping[str, Any]) -> tuple[float, ...]:
        base = self.base_yaw_rad(waypoint_pose)
        return tuple(
            normalize_angle(base + math.radians(offset))
            for offset in self.search_offsets_deg
        )


@dataclass(frozen=True)
class InspectionPlan:
    vision_api_url: str
    result_root: Path
    restore_camera_mode: bool
    speech_ack_timeout_s: float
    points: Mapping[str, InspectionPoint]

    def point(self, waypoint: str) -> Optional[InspectionPoint]:
        return self.points.get(str(waypoint).upper())


def _normalize_search_offsets(value: Any, label: str) -> tuple[float, ...]:
    if value is None:
        value = [0.0, -10.0, 10.0, -20.0, 20.0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InspectionConfigError(f"{label} must be a sequence")
    if not value or len(value) > MAX_SEARCH_HEADINGS:
        raise InspectionConfigError(
            f"{label} must contain between 1 and {MAX_SEARCH_HEADINGS} headings"
        )
    offsets = tuple(_finite_number(item, label) for item in value)
    if abs(offsets[0]) > 1e-9:
        raise InspectionConfigError(f"{label} must start with 0 degrees")
    if any(abs(item) > MAX_SEARCH_OFFSET_DEG for item in offsets):
        raise InspectionConfigError(
            f"{label} is limited to +/-{MAX_SEARCH_OFFSET_DEG:.0f} degrees"
        )
    if len({round(item, 9) for item in offsets}) != len(offsets):
        raise InspectionConfigError(f"{label} must not contain duplicates")
    return offsets


def _load_point(name: str, raw: Any, defaults: Mapping[str, Any]) -> InspectionPoint:
    if not isinstance(raw, Mapping):
        raise InspectionConfigError(f"points.{name} must be a mapping")
    merged = dict(defaults)
    merged.update(raw)
    enabled = bool(merged.get("enabled", False))
    task_id = _nonempty_text(merged.get("task_id", "inspect_meter"), f"points.{name}.task_id")
    kind = _nonempty_text(merged.get("kind", "digital_meter"), f"points.{name}.kind")
    if kind not in SUPPORTED_TASK_KINDS:
        raise InspectionConfigError(
            f"points.{name}.kind must be one of {', '.join(SUPPORTED_TASK_KINDS)}"
        )
    meter_id_value = merged.get("meter_id")
    meter_id = None if meter_id_value in (None, "") else _nonempty_text(
        meter_id_value, f"points.{name}.meter_id"
    )
    if enabled and meter_id is None:
        raise InspectionConfigError(f"points.{name}.meter_id is required when enabled")

    meter_type = METER_TYPE_BY_KIND[kind]
    raw_calibration = merged.get("calibration")
    calibration: dict[str, Any] = {}
    if meter_type == "pointer":
        if not isinstance(raw_calibration, Mapping):
            raise InspectionConfigError(
                f"points.{name}.calibration is required for a pointer meter"
            )
        required = (
            "zero_angle_deg",
            "sweep_deg",
            "min_value",
            "max_value",
            "direction",
        )
        missing = [key for key in required if key not in raw_calibration]
        if missing:
            raise InspectionConfigError(
                f"points.{name}.calibration is missing {', '.join(missing)}"
            )
        calibration = {
            "zero_angle_deg": _finite_number(
                raw_calibration.get("zero_angle_deg"),
                f"points.{name}.calibration.zero_angle_deg",
            ),
            "sweep_deg": _finite_number(
                raw_calibration.get("sweep_deg"),
                f"points.{name}.calibration.sweep_deg",
            ),
            "min_value": _finite_number(
                raw_calibration.get("min_value"),
                f"points.{name}.calibration.min_value",
            ),
            "max_value": _finite_number(
                raw_calibration.get("max_value"),
                f"points.{name}.calibration.max_value",
            ),
            "direction": _nonempty_text(
                raw_calibration.get("direction"),
                f"points.{name}.calibration.direction",
            ).lower(),
            "decimal_places": _bounded_integer(
                raw_calibration.get("decimal_places", 2),
                f"points.{name}.calibration.decimal_places",
                0,
                6,
            ),
        }
        if not 0.0 < calibration["sweep_deg"] <= 360.0:
            raise InspectionConfigError(
                f"points.{name}.calibration.sweep_deg must be in (0, 360]"
            )
        if calibration["max_value"] <= calibration["min_value"]:
            raise InspectionConfigError(
                f"points.{name}.calibration.max_value must exceed min_value"
            )
        if calibration["direction"] not in {"clockwise", "counterclockwise"}:
            raise InspectionConfigError(
                f"points.{name}.calibration.direction must be clockwise or counterclockwise"
            )
    elif raw_calibration not in (None, {}):
        raise InspectionConfigError(
            f"points.{name}.calibration is only valid for a pointer meter"
        )

    yaw_value = merged.get("inspection_yaw_deg")
    target = merged.get("target")
    inspection_yaw_deg = None
    target_x_m = None
    target_y_m = None
    if yaw_value is not None:
        inspection_yaw_deg = _finite_number(yaw_value, f"points.{name}.inspection_yaw_deg")
    if target is not None:
        if not isinstance(target, Mapping):
            raise InspectionConfigError(f"points.{name}.target must be a mapping")
        target_x_m = _finite_number(target.get("x_m"), f"points.{name}.target.x_m")
        target_y_m = _finite_number(target.get("y_m"), f"points.{name}.target.y_m")
    if inspection_yaw_deg is not None and target is not None:
        raise InspectionConfigError(
            f"points.{name} must use inspection_yaw_deg or target, not both"
        )
    if enabled and inspection_yaw_deg is None and target is None:
        raise InspectionConfigError(
            f"points.{name} requires inspection_yaw_deg or target when enabled"
        )

    reads_per_heading = _bounded_integer(
        merged.get("reads_per_heading", 3),
        f"points.{name}.reads_per_heading",
        1,
        10,
    )
    min_consistent_reads = _bounded_integer(
        merged.get("min_consistent_reads", 2),
        f"points.{name}.min_consistent_reads",
        1,
        reads_per_heading,
    )
    min_confidence = _finite_number(
        merged.get("min_confidence", 0.65), f"points.{name}.min_confidence"
    )
    if not 0.0 <= min_confidence <= 1.0:
        raise InspectionConfigError(f"points.{name}.min_confidence must be in [0, 1]")
    numeric_tolerance = _finite_number(
        merged.get("numeric_tolerance", 0.01), f"points.{name}.numeric_tolerance"
    )
    if numeric_tolerance < 0.0:
        raise InspectionConfigError(f"points.{name}.numeric_tolerance must be non-negative")
    normal_min_value = merged.get("normal_min")
    normal_max_value = merged.get("normal_max")
    normal_min = (
        None
        if normal_min_value is None
        else _finite_number(normal_min_value, f"points.{name}.normal_min")
    )
    normal_max = (
        None
        if normal_max_value is None
        else _finite_number(normal_max_value, f"points.{name}.normal_max")
    )
    if enabled and normal_min is None and normal_max is None:
        raise InspectionConfigError(
            f"points.{name} requires normal_min or normal_max when enabled"
        )
    if normal_min is not None and normal_max is not None and normal_min > normal_max:
        raise InspectionConfigError(f"points.{name}.normal_min must not exceed normal_max")
    normal_min_inclusive = _boolean(
        merged.get("normal_min_inclusive", True),
        f"points.{name}.normal_min_inclusive",
    )
    normal_max_inclusive = _boolean(
        merged.get("normal_max_inclusive", True),
        f"points.{name}.normal_max_inclusive",
    )
    failure_policy = _nonempty_text(
        merged.get("failure_policy", "continue"), f"points.{name}.failure_policy"
    )
    if failure_policy not in ALLOWED_FAILURE_POLICIES:
        raise InspectionConfigError(
            f"points.{name}.failure_policy must be abort or continue"
        )
    raw_text_pattern_value = merged.get("raw_text_pattern")
    raw_text_pattern = (
        None
        if raw_text_pattern_value in (None, "")
        else _nonempty_text(raw_text_pattern_value, f"points.{name}.raw_text_pattern")
    )
    if raw_text_pattern is not None:
        if meter_type != "digital":
            raise InspectionConfigError(
                f"points.{name}.raw_text_pattern is only valid for a digital meter"
            )
        try:
            re.compile(raw_text_pattern)
        except re.error as exc:
            raise InspectionConfigError(
                f"points.{name}.raw_text_pattern is invalid: {exc}"
            ) from exc
    return InspectionPoint(
        waypoint=str(name).upper(),
        enabled=enabled,
        task_id=task_id,
        kind=kind,
        meter_id=meter_id,
        meter_type=meter_type,
        calibration=calibration,
        normal_min=normal_min,
        normal_max=normal_max,
        normal_min_inclusive=normal_min_inclusive,
        normal_max_inclusive=normal_max_inclusive,
        inspection_yaw_deg=inspection_yaw_deg,
        target_x_m=target_x_m,
        target_y_m=target_y_m,
        search_offsets_deg=_normalize_search_offsets(
            merged.get("search_offsets_deg"), f"points.{name}.search_offsets_deg"
        ),
        reads_per_heading=reads_per_heading,
        min_consistent_reads=min_consistent_reads,
        min_confidence=min_confidence,
        numeric_tolerance=numeric_tolerance,
        timeout_s=_positive_number(
            merged.get("timeout_s", 45.0), f"points.{name}.timeout_s"
        ),
        failure_policy=failure_policy,
        raw_text_pattern=raw_text_pattern,
    )


def load_inspection_plan(path: str | Path) -> InspectionPlan:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise InspectionConfigError("inspection configuration must contain a mapping")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise InspectionConfigError(
            f"inspection schema_version must be {SCHEMA_VERSION}"
        )
    defaults = document.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise InspectionConfigError("defaults must be a mapping")
    raw_points = document.get("points")
    if not isinstance(raw_points, Mapping) or not raw_points:
        raise InspectionConfigError("inspection configuration contains no points")
    points: dict[str, InspectionPoint] = {}
    for raw_name, raw in raw_points.items():
        name = _nonempty_text(raw_name, "point name").upper()
        if name in points:
            raise InspectionConfigError(f"duplicate inspection point {name}")
        points[name] = _load_point(name, raw, defaults)
    api_url = _nonempty_text(
        document.get("vision_api_url", "http://127.0.0.1:8017"),
        "vision_api_url",
    ).rstrip("/")
    if not api_url.startswith(("http://", "https://")):
        raise InspectionConfigError("vision_api_url must use http:// or https://")
    return InspectionPlan(
        vision_api_url=api_url,
        result_root=Path(
            document.get("result_root", "/data/RVClaw/runs/route-inspections")
        ).expanduser(),
        restore_camera_mode=bool(document.get("restore_camera_mode", True)),
        speech_ack_timeout_s=_positive_number(
            document.get("speech_ack_timeout_s", 12.0),
            "speech_ack_timeout_s",
        ),
        points=points,
    )


class VisionHttpClient:
    """Dependency-free client for the existing RVClaw Studio API."""

    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(
        self,
        path: str,
        method: str = "GET",
        payload: Optional[Mapping[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise VisionApiError(f"vision API HTTP {exc.code}: {detail}") from exc
        except (OSError, URLError, ValueError) as exc:
            raise VisionApiError(f"vision API unavailable: {exc}") from exc
        if not isinstance(value, dict):
            raise VisionApiError("vision API returned a non-object response")
        return value

    def camera_status(self) -> dict[str, Any]:
        return self._request("/api/cameras")

    def meter_configurations(self) -> dict[str, Any]:
        return self._request("/api/meter-configurations")

    def meter_reading(self) -> dict[str, Any]:
        return self._request("/api/meter-reading")

    def current_meter_capabilities(self) -> dict[str, Any]:
        return self._request("/api/meter-current-read")

    def current_read(
        self,
        meter_type: str,
        meter_id: str,
        calibration: Mapping[str, Any],
        timeout_s: float,
        *,
        evidence_session: str,
        waypoint: str,
        capture_index: int,
    ) -> dict[str, Any]:
        return self._request(
            "/api/meter-current-read",
            "POST",
            {
                "meter_type": meter_type,
                "meter_id": meter_id,
                "calibration": dict(calibration),
                "timeout_s": max(0.05, min(10.0, timeout_s)),
                "max_frame_age_ms": 1500,
                "required_camera_mode": INSPECTION_CAMERA_MODE,
                "evidence_session": evidence_session,
                "waypoint": waypoint,
                "capture_index": int(capture_index),
            },
            timeout_s=max(1.0, min(20.0, timeout_s + 1.0)),
        )

    def activate_meter(self, meter_id: str) -> dict[str, Any]:
        return self._request(
            f"/api/meter-configurations/{quote(meter_id, safe='')}/activate",
            "POST",
            {},
            timeout_s=20.0,
        )

    def test_read(self, meter_id: str, timeout_s: float) -> dict[str, Any]:
        return self._request(
            f"/api/meter-configurations/{quote(meter_id, safe='')}/test-read",
            "POST",
            {},
            timeout_s=max(1.0, min(20.0, timeout_s + 1.0)),
        )

    def switch_mode(self, mode: str) -> dict[str, Any]:
        return self._request("/api/camera-mode", "POST", {"mode": mode}, timeout_s=20.0)

    def wait_meter_ready(
        self,
        meter_id: str,
        timeout_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        deadline = clock() + timeout_s
        last: Optional[dict[str, Any]] = None
        while clock() < deadline:
            last = self.meter_reading()
            if (
                last.get("mode") == "astra_meter_digital"
                and last.get("meter_id") == meter_id
                and last.get("ready") is True
            ):
                return last
            sleeper(0.25)
        raise VisionApiError(f"meter {meter_id} did not become ready: {last}")

    def wait_mode(
        self,
        mode: str,
        timeout_s: float = 12.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        deadline = clock() + timeout_s
        last: Optional[dict[str, Any]] = None
        while clock() < deadline:
            last = self.camera_status()
            cameras = last.get("cameras") or []
            camera = cameras[0] if cameras else {}
            inference = camera.get("inference") or {}
            inference_ready = (
                inference.get("enabled") is not True
                or inference.get("ready") is True
            )
            if (
                last.get("active_mode") == mode
                and camera.get("active") is True
                and camera.get("online") is True
                and inference_ready
            ):
                return last
            sleeper(0.25)
        raise VisionApiError(f"vision mode {mode} did not become ready: {last}")


def _reading_key(value: Any, numeric_tolerance: float) -> tuple[str, Any]:
    if isinstance(value, bool) or value is None:
        return ("invalid", None)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return ("text", text)
    if not math.isfinite(numeric):
        return ("invalid", None)
    if numeric_tolerance > 0.0:
        return ("number", int(round(numeric / numeric_tolerance)))
    return ("number", numeric)


def _accepted_trial(
    trial: Mapping[str, Any], point: InspectionPoint
) -> Optional[tuple[str, Any]]:
    if trial.get("status") not in ACCEPTED_METER_STATUSES:
        return None
    key = _reading_key(trial.get("value"), point.numeric_tolerance)
    if key[0] == "invalid" or (key[0] == "text" and not key[1]):
        return None
    try:
        confidence = float(trial.get("confidence"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or confidence < point.min_confidence:
        return None
    if point.raw_text_pattern is not None:
        raw_text = str(trial.get("raw_text", "")).strip()
        if re.fullmatch(point.raw_text_pattern, raw_text) is None:
            return None
    return key


def _resolve_decimal_only_ambiguity(
    trial: Mapping[str, Any], point: InspectionPoint
) -> Optional[dict[str, Any]]:
    """Use a declared display format only for a missing-decimal OCR conflict.

    Both alternatives must describe the same display box and differ only by
    the decimal point.  This does not by itself establish a reading: the route
    requires a second independent frame before accepting a resolved trial.
    """

    if (point.meter_type != "digital" or point.raw_text_pattern is None
            or trial.get("status") != "ambiguous"):
        return None
    scan = trial.get("display_scan")
    if (not isinstance(scan, Mapping) or scan.get("recognised") is not True
            or "crop variants disagreed" not in str(scan.get("refused", ""))):
        return None
    candidates = trial.get("candidates")
    if not isinstance(candidates, (list, tuple)) or len(candidates) != 2:
        return None
    parsed = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("source") != "display-region":
            return None
        try:
            text = str(candidate["text"]).strip()
            score = float(candidate["score"])
            box = tuple(float(value) for value in candidate["bbox"])
        except (KeyError, TypeError, ValueError):
            return None
        if (not math.isfinite(score) or score < point.min_confidence
                or len(box) != 4 or not all(math.isfinite(value) for value in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            return None
        parsed.append((text, score, box))
    if parsed[0][2] != parsed[1][2]:
        return None
    matching = [item for item in parsed if re.fullmatch(point.raw_text_pattern, item[0])]
    if len(matching) != 1:
        return None
    formatted, score, _ = matching[0]
    other = parsed[0][0] if parsed[0] != matching[0] else parsed[1][0]
    if formatted.count(".") != 1 or "." in other or formatted.replace(".", "") != other:
        return None
    try:
        value = float(formatted)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return {
        **trial,
        "source_status": "ambiguous",
        "status": "recognized",
        "value": value,
        "raw_text": formatted,
        "confidence": score,
        "format_resolution": "declared_decimal_point_second_frame_required",
    }


def _uses_threshold_condition_consensus(point: InspectionPoint) -> bool:
    """Use condition consensus for analogue readings that naturally jitter.

    Digital displays with a raw-text contract retain strict value agreement so
    two different plausible OCR values cannot be accepted merely because both
    happen to lie on the same side of a threshold.
    """

    return (
        point.meter_type == "pointer"
        and
        point.raw_text_pattern is None
        and (point.normal_min is not None or point.normal_max is not None)
    )


def _trial_proves_meter_visible(trial: Mapping[str, Any]) -> bool:
    """Return whether the camera has located a meter/display in this frame.

    A bounded heading search is useful when the target is absent from the view.
    Once the target is visible, OCR disagreement should be handled by additional
    stationary frames rather than repeated left-right chassis corrections.
    """

    status = str(trial.get("status", "")).strip().lower()
    if status in {*ACCEPTED_METER_STATUSES, "ambiguous", "out_of_range"}:
        return True
    if trial.get("gauge_box"):
        return True
    display_scan = trial.get("display_scan")
    if isinstance(display_scan, Mapping) and display_scan.get("recognised") is True:
        return True
    candidates = trial.get("candidates")
    return bool(isinstance(candidates, Sequence) and candidates)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        temporary = Path(stream.name)
    os.replace(temporary, path)


class InspectionExecutor:
    """Run a stationary, bounded-search meter inspection at one waypoint."""

    def __init__(
        self,
        plan: InspectionPlan,
        client: Optional[VisionHttpClient] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        evidence_session: Optional[str] = None,
    ):
        self.plan = plan
        self.client = client or VisionHttpClient(plan.vision_api_url)
        self.clock = clock
        self.sleeper = sleeper
        self.evidence_session = evidence_session or datetime.now().astimezone().strftime(
            "%Y%m%d-%H%M%S"
        )
        self._capture_counts: Counter[str] = Counter()

    def readiness(self) -> dict[str, Any]:
        """Read-only service/configuration probe; it never switches the camera."""

        status = self.client.camera_status()
        capabilities = self.client.current_meter_capabilities()
        supported_types = set(capabilities.get("supported_meter_types") or [])
        available_modes = {
            str(item.get("id"))
            for item in (status.get("modes") or [])
            if isinstance(item, Mapping)
        }
        cameras = status.get("cameras") or []
        active_camera = cameras[0] if cameras else {}
        points = []
        failures = []
        if capabilities.get("switches_camera_mode") is not False:
            failures.append("meter_current_read_must_not_switch_camera_mode")
        if capabilities.get("supports_evidence_capture") is not True:
            failures.append("meter_current_read_must_save_inspection_evidence")
        if capabilities.get("required_camera_mode") != INSPECTION_CAMERA_MODE:
            failures.append("meter_current_read_must_require_astra_raw")
        if INSPECTION_CAMERA_MODE not in available_modes:
            failures.append("astra_raw_camera_mode_unavailable")
        if active_camera.get("id") != "astra":
            failures.append("active_camera_is_not_astra")
        if active_camera.get("active") is not True:
            failures.append("active_camera_not_running")
        if active_camera.get("online") is not True:
            failures.append("active_camera_not_online")
        for name, point in sorted(self.plan.points.items()):
            if not point.enabled:
                points.append({"waypoint": name, "status": "DISABLED"})
                continue
            point_failures = []
            if point.meter_type not in supported_types:
                point_failures.append("meter_type_not_supported")
            if point_failures:
                failures.extend(f"{name}:{item}" for item in point_failures)
            points.append(
                {
                    "waypoint": name,
                    "task_id": point.task_id,
                    "meter_id": point.meter_id,
                    "meter_type": point.meter_type,
                    "normal_rule": point.normal_rule(),
                    "status": "PASS" if not point_failures else "FAIL",
                    "failures": point_failures,
                }
            )
        return {
            "status": "PASS" if not failures else "FAIL",
            "motion_authorized": False,
            "vision_api_url": self.plan.vision_api_url,
            "active_camera_mode": status.get("active_mode"),
            "meter_current_read_api_version": capabilities.get("api_version"),
            "required_camera_mode": capabilities.get("required_camera_mode"),
            "inspection_photo_root": capabilities.get("evidence_root"),
            "failures": failures,
            "points": points,
        }

    def execute(
        self,
        waypoint_pose: Mapping[str, Any],
        route_index: int,
        navigation_result: Mapping[str, Any],
        align_heading: Callable[[float], Mapping[str, Any]],
        *,
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        name = str(waypoint_pose.get("name", "")).upper()
        point = self.plan.point(name)
        if point is None or not point.enabled:
            return {
                "status": "SKIPPED",
                "waypoint": name,
                "task_id": None if point is None else point.task_id,
                "failure_policy": "continue",
                "motion_authorized": False,
            }
        request_id = str(uuid4())
        started = self.clock()
        result: dict[str, Any] = {
            "request_id": request_id,
            "route_index": int(route_index),
            "waypoint": name,
            "task_id": point.task_id,
            "kind": point.kind,
            "meter_id": point.meter_id,
            "meter_type": point.meter_type,
            "normal_rule": point.normal_rule(),
            "status": "FAIL",
            "task_succeeded": False,
            "failure_policy": point.failure_policy,
            "motion_authorized": False,
            "heading_attempts": [],
            "trials": [],
            "trial_count": 0,
            "evidence_session": self.evidence_session,
            "failure_stage": None,
            "error": None,
        }
        original_mode: Optional[str] = None
        base_yaw: Optional[float] = None
        last_heading_target: Optional[float] = None
        last_observed_heading: Optional[float] = None
        last_heading_safe = False
        try:
            if navigation_result.get("final_armed") is not False:
                raise RuntimeError("inspection requires a verified DISARMED navigation result")
            status = self.client.camera_status()
            original_mode_value = status.get("active_mode")
            if isinstance(original_mode_value, str) and original_mode_value:
                original_mode = original_mode_value
            capabilities = self.client.current_meter_capabilities()
            if capabilities.get("switches_camera_mode") is not False:
                raise RuntimeError("meter current-read API may switch camera mode")
            if capabilities.get("required_camera_mode") != INSPECTION_CAMERA_MODE:
                raise RuntimeError(
                    f"meter current-read API must require {INSPECTION_CAMERA_MODE}"
                )
            if point.meter_type not in set(capabilities.get("supported_meter_types") or []):
                raise RuntimeError(
                    f"meter type {point.meter_type} is not supported by the vision API"
                )

            transition = {
                "requested": original_mode != INSPECTION_CAMERA_MODE,
                "original_mode": original_mode,
                "target_mode": INSPECTION_CAMERA_MODE,
                "status": "PENDING",
            }
            result["camera_transition"] = transition
            if original_mode != INSPECTION_CAMERA_MODE:
                self.client.switch_mode(INSPECTION_CAMERA_MODE)
            ready_status = self.client.wait_mode(
                INSPECTION_CAMERA_MODE,
                timeout_s=INSPECTION_CAMERA_SWITCH_TIMEOUT_S,
                clock=self.clock,
                sleeper=self.sleeper,
            )
            transition.update(
                {
                    "status": "PASS",
                    "active_mode": ready_status.get("active_mode"),
                }
            )

            deadline = started + point.timeout_s
            counts: Counter[tuple[str, Any]] = Counter()
            accepted_by_key: dict[tuple[str, Any], list[dict[str, Any]]] = {}
            last_failure = "no accepted meter reading"
            # The robot may finish a Nav2 position-only goal away from the
            # catalog centre.  Aim at the configured meter from the measured
            # stopped pose, never from the ideal waypoint alone.
            arrival_pose, _ = trusted_stopped_heading_observation(navigation_result)
            reference_pose = arrival_pose or waypoint_pose
            base_yaw = point.base_yaw_rad(reference_pose)
            heading_candidates = point.heading_candidates_rad(reference_pose)
            if arrival_pose is not None:
                result["aim_reference_pose"] = arrival_pose
            for heading_index, target_yaw in enumerate(heading_candidates, start=1):
                if self.clock() >= deadline:
                    last_failure = "inspection timeout before heading search completed"
                    break
                if point.inspection_yaw_deg is None and result.get("final_amcl_pose"):
                    target_yaw = normalize_angle(
                        point.base_yaw_rad(result["final_amcl_pose"])
                        + math.radians(point.search_offsets_deg[heading_index - 1])
                    )
                last_heading_target = target_yaw
                stationary_capture_fallback = False
                last_heading_safe = False
                heading_result: dict[str, Any] = {}
                observed_pose = None
                observed_heading = None
                safe_stop_retries = 0
                bearing_reaims = 0
                for alignment_attempt in range(1, 4):
                    heading_result = dict(align_heading(target_yaw))
                    heading_result["target_yaw_rad"] = target_yaw
                    heading_result["target_yaw_deg"] = math.degrees(target_yaw)
                    heading_result["alignment_attempt"] = alignment_attempt
                    result["heading_attempts"].append(heading_result)
                    observed_pose, observed_heading = trusted_stopped_heading_observation(
                        heading_result
                    )
                    if observed_pose is not None:
                        result["final_amcl_pose"] = observed_pose
                    if observed_heading is not None:
                        result["final_heading_rad"] = observed_heading
                        result["final_heading_deg"] = math.degrees(observed_heading)
                        result["final_armed"] = False
                        result["final_state"] = heading_result.get(
                            "final_state", "DISARMED"
                        )
                    required_yaw = target_yaw
                    if point.inspection_yaw_deg is None and observed_pose is not None:
                        previous_pose = result.get("aim_reference_pose") or reference_pose
                        shift = math.hypot(
                            observed_pose["x"] - float(previous_pose["x"]),
                            observed_pose["y"] - float(previous_pose["y"]),
                        )
                        if shift > 0.35:
                            result["failure_stage"] = "heading_alignment"
                            last_failure = "localization shifted during inspection turn"
                            break
                        required_yaw = normalize_angle(
                            point.base_yaw_rad(observed_pose)
                            + math.radians(point.search_offsets_deg[heading_index - 1])
                        )
                    heading_error = (
                        None if observed_heading is None
                        else abs(normalize_angle(observed_heading - required_yaw))
                    )
                    if heading_result.get("status") == "PASS":
                        # A successful Spin can still end outside the camera's
                        # bearing tolerance after AMCL or the base position moves.
                        # Never turn a wrapper PASS into an unverified OCR capture.
                        if (heading_error is None
                                or heading_error > INSPECTION_BASE_HEADING_TOLERANCE_RAD):
                            if (heading_error is not None and bearing_reaims == 0
                                    and alignment_attempt < 3 and self.clock() < deadline):
                                bearing_reaims += 1
                                reason = "meter bearing or stopped heading changed"
                                result["heading_alignment_retry_count"] = alignment_attempt
                                result["heading_alignment_retry_reason"] = reason
                                result.setdefault("heading_alignment_retry_reasons", []).append(reason)
                                target_yaw = required_yaw
                                last_heading_target = required_yaw
                                continue
                            result["failure_stage"] = "heading_alignment"
                            last_failure = (
                                "inspection heading observation missing"
                                if heading_error is None
                                else "inspection heading remained outside tolerance"
                            )
                            break
                        last_heading_safe = True
                        break

                    safely_stopped = safe_stationary_capture_after_heading_abort(
                        heading_result,
                        observed_pose,
                        observed_heading,
                    )
                    within_capture_tolerance = (
                        safely_stopped
                        and heading_error is not None
                        and heading_error <= INSPECTION_BASE_HEADING_TOLERANCE_RAD
                    )
                    if within_capture_tolerance:
                        stationary_capture_fallback = True
                        result["heading_alignment_degraded"] = True
                        result["heading_alignment_error"] = heading_result.get("error")
                        result["heading_capture_fallback"] = (
                            "safe_stationary_current_frame_within_tolerance"
                        )
                        last_heading_safe = True
                        break
                    if (safely_stopped and safe_stop_retries == 0
                            and alignment_attempt < 3 and self.clock() < deadline):
                        safe_stop_retries += 1
                        reason = "safe stop remained outside camera heading tolerance"
                        result["heading_alignment_retry_count"] = alignment_attempt
                        result["heading_alignment_retry_reason"] = reason
                        result.setdefault("heading_alignment_retry_reasons", []).append(reason)
                        target_yaw = required_yaw
                        last_heading_target = required_yaw
                        continue
                    result["failure_stage"] = "heading_alignment"
                    last_failure = "inspection heading alignment failed"
                    break
                if not last_heading_safe:
                    break
                if heading_result.get("final_armed") is not False:
                    last_heading_safe = False
                    result["failure_stage"] = "heading_alignment"
                    last_failure = "inspection heading did not finish DISARMED"
                    break
                if observed_heading is None:
                    observed_heading = normalize_angle(target_yaw)
                    result["final_heading_rad"] = observed_heading
                    result["final_heading_deg"] = math.degrees(observed_heading)
                last_observed_heading = observed_heading
                alignment_failed = False
                meter_visible = False
                for _ in range(point.reads_per_heading):
                    remaining = deadline - self.clock()
                    if remaining <= 0.0:
                        last_failure = "inspection timeout while collecting readings"
                        break
                    self._capture_counts[name] += 1
                    response = self.client.current_read(
                        point.meter_type,
                        point.meter_id or "",
                        point.calibration,
                        min(10.0, remaining),
                        evidence_session=self.evidence_session,
                        waypoint=name,
                        capture_index=self._capture_counts[name],
                    )
                    trial = response.get("trial") if isinstance(response, Mapping) else None
                    if not isinstance(trial, Mapping):
                        raise VisionApiError("meter trial response is missing trial data")
                    trial_copy = dict(trial)
                    resolved = _resolve_decimal_only_ambiguity(trial_copy, point)
                    if resolved is not None:
                        trial_copy = resolved
                    trial_copy["heading_index"] = heading_index
                    trial_copy["target_yaw_deg"] = math.degrees(target_yaw)
                    result["trials"].append(trial_copy)
                    meter_visible = meter_visible or _trial_proves_meter_visible(
                        trial_copy
                    )
                    if trial_copy.get("status") == "alignment_failed":
                        alignment_failed = True
                        last_failure = "meter was not aligned in the current camera view"
                        break
                    key = _accepted_trial(trial_copy, point)
                    if key is None:
                        last_failure = str(
                            trial_copy.get("reason")
                            or trial_copy.get("status")
                            or "meter reading rejected"
                        )
                        continue
                    condition, is_anomaly = point.classify_value(
                        trial_copy.get("value")
                    )
                    consensus_mode = "value"
                    consensus_key = key
                    if _uses_threshold_condition_consensus(point):
                        consensus_mode = "threshold_condition"
                        consensus_key = ("condition", condition)
                    counts[consensus_key] += 1
                    accepted_by_key.setdefault(consensus_key, []).append(trial_copy)
                    required_reads = max(
                        point.min_consistent_reads,
                        2 if trial_copy.get("format_resolution") else 1,
                    )
                    if counts[consensus_key] >= required_reads:
                        accepted_trials = accepted_by_key[consensus_key]
                        accepted = accepted_trials[-1]
                        accepted_value = accepted.get("value")
                        consensus_values = []
                        if consensus_mode == "threshold_condition":
                            consensus_values = [
                                float(item["value"]) for item in accepted_trials
                            ]
                            accepted_value = float(median(consensus_values))
                            accepted = min(
                                accepted_trials,
                                key=lambda item: abs(
                                    float(item["value"]) - accepted_value
                                ),
                            )
                        result.update(
                            {
                                "status": "PASS",
                                "task_succeeded": True,
                                "error": None,
                                "reading": accepted,
                                "value": accepted_value,
                                "unit": accepted.get("unit"),
                                "confidence": accepted.get("confidence"),
                                "condition": condition,
                                "is_anomaly": is_anomaly,
                                "speech_key": (
                                    "meter_abnormal" if is_anomaly else "meter_normal"
                                ),
                                "reading_consensus": {
                                    "mode": consensus_mode,
                                    "count": counts[consensus_key],
                                    "required": required_reads,
                                    "condition": condition,
                                    "values": consensus_values,
                                    "representative": (
                                        "median"
                                        if consensus_mode == "threshold_condition"
                                        else "matching_value"
                                    ),
                                },
                            }
                        )
                        break
                if result["task_succeeded"]:
                    break
                if stationary_capture_fallback:
                    break
                if not alignment_failed and meter_visible:
                    result["heading_search_stopped_reason"] = (
                        "meter is already visible; do not oscillate the chassis for "
                        "an OCR or consistency failure"
                    )
                    break
                if not alignment_failed and self.clock() >= deadline:
                    break
            if not result["task_succeeded"]:
                if result["failure_stage"] is None:
                    result["failure_stage"] = "vision"
                result["error"] = last_failure
        except Exception as exc:
            transition = result.get("camera_transition")
            if isinstance(transition, dict) and transition.get("status") == "PENDING":
                transition.update(
                    {
                        "status": "FAIL",
                        "error": str(exc) or type(exc).__name__,
                        "error_type": type(exc).__name__,
                    }
                )
            if result["failure_stage"] is None:
                result["failure_stage"] = "runtime"
            result["error"] = str(exc) or type(exc).__name__
            result["error_type"] = type(exc).__name__
        finally:
            if point.inspection_yaw_deg is None and result.get("final_amcl_pose"):
                try:
                    base_yaw = point.base_yaw_rad(result["final_amcl_pose"])
                except InspectionConfigError:
                    pass
            heading_restoration: dict[str, Any] = {
                "requested": False,
                "base_yaw_rad": base_yaw,
                "base_yaw_deg": None if base_yaw is None else math.degrees(base_yaw),
                "status": "NOT_REQUIRED",
            }
            if base_yaw is not None and last_heading_target is not None:
                heading_error = (
                    None
                    if last_observed_heading is None
                    else abs(normalize_angle(last_observed_heading - base_yaw))
                )
                heading_restoration["pre_restore_error_rad"] = heading_error
                heading_restoration["pre_restore_error_deg"] = (
                    None if heading_error is None else math.degrees(heading_error)
                )
                if (
                    heading_error is None
                    or heading_error > INSPECTION_BASE_HEADING_TOLERANCE_RAD
                ):
                    if not last_heading_safe:
                        heading_restoration.update(
                            {
                                "status": "SKIPPED_UNSAFE",
                                "reason": "last heading attempt did not prove a safe stopped state",
                            }
                        )
                    else:
                        heading_restoration["requested"] = True
                        try:
                            restore_result = dict(align_heading(base_yaw))
                            heading_restoration["result"] = restore_result
                            restored_pose, restored_heading = trusted_stopped_heading_observation(
                                restore_result
                            )
                            restore_error = (
                                None
                                if restored_heading is None
                                else abs(normalize_angle(restored_heading - base_yaw))
                            )
                            heading_restoration["final_error_rad"] = restore_error
                            heading_restoration["final_error_deg"] = (
                                None if restore_error is None else math.degrees(restore_error)
                            )
                            restore_safely_stopped = (
                                restore_result.get("final_armed") is False
                                and restored_heading is not None
                            )
                            restore_within_tolerance = (
                                restore_safely_stopped
                                and restore_error is not None
                                and restore_error
                                <= INSPECTION_BASE_HEADING_TOLERANCE_RAD
                            )
                            if restore_safely_stopped:
                                if restored_pose is not None:
                                    result["final_amcl_pose"] = restored_pose
                                result["final_heading_rad"] = restored_heading
                                result["final_heading_deg"] = math.degrees(
                                    restored_heading
                                )
                                result["final_armed"] = False
                                result["final_state"] = restore_result.get(
                                    "final_state", "DISARMED"
                                )
                            if restore_within_tolerance:
                                heading_restoration["status"] = "PASS"
                                if restore_result.get("status") != "PASS":
                                    heading_restoration[
                                        "terminal_action_failure_accepted"
                                    ] = True
                            elif restore_safely_stopped and result.get("status") == "PASS":
                                # The meter result is already valid. A safely stopped
                                # post-capture heading miss is a navigation warning, not
                                # a vision failure, and the next route leg can plan from
                                # the measured final pose.
                                heading_restoration["status"] = "DEGRADED"
                                heading_restoration["reason"] = (
                                    "safe stopped heading remains outside tolerance"
                                )
                                result["heading_restoration_degraded"] = True
                                result["heading_restore_error"] = (
                                    "inspection base heading restoration remained "
                                    "outside tolerance"
                                )
                            else:
                                heading_restoration["status"] = "FAIL"
                                if result.get("status") == "PASS":
                                    result["status"] = "FAIL"
                                    result["task_succeeded"] = False
                                    result["failure_stage"] = "heading_restore"
                                    result["error"] = "inspection base heading restoration failed"
                                else:
                                    result["heading_restore_error"] = (
                                        "inspection base heading restoration failed"
                                    )
                        except Exception as exc:
                            heading_restoration.update(
                                {
                                    "status": "FAIL",
                                    "error": str(exc) or type(exc).__name__,
                                    "error_type": type(exc).__name__,
                                }
                            )
                            if result.get("status") == "PASS":
                                result["status"] = "FAIL"
                                result["task_succeeded"] = False
                                result["failure_stage"] = "heading_restore"
                                result["error"] = "inspection base heading restoration failed"
                            else:
                                result["heading_restore_error"] = (
                                    "inspection base heading restoration failed"
                                )
            result["heading_restoration"] = heading_restoration

            restoration: dict[str, Any] = {
                "requested": bool(
                    self.plan.restore_camera_mode
                    and original_mode
                    and original_mode != INSPECTION_CAMERA_MODE
                ),
                "original_mode": original_mode,
                "status": "NOT_REQUIRED",
            }
            if restoration["requested"]:
                try:
                    self.client.switch_mode(original_mode)
                    restored_status = self.client.wait_mode(
                        original_mode,
                        timeout_s=INSPECTION_CAMERA_SWITCH_TIMEOUT_S,
                        clock=self.clock,
                        sleeper=self.sleeper,
                    )
                    restoration.update(
                        {
                            "status": "PASS",
                            "active_mode": restored_status.get("active_mode"),
                        }
                    )
                except Exception as exc:
                    restoration.update(
                        {
                            "status": "FAIL",
                            "error": str(exc) or type(exc).__name__,
                            "error_type": type(exc).__name__,
                        }
                    )
            elif not self.plan.restore_camera_mode:
                restoration["status"] = "DISABLED"
                restoration["reason"] = "restore_camera_mode is false"
            elif original_mode == INSPECTION_CAMERA_MODE:
                restoration["reason"] = "camera already used the inspection raw mode"
            else:
                restoration["reason"] = "original camera mode was unavailable"
            result["camera_restoration"] = restoration
            result["trial_count"] = len(result["trials"])
            result["duration_s"] = round(self.clock() - started, 3)
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"{int(route_index):03d}-{name}-{request_id}.json"
            output_path = self.plan.result_root / identifier / filename
            result["result_path"] = str(output_path)
            try:
                _atomic_write_json(output_path, result)
            except OSError as exc:
                result["result_path"] = None
                result["result_persist_error"] = str(exc)
                if result["status"] == "PASS":
                    result["status"] = "FAIL"
                    result["task_succeeded"] = False
                    result["error"] = "inspection result could not be persisted"
        return result


def plan_summary(plan: InspectionPlan) -> dict[str, Any]:
    return {
        "vision_api_url": plan.vision_api_url,
        "result_root": str(plan.result_root),
        "restore_camera_mode": plan.restore_camera_mode,
        "speech_ack_timeout_s": plan.speech_ack_timeout_s,
        "points": {
            name: {
                "enabled": point.enabled,
                "task_id": point.task_id,
                "kind": point.kind,
                "meter_id": point.meter_id,
                "meter_type": point.meter_type,
                "calibration": dict(point.calibration),
                "normal_rule": point.normal_rule(),
                "inspection_yaw_deg": point.inspection_yaw_deg,
                "target": (
                    None
                    if point.target_x_m is None
                    else {"x_m": point.target_x_m, "y_m": point.target_y_m}
                ),
                "search_offsets_deg": list(point.search_offsets_deg),
                "reads_per_heading": point.reads_per_heading,
                "min_consistent_reads": point.min_consistent_reads,
                "min_confidence": point.min_confidence,
                "failure_policy": point.failure_policy,
            }
            for name, point in sorted(plan.points.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a route-inspection plan and probe the local vision API"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    try:
        plan = load_inspection_plan(args.config)
        result = plan_summary(plan) if args.describe else InspectionExecutor(plan).readiness()
    except (InspectionConfigError, VisionApiError, OSError) as exc:
        result = {
            "status": "FAIL",
            "motion_authorized": False,
            "error": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if args.describe or result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
