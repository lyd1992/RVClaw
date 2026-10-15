"""Pure policy and persistence helpers for fail-closed localization startup.

The runtime deliberately separates a *running* navigation stack from a
*trusted* map pose. A lidar or hardware interruption invalidates the live
pose, while a verified cache remains only a candidate until a fresh scan
agrees with it. Explicit charging/reposition events and repeated scan/map
mismatches force full global localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from typing import Dict, Iterable, List, Optional

import yaml


CACHE_SCHEMA_VERSION = 1

HOT_READY = "HOT_READY"
WARM_VALIDATION_REQUIRED = "WARM_VALIDATION_REQUIRED"
GLOBAL_LOCALIZATION_REQUIRED = "GLOBAL_LOCALIZATION_REQUIRED"
WAIT_FOR_SENSORS = "WAIT_FOR_SENSORS"
BLOCKED_CHARGING = "BLOCKED_CHARGING"
TEMPORARY_SCAN_OBSTRUCTION = "TEMPORARY_SCAN_OBSTRUCTION"

REUSE_LIVE_POSE = "REUSE_LIVE_POSE"
VALIDATE_CACHED_POSE = "VALIDATE_CACHED_POSE"
RUN_GLOBAL_LOCALIZATION = "RUN_GLOBAL_LOCALIZATION"


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def map_identity(yaml_path) -> Dict[str, object]:
    """Hash the map metadata and image so cached poses cannot cross maps."""

    yaml_path = Path(yaml_path).resolve()
    metadata_bytes = yaml_path.read_bytes()
    metadata = yaml.safe_load(metadata_bytes.decode("utf-8"))
    if not isinstance(metadata, dict) or "image" not in metadata:
        raise ValueError("map YAML must contain an image field")
    image_path = (yaml_path.parent / str(metadata["image"])).resolve()
    image_bytes = image_path.read_bytes()
    digest = hashlib.sha256()
    digest.update(b"rvclaw-map-v1\0")
    digest.update(metadata_bytes)
    digest.update(b"\0")
    digest.update(image_bytes)
    return {
        "yaml_path": str(yaml_path),
        "image_path": str(image_path),
        "sha256": digest.hexdigest(),
        "resolution": float(metadata["resolution"]),
    }


def build_scan_signature(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    *,
    bin_count: int = 72,
    maximum_range_m: float = 8.0,
) -> Dict[str, object]:
    """Build a compact, noise-resistant signature from one LaserScan."""

    values = list(ranges)
    if bin_count < 12:
        raise ValueError("scan signature requires at least 12 bins")
    if len(values) < bin_count:
        raise ValueError("scan has fewer samples than signature bins")
    if not all(
        _finite(value)
        for value in (angle_min, angle_increment, range_min, range_max)
    ):
        raise ValueError("scan metadata contains non-finite values")
    if angle_increment <= 0.0 or range_min < 0.0 or range_max <= range_min:
        raise ValueError("scan metadata is invalid")

    clipped_max = min(float(range_max), float(maximum_range_m))
    signature: List[Optional[float]] = []
    for index in range(bin_count):
        start = int(index * len(values) / bin_count)
        stop = int((index + 1) * len(values) / bin_count)
        usable = [
            float(value)
            for value in values[start:stop]
            if _finite(value) and float(range_min) <= float(value) <= clipped_max
        ]
        signature.append(
            None if not usable else round(float(statistics.median(usable)), 4)
        )

    angle_max = float(angle_min) + (len(values) - 1) * float(angle_increment)
    return {
        "schema_version": 1,
        "bin_count": int(bin_count),
        "sample_count": len(values),
        "angle_min": float(angle_min),
        "angle_max": angle_max,
        "range_cap_m": clipped_max,
        "values": signature,
    }


def compare_scan_signatures(
    reference: Dict[str, object],
    current: Dict[str, object],
    *,
    maximum_median_delta_m: float = 0.20,
    maximum_p90_delta_m: float = 0.55,
    maximum_close_delta_m: float = 0.25,
    minimum_overlap_ratio: float = 0.65,
    minimum_close_ratio: float = 0.70,
) -> Dict[str, object]:
    """Reject a cached pose unless the surrounding range pattern still agrees."""

    failures: List[str] = []
    ref_values = reference.get("values") if isinstance(reference, dict) else None
    cur_values = current.get("values") if isinstance(current, dict) else None
    if not isinstance(ref_values, list) or not isinstance(cur_values, list):
        return {
            "status": "FAIL",
            "failures": ["scan signature values are missing"],
        }
    if len(ref_values) != len(cur_values) or len(ref_values) < 12:
        return {
            "status": "FAIL",
            "failures": ["scan signature shapes differ"],
        }

    angle_error = max(
        abs(
            float(reference.get("angle_min", math.inf))
            - float(current.get("angle_min", -math.inf))
        ),
        abs(
            float(reference.get("angle_max", math.inf))
            - float(current.get("angle_max", -math.inf))
        ),
    )
    if not math.isfinite(angle_error) or angle_error > math.radians(2.0):
        failures.append("scan angular coverage changed")

    deltas = [
        abs(float(first) - float(second))
        for first, second in zip(ref_values, cur_values)
        if _finite(first) and _finite(second)
    ]
    overlap_ratio = len(deltas) / len(ref_values)
    if overlap_ratio < minimum_overlap_ratio:
        failures.append("scan signature overlap is too small")

    median_delta = math.inf
    p90_delta = math.inf
    close_ratio = 0.0
    if deltas:
        ordered = sorted(deltas)
        median_delta = float(statistics.median(ordered))
        p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
        p90_delta = float(ordered[p90_index])
        close_ratio = (
            sum(delta <= maximum_close_delta_m for delta in ordered) / len(ordered)
        )
        if median_delta > maximum_median_delta_m:
            failures.append("scan signature median delta is too large")
        if p90_delta > maximum_p90_delta_m:
            failures.append("scan signature p90 delta is too large")
        if close_ratio < minimum_close_ratio:
            failures.append("too few scan signature bins remain close")
    else:
        failures.append("scan signatures have no overlapping finite bins")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "overlap_ratio": overlap_ratio,
        "median_delta_m": median_delta,
        "p90_delta_m": p90_delta,
        "close_ratio": close_ratio,
        "thresholds": {
            "maximum_median_delta_m": maximum_median_delta_m,
            "maximum_p90_delta_m": maximum_p90_delta_m,
            "maximum_close_delta_m": maximum_close_delta_m,
            "minimum_overlap_ratio": minimum_overlap_ratio,
            "minimum_close_ratio": minimum_close_ratio,
        },
    }


def build_pose_cache(
    map_info: Dict[str, object],
    pose: Dict[str, float],
    scan_signature: Dict[str, object],
    *,
    saved_at_s: Optional[float] = None,
    source: str = "verified_runtime",
    validation_epoch: int = 0,
) -> Dict[str, object]:
    required = ("x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
    if not all(_finite(pose.get(key)) for key in required):
        raise ValueError("pose cache contains missing or non-finite pose fields")
    if any(
        float(pose[key]) < 0.0 for key in ("cov_x", "cov_y", "cov_yaw")
    ):
        raise ValueError("pose cache covariance cannot be negative")
    map_hash = map_info.get("sha256")
    if not isinstance(map_hash, str) or len(map_hash) != 64:
        raise ValueError("map identity does not contain a SHA-256 digest")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "saved_at_s": float(time.time() if saved_at_s is None else saved_at_s),
        "source": str(source),
        "validation_epoch": int(validation_epoch),
        "map": {
            "sha256": map_hash,
            "yaml_path": str(map_info.get("yaml_path", "")),
        },
        "pose": {key: float(pose[key]) for key in required},
        "scan_signature": scan_signature,
        "trust": "verified_candidate",
        "hard_invalidated": False,
        "invalidation_reason": None,
    }


def validate_pose_cache(
    document: Dict[str, object],
    current_map: Dict[str, object],
    *,
    now_s: Optional[float] = None,
    maximum_age_s: float = 7.0 * 24.0 * 3600.0,
    maximum_position_variance: float = 0.10,
    maximum_yaw_variance: float = math.radians(10.0) ** 2,
) -> Dict[str, object]:
    failures: List[str] = []
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != CACHE_SCHEMA_VERSION
    ):
        return {
            "status": "FAIL",
            "failures": ["pose cache schema is unsupported"],
        }
    if document.get("hard_invalidated"):
        failures.append("pose cache was hard-invalidated")
    cached_map = document.get("map", {})
    if cached_map.get("sha256") != current_map.get("sha256"):
        failures.append("pose cache map identity does not match")

    saved_at = document.get("saved_at_s")
    current_time = time.time() if now_s is None else float(now_s)
    if not _finite(saved_at):
        failures.append("pose cache timestamp is invalid")
        age_s = math.inf
    else:
        age_s = current_time - float(saved_at)
        if age_s < -5.0:
            failures.append("pose cache timestamp is in the future")
        elif age_s > maximum_age_s:
            failures.append("pose cache is older than the admitted window")

    pose = document.get("pose", {})
    required = ("x", "y", "yaw", "cov_x", "cov_y", "cov_yaw")
    if not isinstance(pose, dict) or not all(
        _finite(pose.get(key)) for key in required
    ):
        failures.append("pose cache contains invalid pose values")
    else:
        if (
            float(pose["cov_x"]) > maximum_position_variance
            or float(pose["cov_y"]) > maximum_position_variance
        ):
            failures.append("pose cache position covariance is too large")
        if float(pose["cov_yaw"]) > maximum_yaw_variance:
            failures.append("pose cache yaw covariance is too large")
    if not isinstance(document.get("scan_signature"), dict):
        failures.append("pose cache has no scan signature")

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "age_s": age_s,
        "pose": pose if isinstance(pose, dict) else {},
        "scan_signature": document.get("scan_signature"),
    }


def validate_pose_candidate(
    document: Dict[str, object],
    current_map: Dict[str, object],
    **kwargs,
) -> Dict[str, object]:
    """Validate a cached pose only as a hypothesis for independent matching.

    A charging or manual-reposition event permanently removes direct trust, but
    it does not need to destroy the last verified pose and scan signature.  A
    hard-invalidated cache may therefore seed a bounded local scan search; it
    can never authorize motion or become trusted again without fresh map
    matching, AMCL no-motion updates, and runtime-manager acceptance.
    """

    hard_invalidated = bool(
        isinstance(document, dict) and document.get("hard_invalidated")
    )
    candidate_document = dict(document) if isinstance(document, dict) else document
    if isinstance(candidate_document, dict):
        candidate_document["hard_invalidated"] = False
    result = validate_pose_cache(candidate_document, current_map, **kwargs)
    result.update(
        {
            "candidate_only": True,
            "hard_invalidated": hard_invalidated,
            "invalidation_reason": (
                document.get("invalidation_reason")
                if isinstance(document, dict)
                else None
            ),
        }
    )
    return result


def load_pose_cache(path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pose_cache(path, document: Dict[str, object]) -> None:
    """Atomically replace a cache file; an interrupted write keeps the old one."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def invalidate_pose_cache(
    path,
    reason: str,
    *,
    now_s: Optional[float] = None,
) -> Dict[str, object]:
    document = load_pose_cache(path)
    document["hard_invalidated"] = True
    document["invalidation_reason"] = str(reason)
    document["invalidated_at_s"] = float(
        time.time() if now_s is None else now_s
    )
    document["trust"] = "invalid"
    save_pose_cache(path, document)
    return document


def continuous_pose_reusable(
    *,
    pose_available: bool,
    pose_age_s: float,
    maximum_pose_age_s: float,
    external_verification_epoch: int,
    runtime_gap_observed: bool,
    explicit_reposition: bool,
) -> bool:
    """Keep a stationary verified pose usable while live scans still validate it."""

    if not pose_available:
        return False
    if float(pose_age_s) <= float(maximum_pose_age_s):
        return True
    return (
        int(external_verification_epoch) > 0
        and not bool(runtime_gap_observed)
        and not bool(explicit_reposition)
    )


def controlled_motion_validation_deferred(
    *,
    armed: Optional[bool],
    controller_state: Optional[str],
    last_motion_age_s: Optional[float],
    settle_s: float,
) -> bool:
    """Defer kidnapping checks while commanded motion is active or settling."""

    if armed is True or str(controller_state or "").upper() == "ACTIVE":
        return True
    if last_motion_age_s is None:
        return False
    return 0.0 <= float(last_motion_age_s) < max(0.0, float(settle_s))


def temporary_obstruction_grace_active(
    *,
    safety_pause_age_s: Optional[float],
    grace_s: float,
    explicit_reposition: bool,
    runtime_gap_observed: bool,
    hardware_connected: bool,
    armed: Optional[bool],
    pose_message_fresh: bool,
    covariance_ok: bool,
    position_jump_m: Optional[float],
    yaw_jump_rad: Optional[float],
    maximum_position_jump_m: float,
    maximum_yaw_jump_rad: float,
) -> bool:
    """Keep a recent safety-stop scan mismatch soft, never navigation-valid.

    A person or object that triggered the collision monitor can dominate the
    lidar view for a few seconds after the chassis has stopped. Treating those
    scans as proof that the robot was carried away destroys an otherwise valid
    pose cache and makes a normal ``continue inspection`` command fall into an
    ambiguous global search. This grace only suppresses *hard invalidation*;
    the runtime remains invalid and cannot authorize motion until the live scan
    agrees with the map again.

    Explicit reposition reports, transport gaps, stale/uncertain poses, or a
    significant stationary AMCL jump always bypass the grace and retain the
    existing fail-closed global-localization behavior.
    """

    if safety_pause_age_s is None:
        return False
    try:
        age = float(safety_pause_age_s)
        grace = float(grace_s)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(age) or not math.isfinite(grace):
        return False
    if age < 0.0 or grace <= 0.0 or age > grace:
        return False
    if explicit_reposition or runtime_gap_observed:
        return False
    if not hardware_connected or armed is not False:
        return False
    if not pose_message_fresh or not covariance_ok:
        return False
    if position_jump_m is None or yaw_jump_rad is None:
        return False
    try:
        position_jump = float(position_jump_m)
        yaw_jump = abs(float(yaw_jump_rad))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(position_jump) or not math.isfinite(yaw_jump):
        return False
    return (
        position_jump <= max(0.0, float(maximum_position_jump_m))
        and yaw_jump <= max(0.0, float(maximum_yaw_jump_rad))
    )


@dataclass(frozen=True)
class StartupEvidence:
    charging: bool = False
    explicit_reposition: bool = False
    map_changed: bool = False
    hardware_connected: bool = False
    lidar_fresh: bool = False
    runtime_gap_observed: bool = False
    live_pose_fresh: bool = False
    live_covariance_ok: bool = False
    live_scan_map_ok: bool = False
    cache_valid: bool = False
    cache_signature_ok: bool = False
    consecutive_scan_mismatches: int = 0


def decide_fast_start(evidence: StartupEvidence) -> Dict[str, object]:
    """Choose the fastest safe localization path without authorizing motion."""

    reasons: List[str] = []
    if evidence.charging:
        return {
            "state": BLOCKED_CHARGING,
            "decision": BLOCKED_CHARGING,
            "motion_authorized": False,
            "reasons": ["charging mode explicitly invalidates navigation"],
        }
    if not evidence.hardware_connected or not evidence.lidar_fresh:
        if not evidence.hardware_connected:
            reasons.append("hardware transport is unavailable")
        if not evidence.lidar_fresh:
            reasons.append("lidar scan is unavailable or stale")
        return {
            "state": WAIT_FOR_SENSORS,
            "decision": WAIT_FOR_SENSORS,
            "motion_authorized": False,
            "reasons": reasons,
        }
    if (
        evidence.explicit_reposition
        or evidence.map_changed
        or evidence.consecutive_scan_mismatches >= 3
    ):
        if evidence.explicit_reposition:
            reasons.append("manual reposition or charging was reported")
        if evidence.map_changed:
            reasons.append("map identity changed")
        if evidence.consecutive_scan_mismatches >= 3:
            reasons.append("repeated live scan/map mismatches indicate kidnapping")
        return {
            "state": GLOBAL_LOCALIZATION_REQUIRED,
            "decision": RUN_GLOBAL_LOCALIZATION,
            "motion_authorized": False,
            "reasons": reasons,
        }
    if (
        not evidence.runtime_gap_observed
        and evidence.live_pose_fresh
        and evidence.live_covariance_ok
        and evidence.live_scan_map_ok
    ):
        return {
            "state": HOT_READY,
            "decision": REUSE_LIVE_POSE,
            "motion_authorized": False,
            "reasons": [
                "continuous live pose passed covariance and scan/map checks"
            ],
        }
    if evidence.cache_valid and evidence.cache_signature_ok:
        return {
            "state": WARM_VALIDATION_REQUIRED,
            "decision": VALIDATE_CACHED_POSE,
            "motion_authorized": False,
            "reasons": [
                "cached pose and current scan signature agree; local validation is required"
            ],
        }
    return {
        "state": GLOBAL_LOCALIZATION_REQUIRED,
        "decision": RUN_GLOBAL_LOCALIZATION,
        "motion_authorized": False,
        "reasons": [
            "no continuously trusted or scan-validated cached pose is available"
        ],
    }
