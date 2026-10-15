"""Pure message-contract checks for the navigation readiness observer."""

import math
from typing import Dict, Iterable, List


def inspect_scan(
    frame_id: str,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    ranges: Iterable[float],
    expected_frame: str = "laser",
) -> Dict[str, object]:
    ranges = list(ranges)
    issues: List[str] = []
    if frame_id != expected_frame:
        issues.append(f"scan frame is {frame_id!r}, expected {expected_frame!r}")
    scalars = [angle_min, angle_max, angle_increment, range_min, range_max]
    if not all(math.isfinite(float(value)) for value in scalars):
        issues.append("scan metadata contains a non-finite value")
    elif angle_increment <= 0.0 or angle_max <= angle_min:
        issues.append("scan angles are not increasing")
    if range_min < 0.0 or range_max <= range_min:
        issues.append("scan range bounds are invalid")
    if len(ranges) < 10:
        issues.append("scan has fewer than 10 range bins")

    usable = sum(
        1
        for value in ranges
        if math.isfinite(float(value)) and range_min <= value <= range_max
    )
    usable_ratio = usable / len(ranges) if ranges else 0.0
    if usable < 3:
        issues.append("scan has fewer than 3 usable ranges")
    return {
        "frame_id": frame_id,
        "range_bins": len(ranges),
        "usable_ranges": usable,
        "usable_ratio": usable_ratio,
        "issues": issues,
    }


def inspect_odom(
    frame_id: str,
    child_frame_id: str,
    numeric_values: Iterable[float],
    expected_frame: str = "odom",
    expected_child: str = "base_footprint",
) -> Dict[str, object]:
    issues: List[str] = []
    if frame_id != expected_frame:
        issues.append(f"odom frame is {frame_id!r}, expected {expected_frame!r}")
    if child_frame_id != expected_child:
        issues.append(
            f"odom child frame is {child_frame_id!r}, expected {expected_child!r}"
        )
    if not all(math.isfinite(float(value)) for value in numeric_values):
        issues.append("odometry contains a non-finite value")
    return {
        "frame_id": frame_id,
        "child_frame_id": child_frame_id,
        "issues": issues,
    }


def build_readiness_result(
    scan_count: int,
    raw_odom_count: int,
    minimum_messages: int,
    scan_report: Dict[str, object],
    raw_odom_report: Dict[str, object],
    transforms: Dict[str, bool],
) -> Dict[str, object]:
    failures: List[str] = []
    for name, count in (
        ("scan", scan_count),
        ("raw_odom", raw_odom_count),
    ):
        if count < minimum_messages:
            failures.append(f"{name} received {count}, requires {minimum_messages}")
    for name, report in (
        ("scan", scan_report),
        ("raw_odom", raw_odom_report),
    ):
        failures.extend(f"{name}: {issue}" for issue in report.get("issues", []))
    for name, available in transforms.items():
        if not available:
            failures.append(f"missing transform: {name}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "subscription_only": True,
        "command_publishers": 0,
        "message_counts": {
            "scan": scan_count,
            "raw_odom": raw_odom_count,
        },
        "scan": scan_report,
        "raw_odom": raw_odom_report,
        "transforms": transforms,
        "failures": failures,
    }
