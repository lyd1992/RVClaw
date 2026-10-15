"""Map-independent budget and evidence checks for stopped, forward-only escape.

Every segment is independently admitted by the live scan, current map and
localization in the motion client. This module only prevents a chain of
nominally successful but ineffective or unbounded segments.
"""

import math

from rvclaw_nav2_bringup.safe_egress_policy import (
    EGRESS_DISTANCE_M,
    EGRESS_MIN_DISTANCE_M,
    EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
    EGRESS_MODE_SIDE_NEAR,
)


MAX_STEPS = 3
MAX_TOTAL_DISTANCE_M = 0.90
MIN_PROGRESS_RATIO = 0.60
MAX_LATERAL_M = 0.06
MAX_HEADING_CHANGE_RAD = 0.15


def _pose(value):
    if not isinstance(value, dict):
        return None
    try:
        fields = tuple(float(value[key]) for key in ("x", "y", "yaw"))
    except (KeyError, TypeError, ValueError):
        return None
    return fields if all(math.isfinite(item) for item in fields) else None


def step_budget(accepted_distances):
    """Return whether a fresh, independently checked step may be requested."""
    try:
        distances = tuple(float(value) for value in accepted_distances)
    except (TypeError, ValueError):
        return {"status": "STOP", "reason": "invalid egress history"}
    if any(not math.isfinite(d) or d < EGRESS_MIN_DISTANCE_M - 1e-9
           or d > EGRESS_DISTANCE_M + 1e-9 for d in distances):
        return {"status": "STOP", "reason": "egress history outside single-step bounds"}
    total = sum(distances)
    if len(distances) >= MAX_STEPS or total + EGRESS_MIN_DISTANCE_M > MAX_TOTAL_DISTANCE_M + 1e-9:
        return {"status": "STOP", "reason": "bounded autonomous escape budget exhausted",
                "completed_steps": len(distances), "selected_total_m": total}
    return {"status": "ALLOW_CHECKED_STEP", "completed_steps": len(distances),
            "selected_total_m": total,
            "maximum_next_m": min(EGRESS_DISTANCE_M, MAX_TOTAL_DISTANCE_M - total)}


def validate_completed_step(result):
    """Require real forward progress, stopped hardware, restored protection.

    A DriveOnHeading SUCCESS alone is not a certificate of movement or safety.
    Missing evidence is failure, never a reason to skip the next map check.
    """
    if not isinstance(result, dict) or result.get("status") != "PASS":
        return {"status": "STOP", "reason": "bounded action did not pass"}
    if result.get("final_armed") is not False or result.get("final_state") != "DISARMED":
        return {"status": "STOP", "reason": "hardware not safely disarmed"}
    if result.get("mode") == EGRESS_MODE_SIDE_NEAR and result.get("collision_profile_restored") is not True:
        return {"status": "STOP", "reason": "ordinary collision protection not restored"}
    localization = result.get("relocalization_gate")
    if not isinstance(localization, dict) or localization.get("status") != "PASS":
        return {"status": "STOP", "reason": "new position not independently localized"}
    try:
        scan_checks = int(result.get("runtime_scan_checks") or 0)
    except (TypeError, ValueError):
        scan_checks = 0
    if scan_checks < EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS:
        return {"status": "STOP", "reason": "insufficient distinct live scan checks"}
    start = _pose(result.get("start_raw_pose"))
    end = _pose(result.get("final_raw_pose"))
    amcl = _pose(result.get("final_amcl_pose"))
    if start is None or end is None or amcl is None:
        return {"status": "STOP", "reason": "missing finite before/after odometry or AMCL pose"}
    try:
        selected = float(result["distance_m"])
        feedback = float(result["feedback_distance_m"])
    except (KeyError, TypeError, ValueError):
        return {"status": "STOP", "reason": "missing action distance or feedback"}
    if not all(math.isfinite(value) for value in (selected, feedback)) or not (
        EGRESS_MIN_DISTANCE_M - 1e-9 <= selected <= EGRESS_DISTANCE_M + 1e-9
    ):
        return {"status": "STOP", "reason": "unreviewed action distance"}
    dx, dy = end[0] - start[0], end[1] - start[1]
    forward = math.cos(start[2]) * dx + math.sin(start[2]) * dy
    lateral = -math.sin(start[2]) * dx + math.cos(start[2]) * dy
    yaw_delta = math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2]))
    if forward < selected * MIN_PROGRESS_RATIO or feedback < selected * MIN_PROGRESS_RATIO:
        return {"status": "STOP", "reason": "bounded action made insufficient forward progress"}
    if abs(lateral) > MAX_LATERAL_M or abs(yaw_delta) > MAX_HEADING_CHANGE_RAD:
        return {"status": "STOP", "reason": "bounded action was not straight"}
    return {"status": "PASS", "selected_distance_m": selected,
            "raw_forward_m": forward, "raw_lateral_m": lateral,
            "raw_heading_change_rad": yaw_delta}


def validate_chain_continuity(previous, current, *, maximum_gap_m=0.12):
    """A new motion segment must begin where the previous one stopped."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return {"status": "STOP", "reason": "missing consecutive step evidence"}
    prior = _pose(previous.get("final_raw_pose"))
    start = _pose(current.get("start_raw_pose"))
    if prior is None or start is None:
        return {"status": "STOP", "reason": "missing consecutive raw poses"}
    gap = math.hypot(prior[0] - start[0], prior[1] - start[1])
    yaw_gap = abs(math.atan2(math.sin(prior[2] - start[2]), math.cos(prior[2] - start[2])))
    if gap > maximum_gap_m or yaw_gap > MAX_HEADING_CHANGE_RAD:
        return {"status": "STOP", "reason": "robot pose changed between checked escape steps",
                "position_gap_m": gap, "heading_gap_rad": yaw_gap}
    return {"status": "PASS", "position_gap_m": gap, "heading_gap_rad": yaw_gap}
