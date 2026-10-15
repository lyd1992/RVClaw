"""Fail-closed agreement checks for independently acquired static matches."""
import math


def evaluate_static_match_consensus(summaries):
    failures = []
    poses = []
    stamps = []
    for summary in summaries:
        stamp = summary.get("scan_stamp_ns", 0)
        if not isinstance(stamp, int) or stamp <= 0:
            failures.append("invalid acquisition stamp")
        stamps.append(stamp)
        if summary.get("status") != "PASS":
            continue
        best = summary.get("gate", {}).get("best", {})
        try:
            pose = tuple(float(best[key]) for key in ("x", "y", "yaw"))
            if not all(math.isfinite(v) for v in pose):
                raise ValueError
            poses.append(pose)
        except (KeyError, TypeError, ValueError):
            failures.append("invalid passing pose")
    if any(not isinstance(a, int) or not isinstance(b, int) or b <= a for a,b in zip(stamps, stamps[1:])):
        failures.append("scan acquisition stamps did not advance")
    if len(summaries) < 2 or any(s.get("status") != "PASS" for s in summaries[-2:]):
        failures.append("two consecutive unique static matches are required")
    # A later agreement must not erase an earlier conflicting high-confidence pose.
    max_position = 0.0
    max_yaw = 0.0
    for i, a in enumerate(poses):
        for b in poses[i+1:]:
            max_position = max(max_position, math.hypot(a[0]-b[0], a[1]-b[1]))
            delta = math.atan2(math.sin(a[2]-b[2]), math.cos(a[2]-b[2]))
            max_yaw = max(max_yaw, abs(math.degrees(delta)))
    if max_position > .25 or max_yaw > 12.0:
        failures.append("independent passing static matches disagree")
    return {"status": "FAIL" if failures else "PASS", "failures": failures,
            "attempt_count": len(summaries), "required_consecutive_passes": 2,
            "maximum_position_delta_m": max_position, "maximum_yaw_delta_deg": max_yaw,
            "position_limit_m": .25, "yaw_limit_deg": 12.0}
