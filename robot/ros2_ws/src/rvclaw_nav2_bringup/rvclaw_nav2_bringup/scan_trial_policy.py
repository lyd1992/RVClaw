"""Runtime adapter for the operator-authorized forward-only trial."""
import math
import os

from .scan_map_consistency import evaluate_rotation_clearance as full_rotation_gate


def evaluate_rotation_clearance(ranges, angle_min, angle_increment, range_min, range_max, **kwargs):
    half = float(os.environ.get("RVCLAW_FRONT_SCAN_HALF_ANGLE_DEG", "180.0"))
    if not math.isfinite(half) or not 20 <= half <= 180:
        raise ValueError("invalid RVCLAW_FRONT_SCAN_HALF_ANGLE_DEG")
    if half < 180 and not (
        len(ranges) >= 2
        and all(math.isfinite(float(v)) for v in (angle_min, angle_increment, range_min, range_max))
        and float(angle_increment) != 0.0
        and 0.0 < float(range_min) < float(range_max)
    ):
        result = full_rotation_gate([], 0.0, 1.0, 0.15, 50.0, **kwargs)
        result["failures"].append("invalid forward scan metadata")
        result.update(status="FAIL", coverage_scope="controlled_forward_trial",
                      side_and_rear_clearance_verified=False)
        return result
    if half < 180:
        # A received +Inf beam reports no hit; it is not a missing bearing.
        # Keep the existing radial obstacle test, but measure coverage using
        # received hits AND no-hit beams within the genuinely cropped window.
        kwargs["minimum_coverage_deg"] = 0.0
    result = full_rotation_gate(ranges, angle_min, angle_increment, range_min, range_max, **kwargs)
    result["coverage_scope"] = "controlled_forward_trial" if half < 180 else "full_sweep"
    result["side_and_rear_clearance_verified"] = half == 180 and result["status"] == "PASS"
    if half < 180:
        required_coverage = 2 * half - 5.0
        result["finite_return_coverage_deg"] = result["coverage_deg"]
        result["required_minimum_coverage_deg"] = required_coverage
        result["coverage_method"] = "received_hit_or_positive_infinity"
        first = math.degrees(float(angle_min))
        last = math.degrees(float(angle_min) + (len(ranges) - 1) * float(angle_increment))
        lower, upper = sorted((first, last))
        result["scan_geometry_coverage_deg"] = upper - lower
        received_angles = []
        no_hit_count = 0
        for i, raw in enumerate(ranges):
            distance = float(raw)
            finite_hit = math.isfinite(distance) and max(float(range_min), 0.15) <= distance <= float(range_max)
            if finite_hit or distance == math.inf:
                received_angles.append(math.degrees(float(angle_min) + i * float(angle_increment)))
                no_hit_count += distance == math.inf
        received_angles.sort()
        coverage = 0.0 if not received_angles else received_angles[-1] - received_angles[0]
        largest_gap = max((b - a for a, b in zip(received_angles, received_angles[1:])), default=math.inf)
        result.update(coverage_deg=coverage, no_hit_beam_count=no_hit_count,
                      received_beam_count=len(received_angles), largest_unknown_gap_deg=largest_gap)
        if coverage < required_coverage:
            result["failures"].append("received lidar angular coverage is insufficient for spin clearance (%.1f deg < %.1f deg)" % (coverage, required_coverage))
        # Apply the existing five-degree coverage allowance to internal gaps
        # as well. NaN, -Inf and invalid finite ranges are not clear space.
        if largest_gap > 5.0:
            result["failures"].append("received lidar bearings contain an unknown gap above 5 deg")
        if not len(ranges) or abs(lower + half) > 2.0 or abs(upper - half) > 2.0:
            result["failures"].append("scan geometry does not match the configured forward trial")
        result["status"] = "FAIL" if result["failures"] else "PASS"
    return result
