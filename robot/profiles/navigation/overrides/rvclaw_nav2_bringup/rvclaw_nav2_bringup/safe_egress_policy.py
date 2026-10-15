"""Bounded forward egress policy for ambiguous near-obstacle starts.

The policy is intentionally conservative: it admits only a short, slow,
forward DriveOnHeading action when both the live scan and every plausible
static-map match agree that the swept forward corridor is free and known.
"""

import math


EGRESS_DISTANCE_M = 0.30
EGRESS_MIN_DISTANCE_M = 0.10
EGRESS_DISTANCE_STEP_M = 0.05
EGRESS_SPEED_M_S = 0.03
EGRESS_ACTION_TIME_ALLOWANCE_S = 12
EGRESS_RESULT_TIMEOUT_S = 15.0
EGRESS_BODY_FRONT_M = 0.15440
EGRESS_BODY_REAR_M = 0.15440
EGRESS_LASER_X_M = 0.07725
EGRESS_HALF_WIDTH_M = 0.12725
EGRESS_SAFETY_MARGIN_M = 0.05
# A side-near escape is allowed to use only a 10 mm geometric margin.  This
# is a temporary, explicitly selected mode; ordinary navigation keeps the
# 50 mm margin above.
EGRESS_SIDE_NEAR_SAFETY_MARGIN_M = 0.01
EGRESS_MODE_STANDARD = "standard"
EGRESS_MODE_SIDE_NEAR = "side_near"
# The route Collision Monitor stops for returns up to x=0.35 m in the base
# frame.  An egress candidate must leave obstacles outside that zone at the
# end of the move, otherwise the action would be admitted only to be stopped
# part-way through.
EGRESS_COLLISION_STOP_FRONT_M = 0.35
EGRESS_MIN_FRONT_SECTOR_DEG = 60.0
EGRESS_MIN_FRONT_RAYS = 3
EGRESS_MIN_FRONT_EVIDENCE_RATIO = 0.80
EGRESS_MAP_SAMPLE_STEP_M = 0.025
EGRESS_MINIMUM_CANDIDATE_SCORE = 0.70
EGRESS_MINIMUM_CANDIDATE_RATIO = 0.90
EGRESS_MAXIMUM_CANDIDATES = 3
EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S = 0.75
EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS = 2


def describe_plan():
    return {
        "mode": "bounded_forward_safe_egress_then_relocalize",
        "supported_egress_modes": [EGRESS_MODE_STANDARD, EGRESS_MODE_SIDE_NEAR],
        "default_egress_mode": EGRESS_MODE_STANDARD,
        "distance_m": EGRESS_DISTANCE_M,
        "minimum_distance_m": EGRESS_MIN_DISTANCE_M,
        "distance_step_m": EGRESS_DISTANCE_STEP_M,
        "adaptive_distance": True,
        "speed_m_s": EGRESS_SPEED_M_S,
        "action_name": "/drive_on_heading",
        "action_time_allowance_s": EGRESS_ACTION_TIME_ALLOWANCE_S,
        "straight_only": True,
        "allow_reverse": False,
        "allow_lateral": False,
        "allow_spin": False,
        "requires_live_scan_gate": True,
        "requires_static_map_gate": True,
        "requires_relocalization_after_motion": True,
        "runtime_scan_monitor_required": True,
        "runtime_maximum_scan_age_s": EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
        "runtime_minimum_scan_checks": EGRESS_RUNTIME_MINIMUM_SCAN_CHECKS,
        "runtime_failure_policy": "cancel_estop_zero_disarm",
        "unknown_map_policy": "reject",
        "operator_reposition_fallback": True,
        "side_near_safety_margin_m": EGRESS_SIDE_NEAR_SAFETY_MARGIN_M,
        "side_near_requires_temporary_collision_profile": True,
    }


def egress_safety_margin(mode=EGRESS_MODE_STANDARD, safety_margin_m=None):
    """Resolve and validate the geometric margin for a reviewed mode."""
    if mode not in (EGRESS_MODE_STANDARD, EGRESS_MODE_SIDE_NEAR):
        raise ValueError("unknown egress mode: %s" % mode)
    expected = (
        EGRESS_SIDE_NEAR_SAFETY_MARGIN_M
        if mode == EGRESS_MODE_SIDE_NEAR
        else EGRESS_SAFETY_MARGIN_M
    )
    margin = expected if safety_margin_m is None else float(safety_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("egress safety margin must be finite and nonnegative")
    return margin


def egress_distance_candidates(
    maximum_distance_m=EGRESS_DISTANCE_M,
    minimum_distance_m=EGRESS_MIN_DISTANCE_M,
    step_m=EGRESS_DISTANCE_STEP_M,
):
    """Return descending, stable candidate distances for bounded egress."""
    maximum = float(maximum_distance_m)
    minimum = float(minimum_distance_m)
    step = float(step_m)
    if not all(math.isfinite(value) for value in (maximum, minimum, step)):
        raise ValueError("egress distance bounds must be finite")
    if minimum <= 0.0 or maximum < minimum or step <= 0.0:
        raise ValueError("invalid egress distance bounds")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [maximum - index * step for index in range(count + 1)]
    if values[-1] > minimum + 1e-9:
        values.append(minimum)
    return tuple(round(max(minimum, value), 6) for value in values)


def remaining_egress_distance(selected_distance_m, distance_traveled_m):
    """Return the untraveled portion of a reviewed forward egress."""
    selected = float(selected_distance_m)
    traveled = float(distance_traveled_m)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError("selected egress distance must be finite and nonnegative")
    if not math.isfinite(traveled):
        raise ValueError("traveled egress distance must be finite")
    return max(0.0, selected - max(0.0, traveled))


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def evaluate_front_egress_scan(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    laser_x=EGRESS_LASER_X_M,
    laser_y=0.0,
    laser_yaw=0.0,
    distance_m=EGRESS_DISTANCE_M,
    body_front_m=EGRESS_BODY_FRONT_M,
    body_rear_m=EGRESS_BODY_REAR_M,
    half_width_m=EGRESS_HALF_WIDTH_M,
    safety_margin_m=None,
    collision_stop_front_m=EGRESS_COLLISION_STOP_FRONT_M,
    minimum_sector_deg=EGRESS_MIN_FRONT_SECTOR_DEG,
    minimum_rays=EGRESS_MIN_FRONT_RAYS,
    minimum_evidence_ratio=EGRESS_MIN_FRONT_EVIDENCE_RATIO,
    mode=EGRESS_MODE_STANDARD,
):
    """Check the live scan against the forward swept footprint."""
    safety_margin_m = egress_safety_margin(mode, safety_margin_m)
    front_half_angle = math.radians(float(minimum_sector_deg)) / 2.0
    # Only the area newly swept in front of the current footprint can block a
    # bounded forward escape.  A rear/side return may be historical chassis
    # self-reflection and is not entered by this motion.
    corridor_x_min = body_front_m
    corridor_x_max = max(
        body_front_m + distance_m + safety_margin_m,
        collision_stop_front_m + distance_m,
    )
    corridor_y = half_width_m + safety_margin_m
    front_angles = []
    front_evidence_count = 0
    blocking_returns = []
    valid_returns = 0
    laser_cos = math.cos(laser_yaw)
    laser_sin = math.sin(laser_yaw)

    for index, raw_distance in enumerate(ranges):
        angle = float(angle_min) + index * float(angle_increment)
        base_angle = _normalize_angle(float(laser_yaw) + angle)
        in_front_sector = abs(base_angle) <= front_half_angle + 1e-9
        if in_front_sector:
            front_angles.append(base_angle)
        distance = float(raw_distance)
        if math.isinf(distance) and distance > 0.0:
            if in_front_sector:
                front_evidence_count += 1
            continue
        if not math.isfinite(distance):
            continue
        if distance > float(range_max):
            if in_front_sector:
                front_evidence_count += 1
            continue
        if distance < max(float(range_min), 0.15) or distance > float(range_max):
            continue
        if in_front_sector:
            front_evidence_count += 1
        base_ray_x = math.cos(angle)
        base_ray_y = math.sin(angle)
        point_x = float(laser_x) + distance * (
            laser_cos * base_ray_x - laser_sin * base_ray_y
        )
        point_y = float(laser_y) + distance * (
            laser_sin * base_ray_x + laser_cos * base_ray_y
        )
        point_angle = _normalize_angle(math.atan2(point_y, point_x))
        valid_returns += 1
        if (
            corridor_x_min <= point_x <= corridor_x_max
            and abs(point_y) <= corridor_y
        ):
            blocking_returns.append(
                {
                    "range_m": distance,
                    "x_m": point_x,
                    "y_m": point_y,
                    "angle_deg": math.degrees(point_angle),
                }
            )

    coverage_deg = 0.0
    if front_angles:
        coverage_deg = math.degrees(max(front_angles) - min(front_angles))
    evidence_ratio = (
        0.0 if not front_angles else float(front_evidence_count) / len(front_angles)
    )
    failures = []
    if len(front_angles) < int(minimum_rays):
        failures.append(
            "forward scan has too few valid rays (%d < %d)"
            % (len(front_angles), minimum_rays)
        )
    if coverage_deg < float(minimum_sector_deg) * 0.50:
        failures.append(
            "forward scan angular coverage is too narrow (%.1f deg)" % coverage_deg
        )
    if evidence_ratio < float(minimum_evidence_ratio):
        failures.append(
            "forward scan has too little valid/clear evidence (%.3f < %.3f)"
            % (evidence_ratio, minimum_evidence_ratio)
        )
    if blocking_returns:
        nearest = min(item["range_m"] for item in blocking_returns)
        failures.append(
            "forward egress corridor is occupied (nearest %.3f m)" % nearest
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "valid_return_count": valid_returns,
        "forward_ray_count": len(front_angles),
        "forward_evidence_count": front_evidence_count,
        "forward_evidence_ratio": evidence_ratio,
        "forward_coverage_deg": coverage_deg,
        "blocking_returns": blocking_returns[:8],
        "required_distance_m": float(distance_m),
        "corridor_x_min_m": float(corridor_x_min),
        "corridor_x_max_m": float(corridor_x_max),
        "corridor_half_width_m": float(corridor_y),
        "collision_stop_front_m": float(collision_stop_front_m),
        "mode": mode,
        "safety_margin_m": float(safety_margin_m),
    }


def _world_to_cell(grid, world_x, world_y):
    dx = world_x - float(grid.origin_x)
    dy = world_y - float(grid.origin_y)
    cosine = math.cos(float(grid.origin_yaw))
    sine = math.sin(float(grid.origin_yaw))
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (
        int(math.floor(local_x / float(grid.resolution))),
        int(math.floor(local_y / float(grid.resolution))),
    )


def _candidate_corridor_cells(
    grid,
    candidate,
    distance_m,
    body_front_m,
    body_rear_m,
    half_width_m,
    safety_margin_m,
    collision_stop_front_m,
    sample_step_m,
):
    """Inspect only cells the chassis newly enters during forward motion.

    The current footprint is deliberately excluded from the blocking count:
    the vehicle is already there, and the static map may contain a historical
    self-return or its inflated residue.  The newly swept rectangle starts at
    the front face and extends by ``distance_m``.  Unknown, out-of-map, and
    occupied cells in that rectangle remain hard failures.
    """
    yaw = float(candidate["yaw"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    points = []
    corridor_x_max = max(
        body_front_m + distance_m + safety_margin_m,
        collision_stop_front_m + distance_m,
    )
    x = body_front_m
    while x <= corridor_x_max + 1e-9:
        y = -half_width_m - safety_margin_m
        while y <= half_width_m + safety_margin_m + 1e-9:
            points.append((x, y))
            y += sample_step_m
        y = half_width_m + safety_margin_m
        points.append((x, y))
        x += sample_step_m
    y = -half_width_m - safety_margin_m
    while y <= half_width_m + safety_margin_m + 1e-9:
        points.append((corridor_x_max, y))
        y += sample_step_m

    unknown = 0
    occupied = 0
    out_of_bounds = 0
    for local_x, local_y in points:
        world_x = float(candidate["x"]) + cosine * local_x - sine * local_y
        world_y = float(candidate["y"]) + sine * local_x + cosine * local_y
        cell_x, cell_y = _world_to_cell(grid, world_x, world_y)
        if not (0 <= cell_x < grid.width and 0 <= cell_y < grid.height):
            out_of_bounds += 1
            continue
        value = int(grid.cells[cell_y, cell_x])
        if value < 0:
            unknown += 1
        elif value >= 65:
            occupied += 1
    return {
        "sample_count": len(points),
        "unknown_count": unknown,
        "occupied_count": occupied,
        "out_of_bounds_count": out_of_bounds,
        "initial_body_occupied_count": 0,
        "initial_body_unknown_count": 0,
        "corridor_x_max_m": float(corridor_x_max),
    }


def evaluate_static_map_egress(
    grid,
    matches,
    distance_m=EGRESS_DISTANCE_M,
    body_front_m=EGRESS_BODY_FRONT_M,
    body_rear_m=EGRESS_BODY_REAR_M,
    half_width_m=EGRESS_HALF_WIDTH_M,
    safety_margin_m=None,
    collision_stop_front_m=EGRESS_COLLISION_STOP_FRONT_M,
    sample_step_m=EGRESS_MAP_SAMPLE_STEP_M,
    minimum_candidate_score=EGRESS_MINIMUM_CANDIDATE_SCORE,
    minimum_candidate_ratio=EGRESS_MINIMUM_CANDIDATE_RATIO,
    maximum_candidates=EGRESS_MAXIMUM_CANDIDATES,
    mode=EGRESS_MODE_STANDARD,
):
    """Require every plausible map pose to have known, free egress space."""
    safety_margin_m = egress_safety_margin(mode, safety_margin_m)
    failures = []
    candidate_results = []
    if not matches:
        failures.append("static search produced no plausible map candidates")
    scored_matches = [
        candidate for candidate in matches if "mean_likelihood" in candidate
    ]
    if scored_matches:
        best_score = float(scored_matches[0]["mean_likelihood"])
        plausible_matches = [
            candidate
            for candidate in scored_matches
            if float(candidate["mean_likelihood"]) >= float(minimum_candidate_score)
            and (
                best_score <= 0.0
                or float(candidate["mean_likelihood"]) / best_score
                >= float(minimum_candidate_ratio)
            )
        ][: int(maximum_candidates)]
    else:
        plausible_matches = list(matches)[: int(maximum_candidates)]
    if matches and not plausible_matches:
        failures.append("static search produced no high-confidence egress candidates")
    for index, candidate in enumerate(plausible_matches):
        result = _candidate_corridor_cells(
            grid,
            candidate,
            distance_m,
            body_front_m,
            body_rear_m,
            half_width_m,
            safety_margin_m,
            collision_stop_front_m,
            sample_step_m,
        )
        result["candidate_index"] = index
        result["pose"] = {
            "x": float(candidate["x"]),
            "y": float(candidate["y"]),
            "yaw_deg": float(candidate.get("yaw_deg", math.degrees(candidate["yaw"]))),
        }
        candidate_results.append(result)
        if result["unknown_count"] or result["out_of_bounds_count"]:
            failures.append(
                "candidate %d egress corridor enters unknown/out-of-map cells" % index
            )
        if result["occupied_count"]:
            failures.append(
                "candidate %d egress corridor contains occupied cells" % index
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "input_candidate_count": len(matches),
        "candidate_count": len(candidate_results),
        "candidates": candidate_results,
        "distance_m": float(distance_m),
        "safety_margin_m": float(safety_margin_m),
        "collision_stop_front_m": float(collision_stop_front_m),
        "mode": mode,
    }


def evaluate_safe_egress(
    scan_gate,
    map_gate,
    distance_m=EGRESS_DISTANCE_M,
    mode=EGRESS_MODE_STANDARD,
    safety_margin_m=None,
):
    """Combine independent live-scan and static-map gates."""
    safety_margin_m = egress_safety_margin(mode, safety_margin_m)
    failures = list(scan_gate.get("failures", [])) + list(map_gate.get("failures", []))
    return {
        "status": "PASS" if not failures else "FAIL",
        "decision": "REQUIRE_SAFE_EGRESS" if not failures else "REQUIRE_OPERATOR_REPOSITION",
        "failures": failures,
        "scan_gate": scan_gate,
        "map_gate": map_gate,
        "distance_m": float(distance_m),
        "speed_m_s": EGRESS_SPEED_M_S,
        "mode": mode,
        "safety_margin_m": float(safety_margin_m),
    }


def detect_side_near_obstacle(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    laser_x=EGRESS_LASER_X_M,
    laser_y=0.0,
    laser_yaw=0.0,
    body_front_m=EGRESS_BODY_FRONT_M,
    body_rear_m=EGRESS_BODY_REAR_M,
    half_width_m=EGRESS_HALF_WIDTH_M,
    near_margin_m=EGRESS_SIDE_NEAR_SAFETY_MARGIN_M,
    outer_margin_m=EGRESS_SAFETY_MARGIN_M,
):
    """Detect a return just outside either side of the chassis footprint.

    This is a mode selector only.  It never admits motion by itself: the
    normal live-scan and static-map gates still have to pass in the selected
    mode.  Returns are limited to the chassis longitudinal span to avoid
    treating a distant diagonal object as a side wall.
    """
    inner = float(half_width_m) + float(near_margin_m)
    outer = float(half_width_m) + float(outer_margin_m)
    if not math.isfinite(inner) or not math.isfinite(outer) or outer < inner:
        raise ValueError("invalid side-near obstacle bounds")
    laser_cos = math.cos(float(laser_yaw))
    laser_sin = math.sin(float(laser_yaw))
    detections = []
    unsafe_detections = []
    for index, raw_distance in enumerate(ranges):
        distance = float(raw_distance)
        if not math.isfinite(distance):
            continue
        if distance < max(float(range_min), 0.15) or distance > float(range_max):
            continue
        angle = float(angle_min) + index * float(angle_increment)
        ray_x = math.cos(angle)
        ray_y = math.sin(angle)
        point_x = float(laser_x) + distance * (
            laser_cos * ray_x - laser_sin * ray_y
        )
        point_y = float(laser_y) + distance * (
            laser_sin * ray_x + laser_cos * ray_y
        )
        if -float(body_rear_m) <= point_x <= float(body_front_m):
            item = {
                "range_m": distance,
                "x_m": point_x,
                "y_m": point_y,
                "side": "left" if point_y > 0.0 else "right",
            }
            if abs(point_y) <= inner:
                unsafe_detections.append(item)
            elif abs(point_y) <= outer:
                detections.append(item)
    return {
        "status": "FAIL" if unsafe_detections else "PASS",
        "detected": bool(detections),
        "detections": detections[:8],
        "unsafe_detections": unsafe_detections[:8],
        "inner_lateral_m": inner,
        "outer_lateral_m": outer,
    }


def select_safe_egress(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    grid,
    matches,
    candidate_distances_m=None,
    mode=EGRESS_MODE_STANDARD,
    safety_margin_m=None,
):
    """Select the longest distance admitted by both live and static gates."""
    safety_margin_m = egress_safety_margin(mode, safety_margin_m)
    candidates = (
        egress_distance_candidates()
        if candidate_distances_m is None
        else tuple(
            sorted(
                {float(value) for value in candidate_distances_m},
                reverse=True,
            )
        )
    )
    if not candidates:
        raise ValueError("at least one egress distance candidate is required")
    if any(
        not math.isfinite(value)
        or value < EGRESS_MIN_DISTANCE_M - 1e-9
        or value > EGRESS_DISTANCE_M + 1e-9
        for value in candidates
    ):
        raise ValueError("egress distance candidate is outside reviewed bounds")

    matches = list(matches)
    attempts = []
    last_result = None
    for distance_m in candidates:
        scan_gate = evaluate_front_egress_scan(
            ranges,
            angle_min,
            angle_increment,
            range_min,
            range_max,
            distance_m=distance_m,
            safety_margin_m=safety_margin_m,
            mode=mode,
        )
        map_gate = evaluate_static_map_egress(
            grid,
            matches,
            distance_m=distance_m,
            safety_margin_m=safety_margin_m,
            mode=mode,
        )
        result = evaluate_safe_egress(
            scan_gate,
            map_gate,
            distance_m=distance_m,
            mode=mode,
            safety_margin_m=safety_margin_m,
        )
        attempts.append(
            {
                "distance_m": distance_m,
                "status": result["status"],
                "failures": list(result["failures"]),
            }
        )
        last_result = result
        if result["status"] == "PASS":
            result.update(
                {
                    "adaptive_distance": True,
                    "selected_distance_m": distance_m,
                    "candidate_distances_m": list(candidates),
                    "attempts": attempts,
                    "mode": mode,
                    "safety_margin_m": float(safety_margin_m),
                }
            )
            return result

    failures = ["no reviewed adaptive egress distance is clear"]
    if last_result is not None:
        failures.extend(last_result["failures"])
    return {
        **(last_result or {}),
        "status": "FAIL",
        "decision": "REQUIRE_OPERATOR_REPOSITION",
        "failures": failures,
        "adaptive_distance": True,
        "selected_distance_m": None,
        "candidate_distances_m": list(candidates),
        "attempts": attempts,
        "mode": mode,
        "safety_margin_m": float(safety_margin_m),
    }


def evaluate_runtime_egress_monitor(
    scan_gate,
    scan_age_s,
    maximum_scan_age_s=EGRESS_RUNTIME_MAXIMUM_SCAN_AGE_S,
):
    """Fail closed when the live corridor blocks or scan delivery stalls."""
    failures = list(scan_gate.get("failures", []))
    age = float(scan_age_s)
    if not math.isfinite(age) or age < 0.0:
        failures.append("runtime forward scan age is invalid")
    elif age > float(maximum_scan_age_s):
        failures.append(
            "runtime forward scan is stale (%.3f s > %.3f s)"
            % (age, maximum_scan_age_s)
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scan_age_s": age,
        "maximum_scan_age_s": float(maximum_scan_age_s),
        "scan_gate": scan_gate,
    }
