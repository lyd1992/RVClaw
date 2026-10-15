"""Independent LaserScan-to-static-map localization checks."""

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import yaml


@dataclass
class StaticMap:
    cells: np.ndarray
    occupied_distance_m: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float

    @property
    def height(self):
        return int(self.cells.shape[0])

    @property
    def width(self):
        return int(self.cells.shape[1])


def _chamfer_distance(occupied, resolution):
    """Return an eight-neighbour distance field without a SciPy dependency."""
    height, width = occupied.shape
    distance = np.full((height, width), np.inf, dtype=np.float32)
    distance[occupied] = 0.0
    diagonal = math.sqrt(2.0)
    for y in range(height):
        for x in range(width):
            value = distance[y, x]
            if x:
                value = min(value, distance[y, x - 1] + 1.0)
            if y:
                value = min(value, distance[y - 1, x] + 1.0)
                if x:
                    value = min(value, distance[y - 1, x - 1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y - 1, x + 1] + diagonal)
            distance[y, x] = value
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            value = distance[y, x]
            if x + 1 < width:
                value = min(value, distance[y, x + 1] + 1.0)
            if y + 1 < height:
                value = min(value, distance[y + 1, x] + 1.0)
                if x:
                    value = min(value, distance[y + 1, x - 1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y + 1, x + 1] + diagonal)
            distance[y, x] = value
    return distance * float(resolution)


@lru_cache(maxsize=4)
def _load_static_map_cached(
    yaml_path_text,
    yaml_mtime_ns,
    image_path_text,
    image_mtime_ns,
):
    # The mtimes are part of the cache key.  They are intentionally unused in
    # the body: changing either source file creates a new immutable cache entry.
    del yaml_mtime_ns, image_mtime_ns
    # The geometry and policy tests only need ``StaticMap``.  Delay the image
    # decoder import so those tests remain useful on a host without Pillow;
    # actual map loading still fails clearly unless the runtime dependency is
    # installed (as it is on the K3 ROS image).
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to load a static map image; install python3-pil"
        ) from exc
    yaml_path = Path(yaml_path_text)
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    image_path = Path(image_path_text)
    pixels = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    if int(metadata.get("negate", 0)):
        occupied_probability = pixels / 255.0
    else:
        occupied_probability = (255.0 - pixels) / 255.0
    cells = np.full(pixels.shape, -1, dtype=np.int8)
    cells[occupied_probability >= float(metadata["occupied_thresh"])] = 100
    cells[occupied_probability <= float(metadata["free_thresh"])] = 0
    # PGM rows run top-to-bottom while OccupancyGrid cell Y runs bottom-to-top.
    cells = np.flipud(cells).copy()
    resolution = float(metadata["resolution"])
    origin = metadata["origin"]
    return StaticMap(
        cells=cells,
        occupied_distance_m=_chamfer_distance(cells == 100, resolution),
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
    )


def load_static_map(yaml_path):
    """Load one map and reuse its expensive distance field while unchanged."""

    yaml_path = Path(yaml_path).resolve()
    metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    image_path = (yaml_path.parent / metadata["image"]).resolve()
    return _load_static_map_cached(
        str(yaml_path),
        yaml_path.stat().st_mtime_ns,
        str(image_path),
        image_path.stat().st_mtime_ns,
    )


def clear_static_map_cache():
    """Expose deterministic cache reset for tests and controlled map reloads."""

    _load_static_map_cached.cache_clear()


def scan_endpoints(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    laser_x=0.07725,
    laser_y=0.0,
    laser_yaw=0.0,
    maximum_range_m=8.0,
    maximum_points=180,
):
    values = np.asarray(ranges, dtype=np.float64)
    indices = np.arange(values.size, dtype=np.float64)
    valid = np.isfinite(values)
    valid &= values >= max(float(range_min), 0.18)
    valid &= values <= min(float(range_max), float(maximum_range_m))
    values = values[valid]
    angles = float(angle_min) + indices[valid] * float(angle_increment)
    if values.size > maximum_points:
        selected = np.linspace(0, values.size - 1, maximum_points).astype(int)
        values = values[selected]
        angles = angles[selected]
    angles = angles + float(laser_yaw)
    return np.column_stack(
        (
            float(laser_x) + values * np.cos(angles),
            float(laser_y) + values * np.sin(angles),
        )
    )


def _world_to_cells(grid, world_x, world_y):
    dx = world_x - grid.origin_x
    dy = world_y - grid.origin_y
    cosine = math.cos(grid.origin_yaw)
    sine = math.sin(grid.origin_yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return (
        np.floor(local_x / grid.resolution).astype(np.int64),
        np.floor(local_y / grid.resolution).astype(np.int64),
    )


def score_pose(grid, endpoints, x, y, yaw, sigma_m=0.20):
    endpoints = np.asarray(endpoints, dtype=np.float64)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2 or not len(endpoints):
        raise ValueError("at least one two-dimensional scan endpoint is required")
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    world_x = float(x) + cosine * endpoints[:, 0] - sine * endpoints[:, 1]
    world_y = float(y) + sine * endpoints[:, 0] + cosine * endpoints[:, 1]
    cell_x, cell_y = _world_to_cells(grid, world_x, world_y)
    in_map = (
        (cell_x >= 0)
        & (cell_y >= 0)
        & (cell_x < grid.width)
        & (cell_y < grid.height)
    )
    states = np.full(len(endpoints), -1, dtype=np.int16)
    distances = np.full(len(endpoints), np.inf, dtype=np.float64)
    states[in_map] = grid.cells[cell_y[in_map], cell_x[in_map]]
    known = in_map & (states >= 0)
    distances[known] = grid.occupied_distance_m[cell_y[known], cell_x[known]]
    likelihood = np.zeros(len(endpoints), dtype=np.float64)
    likelihood[known] = np.exp(-0.5 * (distances[known] / float(sigma_m)) ** 2)
    return {
        "endpoint_count": int(len(endpoints)),
        "in_map_ratio": float(np.mean(in_map)),
        "known_endpoint_ratio": float(np.mean(known)),
        "occupied_endpoint_ratio": float(np.mean(states == 100)),
        "near_occupied_ratio": float(np.mean(distances <= sigma_m)),
        "mean_likelihood": float(np.mean(likelihood)),
        "pose": {"x": float(x), "y": float(y), "yaw": float(yaw)},
    }


def evaluate_pose_ray_consistency(
    grid, endpoints, pose, *, laser_x=0.07725, laser_y=0.0,
    endpoint_tolerance_m=0.25, origin_skip_m=0.20,
    maximum_conflicting_ray_ratio=0.25, maximum_unknown_sample_ratio=0.15,
    minimum_ray_count=60, maximum_points=480,
):
    """Veto impossible scan/map poses without authorizing any robot motion.

    Endpoints near a wall are not sufficient: the measured free ray before
    that hit must not cross static occupied cells. Half-cell samples are a
    localization check, NOT the collision monitor or a swept-volume check.
    Unknown/OOB samples are counted, never silently considered free. A
    quarter metre endpoint band tolerates wall rasterization/map error;
    it is not an obstacle clearance, masking radius, or motion threshold.
    """
    points = np.asarray(endpoints, dtype=np.float64)
    required = np.array([pose.get(key, math.nan) for key in ("x", "y", "yaw")], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(required).all()):
        return {"status": "FAIL", "failures": ["invalid pose or scan endpoints for ray consistency"]}
    if grid.resolution <= 0 or not math.isfinite(grid.resolution):
        raise ValueError("ray consistency requires a positive finite map resolution")
    if maximum_points < minimum_ray_count or minimum_ray_count < 1:
        raise ValueError("ray sample bound is below the required evidence count")
    if endpoint_tolerance_m < grid.resolution or origin_skip_m < 0:
        raise ValueError("invalid ray sampling tolerance")
    if len(points) > maximum_points:
        points = points[np.linspace(0, len(points) - 1, maximum_points).astype(int)]
    relative = points - np.array([laser_x, laser_y])
    distances = np.linalg.norm(relative, axis=1)
    usable = distances > origin_skip_m + endpoint_tolerance_m + grid.resolution
    relative, distances = relative[usable], distances[usable]
    failures = []
    if len(distances) < minimum_ray_count:
        return {"status": "FAIL", "failures": ["too few finite free-space rays"],
                "ray_count": int(len(distances)), "required_ray_count": int(minimum_ray_count)}
    yaw = required[2]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    directions = relative / distances[:, None]
    dx = cosine * directions[:, 0] - sine * directions[:, 1]
    dy = sine * directions[:, 0] + cosine * directions[:, 1]
    origin_x = required[0] + cosine * laser_x - sine * laser_y
    origin_y = required[1] + sine * laser_x + cosine * laser_y
    step = grid.resolution / 2.0
    steps = np.arange(origin_skip_m, distances.max() - endpoint_tolerance_m, step)
    if len(steps) * len(distances) > 2_000_000:
        return {"status": "FAIL", "failures": ["ray sampling budget exceeded"]}
    active = steps[None, :] < distances[:, None] - endpoint_tolerance_m
    cell_x, cell_y = _world_to_cells(
        grid, origin_x + dx[:, None] * steps,
        origin_y + dy[:, None] * steps,
    )
    in_map = (cell_x >= 0) & (cell_y >= 0) & (cell_x < grid.width) & (cell_y < grid.height)
    states = np.full(cell_x.shape, -1, dtype=np.int16)
    states[in_map] = grid.cells[cell_y[in_map], cell_x[in_map]]
    conflict = ((states == 100) & active).any(axis=1)
    active_count = int(active.sum())
    unknown_ratio = float(((states < 0) & active).sum() / max(1, active_count))
    conflict_ratio = float(conflict.mean())
    if conflict_ratio > maximum_conflicting_ray_ratio:
        failures.append("too many measured free rays cross static walls (%.3f > %.3f)"
                        % (conflict_ratio, maximum_conflicting_ray_ratio))
    if unknown_ratio > maximum_unknown_sample_ratio:
        failures.append("too much ray space is unknown or outside map (%.3f > %.3f)"
                        % (unknown_ratio, maximum_unknown_sample_ratio))
    return {
        "status": "FAIL" if failures else "PASS", "failures": failures,
        "ray_count": int(len(distances)), "conflicting_ray_count": int(conflict.sum()),
        "conflicting_ray_ratio": conflict_ratio, "unknown_sample_ratio": unknown_ratio,
        "sample_count": active_count, "endpoint_tolerance_m": float(endpoint_tolerance_m),
        "maximum_conflicting_ray_ratio": float(maximum_conflicting_ray_ratio),
        "maximum_unknown_sample_ratio": float(maximum_unknown_sample_ratio),
        "motion_authorized": False,
    }


def filter_ray_inconsistent_matches(grid, endpoints, search, **ray_options):
    """Keep original endpoint scores and uniqueness gates, adding a ray veto.

    This does not assert that the first surviving pose is the true pose.
    Multiple survivors still need the unchanged margin, ratio, fresh-scan
    consensus, AMCL convergence and final independent acceptance checks.
    """
    accepted, rejected = [], []
    for match in search.get("matches", []):
        gate = evaluate_pose_ray_consistency(grid, endpoints, match, **ray_options)
        item = dict(match, ray_consistency=gate)
        (accepted if gate["status"] == "PASS" else rejected).append(item)
    accepted.sort(key=lambda item: item["mean_likelihood"], reverse=True)
    return dict(search, matches=accepted, ray_consistency_required=True,
                ray_checked_candidate_count=len(accepted) + len(rejected),
                ray_rejected_matches=rejected)


def _candidate_world_positions(grid, position_step_m):
    step = max(1, int(round(float(position_step_m) / grid.resolution)))
    rows, columns = np.nonzero(grid.cells[::step, ::step] == 0)
    cell_x = columns * step
    cell_y = rows * step
    local_x = (cell_x + 0.5) * grid.resolution
    local_y = (cell_y + 0.5) * grid.resolution
    cosine = math.cos(grid.origin_yaw)
    sine = math.sin(grid.origin_yaw)
    return np.column_stack(
        (
            grid.origin_x + cosine * local_x - sine * local_y,
            grid.origin_y + sine * local_x + cosine * local_y,
        )
    )


def _score_pose_candidates(
    grid,
    endpoints,
    positions,
    yaw_values,
    sigma_m,
    keep_per_yaw,
):
    endpoints = np.asarray(endpoints, dtype=np.float64)
    candidates = []
    for yaw_deg in yaw_values:
        yaw = math.radians(float(yaw_deg))
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotated_x = cosine * endpoints[:, 0] - sine * endpoints[:, 1]
        rotated_y = sine * endpoints[:, 0] + cosine * endpoints[:, 1]
        scores = np.zeros(len(positions), dtype=np.float64)
        known_ratios = np.zeros(len(positions), dtype=np.float64)
        for start in range(0, len(positions), 512):
            batch = positions[start : start + 512]
            world_x = batch[:, 0, None] + rotated_x[None, :]
            world_y = batch[:, 1, None] + rotated_y[None, :]
            cell_x, cell_y = _world_to_cells(grid, world_x, world_y)
            in_map = (
                (cell_x >= 0)
                & (cell_y >= 0)
                & (cell_x < grid.width)
                & (cell_y < grid.height)
            )
            safe_x = np.clip(cell_x, 0, grid.width - 1)
            safe_y = np.clip(cell_y, 0, grid.height - 1)
            known = in_map & (grid.cells[safe_y, safe_x] >= 0)
            distance = grid.occupied_distance_m[safe_y, safe_x]
            likelihood = np.exp(-0.5 * (distance / float(sigma_m)) ** 2)
            likelihood[~known] = 0.0
            scores[start : start + len(batch)] = np.mean(likelihood, axis=1)
            known_ratios[start : start + len(batch)] = np.mean(known, axis=1)
        count = min(int(keep_per_yaw), len(scores))
        if count == 0:
            continue
        selected = np.argpartition(scores, -count)[-count:]
        candidates.extend(
            {
                "x": float(positions[index, 0]),
                "y": float(positions[index, 1]),
                "yaw": yaw,
                "yaw_deg": float(yaw_deg),
                "mean_likelihood": float(scores[index]),
                "known_endpoint_ratio": float(known_ratios[index]),
            }
            for index in selected
        )
    return candidates


def _select_distinct_matches(
    candidates,
    *,
    minimum_position_separation_m=0.60,
    minimum_yaw_separation_deg=35.0,
    maximum_matches=8,
):
    candidates.sort(key=lambda item: item["mean_likelihood"], reverse=True)
    distinct = []
    for candidate in candidates:
        if all(
            math.hypot(candidate["x"] - prior["x"], candidate["y"] - prior["y"])
            >= float(minimum_position_separation_m)
            or abs(
                math.degrees(
                    math.atan2(
                        math.sin(candidate["yaw"] - prior["yaw"]),
                        math.cos(candidate["yaw"] - prior["yaw"]),
                    )
                )
            )
            >= float(minimum_yaw_separation_deg)
            for prior in distinct
        ):
            distinct.append(candidate)
        if len(distinct) >= int(maximum_matches):
            break
    return distinct


def search_global_matches(
    grid,
    endpoints,
    position_step_m=0.20,
    yaw_step_deg=10.0,
    sigma_m=0.20,
    keep_per_yaw=24,
    maximum_matches=8,
):
    endpoints = np.asarray(endpoints, dtype=np.float64)
    positions = _candidate_world_positions(grid, position_step_m)
    yaw_values = np.arange(-180.0, 180.0, float(yaw_step_deg))
    candidates = _score_pose_candidates(
        grid,
        endpoints,
        positions,
        yaw_values,
        sigma_m,
        keep_per_yaw,
    )
    distinct = _select_distinct_matches(candidates, maximum_matches=maximum_matches)
    return {
        "position_step_m": float(position_step_m),
        "yaw_step_deg": float(yaw_step_deg),
        "evaluated_position_count": int(len(positions)),
        "evaluated_pose_count": int(len(positions) * len(yaw_values)),
        "matches": distinct,
    }


def _local_world_positions(grid, center_x, center_y, radius_m, step_m):
    radius_m = max(0.0, float(radius_m))
    step_m = max(float(grid.resolution), float(step_m))
    x_values = np.arange(
        float(center_x) - radius_m,
        float(center_x) + radius_m + step_m * 0.5,
        step_m,
    )
    y_values = np.arange(
        float(center_y) - radius_m,
        float(center_y) + radius_m + step_m * 0.5,
        step_m,
    )
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    positions = np.column_stack((mesh_x.ravel(), mesh_y.ravel()))
    cell_x, cell_y = _world_to_cells(grid, positions[:, 0], positions[:, 1])
    in_map = (
        (cell_x >= 0)
        & (cell_y >= 0)
        & (cell_x < grid.width)
        & (cell_y < grid.height)
    )
    safe_x = np.clip(cell_x, 0, grid.width - 1)
    safe_y = np.clip(cell_y, 0, grid.height - 1)
    free = in_map & (grid.cells[safe_y, safe_x] == 0)
    return positions[free]


def search_local_matches(
    grid,
    endpoints,
    center_x,
    center_y,
    center_yaw,
    *,
    position_radius_m=0.40,
    yaw_radius_deg=30.0,
    position_step_m=0.05,
    yaw_step_deg=2.5,
    sigma_m=0.20,
    keep_per_yaw=12,
    maximum_matches=8,
):
    """Search only around a cached pose; this never proves a global pose alone."""

    endpoints = np.asarray(endpoints, dtype=np.float64)
    positions = _local_world_positions(
        grid,
        center_x,
        center_y,
        position_radius_m,
        position_step_m,
    )
    center_yaw_deg = math.degrees(float(center_yaw))
    yaw_values = np.arange(
        center_yaw_deg - float(yaw_radius_deg),
        center_yaw_deg + float(yaw_radius_deg) + float(yaw_step_deg) * 0.5,
        float(yaw_step_deg),
    )
    candidates = _score_pose_candidates(
        grid,
        endpoints,
        positions,
        yaw_values,
        sigma_m,
        keep_per_yaw,
    )
    distinct = _select_distinct_matches(
        candidates,
        minimum_position_separation_m=0.25,
        minimum_yaw_separation_deg=15.0,
        maximum_matches=maximum_matches,
    )
    return {
        "search_mode": "local",
        "center": {
            "x": float(center_x),
            "y": float(center_y),
            "yaw": float(center_yaw),
        },
        "position_radius_m": float(position_radius_m),
        "yaw_radius_deg": float(yaw_radius_deg),
        "position_step_m": float(position_step_m),
        "yaw_step_deg": float(yaw_step_deg),
        "evaluated_position_count": int(len(positions)),
        "evaluated_pose_count": int(len(positions) * len(yaw_values)),
        "matches": distinct,
    }


def search_global_matches_coarse_to_fine(
    grid,
    endpoints,
    *,
    coarse_position_step_m=0.40,
    coarse_yaw_step_deg=20.0,
    refine_position_radius_m=0.40,
    refine_yaw_radius_deg=20.0,
    refine_position_step_m=0.05,
    refine_yaw_step_deg=2.5,
    sigma_m=0.20,
    maximum_seeds=8,
    maximum_matches=8,
    ray_endpoints=None,
):
    """Refine global regions and veto impossible rays before deduplication.

    Endpoint-only peaks can be behind a wall. Such a peak must not suppress
    a nearby valid pose before ray validation. Callers can widen the bounded
    seed pool without changing the admission score or ambiguity thresholds.
    The default endpoint-only search used by existing consumers is retained.
    """
    if int(maximum_seeds) < 1 or int(maximum_matches) < 1:
        raise ValueError("global search needs a positive candidate budget")

    coarse = search_global_matches(
        grid,
        endpoints,
        position_step_m=coarse_position_step_m,
        yaw_step_deg=coarse_yaw_step_deg,
        sigma_m=sigma_m,
        keep_per_yaw=max(24, int(maximum_seeds) * 4),
        maximum_matches=int(maximum_seeds),
    )
    refined_candidates = []
    refined_pose_count = 0
    for seed in coarse.get("matches", [])[: int(maximum_seeds)]:
        refined = search_local_matches(
            grid,
            endpoints,
            seed["x"],
            seed["y"],
            seed["yaw"],
            position_radius_m=refine_position_radius_m,
            yaw_radius_deg=refine_yaw_radius_deg,
            position_step_m=refine_position_step_m,
            yaw_step_deg=refine_yaw_step_deg,
            sigma_m=sigma_m,
            keep_per_yaw=16,
            maximum_matches=8,
        )
        refined_pose_count += int(refined["evaluated_pose_count"])
        refined_candidates.extend(refined.get("matches", []))

    if not refined_candidates:
        refined_candidates = list(coarse.get("matches", []))
    ray_report = {}
    if ray_endpoints is not None:
        # Overlapping refinement windows frequently return the same pose.
        # Remove only exact numerical duplicates before expensive ray checks;
        # spatial suppression still happens after the checks.
        unique = {}
        for item in refined_candidates:
            key = tuple(round(item[field], 9) for field in ("x", "y", "yaw"))
            if key not in unique or item["mean_likelihood"] > unique[key]["mean_likelihood"]:
                unique[key] = item
        refined_candidates = list(unique.values())
        checked = filter_ray_inconsistent_matches(
            grid, ray_endpoints, {"matches": refined_candidates},
        )
        refined_candidates = checked["matches"]
        for item in refined_candidates:
            ray = item["ray_consistency"]
            # A surviving ray veto still permits up to 25% conflicts. Keep
            # that hard gate, and also retain continuous free-space evidence
            # for endpoint ties. This score can only reduce endpoint support.
            item["ray_joint_likelihood"] = (
                item["mean_likelihood"]
                * (1.0 - ray["conflicting_ray_ratio"])
                * (1.0 - ray["unknown_sample_ratio"])
            )
        ray_report = {key: checked[key] for key in (
            "ray_consistency_required", "ray_checked_candidate_count",
            "ray_rejected_matches",
        )}
    distinct = _select_distinct_matches(
        refined_candidates, maximum_matches=int(maximum_matches),
    )
    return {
        "search_mode": "coarse_to_fine",
        "coarse_position_step_m": float(coarse_position_step_m),
        "coarse_yaw_step_deg": float(coarse_yaw_step_deg),
        "refine_position_step_m": float(refine_position_step_m),
        "refine_yaw_step_deg": float(refine_yaw_step_deg),
        "coarse_evaluated_position_count": coarse["evaluated_position_count"],
        "coarse_evaluated_pose_count": coarse["evaluated_pose_count"],
        "refined_evaluated_pose_count": refined_pose_count,
        "evaluated_pose_count": coarse["evaluated_pose_count"] + refined_pose_count,
        "seed_count": min(len(coarse.get("matches", [])), int(maximum_seeds)),
        "matches": distinct,
        **ray_report,
    }


def compare_candidate_to_search(candidate_score, search):
    matches = search.get("matches", [])
    best = matches[0] if matches else None
    second = matches[1] if len(matches) > 1 else None
    if best is None:
        return {"best_score": None, "candidate_to_best_ratio": 0.0}
    best_score = float(best["mean_likelihood"])
    candidate_value = float(candidate_score["mean_likelihood"])
    return {
        "best_score": best_score,
        "second_distinct_score": (
            None if second is None else float(second["mean_likelihood"])
        ),
        "second_to_best_ratio": (
            None
            if second is None or best_score <= 0.0
            else float(second["mean_likelihood"]) / best_score
        ),
        "candidate_to_best_ratio": (
            0.0 if best_score <= 0.0 else candidate_value / best_score
        ),
        "candidate_to_best_distance_m": math.hypot(
            candidate_score["pose"]["x"] - best["x"],
            candidate_score["pose"]["y"] - best["y"],
        ),
        "candidate_to_best_yaw_deg": abs(
            math.degrees(
                math.atan2(
                    math.sin(candidate_score["pose"]["yaw"] - best["yaw"]),
                    math.cos(candidate_score["pose"]["yaw"] - best["yaw"]),
                )
            )
        ),
    }


def evaluate_in_motion_pose_disagreement(candidate_score, search):
    """Conservative stop-only signal; never use a scan match to set AMCL pose.

    An unknown/dynamic scene is INDETERMINATE, not an alternative pose. The
    caller must require several fresh, separated scans before stopping.
    """
    best = next(iter(search.get("matches", [])), None)
    candidate_value = float(candidate_score.get("mean_likelihood", 0.0))
    known = float(candidate_score.get("known_endpoint_ratio", 0.0))
    if best is None:
        return {"status": "INDETERMINATE", "reason": "no alternative map match"}
    best_value = float(best.get("mean_likelihood", 0.0))
    best_known = float(best.get("known_endpoint_ratio", 0.0))
    current = candidate_score["pose"]
    separation = math.hypot(float(current["x"]) - float(best["x"]),
                            float(current["y"]) - float(best["y"]))
    evidence = {
        "current_score": candidate_value,
        "best_score": best_value,
        "score_gap": best_value - candidate_value,
        "separation_m": separation,
        "known_ratio": known,
        "best_known_ratio": best_known,
    }
    if known < 0.90 or best_known < 0.90 or best_value < 0.90:
        return dict(evidence, status="INDETERMINATE", reason="insufficient static-map evidence")
    if candidate_value < 0.82 and best_value - candidate_value >= 0.12 and separation >= 0.40:
        return dict(evidence, status="DISAGREEMENT", reason="AMCL disagrees with a strong separated local match")
    return dict(evidence, status="CLEAR", reason="no strong separated disagreement")


def predict_map_pose_from_odometry(start_map_pose, start_odom_pose, current_odom_pose):
    """Project relative wheel odometry from a verified segment start into map."""
    raw_yaw = float(start_odom_pose["yaw"])
    map_yaw = float(start_map_pose["yaw"])
    dx = float(current_odom_pose["x"]) - float(start_odom_pose["x"])
    dy = float(current_odom_pose["y"]) - float(start_odom_pose["y"])
    forward = math.cos(raw_yaw) * dx + math.sin(raw_yaw) * dy
    lateral = -math.sin(raw_yaw) * dx + math.cos(raw_yaw) * dy
    return {
        "x": float(start_map_pose["x"]) + math.cos(map_yaw) * forward
             - math.sin(map_yaw) * lateral,
        "y": float(start_map_pose["y"]) + math.sin(map_yaw) * forward
             + math.cos(map_yaw) * lateral,
        "yaw": math.atan2(
            math.sin(map_yaw + float(current_odom_pose["yaw"]) - raw_yaw),
            math.cos(map_yaw + float(current_odom_pose["yaw"]) - raw_yaw),
        ),
    }


def qualify_motion_disagreement_geometry(evidence, current_ray, ray_checked_search):
    """An ambiguous endpoint optimum alone cannot establish a lost pose.

    Keep a stop when free-space rays contradict the current pose. Otherwise
    require a ray-consistent, distinct alternative before treating a score
    improvement as confirmed disagreement. This never sets a pose or accepts
    a navigation result; hardware/collision/freshness gates are independent.
    """
    if evidence.get("status") != "DISAGREEMENT":
        return evidence
    result = dict(evidence, current_ray_consistency=current_ray)
    if current_ray.get("status") != "PASS":
        return result
    if ray_checked_search.get("ray_consistency_required") is not True:
        return result  # Missing evidence must not silently disable a stop.
    uniqueness = evaluate_static_match_gate(
        ray_checked_search, minimum_best_score=0.90,
        minimum_known_endpoint_ratio=0.90,
    )
    result["alternative_uniqueness"] = uniqueness
    if uniqueness["status"] != "PASS":
        result.update(status="INDETERMINATE",
                      reason="current free-space rays are consistent; alternative map matches are ambiguous or impossible")
    return result


def corroborate_motion_map_disagreement(evidence, current_pose, search,
                                        odom_prediction, maximum_odom_error_m=0.45,
                                        contradiction_margin_m=0.20):
    """Do not treat a geometrically aliased scan match as a confirmed pose loss.

    The scan score alone is stop-only evidence, not ground truth.  Suppress
    that one reason only when AMCL still follows the independent wheel-odom
    trajectory from a verified segment start, and the proposed alternative
    is materially *farther* from that trajectory.  All other stop reasons,
    collision checks, and stale-AMCL checks remain unchanged.
    """
    if evidence.get("status") != "DISAGREEMENT" or odom_prediction is None:
        return evidence
    best = next(iter(search.get("matches", [])), None)
    if best is None:
        return evidence
    amcl_error = math.hypot(float(current_pose["x"]) - odom_prediction["x"],
                            float(current_pose["y"]) - odom_prediction["y"])
    alternate_error = math.hypot(float(best["x"]) - odom_prediction["x"],
                                 float(best["y"]) - odom_prediction["y"])
    result = dict(evidence, odom_predicted_pose=dict(odom_prediction),
                  amcl_to_odom_m=amcl_error, best_to_odom_m=alternate_error)
    if (math.isfinite(amcl_error) and math.isfinite(alternate_error)
            and amcl_error <= maximum_odom_error_m
            and alternate_error >= amcl_error + contradiction_margin_m):
        result.update(status="INDETERMINATE",
                      reason="scan match contradicts verified odometry trajectory")
    return result


def advance_in_motion_disagreement_streak(previous, status, required=3):
    """Trip only on consecutive independent scan checks; clear on uncertainty."""
    if int(required) < 2:
        raise ValueError("motion disagreement needs at least two observations")
    count = int(previous) + 1 if status == "DISAGREEMENT" else 0
    return count, count >= int(required)


def evaluate_match_consensus(
    searches,
    *,
    minimum_scan_count=2,
    minimum_best_score=0.80,
    minimum_known_endpoint_ratio=0.85,
    maximum_pair_distance_m=0.25,
    maximum_pair_yaw_deg=12.0,
    require_global_uniqueness=False,
    minimum_unique_scan_count=None,
    minimum_score_margin=0.05,
    maximum_second_to_best_ratio=0.95,
):
    """Require independent scans to agree before accepting a pose hypothesis."""

    searches = list(searches)
    failures = []
    best_matches = []
    gates = []
    uniqueness_failures = []
    if len(searches) < int(minimum_scan_count):
        failures.append(
            "too few independent scan searches (%d < %d)"
            % (len(searches), int(minimum_scan_count))
        )
    for index, search in enumerate(searches):
        matches = search.get("matches", []) if isinstance(search, dict) else []
        if not matches:
            failures.append("scan %d produced no map match" % (index + 1))
            continue
        best = matches[0]
        best_matches.append(best)
        score = float(best.get("mean_likelihood", 0.0))
        known_ratio = float(best.get("known_endpoint_ratio", 0.0))
        if score < float(minimum_best_score):
            failures.append(
                "scan %d best score is too weak (%.3f < %.3f)"
                % (index + 1, score, minimum_best_score)
            )
        if known_ratio < float(minimum_known_endpoint_ratio):
            failures.append(
                "scan %d known-endpoint ratio is too low (%.3f < %.3f)"
                % (index + 1, known_ratio, minimum_known_endpoint_ratio)
            )
        if require_global_uniqueness:
            gate = evaluate_static_match_gate(
                search,
                minimum_best_score=minimum_best_score,
                minimum_known_endpoint_ratio=minimum_known_endpoint_ratio,
                minimum_score_margin=minimum_score_margin,
                maximum_second_to_best_ratio=maximum_second_to_best_ratio,
            )
            gates.append(gate)
            uniqueness_failures.extend(
                "scan %d: %s" % (index + 1, failure)
                for failure in gate.get("failures", [])
            )

    if require_global_uniqueness:
        required_unique = (
            len(searches)
            if minimum_unique_scan_count is None
            else int(minimum_unique_scan_count)
        )
        unique_pass_count = sum(gate.get("status") == "PASS" for gate in gates)
        if unique_pass_count < required_unique:
            failures.append(
                "too few globally unique scans (%d < %d)"
                % (unique_pass_count, required_unique)
            )
            failures.extend(uniqueness_failures)
    else:
        required_unique = 0
        unique_pass_count = 0

    maximum_distance = 0.0
    maximum_yaw = 0.0
    for first_index, first in enumerate(best_matches):
        for second in best_matches[first_index + 1 :]:
            distance = math.hypot(
                float(first["x"]) - float(second["x"]),
                float(first["y"]) - float(second["y"]),
            )
            yaw_delta = abs(
                math.degrees(
                    math.atan2(
                        math.sin(float(first["yaw"]) - float(second["yaw"])),
                        math.cos(float(first["yaw"]) - float(second["yaw"])),
                    )
                )
            )
            maximum_distance = max(maximum_distance, distance)
            maximum_yaw = max(maximum_yaw, yaw_delta)
    if maximum_distance > float(maximum_pair_distance_m):
        failures.append(
            "independent scan matches disagree in position (%.3f m > %.3f m)"
            % (maximum_distance, maximum_pair_distance_m)
        )
    if maximum_yaw > float(maximum_pair_yaw_deg):
        failures.append(
            "independent scan matches disagree in yaw (%.1f deg > %.1f deg)"
            % (maximum_yaw, maximum_pair_yaw_deg)
        )

    pose = None
    if best_matches:
        pose = {
            "x": sum(float(match["x"]) for match in best_matches)
            / len(best_matches),
            "y": sum(float(match["y"]) for match in best_matches)
            / len(best_matches),
            "yaw": math.atan2(
                sum(math.sin(float(match["yaw"])) for match in best_matches),
                sum(math.cos(float(match["yaw"])) for match in best_matches),
            ),
        }
        pose["yaw_deg"] = math.degrees(pose["yaw"])
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scan_count": len(searches),
        "best_matches": best_matches,
        "maximum_pair_distance_m": maximum_distance,
        "maximum_pair_yaw_deg": maximum_yaw,
        "require_global_uniqueness": bool(require_global_uniqueness),
        "required_unique_scan_count": required_unique,
        "unique_pass_count": unique_pass_count,
        "uniqueness_warnings": uniqueness_failures,
        "global_gates": gates,
        "pose": pose,
    }


def evaluate_static_match_gate(
    search,
    minimum_best_score=0.80,
    minimum_known_endpoint_ratio=0.85,
    minimum_score_margin=0.05,
    maximum_second_to_best_ratio=0.95,
):
    """Admit a stationary scan only when its best map match is distinct."""
    failures = []
    matches = search.get("matches", [])
    best = matches[0] if matches else None
    second = matches[1] if len(matches) > 1 else None
    best_score = None if best is None else float(best.get("mean_likelihood", 0.0))
    best_known_ratio = (
        0.0 if best is None else float(best.get("known_endpoint_ratio", 0.0))
    )
    second_score = None if second is None else float(second.get("mean_likelihood", 0.0))
    margin = None if second_score is None else best_score - second_score
    ratio = (
        None
        if second_score is None or best_score <= 0.0
        else second_score / best_score
    )

    if best is None or best_score < float(minimum_best_score):
        failures.append(
            "best stationary scan match is too weak (%.3f < %.3f)"
            % (0.0 if best_score is None else best_score, minimum_best_score)
        )
    if best is None or best_known_ratio < float(minimum_known_endpoint_ratio):
        failures.append(
            "best stationary match has too few known endpoints (%.3f < %.3f)"
            % (best_known_ratio, minimum_known_endpoint_ratio)
        )
    if second is None:
        failures.append("stationary scan did not produce a distinct second match")
    else:
        if margin < float(minimum_score_margin):
            failures.append(
                "stationary scan match margin is too small (%.3f < %.3f)"
                % (margin, minimum_score_margin)
            )
        if ratio > float(maximum_second_to_best_ratio):
            failures.append(
                "stationary scan has an ambiguous second match (ratio %.3f > %.3f)"
                % (ratio, maximum_second_to_best_ratio)
            )

    result = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "best": best,
        "second": second,
        "best_score": best_score,
        "best_known_endpoint_ratio": best_known_ratio,
        "score_margin": margin,
        "second_to_best_ratio": ratio,
        "thresholds": {
            "minimum_best_score": float(minimum_best_score),
            "minimum_known_endpoint_ratio": float(minimum_known_endpoint_ratio),
            "minimum_score_margin": float(minimum_score_margin),
            "maximum_second_to_best_ratio": float(maximum_second_to_best_ratio),
        },
    }
    # Endpoint separation remains sufficient, exactly as before. If only
    # endpoint ties prevent admission, independent free-ray support may
    # resolve them, using the same score, margin and ratio thresholds. Both
    # rankings must identify the same best pose; weak or missing evidence
    # never gains an unconditional pass.
    if failures and matches and all(
        math.isfinite(item.get("ray_joint_likelihood", math.nan))
        and 0.0 <= item["ray_joint_likelihood"] <= item["mean_likelihood"]
        for item in matches
    ):
        joint = [dict(item, mean_likelihood=item["ray_joint_likelihood"])
                 for item in matches]
        for item in joint:
            item.pop("ray_joint_likelihood", None)
        joint.sort(key=lambda item: item["mean_likelihood"], reverse=True)
        joint_gate = evaluate_static_match_gate(
            {"matches": joint}, minimum_best_score, minimum_known_endpoint_ratio,
            minimum_score_margin, maximum_second_to_best_ratio,
        )
        same_best = all(joint[0][key] == best[key] for key in ("x", "y", "yaw"))
        result["free_ray_uniqueness"] = joint_gate
        result["free_ray_best_agrees"] = same_best
        if joint_gate["status"] == "PASS" and same_best:
            result.update(status="PASS", failures=[],
                          endpoint_uniqueness_failures=failures,
                          uniqueness_evidence="endpoints_and_free_rays")
    return result


def evaluate_rotation_clearance(
    ranges,
    angle_min,
    angle_increment,
    range_min,
    range_max,
    laser_x=0.07725,
    laser_y=0.0,
    minimum_radial_m=0.25,
    minimum_coverage_deg=330.0,
):
    """Check whether lidar evidence covers and clears an in-place sweep."""
    nearest = math.inf
    nearest_angle_deg = None
    valid_angles = []
    for index, raw_distance in enumerate(ranges):
        distance = float(raw_distance)
        if not math.isfinite(distance):
            continue
        if distance < max(float(range_min), 0.15) or distance > float(range_max):
            continue
        angle = float(angle_min) + index * float(angle_increment)
        base_x = float(laser_x) + distance * math.cos(angle)
        base_y = float(laser_y) + distance * math.sin(angle)
        radial = math.hypot(base_x, base_y)
        angle_deg = math.degrees(angle)
        valid_angles.append(angle_deg)
        if radial < nearest:
            nearest = radial
            nearest_angle_deg = angle_deg

    coverage_deg = (
        0.0 if not valid_angles else max(valid_angles) - min(valid_angles)
    )
    failures = []
    if not valid_angles:
        failures.append("no valid lidar returns for spin clearance")
    if nearest < float(minimum_radial_m):
        failures.append(
            "spin swept envelope is occupied (%.3f m < %.3f m)"
            % (nearest, minimum_radial_m)
        )
    if coverage_deg < float(minimum_coverage_deg):
        failures.append(
            "lidar angular coverage is insufficient for spin clearance (%.1f deg < %.1f deg)"
            % (coverage_deg, minimum_coverage_deg)
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "valid_return_count": len(valid_angles),
        "minimum_radial_m": None if math.isinf(nearest) else nearest,
        "minimum_return_angle_deg": nearest_angle_deg,
        "coverage_deg": coverage_deg,
        "required_minimum_radial_m": float(minimum_radial_m),
        "required_minimum_coverage_deg": float(minimum_coverage_deg),
    }


def evaluate_localization_gate(
    candidate_score,
    comparison,
    minimum_known_endpoint_ratio=0.85,
    minimum_best_score=0.80,
    minimum_candidate_to_best_ratio=0.78,
    maximum_candidate_to_best_distance_m=0.50,
    maximum_candidate_to_best_yaw_deg=35.0,
):
    """Admit an AMCL pose only when it agrees with an independent map search."""
    failures = []
    known_ratio = float(candidate_score.get("known_endpoint_ratio", 0.0))
    if known_ratio < float(minimum_known_endpoint_ratio):
        failures.append(
            "too few scan endpoints are in known static-map cells (%.3f < %.3f)"
            % (known_ratio, minimum_known_endpoint_ratio)
        )

    best_score = comparison.get("best_score")
    if best_score is None or float(best_score) < float(minimum_best_score):
        failures.append(
            "best static-map scan match is too weak (%.3f < %.3f)"
            % (
                0.0 if best_score is None else float(best_score),
                minimum_best_score,
            )
        )

    candidate_ratio = comparison.get("candidate_to_best_ratio")
    if candidate_ratio is None or float(candidate_ratio) < float(
        minimum_candidate_to_best_ratio
    ):
        failures.append(
            "AMCL pose is not close enough to the best static-map match (ratio %.3f < %.3f)"
            % (
                0.0 if candidate_ratio is None else float(candidate_ratio),
                minimum_candidate_to_best_ratio,
            )
        )

    candidate_distance = comparison.get("candidate_to_best_distance_m")
    if candidate_distance is None or float(candidate_distance) > float(
        maximum_candidate_to_best_distance_m
    ):
        failures.append(
            "AMCL pose is displaced from the best static-map match (%.3f m > %.3f m)"
            % (
                float("inf") if candidate_distance is None else float(candidate_distance),
                maximum_candidate_to_best_distance_m,
            )
        )

    candidate_yaw = comparison.get("candidate_to_best_yaw_deg")
    if candidate_yaw is None or float(candidate_yaw) > float(
        maximum_candidate_to_best_yaw_deg
    ):
        failures.append(
            "AMCL heading disagrees with the best static-map match (%.1f deg > %.1f deg)"
            % (
                float("inf") if candidate_yaw is None else float(candidate_yaw),
                maximum_candidate_to_best_yaw_deg,
            )
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "thresholds": {
            "minimum_known_endpoint_ratio": float(minimum_known_endpoint_ratio),
            "minimum_best_score": float(minimum_best_score),
            "minimum_candidate_to_best_ratio": float(
                minimum_candidate_to_best_ratio
            ),
            "maximum_candidate_to_best_distance_m": float(
                maximum_candidate_to_best_distance_m
            ),
            "maximum_candidate_to_best_yaw_deg": float(
                maximum_candidate_to_best_yaw_deg
            ),
        },
    }
