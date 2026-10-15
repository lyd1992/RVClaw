"""Static-map and mission-corridor acceptance checks without ROS or hardware."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .mission_contract import MissionCatalog, MissionContractError, Pose2D, load_mission_catalog


@dataclass(frozen=True)
class MapGrid:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    cells: tuple[tuple[int, ...], ...]

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        dx = x_m - self.origin_x
        dy = y_m - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return math.floor(local_x / self.resolution), math.floor(local_y / self.resolution)

    def state_at(self, x_m: float, y_m: float) -> int:
        cell_x, cell_y = self.world_to_cell(x_m, y_m)
        return self.state_at_cell(cell_x, cell_y)

    def state_at_cell(self, cell_x: int, cell_y: int) -> int:
        if not (0 <= cell_x < self.width and 0 <= cell_y < self.height):
            return -2
        return self.cells[self.height - 1 - cell_y][cell_x]

    def cell_to_world(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        local_x = (cell_x + 0.5) * self.resolution
        local_y = (cell_y + 0.5) * self.resolution
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        return (
            self.origin_x + cosine * local_x - sine * local_y,
            self.origin_y + sine * local_x + cosine * local_y,
        )


def _pgm_tokens(payload: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    index = 0
    length = len(payload)
    while len(tokens) < 4:
        while index < length and payload[index] in b" \t\r\n":
            index += 1
        if index < length and payload[index] == ord("#"):
            while index < length and payload[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < length and payload[index] not in b" \t\r\n":
            index += 1
        if start == index:
            raise MissionContractError("PGM header is incomplete")
        tokens.append(payload[start:index])
    while index < length and payload[index] in b" \t\r\n":
        index += 1
    return tokens, index


def _load_pgm(path: Path) -> tuple[int, int, list[int]]:
    payload = path.read_bytes()
    header, data_start = _pgm_tokens(payload)
    magic, width_raw, height_raw, max_value_raw = header
    width = int(width_raw)
    height = int(height_raw)
    max_value = int(max_value_raw)
    if magic == b"P5":
        if max_value > 255:
            expected = width * height * 2
            data = payload[data_start : data_start + expected]
            values = [int.from_bytes(data[index : index + 2], "big") for index in range(0, len(data), 2)]
        else:
            values = list(payload[data_start : data_start + width * height])
    elif magic == b"P2":
        values = [int(value) for value in payload[data_start:].split()]
    else:
        raise MissionContractError("map image must be a PGM file")
    if len(values) != width * height:
        raise MissionContractError("PGM pixel count does not match dimensions")
    if max_value != 255:
        values = [round(value * 255 / max_value) for value in values]
    return width, height, values


def load_map(path: str | Path) -> MapGrid:
    yaml_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MissionContractError("map YAML is not readable") from exc
    if not isinstance(document, Mapping):
        raise MissionContractError("map YAML must contain a mapping")
    image_name = document.get("image")
    if not isinstance(image_name, str) or not image_name:
        raise MissionContractError("map YAML image is missing")
    image_path = (yaml_path.parent / image_name).resolve()
    width, height, pixels = _load_pgm(image_path)
    resolution = float(document["resolution"])
    origin = document["origin"]
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        raise MissionContractError("map origin must contain x, y and yaw")
    occupied_threshold = float(document.get("occupied_thresh", 0.65))
    free_threshold = float(document.get("free_thresh", 0.25))
    negate = int(document.get("negate", 0))
    cells: list[tuple[int, ...]] = []
    for row in range(height):
        row_values: list[int] = []
        for column in range(width):
            pixel = pixels[row * width + column]
            probability = pixel / 255.0 if negate else (255.0 - pixel) / 255.0
            if probability >= occupied_threshold:
                row_values.append(100)
            elif probability <= free_threshold:
                row_values.append(0)
            else:
                row_values.append(-1)
        cells.append(tuple(row_values))
    return MapGrid(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        cells=tuple(cells),
    )


@dataclass(frozen=True)
class Footprint:
    front_m: float = 0.13275
    rear_m: float = 0.13440
    left_m: float = 0.10725
    right_m: float = 0.10725
    margin_m: float = 0.02000


def _sample_pose(grid: MapGrid, pose: Pose2D, footprint: Footprint, step_m: float) -> dict[str, Any]:
    x_values = _axis_values(-footprint.rear_m - footprint.margin_m, footprint.front_m + footprint.margin_m, step_m)
    y_values = _axis_values(-footprint.right_m - footprint.margin_m, footprint.left_m + footprint.margin_m, step_m)
    cosine = math.cos(math.radians(pose.yaw_deg))
    sine = math.sin(math.radians(pose.yaw_deg))
    counts = Counter()
    examples: dict[str, list[dict[str, float]]] = {"occupied": [], "unknown": [], "outside": []}
    for local_x in x_values:
        for local_y in y_values:
            world_x = pose.x_m + cosine * local_x - sine * local_y
            world_y = pose.y_m + sine * local_x + cosine * local_y
            state = grid.state_at(world_x, world_y)
            key = {100: "occupied", 0: "free", -1: "unknown", -2: "outside"}.get(state, "unknown")
            counts[key] += 1
            if key in examples and len(examples[key]) < 4:
                examples[key].append({"x_m": world_x, "y_m": world_y})
    failures = [key for key in ("occupied", "unknown", "outside") if counts[key]]
    return {
        "status": "PASS" if not failures else "FAIL",
        "counts": dict(counts),
        "failures": failures,
        "examples": examples,
    }


def _axis_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0.0 or not math.isfinite(step):
        raise MissionContractError("sample step must be positive and finite")
    count = max(1, int(math.ceil((end - start) / step)))
    return [start + (end - start) * index / count for index in range(count + 1)]


def _resolve_map_path(catalog_path: Path, catalog: MissionCatalog, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if not catalog.map_yaml:
        raise MissionContractError("catalog has no map_yaml; pass --map")
    candidates = [
        (catalog_path.parent / catalog.map_yaml).resolve(),
        (catalog_path.parent.parent.parent.parent.parent / catalog.map_yaml).resolve(),
        (catalog_path.parent.parent.parent.parent.parent.parent / catalog.map_yaml).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MissionContractError("map_yaml path was not found; pass --map")


def _clearance_offsets(grid: MapGrid, clearance_m: float) -> tuple[tuple[int, int], ...]:
    radius_cells = max(0, int(math.ceil(clearance_m / grid.resolution)))
    tolerance = grid.resolution * 0.5
    return tuple(
        (dx, dy)
        for dx in range(-radius_cells, radius_cells + 1)
        for dy in range(-radius_cells, radius_cells + 1)
        if math.hypot(dx * grid.resolution, dy * grid.resolution)
        <= clearance_m + tolerance
    )


def _find_grid_path(
    grid: MapGrid,
    start: Pose2D,
    end: Pose2D,
    footprint: Footprint,
) -> dict[str, Any]:
    """Find a conservative free-space route; unknown cells are never traversed."""

    start_cell = grid.world_to_cell(start.x_m, start.y_m)
    goal_cell = grid.world_to_cell(end.x_m, end.y_m)
    clearance_m = math.hypot(
        max(footprint.front_m, footprint.rear_m),
        max(footprint.left_m, footprint.right_m),
    ) + footprint.margin_m
    offsets = _clearance_offsets(grid, clearance_m)
    traversable_cache: dict[tuple[int, int], bool] = {}

    def traversable(cell: tuple[int, int]) -> bool:
        if cell not in traversable_cache:
            cell_x, cell_y = cell
            traversable_cache[cell] = all(
                grid.state_at_cell(cell_x + dx, cell_y + dy) == 0
                for dx, dy in offsets
            )
        return traversable_cache[cell]

    base = {
        "clearance_m": clearance_m,
        "start_cell": list(start_cell),
        "goal_cell": list(goal_cell),
    }
    if not traversable(start_cell):
        return {"status": "FAIL", "reason": "start_not_clear", **base}
    if not traversable(goal_cell):
        return {"status": "FAIL", "reason": "goal_not_clear", **base}

    neighbor_steps = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    queue: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(queue, (0.0, 0.0, start_cell))
    costs = {start_cell: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    expanded = 0

    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost != costs.get(current):
            continue
        expanded += 1
        if current == goal_cell:
            break
        for dx, dy, step_cost in neighbor_steps:
            candidate = (current[0] + dx, current[1] + dy)
            if not traversable(candidate):
                continue
            if dx and dy:
                if not traversable((current[0] + dx, current[1])):
                    continue
                if not traversable((current[0], current[1] + dy)):
                    continue
            candidate_cost = current_cost + step_cost
            if candidate_cost >= costs.get(candidate, math.inf):
                continue
            costs[candidate] = candidate_cost
            previous[candidate] = current
            heuristic = math.hypot(
                goal_cell[0] - candidate[0], goal_cell[1] - candidate[1]
            )
            heapq.heappush(
                queue,
                (candidate_cost + heuristic, candidate_cost, candidate),
            )

    if goal_cell not in costs:
        return {
            "status": "FAIL",
            "reason": "no_connected_free_path",
            "expanded_cells": expanded,
            **base,
        }

    path = [goal_cell]
    while path[-1] != start_cell:
        path.append(previous[path[-1]])
    path.reverse()
    preview_stride = max(1, len(path) // 32)
    preview_cells = path[::preview_stride]
    if preview_cells[-1] != path[-1]:
        preview_cells.append(path[-1])
    return {
        "status": "PASS",
        "reason": "connected_free_path",
        "expanded_cells": expanded,
        "path_cell_count": len(path),
        "path_length_m": costs[goal_cell] * grid.resolution,
        "path_preview_world": [
            {"x_m": x_m, "y_m": y_m}
            for x_m, y_m in (grid.cell_to_world(*cell) for cell in preview_cells)
        ],
        **base,
    }


def validate_mission_map(
    catalog: MissionCatalog,
    mission_id: str,
    grid: MapGrid,
    footprint: Footprint = Footprint(),
    sample_step_m: float = 0.025,
) -> dict[str, Any]:
    plan = catalog.mission(mission_id)
    locations: list[dict[str, Any]] = []
    failures: list[str] = []
    for step_index, step in enumerate(plan.steps, start=1):
        location = catalog.location(step.location_id)
        result = _sample_pose(grid, location.pose, footprint, sample_step_m)
        entry = {
            "route_index": step_index,
            "location_id": step.location_id,
            "task_id": step.task_id,
            "pose": location.pose.to_dict(),
            **result,
        }
        locations.append(entry)
        if result["status"] != "PASS":
            failures.append("location:%s" % step.location_id)
    corridors: list[dict[str, Any]] = []
    for index, pair in enumerate(zip(plan.steps, plan.steps[1:]), start=1):
        start = catalog.location(pair[0].location_id).pose
        end = catalog.location(pair[1].location_id).pose
        distance = math.hypot(end.x_m - start.x_m, end.y_m - start.y_m)
        count = max(1, int(math.ceil(distance / sample_step_m)))
        segment_failures = []
        samples = []
        bearing = math.degrees(math.atan2(end.y_m - start.y_m, end.x_m - start.x_m)) if distance else start.yaw_deg
        for sample_index in range(count + 1):
            ratio = sample_index / count
            pose = Pose2D(
                start.x_m + ratio * (end.x_m - start.x_m),
                start.y_m + ratio * (end.y_m - start.y_m),
                bearing if sample_index < count else end.yaw_deg,
            )
            result = _sample_pose(grid, pose, footprint, sample_step_m)
            samples.append({"ratio": ratio, **result})
            if result["status"] != "PASS":
                segment_failures.append(sample_index)
        direct_status = "PASS" if not segment_failures else "FAIL"
        planned_path = _find_grid_path(grid, start, end, footprint)
        route_status = (
            "PASS"
            if direct_status == "PASS" or planned_path["status"] == "PASS"
            else "FAIL"
        )
        corridors.append({
            "from": pair[0].location_id,
            "to": pair[1].location_id,
            "distance_m": distance,
            "status": route_status,
            "route_method": (
                "DIRECT"
                if direct_status == "PASS"
                else "GRID_PATH" if route_status == "PASS" else "NONE"
            ),
            "direct_status": direct_status,
            "failed_sample_indices": segment_failures[:8],
            "sample_count": len(samples),
            "samples": samples,
            "planned_path": planned_path,
        })
        if route_status != "PASS":
            failures.append("route:%s->%s" % (pair[0].location_id, pair[1].location_id))
    return {
        "status": "PASS" if not failures else "FAIL",
        "motion_authorized": False,
        "map_id": catalog.map_id,
        "mission_id": plan.mission_id,
        "footprint": {
            "front_m": footprint.front_m,
            "rear_m": footprint.rear_m,
            "left_m": footprint.left_m,
            "right_m": footprint.right_m,
            "margin_m": footprint.margin_m,
        },
        "failures": failures,
        "locations": locations,
        "corridors": corridors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--mission", default="inspection_route")
    parser.add_argument("--map", dest="map_path")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-step-m", type=float, default=0.025)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    catalog = load_mission_catalog(args.catalog)
    map_path = _resolve_map_path(args.catalog.resolve(), catalog, args.map_path)
    result = validate_mission_map(catalog, args.mission, load_map(map_path), sample_step_m=args.sample_step_m)
    result["map_path"] = str(map_path)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output exists; use a new evidence path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not args.quiet:
        print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
