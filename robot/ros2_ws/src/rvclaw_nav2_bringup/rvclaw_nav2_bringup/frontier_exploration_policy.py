"""Pure frontier selection policy for online autonomous mapping.

The policy consumes an occupancy grid and returns a safe NavigateToPose target.
It has no ROS or hardware side effects, which keeps exploration decisions
reviewable and straightforward to test on a workstation.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Sequence, Tuple


UNKNOWN = -1
FREE_MAX = 25
OCCUPIED_MIN = 65


@dataclass(frozen=True)
class ExplorationLimits:
    max_runtime_s: float = 60.0
    max_goals: int = 3
    max_exploration_radius_m: float = 5.0
    min_frontier_cells: int = 5
    goal_clearance_m: float = 0.30
    sample_step_m: float = 0.05

    def validate(self) -> None:
        if not math.isfinite(self.max_runtime_s) or self.max_runtime_s <= 0:
            raise ValueError("max_runtime_s must be positive and finite")
        if isinstance(self.max_goals, bool) or self.max_goals < 1:
            raise ValueError("max_goals must be a positive integer")
        if not math.isfinite(self.max_exploration_radius_m) or self.max_exploration_radius_m <= 0:
            raise ValueError("max_exploration_radius_m must be positive and finite")
        if isinstance(self.min_frontier_cells, bool) or self.min_frontier_cells < 1:
            raise ValueError("min_frontier_cells must be a positive integer")
        if not math.isfinite(self.goal_clearance_m) or self.goal_clearance_m <= 0:
            raise ValueError("goal_clearance_m must be positive and finite")
        if not math.isfinite(self.sample_step_m) or self.sample_step_m <= 0:
            raise ValueError("sample_step_m must be positive and finite")


@dataclass(frozen=True)
class FrontierCandidate:
    x: float
    y: float
    yaw: float
    cell_count: int
    distance_m: float
    score: float


@dataclass(frozen=True)
class ExplorationDecision:
    status: str
    reason: str
    candidate: Optional[FrontierCandidate] = None
    candidates: Tuple[FrontierCandidate, ...] = ()


def _shape(grid) -> Tuple[int, int]:
    cells = grid.cells
    if len(cells) == 0 or len(cells[0]) == 0:
        raise ValueError("occupancy grid must not be empty")
    return len(cells), len(cells[0])


def _cell(grid, x: int, y: int) -> int:
    height, width = _shape(grid)
    if x < 0 or y < 0 or x >= width or y >= height:
        return OCCUPIED_MIN
    return int(grid.cells[y][x])


def _world_to_cell(grid, x: float, y: float) -> Tuple[int, int]:
    dx = float(x) - float(grid.origin_x)
    dy = float(y) - float(grid.origin_y)
    cosine = math.cos(float(getattr(grid, "origin_yaw", 0.0)))
    sine = math.sin(float(getattr(grid, "origin_yaw", 0.0)))
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return int(math.floor(local_x / float(grid.resolution))), int(
        math.floor(local_y / float(grid.resolution))
    )


def _cell_to_world(grid, x: int, y: int) -> Tuple[float, float]:
    local_x = (float(x) + 0.5) * float(grid.resolution)
    local_y = (float(y) + 0.5) * float(grid.resolution)
    cosine = math.cos(float(getattr(grid, "origin_yaw", 0.0)))
    sine = math.sin(float(getattr(grid, "origin_yaw", 0.0)))
    return (
        float(grid.origin_x) + cosine * local_x - sine * local_y,
        float(grid.origin_y) + sine * local_x + cosine * local_y,
    )


def _is_free(value: int) -> bool:
    return 0 <= int(value) <= FREE_MAX


def _is_frontier(grid, x: int, y: int) -> bool:
    if _cell(grid, x, y) != UNKNOWN:
        return False
    return any(_is_free(_cell(grid, x + dx, y + dy)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def frontier_cells(grid) -> set:
    height, width = _shape(grid)
    return {
        (x, y)
        for y in range(height)
        for x in range(width)
        if _is_frontier(grid, x, y)
    }


def _clusters(cells: set) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    remaining = set(cells)
    clusters = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        cluster = [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    point = (x + dx, y + dy)
                    if point in remaining:
                        remaining.remove(point)
                        stack.append(point)
                        cluster.append(point)
        clusters.append(tuple(cluster))
    return tuple(clusters)


def _footprint_is_clear(grid, x: float, y: float, limits: ExplorationLimits) -> bool:
    radius = float(limits.goal_clearance_m)
    step = min(float(limits.sample_step_m), radius)
    samples = max(1, int(math.ceil(radius / step)))
    for ix in range(-samples, samples + 1):
        for iy in range(-samples, samples + 1):
            local_x = ix * step
            local_y = iy * step
            if local_x * local_x + local_y * local_y > radius * radius:
                continue
            cell_x, cell_y = _world_to_cell(grid, x + local_x, y + local_y)
            if not _is_free(_cell(grid, cell_x, cell_y)):
                return False
    return True


def _cluster_candidate(grid, cluster: Sequence[Tuple[int, int]], current_pose, limits: ExplorationLimits):
    free_neighbors = []
    for x, y in cluster:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if _is_free(_cell(grid, nx, ny)):
                free_neighbors.append((nx, ny))
    if not free_neighbors:
        return None
    center_x = sum(point[0] for point in free_neighbors) / len(free_neighbors)
    center_y = sum(point[1] for point in free_neighbors) / len(free_neighbors)
    # A frontier's unknown cell is not a valid goal.  Prefer the free cell
    # closest to the cluster center that still contains the complete vehicle
    # footprint in known free space.
    # Walk inward through known free cells.  Immediate frontier neighbours are
    # intentionally not sufficient because their footprint touches unknown
    # space; a valid goal must be far enough inside the mapped free region.
    unique_neighbors = set(free_neighbors)
    queue = list(unique_neighbors)
    visited = set(unique_neighbors)
    max_steps = max(1, int(math.ceil(float(limits.goal_clearance_m) / float(grid.resolution))) + 2)
    for _ in range(max_steps):
        next_queue = []
        for cell_x, cell_y in queue:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                point = (cell_x + dx, cell_y + dy)
                if point not in visited and _is_free(_cell(grid, *point)):
                    visited.add(point)
                    next_queue.append(point)
        queue = next_queue
        if not queue:
            break
    clear_neighbors = []
    for cell_x, cell_y in sorted(visited):
        world_x, world_y = _cell_to_world(grid, cell_x, cell_y)
        if _footprint_is_clear(grid, world_x, world_y, limits):
            clear_neighbors.append((cell_x, cell_y, world_x, world_y))
    if not clear_neighbors:
        return None
    cell_x, cell_y, world_x, world_y = min(
        clear_neighbors,
        key=lambda item: (item[0] - center_x) ** 2 + (item[1] - center_y) ** 2,
    )
    pose_x, pose_y, pose_yaw = current_pose
    distance = math.hypot(world_x - pose_x, world_y - pose_y)
    if distance <= float(limits.goal_clearance_m) * 0.5:
        return None
    if distance > float(limits.max_exploration_radius_m):
        return None
    yaw = math.atan2(world_y - pose_y, world_x - pose_x)
    # Larger frontiers are preferred, with a mild distance penalty.
    score = 2.0 * math.log1p(len(cluster)) - distance
    return FrontierCandidate(world_x, world_y, yaw, len(cluster), distance, score)


def choose_frontier(grid, current_pose: Iterable[float], limits: ExplorationLimits = ExplorationLimits()) -> ExplorationDecision:
    """Return the highest-scoring safe frontier, or a fail-closed decision."""
    limits.validate()
    pose = tuple(float(value) for value in current_pose)
    if len(pose) != 3 or not all(math.isfinite(value) for value in pose):
        raise ValueError("current_pose must contain finite x, y, yaw")
    candidates = []
    for cluster in _clusters(frontier_cells(grid)):
        if len(cluster) < limits.min_frontier_cells:
            continue
        candidate = _cluster_candidate(grid, cluster, pose, limits)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (-item.score, item.distance_m, -item.cell_count))
    if not candidates:
        return ExplorationDecision("STOP", "no_safe_reachable_frontier")
    return ExplorationDecision("GOAL", "frontier_selected", candidates[0], tuple(candidates))


def describe_policy() -> dict:
    limits = ExplorationLimits()
    return {
        "policy": "frontier_exploration_v1",
        "map_input": "nav_msgs/OccupancyGrid",
        "target_output": "Nav2 NavigateToPose only",
        "direct_cmd_vel_publishers": 0,
        "unknown_internal_cells_as_goals": False,
        "occupied_or_unclear_footprint": "reject",
        "default_limits": {
            "max_runtime_s": limits.max_runtime_s,
            "max_goals": limits.max_goals,
            "max_exploration_radius_m": limits.max_exploration_radius_m,
            "min_frontier_cells": limits.min_frontier_cells,
            "goal_clearance_m": limits.goal_clearance_m,
        },
        "failure_policy": "cancel_goal_disarm_zero_estop",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe_policy(), indent=2))
