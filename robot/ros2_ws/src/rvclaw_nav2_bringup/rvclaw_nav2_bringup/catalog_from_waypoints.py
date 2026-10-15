"""Create a taskful mission catalog from a saved map waypoint file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

from .mission_contract import MissionContractError, load_mission_catalog


REQUIRED_ROUTE = ("A", "B", "C", "ORIGIN")
ALIASES = {
    "A": ["A点", "A位置"],
    "B": ["B点", "B位置"],
    "C": ["C点", "C位置"],
    "ORIGIN": ["原点", "起点"],
}


def build_catalog(waypoints_path: str | Path) -> dict[str, Any]:
    source = Path(waypoints_path)
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise MissionContractError("waypoint file must contain a mapping")
    map_id = document.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        raise MissionContractError("waypoint file has no map_id")
    raw_waypoints = document.get("waypoints")
    if not isinstance(raw_waypoints, Mapping):
        raise MissionContractError("waypoint file has no waypoints mapping")
    missing = [name for name in REQUIRED_ROUTE if name not in raw_waypoints]
    if missing:
        raise MissionContractError("missing required waypoints: %s" % ",".join(missing))

    locations: dict[str, Any] = {}
    for name in REQUIRED_ROUTE:
        raw = raw_waypoints[name]
        if not isinstance(raw, Mapping):
            raise MissionContractError("waypoint %s must be a mapping" % name)
        locations[name] = {
            "x_m": float(raw["x_m"]),
            "y_m": float(raw["y_m"]),
            "yaw_deg": float(raw.get("yaw_deg", 0.0)),
            "arrival_position_tolerance_m": 0.20,
            "arrival_yaw_tolerance_deg": 15.0,
            "aliases": ALIASES[name],
        }

    return {
        "schema_version": 1,
        "map_id": map_id.strip(),
        "map_yaml": "map.yaml",
        "frame_id": "map",
        "locations": locations,
        "tasks": {
            "inspect_meter": {
                "kind": "inspection",
                "aliases": ["检查仪表", "检查电表", "读仪表"],
                "timeout_s": 45.0,
                "max_attempts": 2,
                "failure_policy": "abort",
            },
            "inspect_scene": {
                "kind": "inspection",
                "aliases": ["检查现场", "巡检", "观察环境"],
                "timeout_s": 60.0,
                "max_attempts": 1,
                "failure_policy": "abort",
            },
        },
        "missions": {
            "inspection_route": [
                {"location": "A", "task": "inspect_meter"},
                {"location": "B", "task": "inspect_scene"},
                {"location": "C", "task": "inspect_scene"},
                {"location": "ORIGIN"},
            ]
        },
        "mission_aliases": {
            "inspection_route": ["巡检路线", "完整巡检", "依次巡检ABC并返回原点"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waypoints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new path")
    document = build_catalog(args.waypoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    load_mission_catalog(args.output)
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
