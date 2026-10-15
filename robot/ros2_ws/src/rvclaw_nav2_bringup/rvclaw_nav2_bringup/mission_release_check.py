"""Combine map, mission, and task contracts into one offline release report."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from .map_acceptance import load_map, validate_mission_map
from .mission_contract import load_mission_catalog
from .offline_mission_validator import validate_catalog
from .task_worker_contract import load_task_bindings, validate_task_bindings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_image_path(map_path: Path) -> Path:
    document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    return (map_path.parent / document["image"]).resolve()


def run_release_check(
    catalog_path: str | Path,
    map_path: str | Path,
    mission_id: str,
    *,
    task_bindings_path: Optional[str | Path] = None,
    navigation_only: bool = False,
) -> dict[str, Any]:
    catalog_path = Path(catalog_path).resolve()
    map_path = Path(map_path).resolve()
    catalog = load_mission_catalog(catalog_path)
    grid = load_map(map_path)
    mission_result = validate_catalog(catalog, mission_id)
    map_result = validate_mission_map(catalog, mission_id, grid)

    required_tasks = tuple(
        dict.fromkeys(
            step.task_id
            for step in catalog.mission(mission_id).steps
            if step.task_id is not None
        )
    )
    if navigation_only:
        task_result: dict[str, Any] = {
            "status": "SKIPPED",
            "reason": "navigation_only",
            "required_tasks": list(required_tasks),
            "motion_authorized": False,
        }
    elif required_tasks and task_bindings_path is None:
        task_result = {
            "status": "FAIL",
            "failures": ["task_bindings_missing"],
            "required_tasks": list(required_tasks),
            "motion_authorized": False,
        }
    elif task_bindings_path is not None:
        task_result = validate_task_bindings(
            catalog,
            mission_id,
            load_task_bindings(task_bindings_path),
        )
    else:
        task_result = {
            "status": "PASS",
            "required_tasks": [],
            "motion_authorized": False,
        }

    blocking_components = {
        "mission_contract": mission_result["status"],
        "map_acceptance": map_result["status"],
        "task_bindings": task_result["status"],
    }
    failures = [
        "%s:%s" % (name, status)
        for name, status in blocking_components.items()
        if status not in ("PASS", "SKIPPED")
    ]
    cell_counts = Counter(cell for row in grid.cells for cell in row)
    map_image = _map_image_path(map_path)
    hashes = {
        "catalog_sha256": _sha256(catalog_path),
        "map_yaml_sha256": _sha256(map_path),
        "map_image_sha256": _sha256(map_image),
    }
    if task_bindings_path is not None:
        hashes["task_bindings_sha256"] = _sha256(Path(task_bindings_path).resolve())

    passed = not failures
    return {
        "status": "PASS" if passed else "FAIL",
        "release_stage": (
            "READY_FOR_SUPERVISED_FIELD_TEST" if passed else "BLOCKED_OFFLINE"
        ),
        "first_run_guaranteed": False,
        "field_test_required": True,
        "field_motion_authorized": False,
        "map_id": catalog.map_id,
        "mission_id": mission_id,
        "navigation_only": navigation_only,
        "failures": failures,
        "components": blocking_components,
        "map_summary": {
            "width": grid.width,
            "height": grid.height,
            "resolution": grid.resolution,
            "free_cells": cell_counts[0],
            "occupied_cells": cell_counts[100],
            "unknown_cells": cell_counts[-1],
        },
        "hashes": hashes,
        "mission_contract": mission_result,
        "map_acceptance": map_result,
        "task_bindings": task_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--map", dest="map_path", required=True, type=Path)
    parser.add_argument("--mission", default="inspection_route")
    parser.add_argument("--task-bindings", type=Path)
    parser.add_argument("--navigation-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_release_check(
        args.catalog,
        args.map_path,
        args.mission,
        task_bindings_path=args.task_bindings,
        navigation_only=args.navigation_only,
    )
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
