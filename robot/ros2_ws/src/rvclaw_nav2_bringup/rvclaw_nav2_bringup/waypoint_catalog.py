"""Load and validate named absolute map-frame poses."""

import math
from pathlib import Path
from typing import Dict, Mapping

import yaml


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def load_waypoints(path: str) -> Dict[str, Dict[str, float]]:
    """Return finite ``x``, ``y`` and ``yaw`` values from a catalog."""
    catalog_path = Path(path).expanduser().resolve()
    with catalog_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("waypoint catalog must contain a mapping")
    if document.get("frame_id", "map") != "map":
        raise ValueError("waypoint catalog frame_id must be map")
    entries = document.get("waypoints")
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("waypoint catalog contains no waypoints")

    result: Dict[str, Dict[str, float]] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("waypoint names must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise ValueError("waypoint %s is not a mapping" % name)
        try:
            x = float(entry["x_m"])
            y = float(entry["y_m"])
            yaw_deg = float(entry["yaw_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("waypoint %s is missing numeric pose fields" % name) from exc
        if not all(math.isfinite(value) for value in (x, y, yaw_deg)):
            raise ValueError("waypoint %s contains a non-finite value" % name)
        result[name] = {
            "name": name,
            "x": x,
            "y": y,
            "yaw_deg": yaw_deg,
            "yaw": normalize_angle(math.radians(yaw_deg)),
        }
    return result


def select_waypoint(path: str, name: str) -> Dict[str, float]:
    waypoints = load_waypoints(path)
    try:
        return waypoints[name]
    except KeyError as exc:
        available = ", ".join(sorted(waypoints))
        raise ValueError("unknown waypoint %s; available: %s" % (name, available)) from exc
