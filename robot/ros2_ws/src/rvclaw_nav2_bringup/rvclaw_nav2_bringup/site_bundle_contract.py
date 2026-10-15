"""Map-agnostic binding between a selected map and its waypoint catalog.

This is a configuration check, not a physical clearance or localization proof.
It prevents a stale waypoint catalog from being silently paired with another
map merely because both files are individually readable.
"""

from pathlib import Path

import yaml


def validate_map_waypoint_binding(map_yaml, waypoint_catalog):
    selected = Path(map_yaml).expanduser().resolve()
    catalog = Path(waypoint_catalog).expanduser().resolve()
    if selected.suffix.lower() not in (".yaml", ".yml") or not selected.is_file():
        raise ValueError("selected map YAML is unavailable")
    if not catalog.is_file():
        raise ValueError("selected waypoint catalog is unavailable")
    document = yaml.safe_load(catalog.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("waypoint catalog must be a YAML mapping")
    if document.get("frame_id") != "map":
        raise ValueError("waypoint catalog must use the map frame")
    map_id = document.get("map_id")
    if not isinstance(map_id, str) or not map_id.strip():
        raise ValueError("waypoint catalog must declare a map_id")
    declared = document.get("map_yaml")
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("waypoint catalog must declare its map_yaml")
    declared_path = (catalog.parent / declared).resolve()
    if declared_path != selected:
        raise ValueError("waypoint catalog belongs to a different map")
    metadata = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("selected map metadata is invalid")
    image = metadata.get("image")
    if not isinstance(image, str) or not image.strip() or not (
        selected.parent / image
    ).resolve().is_file():
        raise ValueError("selected map image is unavailable")
    return {
        "map_id": map_id,
        "map_yaml": str(selected),
        "waypoint_catalog": str(catalog),
        "map_image": str((selected.parent / image).resolve()),
    }
