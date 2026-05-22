from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_ZONES = ["A-03", "B-01", "BASE"]


def load_zone_ids(path: str | Path | None = None) -> list[str]:
    config_path = Path(path or os.environ.get("RVCLAW_ZONES_CONFIG", "configs/zones.yaml"))
    if not config_path.exists():
        return list(DEFAULT_ZONES)
    text = config_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _load_zone_ids_from_simple_yaml(text)
    zones = payload.get("zones", [])
    ids = [_zone_id(zone) for zone in zones]
    return [zone_id for zone_id in ids if zone_id] or list(DEFAULT_ZONES)


def _zone_id(zone: Any) -> str:
    if isinstance(zone, str):
        return zone
    if isinstance(zone, dict):
        return str(zone.get("id", ""))
    return ""


def _load_zone_ids_from_simple_yaml(text: str) -> list[str]:
    ids = []
    for line in text.splitlines():
        match = re.match(r"\s*-\s*id:\s*['\"]?([^'\"\s#]+)", line)
        if match:
            ids.append(match.group(1))
    return ids or list(DEFAULT_ZONES)
