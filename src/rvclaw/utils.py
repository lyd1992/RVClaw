from __future__ import annotations

import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_zone(goal: str, default: str = "A-03") -> str:
    match = re.search(r"\b([A-Z]-\d{2})\b", goal.upper())
    return match.group(1) if match else default


def platform_profile() -> dict[str, Any]:
    rvv_vlen = os.environ.get("RVCLAW_RVV_VLEN", "unknown")
    if rvv_vlen.isdigit():
        rvv_vlen_value: int | str = int(rvv_vlen)
    else:
        rvv_vlen_value = rvv_vlen
    return {
        "soc": os.environ.get("RVCLAW_PLATFORM_SOC", "unknown"),
        "isa": os.environ.get("RVCLAW_PLATFORM_ISA", platform.machine() or "unknown"),
        "rvv_vlen": rvv_vlen_value,
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def dump_simple_yaml(data: Any, indent: int = 0) -> str:
    """Write a small YAML subset without pulling in PyYAML.

    The output is intentionally conservative: scalars are JSON encoded, and
    nested dict/list values use block indentation.
    """

    pad = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(dump_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {json.dumps(value, ensure_ascii=False)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {json.dumps(item, ensure_ascii=False)}")
        return "\n".join(lines)
    return f"{pad}{json.dumps(data, ensure_ascii=False)}"


def write_yaml(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.write_text(dump_simple_yaml(data) + "\n", encoding="utf-8")
    return path
