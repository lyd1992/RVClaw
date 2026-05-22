from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rvclaw.utils import read_json


TEXT_CONTENT_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/jsonl",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def list_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if run_dir.is_dir() and (run_dir / "metrics.json").exists():
            runs.append(_run_summary_from_dir(run_dir))
    return runs


def get_run_detail(runs_dir: str | Path, run_id: str) -> dict[str, Any]:
    run_dir = _safe_run_dir(runs_dir, run_id)
    return {
        "summary": _run_summary_from_dir(run_dir),
        "metrics": _read_json_if_exists(run_dir / "metrics.json"),
        "trace": _read_trace(run_dir / "trace.jsonl"),
        "files": list_run_files(runs_dir, run_id),
    }


def list_run_files(runs_dir: str | Path, run_id: str) -> list[str]:
    run_dir = _safe_run_dir(runs_dir, run_id)
    files: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(run_dir).as_posix())
    return files


def read_run_file(runs_dir: str | Path, run_id: str, name: str) -> dict[str, Any]:
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("run file path must stay inside the run directory")
    run_dir = _safe_run_dir(runs_dir, run_id)
    path = (run_dir / name).resolve()
    if not path.is_file() or run_dir.resolve() not in [path.parent, *path.parents]:
        raise FileNotFoundError(name)
    content_type = TEXT_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    if content_type == "application/octet-stream":
        return {"name": name, "content_type": content_type, "bytes": path.read_bytes()}
    return {"name": name, "content_type": content_type, "content": path.read_text(encoding="utf-8")}


def read_benchmark_rows(runs_dir: str | Path) -> list[dict[str, str]]:
    path = Path(runs_dir) / "benchmark_agent_e2e.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _run_summary_from_dir(run_dir: Path) -> dict[str, Any]:
    metrics = _read_json_if_exists(run_dir / "metrics.json")
    return {
        "run_id": run_dir.name,
        "status": metrics.get("status", "unknown"),
        "planner": metrics.get("planner", "unknown"),
        "planner_mode": metrics.get("planner_mode", "unknown"),
        "task_success": metrics.get("task_success", False),
        "tool_call_count": metrics.get("tool_call_count", 0),
        "latency_ms": metrics.get("latency_ms"),
        "run_dir": str(run_dir),
    }


def _safe_run_dir(runs_dir: str | Path, run_id: str) -> Path:
    if "/" in run_id or "\\" in run_id or run_id in {"", ".", ".."}:
        raise ValueError("invalid run_id")
    run_dir = (Path(runs_dir) / run_id).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_id)
    return run_dir


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
