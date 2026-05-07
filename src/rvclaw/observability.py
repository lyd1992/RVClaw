from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .utils import ensure_dir, utc_now_iso, write_json, write_yaml


class RunRecorder:
    def __init__(self, run_id: str, run_dir: str | Path):
        self.run_id = run_id
        self.run_dir = ensure_dir(run_dir)
        self.task_path = self.run_dir / "task.yaml"
        self.metrics_path = self.run_dir / "metrics.json"
        self.trace_path = self.run_dir / "trace.jsonl"
        self.report_path = self.run_dir / "report.md"
        self.raw_log_path = self.run_dir / "raw.log"
        self.started_at = time.perf_counter()
        self.trace_path.write_text("", encoding="utf-8")
        self.raw_log_path.write_text("", encoding="utf-8")

    def trace(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        row = {
            "ts": utc_now_iso(),
            "run_id": self.run_id,
            "event": event_type,
            "payload": payload or {},
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def raw(self, message: str) -> None:
        with self.raw_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{utc_now_iso()} {message}\n")

    def write_task(self, task: dict[str, Any]) -> Path:
        return write_yaml(self.task_path, task)

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        metrics = dict(metrics)
        metrics.setdefault("latency_ms", round((time.perf_counter() - self.started_at) * 1000, 3))
        return write_json(self.metrics_path, metrics)

    def write_report(self, report: str) -> Path:
        self.report_path.write_text(report, encoding="utf-8")
        return self.report_path
