from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--planner", default="mock")
    parser.add_argument("--goal", default="检查 A-03 区域设备状态并生成报告")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = parser.parse_args()

    output = Path(args.runs_dir) / "benchmark_agent_e2e.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        summary = run_demo(goal=args.goal, runs_dir=args.runs_dir, planner_name=args.planner)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        rows.append(
            {
                "run_id": summary.run_id,
                "planner": args.planner,
                "status": summary.status,
                "latency_ms": elapsed_ms,
                "task_success": summary.status == "completed",
                "run_dir": summary.run_dir,
            }
        )

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
