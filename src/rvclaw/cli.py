from __future__ import annotations

import argparse
import json
from pathlib import Path

from rvclaw.api import run_demo


DEFAULT_GOAL = "检查 A-03 区域设备状态并生成报告"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rvclaw", description="RVClaw Demo Claw v0.1 runtime CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run a Demo Claw inspection task")
    run_parser.add_argument("goal", nargs="?", default=DEFAULT_GOAL, help="natural-language task goal")
    run_parser.add_argument("--planner", default="auto", choices=["auto", "mock", "claude"], help="planner backend")
    run_parser.add_argument("--runs-dir", default="runs", help="directory for run artifacts")
    run_parser.add_argument("--memory-db", default=None, help="SQLite memory database path")
    run_parser.add_argument("--json", action="store_true", help="print machine-readable summary")

    replay_parser = subparsers.add_parser("replay", help="print a saved trace.jsonl")
    replay_parser.add_argument("trace", help="path to trace.jsonl")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "run"}:
        summary = run_demo(
            goal=getattr(args, "goal", DEFAULT_GOAL),
            runs_dir=getattr(args, "runs_dir", "runs"),
            planner_name=getattr(args, "planner", "auto"),
            memory_db=getattr(args, "memory_db", None),
        )
        if getattr(args, "json", False):
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"RVClaw run {summary.run_id} finished: {summary.status}")
            print(f"  run_dir: {summary.run_dir}")
            print(f"  report:  {summary.report_path}")
            print(f"  metrics: {summary.metrics_path}")
            print(f"  trace:   {summary.trace_path}")
        return 0 if summary.status == "completed" else 1

    if args.command == "replay":
        print(Path(args.trace).read_text(encoding="utf-8"))
        return 0

    parser.print_help()
    return 1
