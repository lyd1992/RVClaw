from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo


class AgentGraphTraceTest(unittest.TestCase):
    def test_trace_records_safety_guard_approved_events_for_graph(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=Path(scratch),
                planner_name="mock",
                run_id="graph-approved",
            )

            rows = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            approvals = [row for row in rows if row["event"] == "safety_guard.approved"]
            self.assertEqual(len(approvals), summary.tool_calls.__len__())
            self.assertEqual(approvals[0]["payload"]["call"]["name"], "memory_query")

    def test_trace_records_safety_guard_rejected_events_for_graph(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run_demo(
                goal="移动到 Z-99 区域并拍照",
                runs_dir=Path(scratch),
                planner_name="mock",
                run_id="graph-rejected",
            )

            rows = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(summary.status, "failed")
            self.assertTrue(any(row["event"] == "safety_guard.rejected" for row in rows))


if __name__ == "__main__":
    unittest.main()
