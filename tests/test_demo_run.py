from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo


class DemoRunTest(unittest.TestCase):
    def test_demo_run_outputs_acceptance_artifacts(self) -> None:
        scratch_root = ROOT / "runs" / "test-unittest"
        scratch_root.mkdir(parents=True, exist_ok=True)
        summary = run_demo(
            goal="检查 A-03 区域设备状态并生成报告",
            runs_dir=scratch_root,
            planner_name="mock",
            run_id="test-run",
        )

        self.assertEqual(summary.status, "completed")
        for artifact in [
            summary.task_path,
            summary.metrics_path,
            summary.trace_path,
            summary.report_path,
            str(Path(summary.run_dir) / "raw.log"),
        ]:
            self.assertTrue(Path(artifact).exists(), artifact)

        metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
        self.assertTrue(metrics["task_success"])
        self.assertEqual(metrics["tool_call_count"], 6)


if __name__ == "__main__":
    unittest.main()
