from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo
from rvclaw.web.service import get_run_detail, list_run_files, list_runs, read_benchmark_rows, read_run_file


class WebServiceTest(unittest.TestCase):
    def test_run_history_detail_and_files_are_read_from_runs_dir(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            runs_dir = Path(scratch)
            summary = run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=runs_dir,
                planner_name="mock",
                run_id="test-web-run",
            )

            runs = list_runs(runs_dir)
            self.assertEqual(runs[0]["run_id"], "test-web-run")
            self.assertEqual(runs[0]["status"], "completed")

            detail = get_run_detail(runs_dir, "test-web-run")
            self.assertEqual(detail["summary"]["run_id"], summary.run_id)
            self.assertEqual(detail["metrics"]["tool_call_count"], 6)
            self.assertGreaterEqual(len(detail["trace"]), 1)

            files = list_run_files(runs_dir, "test-web-run")
            self.assertIn("metrics.json", files)
            self.assertIn("trace.jsonl", files)
            report = read_run_file(runs_dir, "test-web-run", "report.md")
            self.assertIn("RVClaw Inspection Report", report["content"])
            self.assertEqual(report["content_type"], "text/markdown")

    def test_run_file_reader_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            runs_dir = Path(scratch)
            run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=runs_dir,
                planner_name="mock",
                run_id="test-web-safe",
            )

            with self.assertRaises(ValueError):
                read_run_file(runs_dir, "test-web-safe", "../memory.sqlite3")

    def test_benchmark_rows_return_empty_list_when_csv_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            self.assertEqual(read_benchmark_rows(Path(scratch)), [])

    def test_run_history_summary_exposes_vision_fields_for_drawer_filters(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            runs_dir = Path(scratch)
            run_demo(
                goal="检测图片中的目标并生成结论",
                runs_dir=runs_dir,
                planner_name="mock",
                run_id="test-web-vision",
            )

            runs = list_runs(runs_dir)
            self.assertEqual(runs[0]["vision_task"], "object_detection")
            self.assertEqual(runs[0]["vision_backend"], "mock")


if __name__ == "__main__":
    unittest.main()
