from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)


class CVSampleDeviceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = {
            "RVCLAW_DEVICE_BACKEND": os.environ.get("RVCLAW_DEVICE_BACKEND"),
            "RVCLAW_VISION_SOURCE": os.environ.get("RVCLAW_VISION_SOURCE"),
        }

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_cv_sample_backend_writes_capture_and_annotated_png_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "a03_normal.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            summary = run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=Path(scratch) / "runs",
                planner_name="mock",
                run_id="test-cv-sample",
            )

            self.assertEqual(summary.status, "completed")
            capture = Path(summary.run_dir) / "artifacts" / "a03_capture.png"
            annotated = Path(summary.run_dir) / "artifacts" / "a03_annotated.png"
            self.assertTrue(capture.exists(), capture)
            self.assertTrue(annotated.exists(), annotated)
            self.assertEqual(capture.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(metrics["device_backend"], "cv_sample")
            self.assertEqual(metrics["vision_source"], str(source))

            trace_rows = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            detect_events = [
                row
                for row in trace_rows
                if row["event"] == "skill_call.completed"
                and row["payload"]["call"]["name"] == "detect_status"
            ]
            self.assertEqual(detect_events[0]["payload"]["result"]["output"]["status"], "normal")
            self.assertIn("annotated_image_ref", detect_events[0]["payload"]["result"]["output"])

    def test_cv_sample_backend_falls_back_to_mock_when_source_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_SOURCE"] = str(Path(scratch) / "missing.png")

            summary = run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=Path(scratch) / "runs",
                planner_name="mock",
                run_id="test-cv-fallback",
            )

            self.assertEqual(summary.status, "completed")
            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(metrics["device_backend"], "mock_fallback")
            self.assertTrue(metrics["task_success"])


if __name__ == "__main__":
    unittest.main()
