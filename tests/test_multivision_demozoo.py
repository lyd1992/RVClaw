from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.adapters.demozoo_device import DemoZooClient
from rvclaw.adapters.vision import local_vision_result
from rvclaw.api import run_demo
from rvclaw.agent.safety_guard import SafetyGuard, SkillRegistry
from rvclaw.models import ToolCall


ONE_PIXEL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLv"
    "AAAAAElFTkSuQmCC"
)


class MultiVisionDemoZooTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = {
            "RVCLAW_DEVICE_BACKEND": os.environ.get("RVCLAW_DEVICE_BACKEND"),
            "RVCLAW_VISION_BACKEND": os.environ.get("RVCLAW_VISION_BACKEND"),
            "RVCLAW_VISION_SOURCE": os.environ.get("RVCLAW_VISION_SOURCE"),
            "RVCLAW_DEMOZOO_BASE_URL": os.environ.get("RVCLAW_DEMOZOO_BASE_URL"),
            "RVCLAW_REQUIRE_REAL_VISION": os.environ.get("RVCLAW_REQUIRE_REAL_VISION"),
        }

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_cv_sample_supports_all_multivision_tasks(self) -> None:
        goals = {
            "classification": "分类这张图片并说明结果",
            "object_detection": "检测图片中的目标并生成结论",
            "segmentation": "分割画面中的主要区域",
            "face_detection": "检测画面中是否有人脸",
        }
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            for task, goal in goals.items():
                summary = run_demo(goal=goal, runs_dir=Path(scratch) / "runs", planner_name="mock", run_id=f"test-{task}")
                metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))

                self.assertEqual(summary.status, "completed")
                self.assertEqual(metrics["vision_task"], task)
                self.assertEqual(metrics["vision_backend"], "cv_sample")
                self.assertTrue((Path(summary.run_dir) / "artifacts" / "vision_result.json").exists())

    def test_demozoo_backend_normalizes_object_detection_response(self) -> None:
        payload = {"objects": [{"label": "person", "confidence": 0.92, "bbox": [0.1, 0.2, 0.4, 0.8]}]}
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", return_value=payload):
                summary = run_demo(
                    goal="检测图片中的目标并生成结论",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-detect",
                )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            result = json.loads((Path(summary.run_dir) / "artifacts" / "vision_result.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["vision_backend"], "demozoo")
            self.assertEqual(metrics["vision_model"], "yolov8")
            self.assertEqual(metrics["objects_count"], 1)
            self.assertEqual(result["objects"][0]["label"], "person")

    def test_demozoo_unavailable_falls_back_to_mock_result(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", side_effect=OSError("demozoo down")):
                summary = run_demo(
                    goal="检测图片中的目标并生成结论",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-fallback",
                )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(summary.status, "completed")
            self.assertEqual(metrics["vision_backend"], "mock_fallback")
            self.assertGreater(metrics["objects_count"], 0)

    def test_real_vision_required_does_not_fall_back_to_cv_sample(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", side_effect=OSError("demozoo down")):
                summary = run_demo(
                    goal="检测图片中的目标并生成结论",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-required",
                )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            trace = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(summary.status, "failed")
            self.assertEqual(metrics["vision_backend"], "demozoo")
            failures = [row for row in trace if row["event"] == "skill_call.failed"]
            self.assertTrue(failures)
            self.assertIn("DemoZoo real vision backend is required", failures[-1]["payload"]["result"]["error"])

    def test_real_vision_required_rejects_cv_sample_backend(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            summary = run_demo(
                goal="检测图片中的目标并生成结论",
                runs_dir=Path(scratch) / "runs",
                planner_name="mock",
                run_id="test-cv-sample-required",
            )

            trace = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(summary.status, "failed")
            failures = [row for row in trace if row["event"] == "skill_call.failed"]
            self.assertIn("requires a real vision backend", failures[-1]["payload"]["result"]["error"])

    def test_cv_sample_classification_is_explicitly_heuristic_for_uploaded_images(self) -> None:
        result = local_vision_result(task="classification", model="resnet", source=Path("机器人.png"))

        self.assertEqual(result["backend"], "cv_sample")
        self.assertEqual(result["backend_detail"], "cv_sample_heuristic")
        self.assertTrue(result["requires_real_model"])
        self.assertNotIn("工业控制设备", result["summary"])
        self.assertTrue(any("robot" in item["label"] for item in result["labels"]))

    def test_safety_guard_rejects_unknown_vision_task_and_model(self) -> None:
        guard = SafetyGuard(SkillRegistry.from_default())

        with self.assertRaisesRegex(ValueError, "outside whitelist"):
            guard.validate(ToolCall("analyze_image", {"image_ref": "latest", "task": "ocr", "model": "yolov8"}))
        with self.assertRaisesRegex(ValueError, "outside whitelist"):
            guard.validate(ToolCall("analyze_image", {"image_ref": "latest", "task": "object_detection", "model": "remote_model"}))
        with self.assertRaisesRegex(ValueError, "outside whitelist"):
            guard.validate(ToolCall("analyze_image", {"image_ref": "http://example.com/a.png", "task": "object_detection", "model": "yolov8"}))


if __name__ == "__main__":
    unittest.main()
