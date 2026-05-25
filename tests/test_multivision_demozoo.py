from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.adapters.demozoo_device import DemoZooClient, _multipart_body
from rvclaw.adapters.vision import local_vision_result, normalize_demozoo_payload
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

    def test_demozoo_normalizes_nested_detection_response(self) -> None:
        payload = {
            "success": True,
            "data": {
                "detections": [
                    {
                        "class_name": "robot",
                        "score": 0.91,
                        "bbox": {"xmin": 120, "ymin": 40, "xmax": 330, "ymax": 460},
                    }
                ]
            },
        }

        result = normalize_demozoo_payload(payload, task="object_detection", model="yolov5")

        self.assertEqual(result["objects"][0]["label"], "robot")
        self.assertEqual(result["objects"][0]["confidence"], 0.91)
        self.assertEqual(result["objects"][0]["bbox"], [120.0, 40.0, 330.0, 460.0])

    def test_demozoo_uses_same_task_real_model_fallback_when_default_model_fails(self) -> None:
        payload = {"objects": [{"label": "robot", "confidence": 0.88, "bbox": [0.2, 0.1, 0.7, 0.9]}]}
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", side_effect=[RuntimeError("yolov8 failed"), payload]) as mocked:
                summary = run_demo(
                    goal="detect objects in this image and generate a conclusion",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-model-fallback",
                )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            result = json.loads((Path(summary.run_dir) / "artifacts" / "vision_result.json").read_text(encoding="utf-8"))
            called_models = [call.kwargs["model"] for call in mocked.call_args_list]
            self.assertEqual(summary.status, "completed")
            self.assertEqual(called_models[:2], ["yolov8", "yolov11"])
            self.assertEqual(metrics["vision_backend"], "demozoo")
            self.assertEqual(metrics["vision_model"], "yolov11")
            self.assertEqual(metrics["vision_requested_model"], "yolov8")
            self.assertEqual(metrics["vision_model_fallback_reason"], "yolov8: yolov8 failed")
            self.assertEqual(result["requested_model"], "yolov8")
            self.assertEqual(result["model_fallback_reason"], "yolov8: yolov8 failed")

    def test_demozoo_retries_when_detection_response_is_uninformative(self) -> None:
        payload = {"objects": [{"label": "robot", "confidence": 0.88, "bbox": [0.2, 0.1, 0.7, 0.9]}]}
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", side_effect=[{"results": []}, payload]) as mocked:
                summary = run_demo(
                    goal="detect objects in this image and generate a conclusion",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-empty-retry",
                )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            called_models = [call.kwargs["model"] for call in mocked.call_args_list]
            self.assertEqual(summary.status, "completed")
            self.assertEqual(called_models[:2], ["yolov8", "yolov11"])
            self.assertEqual(metrics["objects_count"], 1)
            self.assertEqual(metrics["vision_model"], "yolov11")

    def test_demozoo_uninformative_error_includes_payload_preview(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", return_value={"success": True, "result": "no boxes"}):
                summary = run_demo(
                    goal="detect objects in this image and generate a conclusion",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-preview",
                )

            trace = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failures = [row for row in trace if row["event"] == "skill_call.failed"]
            self.assertEqual(summary.status, "failed")
            self.assertIn("payload=", failures[-1]["payload"]["result"]["error"])
            self.assertIn("no boxes", failures[-1]["payload"]["result"]["error"])

    def test_demozoo_json_image_result_is_treated_as_usable_annotation(self) -> None:
        payload = {
            "success": True,
            "data": {
                "result_image": f"data:image/png;base64,{ONE_PIXEL_PNG}",
            },
        }

        result = normalize_demozoo_payload(payload, task="object_detection", model="yolov8_pose")

        self.assertEqual(result["backend_detail"], "demozoo_image_result")
        self.assertEqual(result["image_base64"], ONE_PIXEL_PNG)

    def test_demozoo_json_result_image_path_is_treated_as_container_annotation(self) -> None:
        payload = {
            "success": True,
            "output": "Results saved to result.jpg\n",
            "result_image_generated": True,
            "result_image_path": "/app/examples/CV/yolov11/python/result.jpg",
        }

        result = normalize_demozoo_payload(payload, task="object_detection", model="yolov11")

        self.assertEqual(result["backend_detail"], "demozoo_container_image_result")
        self.assertEqual(result["container_image_ref"], "/app/examples/CV/yolov11/python/result.jpg")

    def test_demozoo_copies_container_result_image_for_model_fallback(self) -> None:
        payload = {
            "success": True,
            "output": "Results saved to result.jpg\n",
            "result_image_generated": True,
            "result_image_path": "/app/examples/CV/yolov11/python/result.jpg",
        }

        def fake_docker_cp(args: list[str], **_kwargs: object):
            Path(args[-1]).write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            return type("Completed", (), {"returncode": 0, "stderr": "", "stdout": ""})()

        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", side_effect=[RuntimeError("yolov8 failed"), payload]) as mocked:
                with patch("rvclaw.adapters.demozoo_device.subprocess.run", side_effect=fake_docker_cp) as docker_cp:
                    summary = run_demo(
                        goal="detect objects in this image and generate a conclusion",
                        runs_dir=Path(scratch) / "runs",
                        planner_name="mock",
                        run_id="test-demozoo-container-image",
                    )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            result = json.loads((Path(summary.run_dir) / "artifacts" / "vision_result.json").read_text(encoding="utf-8"))
            called_models = [call.kwargs["model"] for call in mocked.call_args_list]
            self.assertEqual(summary.status, "completed")
            self.assertEqual(called_models[:2], ["yolov8", "yolov11"])
            self.assertEqual(metrics["vision_model"], "yolov11")
            self.assertEqual(metrics["vision_backend_detail"], "demozoo_container_image_result")
            self.assertTrue(Path(result["annotated_image_ref"]).exists())
            self.assertIn("spacemit-demo-container:/app/examples/CV/yolov11/python/result.jpg", docker_cp.call_args.args[0])

    def test_demozoo_real_detection_required_fails_when_all_models_are_uninformative(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            os.environ["RVCLAW_DEVICE_BACKEND"] = "cv_sample"
            os.environ["RVCLAW_VISION_BACKEND"] = "demozoo"
            os.environ["RVCLAW_REQUIRE_REAL_VISION"] = "1"
            os.environ["RVCLAW_VISION_SOURCE"] = str(source)

            with patch.object(DemoZooClient, "predict", return_value={"results": []}):
                summary = run_demo(
                    goal="detect objects in this image and generate a conclusion",
                    runs_dir=Path(scratch) / "runs",
                    planner_name="mock",
                    run_id="test-demozoo-empty-required",
                )

            trace = [
                json.loads(line)
                for line in Path(summary.trace_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            failures = [row for row in trace if row["event"] == "skill_call.failed"]
            self.assertEqual(summary.status, "failed")
            self.assertIn("returned no usable object_detection result", failures[-1]["payload"]["result"]["error"])

    def test_demozoo_client_prefers_model_registry_endpoint_and_trailing_slash_retry(self) -> None:
        client = DemoZooClient(base_url="http://demo.local", endpoint_template="/fallback/{model}")
        client._model_endpoint_cache = {"yolov8": "/predict/yolov8"}

        urls = client._candidate_urls(task="object_detection", model="yolov8")

        self.assertEqual(
            urls,
            [
                "http://demo.local/predict/yolov8",
                "http://demo.local/predict/yolov8/",
                "http://demo.local/fallback/yolov8",
                "http://demo.local/fallback/yolov8/",
            ],
        )

    def test_demozoo_client_falls_back_when_endpoint_template_is_malformed(self) -> None:
        client = DemoZooClient(base_url="http://demo.local", endpoint_template="/predict/{model}}")
        client._model_endpoint_cache = {}

        urls = client._candidate_urls(task="object_detection", model="yolov8")

        self.assertEqual(urls[:2], ["http://demo.local/predict/yolov8", "http://demo.local/predict/yolov8/"])

    def test_demozoo_binary_image_payload_keeps_real_backend_summary(self) -> None:
        payload = {
            "image_base64": ONE_PIXEL_PNG,
            "content_type": "image/png",
            "_rvclaw_demozoo_url": "http://demo.local/predict/yolov8/",
        }

        result = normalize_demozoo_payload(payload, task="object_detection", model="yolov8")

        self.assertEqual(result["backend"], "demozoo")
        self.assertEqual(result["backend_detail"], "demozoo_image_result")
        self.assertIn("annotated image", result["summary"])
        self.assertEqual(result["raw"]["_rvclaw_demozoo_url"], "http://demo.local/predict/yolov8/")

    def test_demozoo_multipart_sends_image_and_file_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))

            body = _multipart_body(boundary="rvclaw-test", image_path=source, fields={"task": "object_detection"})

        self.assertIn(b'name="image"; filename="sample.png"', body)
        self.assertIn(b'name="file"; filename="sample.png"', body)

    def test_demozoo_predict_retries_after_404(self) -> None:
        class Response:
            headers = type("Headers", (), {"get_content_type": lambda self: "application/json"})()

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"objects":[{"label":"robot","score":0.9}]}'

        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / "sample.png"
            source.write_bytes(base64.b64decode(ONE_PIXEL_PNG))
            client = DemoZooClient(base_url="http://demo.local", endpoint_template="/fallback/{model}")
            client._model_endpoint_cache = {"yolov8": "/predict/yolov8"}
            not_found = HTTPError(
                url="http://demo.local/predict/yolov8",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None,
            )

            with patch("rvclaw.adapters.demozoo_device.urlopen", side_effect=[not_found, Response()]) as mocked:
                payload = client.predict(source, task="object_detection", model="yolov8")
            not_found.close()

        self.assertEqual(payload["objects"][0]["label"], "robot")
        self.assertEqual(payload["_rvclaw_demozoo_url"], "http://demo.local/predict/yolov8/")
        self.assertEqual(mocked.call_count, 2)

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
