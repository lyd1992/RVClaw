from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.api import run_demo
from rvclaw.agent.planner import LlamaCppPlannerBackend
from rvclaw.models import Task


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class PlannerModeTest(unittest.TestCase):
    def test_mock_run_records_mock_planner_mode(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            summary = run_demo(
                goal="检查 A-03 区域设备状态并生成报告",
                runs_dir=Path(scratch),
                planner_name="mock",
                run_id="test-planner-mode-mock",
            )

            metrics = json.loads(Path(summary.metrics_path).read_text(encoding="utf-8"))
            self.assertEqual(metrics["planner_mode"], "mock")

    def test_llama_cpp_records_direct_mode_for_complete_json(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "tool_calls": [
                                    {"name": "memory_query", "arguments": {"query": "检查 A-03", "limit": 5}},
                                    {"name": "move_to", "arguments": {"target": "A-03"}},
                                    {"name": "capture_image", "arguments": {"target": "A-03", "mode": "inspection"}},
                                    {"name": "detect_status", "arguments": {"target": "A-03", "image_ref": "latest"}},
                                    {"name": "speak", "arguments": {"text": "A-03 inspection complete."}},
                                    {"name": "upload_report", "arguments": {"title": "A-03 inspection report"}},
                                ]
                            }
                        )
                    }
                }
            ]
        }
        task = Task(task_id="run-test", goal="检查 A-03 区域设备状态并生成报告", created_at="2026-05-22T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            calls = planner.plan(task, memory_context=[])

        self.assertEqual(len(calls), 6)
        self.assertEqual(planner.last_mode, "direct")

    def test_llama_cpp_records_repaired_mode_for_incomplete_inspection(self) -> None:
        response = {"choices": [{"message": {"content": '{"tool_calls":[{"name":"speak","arguments":{"text":"done"}}]}'}}]}
        task = Task(task_id="run-test", goal="检查 A-03 区域设备状态并生成报告", created_at="2026-05-22T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            planner.plan(task, memory_context=[])

        self.assertEqual(planner.last_mode, "repaired_incomplete")

    def test_llama_cpp_repairs_incomplete_photo_task_for_configured_zone(self) -> None:
        response = {"choices": [{"message": {"content": '{"tool_calls":[{"name":"move_to","arguments":{"target":"B-01"}}]}'}}]}
        task = Task(task_id="run-test", goal="移动到 B-01 区域并拍照", created_at="2026-05-22T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            calls = planner.plan(task, memory_context=[])

        self.assertEqual(planner.last_mode, "repaired_incomplete")
        self.assertEqual(
            [call.name for call in calls],
            ["memory_query", "move_to", "capture_image", "detect_status", "speak", "upload_report"],
        )
        self.assertEqual(calls[1].arguments["target"], "B-01")

    def test_llama_cpp_records_fallback_mode_for_malformed_inspection_json(self) -> None:
        response = {"choices": [{"message": {"content": '{"tool_calls":['}}]}
        task = Task(task_id="run-test", goal="检查 A-03 区域设备状态并生成报告", created_at="2026-05-22T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            planner.plan(task, memory_context=[])

        self.assertEqual(planner.last_mode, "fallback_malformed_json")


if __name__ == "__main__":
    unittest.main()
