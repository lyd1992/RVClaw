from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.cli import build_parser
from rvclaw.agent.planner import LlamaCppPlannerBackend, planner_from_name
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


class LlamaCppPlannerTest(unittest.TestCase):
    def test_cli_accepts_llama_cpp_planner(self) -> None:
        args = build_parser().parse_args(["run", "--planner", "llama_cpp"])

        self.assertEqual(args.planner, "llama_cpp")

    def test_planner_from_name_supports_llama_cpp_alias(self) -> None:
        planner = planner_from_name("llama_cpp")

        self.assertEqual(planner.name, "llama_cpp")

    def test_llama_cpp_planner_parses_openai_compatible_chat_response(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "tool_calls": [
                                    {
                                        "name": "memory_query",
                                        "arguments": {"query": "检查 A-03", "limit": 5},
                                    },
                                    {"name": "move_to", "arguments": {"target": "A-03"}},
                                    {
                                        "name": "capture_image",
                                        "arguments": {"target": "A-03", "mode": "inspection"},
                                    },
                                    {
                                        "name": "detect_status",
                                        "arguments": {"target": "A-03", "image_ref": "latest"},
                                    },
                                ]
                            }
                        )
                    }
                }
            ]
        }
        task = Task(task_id="run-test", goal="检查 A-03 区域设备状态", created_at="2026-05-21T00:00:00Z")
        planner = LlamaCppPlannerBackend(
            base_url="http://127.0.0.1:9090/v1",
            model="Qwen3-0.6B",
            timeout_s=3,
        )

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)) as urlopen:
            calls = planner.plan(task, memory_context=[{"device_id": "A-03", "status": "normal"}])

        self.assertEqual(
            [call.name for call in calls],
            ["memory_query", "move_to", "capture_image", "detect_status", "speak", "upload_report"],
        )
        self.assertEqual(calls[0].arguments["limit"], 5)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9090/v1/chat/completions")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "Qwen3-0.6B")
        self.assertFalse(body["stream"])

    def test_llama_cpp_planner_accepts_raw_tool_call_list(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            [
                                {"name": "memory_query", "arguments": {"query": "检查 A-03", "limit": 5}},
                                {"name": "move_to", "arguments": {"target": "A-03"}},
                            ]
                        )
                    }
                }
            ]
        }
        task = Task(task_id="run-test", goal="移动到 A-03", created_at="2026-05-21T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            calls = planner.plan(task, memory_context=[])

        self.assertEqual([call.name for call in calls], ["memory_query", "move_to"])

    def test_llama_cpp_planner_repairs_incomplete_inspection_plan(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "tool_calls": [
                                    {
                                        "name": "speak",
                                        "arguments": {"text": "A-03区域设备状态检查已完成，检测结果：无异常发现。"},
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        task = Task(task_id="run-test", goal="检查 A-03 区域设备状态并生成报告", created_at="2026-05-21T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            calls = planner.plan(task, memory_context=[])

        self.assertEqual(
            [call.name for call in calls],
            ["memory_query", "move_to", "capture_image", "detect_status", "speak", "upload_report"],
        )

    def test_llama_cpp_planner_does_not_repair_non_inspection_plan(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "tool_calls": [
                                    {
                                        "name": "speak",
                                        "arguments": {"text": "hello"},
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        task = Task(task_id="run-test", goal="播报 hello", created_at="2026-05-21T00:00:00Z")
        planner = LlamaCppPlannerBackend(base_url="http://127.0.0.1:9090/v1", model="Qwen3-0.6B", timeout_s=3)

        with patch("rvclaw.agent.planner.urlopen", return_value=_FakeResponse(response)):
            calls = planner.plan(task, memory_context=[])

        self.assertEqual([call.name for call in calls], ["speak"])


if __name__ == "__main__":
    unittest.main()
