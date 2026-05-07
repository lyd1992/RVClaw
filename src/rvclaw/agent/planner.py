from __future__ import annotations

import json
import shutil
import subprocess
from typing import Protocol

from rvclaw.models import Task, ToolCall
from rvclaw.utils import detect_zone


class PlannerBackend(Protocol):
    name: str

    def plan(self, task: Task, memory_context: list[dict]) -> list[ToolCall]:
        ...


class MockPlannerBackend:
    name = "mock"

    def plan(self, task: Task, memory_context: list[dict]) -> list[ToolCall]:
        zone = detect_zone(task.goal)
        return [
            ToolCall("memory_query", {"query": task.goal, "limit": 5}),
            ToolCall("move_to", {"target": zone}),
            ToolCall("capture_image", {"target": zone, "mode": "inspection"}),
            ToolCall("detect_status", {"target": zone, "image_ref": "latest"}),
            ToolCall("speak", {"text": f"{zone} inspection complete. Status is normal."}),
            ToolCall("upload_report", {"title": f"{zone} inspection report"}),
        ]


class ClaudeCliPlannerBackend:
    name = "claude_cli"

    def __init__(self, command: str = "claude", timeout_s: int = 60):
        self.command = command
        self.timeout_s = timeout_s

    def plan(self, task: Task, memory_context: list[dict]) -> list[ToolCall]:
        prompt = self._build_prompt(task, memory_context)
        completed = subprocess.run(
            [self.command],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Claude CLI planner failed")
        payload = _extract_json(completed.stdout)
        calls = payload.get("tool_calls", payload if isinstance(payload, list) else [])
        return [ToolCall.from_dict(call) for call in calls]

    @staticmethod
    def _build_prompt(task: Task, memory_context: list[dict]) -> str:
        allowed = [
            "memory_query",
            "move_to",
            "capture_image",
            "detect_status",
            "speak",
            "upload_report",
            "stop",
        ]
        return (
            "You are the RVClaw Demo Claw planner. Return strict JSON only.\n"
            f"Allowed skills: {allowed}\n"
            "Schema: {\"tool_calls\":[{\"name\":\"skill\",\"arguments\":{...}}]}\n"
            f"Task: {json.dumps(task.to_dict(), ensure_ascii=False)}\n"
            f"Memory context: {json.dumps(memory_context, ensure_ascii=False)}\n"
        )


class AutoPlannerBackend:
    name = "auto"

    def __init__(self):
        self.backend = ClaudeCliPlannerBackend() if shutil.which("claude") else MockPlannerBackend()
        self.name = f"auto/{self.backend.name}"

    def plan(self, task: Task, memory_context: list[dict]) -> list[ToolCall]:
        return self.backend.plan(task, memory_context)


def planner_from_name(name: str) -> PlannerBackend:
    normalized = name.lower().replace("-", "_")
    if normalized == "auto":
        return AutoPlannerBackend()
    if normalized in {"mock", "mock_planner"}:
        return MockPlannerBackend()
    if normalized in {"claude", "claude_cli"}:
        return ClaudeCliPlannerBackend()
    raise ValueError(f"Unknown planner backend: {name}")


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min(i for i in [text.find("{"), text.find("[")] if i >= 0)
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start : end + 1])
