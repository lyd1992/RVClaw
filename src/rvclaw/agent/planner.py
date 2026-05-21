from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

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


class LlamaCppPlannerBackend:
    name = "llama_cpp"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: int | None = None,
        max_tokens: int | None = None,
    ):
        self.base_url = (base_url or os.environ.get("RVCLAW_LLAMA_BASE_URL") or "http://127.0.0.1:9090/v1").rstrip("/")
        self.model = model or os.environ.get("RVCLAW_LLAMA_MODEL") or "Qwen3-0.6B"
        self.timeout_s = timeout_s if timeout_s is not None else int(os.environ.get("RVCLAW_LLAMA_TIMEOUT_S", "120"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("RVCLAW_LLAMA_MAX_TOKENS", "512"))

    def plan(self, task: Task, memory_context: list[dict]) -> list[ToolCall]:
        request = self._build_request(task, memory_context)
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"llama.cpp planner request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("llama.cpp planner returned non-JSON HTTP response") from exc

        content = _extract_openai_message_content(payload)
        parsed = _extract_json(content)
        calls = _extract_tool_calls(parsed)
        return [_tool_call_from_planner_payload(call) for call in calls]

    def _build_request(self, task: Task, memory_context: list[dict]) -> Request:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {json.dumps(task.to_dict(), ensure_ascii=False)}\n"
                        f"Memory context: {json.dumps(memory_context, ensure_ascii=False)}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        return Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    @staticmethod
    def _system_prompt() -> str:
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
            "You are the RVClaw Demo Claw planner running on a local llama.cpp backend. "
            "Return strict JSON only, with no markdown and no explanation. "
            f"Allowed skills: {allowed}. "
            "Schema: {\"tool_calls\":[{\"name\":\"skill\",\"arguments\":{...}}]}. "
            "For inspection tasks, prefer memory_query, move_to, capture_image, detect_status, speak, upload_report. "
            "Use only allowed skill names and keep arguments inside the registry whitelist."
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
    if normalized in {"llama", "llama_cpp", "llamacpp"}:
        return LlamaCppPlannerBackend()
    raise ValueError(f"Unknown planner backend: {name}")


def _extract_openai_message_content(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("llama.cpp planner response missing choices")
    first = choices[0]
    message = first.get("message") or {}

    openai_tool_calls = message.get("tool_calls") or []
    if openai_tool_calls:
        return json.dumps({"tool_calls": [_openai_tool_call_to_dict(call) for call in openai_tool_calls]}, ensure_ascii=False)

    content = message.get("content") or first.get("text")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("llama.cpp planner response missing message content")
    return content


def _openai_tool_call_to_dict(call: dict) -> dict:
    function = call.get("function") or {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    return {"name": function.get("name") or call.get("name"), "arguments": arguments}


def _extract_tool_calls(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        calls = payload.get("tool_calls", [])
        return calls if isinstance(calls, list) else []
    return []


def _tool_call_from_planner_payload(call: dict) -> ToolCall:
    if "function" in call:
        call = _openai_tool_call_to_dict(call)
    return ToolCall.from_dict(call)


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [i for i in [text.find("{"), text.find("[")] if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start : end + 1])
