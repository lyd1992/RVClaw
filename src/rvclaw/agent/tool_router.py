from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rvclaw.agent.safety_guard import SafetyGuard
from rvclaw.models import SkillResult, ToolCall
from rvclaw.observability import RunRecorder


class ToolRouter:
    def __init__(
        self,
        skills: dict[str, Callable[..., dict[str, Any]]],
        safety_guard: SafetyGuard,
        recorder: RunRecorder,
    ):
        self.skills = skills
        self.safety_guard = safety_guard
        self.recorder = recorder

    def execute(self, call: ToolCall) -> SkillResult:
        started_at = time.perf_counter()
        self.recorder.trace("skill_call.started", call.to_dict())
        try:
            checked = self.safety_guard.validate(call)
            skill = self.skills[checked.name]
            output = skill(**checked.arguments)
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            result = SkillResult(ok=True, output={**output, "latency_ms": elapsed_ms})
            self.recorder.trace(
                "skill_call.completed",
                {"call": checked.to_dict(), "result": result.to_dict()},
            )
            return result
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
            result = SkillResult(ok=False, output={"latency_ms": elapsed_ms}, error=str(exc))
            self.recorder.trace("skill_call.failed", {"call": call.to_dict(), "result": result.to_dict()})
            return result
