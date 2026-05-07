from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(slots=True)
class Task:
    task_id: str
    goal: str
    created_at: str
    source: str = "cli"
    platform: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: JsonDict = field(default_factory=dict)
    timeout_s: int | None = None

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ToolCall":
        return cls(
            name=str(data["name"]),
            arguments=dict(data.get("arguments", {})),
            timeout_s=data.get("timeout_s"),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class SkillResult:
    ok: bool
    output: JsonDict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class RunSummary:
    run_id: str
    status: str
    run_dir: str
    report_path: str
    metrics_path: str
    trace_path: str
    task_path: str
    tool_calls: list[JsonDict]

    def to_dict(self) -> JsonDict:
        return asdict(self)
