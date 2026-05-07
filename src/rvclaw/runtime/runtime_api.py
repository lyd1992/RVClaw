from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class RuntimeRequest:
    prompt: str
    model: str = "mock-local"
    options: dict[str, Any] | None = None


@dataclass(slots=True)
class RuntimeResponse:
    text: str
    metrics: dict[str, Any]


class RuntimeBackend(Protocol):
    name: str

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        ...
