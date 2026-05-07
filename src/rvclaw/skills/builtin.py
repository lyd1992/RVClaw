from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rvclaw.adapters.mock_device import MockDevice
from rvclaw.memory.memory_api import MemoryManager


SkillCallable = Callable[..., dict[str, Any]]


def build_builtin_skills(device: MockDevice, memory: MemoryManager, run_id: str) -> dict[str, SkillCallable]:
    def memory_query(query: str, limit: int = 5) -> dict[str, Any]:
        return {"matches": memory.query(query, limit=limit)}

    def move_to(target: str) -> dict[str, Any]:
        return device.move_to(target=target)

    def capture_image(target: str, mode: str = "inspection") -> dict[str, Any]:
        return device.capture_image(target=target, mode=mode)

    def detect_status(target: str, image_ref: str | None = None) -> dict[str, Any]:
        return device.detect_status(target=target, image_ref=image_ref)

    def speak(text: str) -> dict[str, Any]:
        return device.speak(text=text)

    def upload_report(title: str) -> dict[str, Any]:
        return device.upload_report(title=title, run_id=run_id)

    def stop(reason: str = "safety policy triggered") -> dict[str, Any]:
        return device.stop(reason=reason)

    return {
        "memory_query": memory_query,
        "move_to": move_to,
        "capture_image": capture_image,
        "detect_status": detect_status,
        "speak": speak,
        "upload_report": upload_report,
        "stop": stop,
    }
