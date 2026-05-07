from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from rvclaw.models import ToolCall


class SkillRegistry:
    def __init__(self, registry: dict[str, Any]):
        self.version = registry.get("version", "0.0.0")
        self.skills = {skill["name"]: skill for skill in registry.get("skills", [])}

    @classmethod
    def from_default(cls) -> "SkillRegistry":
        with resources.files("rvclaw.skills").joinpath("registry.yaml").open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_file(cls, path: str | Path) -> "SkillRegistry":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, name: str) -> dict[str, Any] | None:
        return self.skills.get(name)


class SafetyGuard:
    def __init__(self, registry: SkillRegistry, require_confirmation: bool = False):
        self.registry = registry
        self.require_confirmation = require_confirmation

    def validate(self, call: ToolCall) -> ToolCall:
        spec = self.registry.get(call.name)
        if spec is None:
            raise PermissionError(f"Skill is not in registry whitelist: {call.name}")

        parameters = spec.get("parameters", {})
        arguments = dict(call.arguments)
        for required in parameters.get("required", []):
            if required not in arguments:
                raise ValueError(f"Skill {call.name} missing required argument: {required}")

        properties = parameters.get("properties", {})
        for key, value in arguments.items():
            prop = properties.get(key)
            if not prop:
                continue
            if "enum" in prop and value not in prop["enum"]:
                raise ValueError(f"Skill {call.name} argument {key}={value!r} is outside whitelist")
            if prop.get("type") == "string" and not isinstance(value, str):
                raise TypeError(f"Skill {call.name} argument {key} must be string")
            if prop.get("type") == "integer" and not isinstance(value, int):
                raise TypeError(f"Skill {call.name} argument {key} must be integer")
            if "max_length" in prop and isinstance(value, str) and len(value) > int(prop["max_length"]):
                raise ValueError(f"Skill {call.name} argument {key} exceeds max_length")
            if "minimum" in prop and isinstance(value, int) and value < int(prop["minimum"]):
                raise ValueError(f"Skill {call.name} argument {key} below minimum")
            if "maximum" in prop and isinstance(value, int) and value > int(prop["maximum"]):
                raise ValueError(f"Skill {call.name} argument {key} above maximum")

        if spec.get("requires_confirmation") and self.require_confirmation:
            raise PermissionError(f"Skill {call.name} requires external confirmation in this profile")

        if call.timeout_s is None:
            call.timeout_s = int(spec.get("timeout_s", 5))
        return call
