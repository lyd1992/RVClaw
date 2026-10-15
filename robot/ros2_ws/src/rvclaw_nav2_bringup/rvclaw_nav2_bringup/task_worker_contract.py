"""ROS-free contract for waypoint inspection workers.

This validates configuration and request/result identity only. It does not
call a camera, microphone, ROS endpoint, serial port, or velocity topic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import yaml

from .mission_contract import MissionCatalog, MissionContractError, load_mission_catalog


TASK_BINDING_SCHEMA_VERSION = 1
ALLOWED_TRANSPORTS = ("ros2_service", "ros2_action")
ALLOWED_RESULT_STATUSES = ("PASS", "FAIL", "CANCELLED")
INTERFACE_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9_]*/(?:srv|action)/[A-Za-z][A-Za-z0-9_]*$"
)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MissionContractError("%s must be a non-empty string" % label)
    return value.strip()


@dataclass(frozen=True)
class TaskBinding:
    task_id: str
    transport: str
    endpoint: str
    interface_type: str
    enabled: bool
    timeout_s: float
    may_publish_motion: bool = False
    readiness_endpoint: Optional[str] = None

    def readiness_failures(self, catalog: MissionCatalog) -> tuple[str, ...]:
        failures: list[str] = []
        if self.task_id not in catalog.tasks:
            failures.append("unknown_task")
        if not self.enabled:
            failures.append("binding_disabled")
        if self.transport not in ALLOWED_TRANSPORTS:
            failures.append("unsupported_transport")
        if not self.endpoint.startswith("/") or self.endpoint == "/":
            failures.append("endpoint_not_absolute")
        if (
            not INTERFACE_PATTERN.fullmatch(self.interface_type)
            or "TODO" in self.interface_type.upper()
        ):
            failures.append("interface_type_invalid_or_placeholder")
        expected_kind = "/srv/" if self.transport == "ros2_service" else "/action/"
        if self.transport in ALLOWED_TRANSPORTS and expected_kind not in self.interface_type:
            failures.append("transport_interface_mismatch")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            failures.append("timeout_invalid")
        if self.task_id in catalog.tasks and self.timeout_s > catalog.tasks[self.task_id].timeout_s:
            failures.append("binding_timeout_exceeds_task_timeout")
        if self.may_publish_motion:
            failures.append("task_worker_may_publish_motion")
        if self.readiness_endpoint is not None and not self.readiness_endpoint.startswith("/"):
            failures.append("readiness_endpoint_not_absolute")
        return tuple(failures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "interface_type": self.interface_type,
            "enabled": self.enabled,
            "timeout_s": self.timeout_s,
            "may_publish_motion": self.may_publish_motion,
            "readiness_endpoint": self.readiness_endpoint,
        }


@dataclass(frozen=True)
class TaskBindingCatalog:
    bindings: dict[str, TaskBinding]


@dataclass(frozen=True)
class TaskRequest:
    request_id: str
    mission_id: str
    map_id: str
    route_index: int
    location_id: str
    task_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "map_id": self.map_id,
            "route_index": self.route_index,
            "location_id": self.location_id,
            "task_id": self.task_id,
            "motion_authorized": False,
        }


@dataclass(frozen=True)
class TaskResult:
    request_id: str
    location_id: str
    task_id: str
    status: str
    result: Mapping[str, Any]
    error: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskResult":
        if not isinstance(value, Mapping):
            raise MissionContractError("task result must be a mapping")
        payload = value.get("result", {})
        if not isinstance(payload, Mapping):
            raise MissionContractError("task result.result must be a mapping")
        error = value.get("error")
        if error is not None and not isinstance(error, str):
            raise MissionContractError("task result.error must be a string or null")
        return cls(
            request_id=_required_text(value.get("request_id"), "request_id"),
            location_id=_required_text(value.get("location_id"), "location_id"),
            task_id=_required_text(value.get("task_id"), "task_id"),
            status=_required_text(value.get("status"), "status").upper(),
            result=dict(payload),
            error=error,
        )


def load_task_bindings(path: str | Path) -> TaskBindingCatalog:
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MissionContractError("task bindings are not readable") from exc
    if not isinstance(document, Mapping):
        raise MissionContractError("task bindings must contain a mapping")
    if int(document.get("schema_version", -1)) != TASK_BINDING_SCHEMA_VERSION:
        raise MissionContractError("unsupported task binding schema_version")
    raw_bindings = document.get("bindings")
    if not isinstance(raw_bindings, Mapping):
        raise MissionContractError("task bindings.bindings must be a mapping")
    bindings: dict[str, TaskBinding] = {}
    for task_id, raw in raw_bindings.items():
        task_id = _required_text(task_id, "task_id")
        if not isinstance(raw, Mapping):
            raise MissionContractError("binding %s must be a mapping" % task_id)
        bindings[task_id] = TaskBinding(
            task_id=task_id,
            transport=_required_text(raw.get("transport"), "transport"),
            endpoint=_required_text(raw.get("endpoint"), "endpoint"),
            interface_type=_required_text(raw.get("interface_type"), "interface_type"),
            enabled=bool(raw.get("enabled", False)),
            timeout_s=float(raw.get("timeout_s", 30.0)),
            may_publish_motion=bool(raw.get("may_publish_motion", False)),
            readiness_endpoint=(
                None
                if raw.get("readiness_endpoint") is None
                else _required_text(raw.get("readiness_endpoint"), "readiness_endpoint")
            ),
        )
    return TaskBindingCatalog(bindings)


def validate_task_bindings(
    catalog: MissionCatalog,
    mission_id: str,
    binding_catalog: TaskBindingCatalog,
) -> dict[str, Any]:
    plan = catalog.mission(mission_id)
    required_tasks = tuple(
        dict.fromkeys(step.task_id for step in plan.steps if step.task_id is not None)
    )
    failures: list[str] = []
    entries: list[dict[str, Any]] = []
    for task_id in required_tasks:
        binding = binding_catalog.bindings.get(task_id)
        if binding is None:
            failures.append("missing_binding:%s" % task_id)
            entries.append(
                {"task_id": task_id, "status": "FAIL", "failures": ["missing_binding"]}
            )
            continue
        binding_failures = binding.readiness_failures(catalog)
        if binding_failures:
            failures.extend(
                "binding:%s:%s" % (task_id, item) for item in binding_failures
            )
        entries.append(
            {
                **binding.to_dict(),
                "status": "PASS" if not binding_failures else "FAIL",
                "failures": list(binding_failures),
            }
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "motion_authorized": False,
        "map_id": catalog.map_id,
        "mission_id": mission_id,
        "required_tasks": list(required_tasks),
        "failures": failures,
        "bindings": entries,
    }


def validate_task_result(request: TaskRequest, result: TaskResult) -> dict[str, Any]:
    failures: list[str] = []
    if result.request_id != request.request_id:
        failures.append("request_id_mismatch")
    if result.location_id != request.location_id:
        failures.append("location_id_mismatch")
    if result.task_id != request.task_id:
        failures.append("task_id_mismatch")
    if result.status not in ALLOWED_RESULT_STATUSES:
        failures.append("unsupported_status")
    if result.status == "PASS" and result.error:
        failures.append("pass_result_contains_error")
    if result.status != "PASS" and not result.error:
        failures.append("failed_result_missing_error")
    confidence = result.result.get("confidence")
    if confidence is not None:
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            failures.append("confidence_not_numeric")
        else:
            if not math.isfinite(numeric_confidence) or not 0.0 <= numeric_confidence <= 1.0:
                failures.append("confidence_out_of_range")
    return {
        "status": "PASS" if not failures else "FAIL",
        "task_succeeded": result.status == "PASS" and not failures,
        "motion_authorized": False,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path)
    parser.add_argument("--mission", default="inspection_route")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = validate_task_bindings(
        load_mission_catalog(args.catalog),
        args.mission,
        load_task_bindings(args.bindings),
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output exists; use a new evidence path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not args.quiet:
        print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
