"""ROS-free mission sequencing and readiness contracts for RVClaw.

The contract layer describes what the navigation stack and task workers must
do.  It never publishes a velocity command, opens a serial device, or
authorizes motion by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import yaml

from .dynamic_obstacle_policy import RecoveryContext, decide_recovery


MISSION_SCHEMA_VERSION = 1
DEFAULT_MAX_POSITION_STD_M = 0.15
DEFAULT_MAX_YAW_STD_DEG = 10.0
DEFAULT_MAX_POSE_AGE_S = 0.50
DEFAULT_MIN_STABLE_SAMPLES = 3
DEFAULT_ARRIVAL_POSITION_TOLERANCE_M = 0.20
DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG = 15.0
DEFAULT_MAX_ARRIVAL_REFINEMENTS = 2
DEFAULT_MAX_ARRIVAL_LOCALIZATION_FAILURES = 5


class MissionContractError(ValueError):
    """Raised when a mission catalog or runtime event is invalid."""


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MissionContractError("%s must be numeric" % label) from exc
    if not math.isfinite(result):
        raise MissionContractError("%s must be finite" % label)
    return result


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MissionContractError("%s must be a non-empty string" % label)
    return value.strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().casefold())


def _yaw_error_deg(actual_deg: float, expected_deg: float) -> float:
    return abs((actual_deg - expected_deg + 180.0) % 360.0 - 180.0)


def _tuple_strings(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise MissionContractError("%s must be a list" % label)
    result = []
    for item in value:
        result.append(_nonempty(item, label + " item"))
    return tuple(result)


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_deg: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.x_m, "x_m"),
            (self.y_m, "y_m"),
            (self.yaw_deg, "yaw_deg"),
        ):
            _finite(value, label)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], label: str = "pose") -> "Pose2D":
        if not isinstance(value, Mapping):
            raise MissionContractError("%s must be a mapping" % label)
        return cls(
            _finite(value.get("x_m"), label + ".x_m"),
            _finite(value.get("y_m"), label + ".y_m"),
            _finite(value.get("yaw_deg", 0.0), label + ".yaw_deg"),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x_m": float(self.x_m),
            "y_m": float(self.y_m),
            "yaw_deg": float(self.yaw_deg),
        }


@dataclass(frozen=True)
class LocationSpec:
    location_id: str
    pose: Pose2D
    aliases: tuple[str, ...] = ()
    arrival_position_tolerance_m: float = DEFAULT_ARRIVAL_POSITION_TOLERANCE_M
    arrival_yaw_tolerance_deg: float = DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG

    def __post_init__(self) -> None:
        _nonempty(self.location_id, "location_id")
        if (
            not math.isfinite(float(self.arrival_position_tolerance_m))
            or float(self.arrival_position_tolerance_m) <= 0.0
        ):
            raise MissionContractError(
                "arrival_position_tolerance_m must be positive and finite"
            )
        if (
            not math.isfinite(float(self.arrival_yaw_tolerance_deg))
            or float(self.arrival_yaw_tolerance_deg) <= 0.0
            or float(self.arrival_yaw_tolerance_deg) > 180.0
        ):
            raise MissionContractError(
                "arrival_yaw_tolerance_deg must be in (0, 180]"
            )

    def all_aliases(self) -> tuple[str, ...]:
        values = (self.location_id,) + self.aliases
        return tuple(dict.fromkeys(_normalize_text(item) for item in values))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    kind: str = "inspection"
    aliases: tuple[str, ...] = ()
    timeout_s: float = 30.0
    max_attempts: int = 1
    failure_policy: str = "abort"

    def __post_init__(self) -> None:
        _nonempty(self.task_id, "task_id")
        _nonempty(self.kind, "task kind")
        if not math.isfinite(float(self.timeout_s)) or float(self.timeout_s) <= 0.0:
            raise MissionContractError("task timeout_s must be positive and finite")
        if isinstance(self.max_attempts, bool) or int(self.max_attempts) != self.max_attempts:
            raise MissionContractError("task max_attempts must be an integer")
        if self.max_attempts < 1:
            raise MissionContractError("task max_attempts must be at least one")
        if self.failure_policy not in ("abort", "continue"):
            raise MissionContractError("task failure_policy must be abort or continue")

    def all_aliases(self) -> tuple[str, ...]:
        values = (self.task_id,) + self.aliases
        return tuple(dict.fromkeys(_normalize_text(item) for item in values))


@dataclass(frozen=True)
class MissionStep:
    location_id: str
    task_id: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        return {"location_id": self.location_id, "task_id": self.task_id}


@dataclass(frozen=True)
class MissionPlan:
    mission_id: str
    map_id: str
    frame_id: str
    steps: tuple[MissionStep, ...]
    map_yaml: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty(self.mission_id, "mission_id")
        _nonempty(self.map_id, "map_id")
        if self.frame_id != "map":
            raise MissionContractError("mission frame_id must be map")
        if not self.steps:
            raise MissionContractError("mission must contain at least one step")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mission_id": self.mission_id,
            "map_id": self.map_id,
            "frame_id": self.frame_id,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.map_yaml is not None:
            result["map_yaml"] = self.map_yaml
        return result


@dataclass(frozen=True)
class MissionCatalog:
    map_id: str
    frame_id: str
    locations: dict[str, LocationSpec]
    tasks: dict[str, TaskSpec]
    missions: dict[str, MissionPlan]
    map_yaml: Optional[str] = None
    mission_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def location(self, location_id: str) -> LocationSpec:
        try:
            return self.locations[location_id]
        except KeyError as exc:
            raise MissionContractError("unknown location %s" % location_id) from exc

    def task(self, task_id: str) -> TaskSpec:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise MissionContractError("unknown task %s" % task_id) from exc

    def mission(self, mission_id: str) -> MissionPlan:
        try:
            return self.missions[mission_id]
        except KeyError as exc:
            raise MissionContractError("unknown mission %s" % mission_id) from exc

    def resolve_location(self, text: str) -> str:
        return _resolve_alias(text, self.locations, "location")

    def resolve_task(self, text: str) -> str:
        return _resolve_alias(text, self.tasks, "task")


def _resolve_alias(text: str, entries: Mapping[str, Any], kind: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        raise MissionContractError("cannot resolve empty %s" % kind)
    matches: list[tuple[int, str]] = []
    for entry_id, entry in entries.items():
        for alias in entry.all_aliases():
            if alias and alias in normalized:
                matches.append((len(alias), entry_id))
    if not matches:
        raise MissionContractError("no %s found in command" % kind)
    longest = max(length for length, _ in matches)
    selected = sorted({entry_id for length, entry_id in matches if length == longest})
    if len(selected) != 1:
        raise MissionContractError("ambiguous %s: %s" % (kind, ",".join(selected)))
    return selected[0]


def _parse_step(value: Any, label: str) -> MissionStep:
    if isinstance(value, str):
        return MissionStep(_nonempty(value, label))
    if not isinstance(value, Mapping):
        raise MissionContractError("%s must be a location name or mapping" % label)
    location_id = _nonempty(value.get("location"), label + ".location")
    task_id = value.get("task")
    if task_id is not None:
        task_id = _nonempty(task_id, label + ".task")
    return MissionStep(location_id, task_id)


def load_mission_catalog(path: str | Path) -> MissionCatalog:
    catalog_path = Path(path).expanduser().resolve()
    try:
        document = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MissionContractError("mission catalog is not readable") from exc
    if not isinstance(document, Mapping):
        raise MissionContractError("mission catalog must contain a mapping")
    if int(document.get("schema_version", -1)) != MISSION_SCHEMA_VERSION:
        raise MissionContractError("unsupported mission schema_version")
    map_id = _nonempty(document.get("map_id"), "map_id")
    frame_id = _nonempty(document.get("frame_id", "map"), "frame_id")
    if frame_id != "map":
        raise MissionContractError("frame_id must be map")
    map_yaml = document.get("map_yaml")
    if map_yaml is not None:
        map_yaml = _nonempty(map_yaml, "map_yaml")

    raw_locations = document.get("locations")
    if not isinstance(raw_locations, Mapping) or not raw_locations:
        raise MissionContractError("catalog contains no locations")
    locations: dict[str, LocationSpec] = {}
    alias_owner: dict[str, str] = {}
    for location_id, raw in raw_locations.items():
        location_id = _nonempty(location_id, "location id")
        if location_id in locations:
            raise MissionContractError("duplicate location %s" % location_id)
        if not isinstance(raw, Mapping):
            raise MissionContractError("location %s must be a mapping" % location_id)
        entry = LocationSpec(
            location_id=location_id,
            pose=Pose2D.from_mapping(raw, "location %s" % location_id),
            aliases=_tuple_strings(raw.get("aliases"), "location aliases"),
            arrival_position_tolerance_m=_finite(
                raw.get(
                    "arrival_position_tolerance_m",
                    DEFAULT_ARRIVAL_POSITION_TOLERANCE_M,
                ),
                "location arrival_position_tolerance_m",
            ),
            arrival_yaw_tolerance_deg=_finite(
                raw.get(
                    "arrival_yaw_tolerance_deg",
                    DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG,
                ),
                "location arrival_yaw_tolerance_deg",
            ),
        )
        locations[location_id] = entry
        for alias in entry.all_aliases():
            previous = alias_owner.get(alias)
            if previous is not None and previous != location_id:
                raise MissionContractError("location alias %s is ambiguous" % alias)
            alias_owner[alias] = location_id

    raw_tasks = document.get("tasks", {})
    if not isinstance(raw_tasks, Mapping):
        raise MissionContractError("tasks must be a mapping")
    tasks: dict[str, TaskSpec] = {}
    task_alias_owner: dict[str, str] = {}
    for task_id, raw in raw_tasks.items():
        task_id = _nonempty(task_id, "task id")
        if not isinstance(raw, Mapping):
            raise MissionContractError("task %s must be a mapping" % task_id)
        task = TaskSpec(
            task_id=task_id,
            kind=_nonempty(raw.get("kind", "inspection"), "task kind"),
            aliases=_tuple_strings(raw.get("aliases"), "task aliases"),
            timeout_s=_finite(raw.get("timeout_s", 30.0), "task timeout_s"),
            max_attempts=int(raw.get("max_attempts", 1)),
            failure_policy=_nonempty(raw.get("failure_policy", "abort"), "task failure_policy"),
        )
        tasks[task_id] = task
        for alias in task.all_aliases():
            previous = task_alias_owner.get(alias)
            if previous is not None and previous != task_id:
                raise MissionContractError("task alias %s is ambiguous" % alias)
            task_alias_owner[alias] = task_id

    raw_missions = document.get("missions")
    if not isinstance(raw_missions, Mapping) or not raw_missions:
        raise MissionContractError("catalog contains no missions")
    missions: dict[str, MissionPlan] = {}
    for mission_id, raw_steps in raw_missions.items():
        mission_id = _nonempty(mission_id, "mission id")
        if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
            raise MissionContractError("mission %s must contain steps" % mission_id)
        steps = tuple(
            _parse_step(value, "mission %s step %d" % (mission_id, index))
            for index, value in enumerate(raw_steps, start=1)
        )
        for step in steps:
            if step.location_id not in locations:
                raise MissionContractError(
                    "mission %s references unknown location %s"
                    % (mission_id, step.location_id)
                )
            if step.task_id is not None and step.task_id not in tasks:
                raise MissionContractError(
                    "mission %s references unknown task %s" % (mission_id, step.task_id)
                )
        missions[mission_id] = MissionPlan(
            mission_id=mission_id,
            map_id=map_id,
            frame_id=frame_id,
            steps=steps,
            map_yaml=map_yaml,
        )
    raw_mission_aliases = document.get("mission_aliases", {})
    if not isinstance(raw_mission_aliases, Mapping):
        raise MissionContractError("mission_aliases must be a mapping")
    mission_aliases: dict[str, tuple[str, ...]] = {}
    alias_owner: dict[str, str] = {}
    for mission_id, aliases in raw_mission_aliases.items():
        mission_id = _nonempty(mission_id, "mission alias owner")
        if mission_id not in missions:
            raise MissionContractError(
                "mission_aliases references unknown mission %s" % mission_id
            )
        parsed = _tuple_strings(aliases, "mission aliases")
        mission_aliases[mission_id] = parsed
        for alias in parsed:
            normalized = _normalize_text(alias)
            previous = alias_owner.get(normalized)
            if previous is not None and previous != mission_id:
                raise MissionContractError("mission alias %s is ambiguous" % alias)
            alias_owner[normalized] = mission_id
    return MissionCatalog(
        map_id,
        frame_id,
        locations,
        tasks,
        missions,
        map_yaml,
        mission_aliases,
    )


@dataclass(frozen=True)
class LocalizationEvidence:
    map_id: str
    source: str
    pose: Pose2D
    converged: bool
    scan_fresh: bool
    pose_age_s: float
    position_std_m: float
    yaw_std_deg: float
    stable_samples: int
    global_search_complete: bool = False


@dataclass(frozen=True)
class GateResult:
    status: str
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons)}


class LocalizationGate:
    def __init__(
        self,
        max_position_std_m: float = DEFAULT_MAX_POSITION_STD_M,
        max_yaw_std_deg: float = DEFAULT_MAX_YAW_STD_DEG,
        max_pose_age_s: float = DEFAULT_MAX_POSE_AGE_S,
        min_stable_samples: int = DEFAULT_MIN_STABLE_SAMPLES,
    ) -> None:
        self.max_position_std_m = _finite(max_position_std_m, "max_position_std_m")
        self.max_yaw_std_deg = _finite(max_yaw_std_deg, "max_yaw_std_deg")
        self.max_pose_age_s = _finite(max_pose_age_s, "max_pose_age_s")
        if self.max_position_std_m <= 0.0 or self.max_yaw_std_deg <= 0.0:
            raise MissionContractError("localization thresholds must be positive")
        if isinstance(min_stable_samples, bool) or int(min_stable_samples) != min_stable_samples:
            raise MissionContractError("min_stable_samples must be an integer")
        if min_stable_samples < 1:
            raise MissionContractError("min_stable_samples must be positive")
        self.min_stable_samples = int(min_stable_samples)

    def evaluate(
        self,
        evidence: LocalizationEvidence,
        expected_map_id: str,
        require_global_search: bool = True,
    ) -> GateResult:
        reasons: list[str] = []
        if evidence.map_id != expected_map_id:
            reasons.append("map_id_mismatch")
        if not evidence.converged:
            reasons.append("localization_not_converged")
        if not evidence.scan_fresh:
            reasons.append("scan_not_fresh")
        if not math.isfinite(evidence.pose_age_s) or evidence.pose_age_s > self.max_pose_age_s:
            reasons.append("pose_stale")
        if not math.isfinite(evidence.position_std_m) or evidence.position_std_m > self.max_position_std_m:
            reasons.append("position_uncertainty_too_large")
        if not math.isfinite(evidence.yaw_std_deg) or evidence.yaw_std_deg > self.max_yaw_std_deg:
            reasons.append("heading_uncertainty_too_large")
        if evidence.stable_samples < self.min_stable_samples:
            reasons.append("insufficient_stable_samples")
        if require_global_search and not evidence.global_search_complete:
            reasons.append("global_scan_not_complete")
        if not evidence.source:
            reasons.append("localization_source_missing")
        return GateResult("PASS" if not reasons else "FAIL", tuple(reasons))


@dataclass(frozen=True)
class ArrivalGateResult:
    status: str
    reasons: tuple[str, ...]
    position_error_m: float
    yaw_error_deg: float
    localization_passed: bool

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "position_error_m": self.position_error_m,
            "yaw_error_deg": self.yaw_error_deg,
            "localization_passed": self.localization_passed,
        }


class ArrivalGate:
    """Verify a fresh, stable pose at the intended waypoint before continuing."""

    def __init__(self, localization_gate: Optional[LocalizationGate] = None) -> None:
        self.localization_gate = localization_gate or LocalizationGate()

    def evaluate(
        self,
        evidence: LocalizationEvidence,
        expected_map_id: str,
        location: LocationSpec,
    ) -> ArrivalGateResult:
        localization = self.localization_gate.evaluate(
            evidence, expected_map_id, require_global_search=False
        )
        reasons = list(localization.reasons)
        position_error_m = math.hypot(
            evidence.pose.x_m - location.pose.x_m,
            evidence.pose.y_m - location.pose.y_m,
        )
        yaw_error_deg = _yaw_error_deg(
            evidence.pose.yaw_deg, location.pose.yaw_deg
        )
        if position_error_m > location.arrival_position_tolerance_m:
            reasons.append("arrival_position_out_of_tolerance")
        if yaw_error_deg > location.arrival_yaw_tolerance_deg:
            reasons.append("arrival_heading_out_of_tolerance")
        return ArrivalGateResult(
            status="PASS" if not reasons else "FAIL",
            reasons=tuple(reasons),
            position_error_m=position_error_m,
            yaw_error_deg=yaw_error_deg,
            localization_passed=localization.passed,
        )


class MissionState(str, Enum):
    WAIT_LOCALIZATION = "WAIT_LOCALIZATION"
    NAVIGATING = "NAVIGATING"
    VERIFYING_ARRIVAL = "VERIFYING_ARRIVAL"
    TASK = "TASK"
    RECOVERY = "RECOVERY"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class MissionCommand:
    kind: str
    mission_id: str
    route_index: int
    location_id: Optional[str] = None
    task_id: Optional[str] = None
    reason: Optional[str] = None
    requires_safety_gate: bool = True
    motion_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "mission_id": self.mission_id,
            "route_index": self.route_index,
            "location_id": self.location_id,
            "task_id": self.task_id,
            "reason": self.reason,
            "requires_safety_gate": self.requires_safety_gate,
            "motion_authorized": self.motion_authorized,
        }


class MissionRunner:
    """Deterministic mission state machine for offline and adapter tests."""

    def __init__(
        self,
        catalog: MissionCatalog,
        mission_id: str,
        localization_gate: Optional[LocalizationGate] = None,
        arrival_gate: Optional[ArrivalGate] = None,
        max_arrival_refinements: int = DEFAULT_MAX_ARRIVAL_REFINEMENTS,
        max_arrival_localization_failures: int = DEFAULT_MAX_ARRIVAL_LOCALIZATION_FAILURES,
    ) -> None:
        self.catalog = catalog
        self.plan = catalog.mission(mission_id)
        self.gate = localization_gate or LocalizationGate()
        self.arrival_gate = arrival_gate or ArrivalGate(self.gate)
        if (
            isinstance(max_arrival_refinements, bool)
            or int(max_arrival_refinements) != max_arrival_refinements
            or max_arrival_refinements < 0
        ):
            raise MissionContractError("max_arrival_refinements must be a non-negative integer")
        if (
            isinstance(max_arrival_localization_failures, bool)
            or int(max_arrival_localization_failures)
            != max_arrival_localization_failures
            or max_arrival_localization_failures < 1
        ):
            raise MissionContractError(
                "max_arrival_localization_failures must be a positive integer"
            )
        self.max_arrival_refinements = int(max_arrival_refinements)
        self.max_arrival_localization_failures = int(
            max_arrival_localization_failures
        )
        self.state = MissionState.WAIT_LOCALIZATION
        self.route_index = 0
        self.task_attempts = 0
        self.recovery_attempts = 0
        self.wait_attempts = 0
        self.spin_attempts = 0
        self.arrival_refinements = 0
        self.arrival_localization_failures = 0
        self.completed: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.command: Optional[MissionCommand] = None
        self.last_gate: Optional[GateResult] = None
        self.last_arrival_gate: Optional[ArrivalGateResult] = None
        self.last_recovery: Optional[dict[str, Any]] = None

    @property
    def current_step(self) -> MissionStep:
        if self.route_index >= len(self.plan.steps):
            raise MissionContractError("mission has no current step")
        return self.plan.steps[self.route_index]

    def _record(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})

    def _set_command(self, kind: str, reason: Optional[str] = None) -> MissionCommand:
        step = self.current_step if self.route_index < len(self.plan.steps) else None
        self.command = MissionCommand(
            kind=kind,
            mission_id=self.plan.mission_id,
            route_index=self.route_index + 1,
            location_id=None if step is None else step.location_id,
            task_id=None if step is None else step.task_id,
            reason=reason,
        )
        self._record("command", **self.command.to_dict())
        return self.command

    def start(self, evidence: LocalizationEvidence) -> MissionCommand:
        if self.state != MissionState.WAIT_LOCALIZATION:
            raise MissionContractError("mission can only start from WAIT_LOCALIZATION")
        self.last_gate = self.gate.evaluate(
            evidence, self.plan.map_id, require_global_search=True
        )
        self._record("localization_gate", **self.last_gate.to_dict())
        if not self.last_gate.passed:
            return self._set_command(
                "REQUEST_GLOBAL_LOCALIZATION", ";".join(self.last_gate.reasons)
            )
        self.state = MissionState.NAVIGATING
        self._record("state", value=self.state.value)
        return self._set_command("NAVIGATE")

    def on_navigation_result(
        self,
        success: bool,
        recovery_context: Optional[RecoveryContext] = None,
        arrival_evidence: Optional[LocalizationEvidence] = None,
    ) -> MissionCommand:
        if self.state != MissionState.NAVIGATING:
            raise MissionContractError("navigation result received outside NAVIGATING")
        if success:
            self._record("navigation_result", status="PASS")
            self.state = MissionState.VERIFYING_ARRIVAL
            self._record("state", value=self.state.value)
            if arrival_evidence is None:
                return self._set_command(
                    "REQUEST_ARRIVAL_LOCALIZATION",
                    "navigation_success_requires_fresh_stable_arrival_pose",
                )
            return self.on_arrival_localization(arrival_evidence)
        if recovery_context is None:
            return self.abort("navigation_failed_without_recovery_evidence")
        decision = decide_recovery(recovery_context)
        self.last_recovery = dict(decision)
        self._record("recovery_decision", **decision)
        action = decision["action"]
        if action == "PROCEED":
            return self.abort("failed_navigation_cannot_proceed")
        if action == "ABORT_GOAL":
            return self.abort(decision.get("reason", "recovery_aborted"))
        self.state = MissionState.RECOVERY
        self._record("state", value=self.state.value)
        if action == "SPIN_SEARCH_THEN_REPLAN":
            self.spin_attempts += 1
            kind = "SPIN_THEN_RELOCALIZE"
        elif action == "STOP_WAIT_REPLAN":
            self.wait_attempts += 1
            kind = "STOP_REFRESH_REPLAN"
        elif action == "REPLAN_ALTERNATE":
            self.recovery_attempts += 1
            kind = "REPLAN_ALTERNATE"
        elif action == "HOLD_STOP":
            kind = "HOLD_STOP_REFRESH"
        else:
            self.recovery_attempts += 1
            kind = "STOP_SLOW_REPLAN"
        return self._set_command(kind, decision.get("reason"))

    def on_arrival_localization(
        self, evidence: LocalizationEvidence
    ) -> MissionCommand:
        if self.state != MissionState.VERIFYING_ARRIVAL:
            raise MissionContractError(
                "arrival localization received outside VERIFYING_ARRIVAL"
            )
        location = self.catalog.location(self.current_step.location_id)
        self.last_arrival_gate = self.arrival_gate.evaluate(
            evidence, self.plan.map_id, location
        )
        self._record(
            "arrival_gate",
            location_id=location.location_id,
            **self.last_arrival_gate.to_dict(),
        )
        if self.last_arrival_gate.passed:
            self.arrival_localization_failures = 0
            if self.current_step.task_id is not None:
                self.state = MissionState.TASK
                self.task_attempts = 0
                self._record("state", value=self.state.value)
                return self._set_command("RUN_TASK")
            return self._advance()

        reason = ";".join(self.last_arrival_gate.reasons)
        if not self.last_arrival_gate.localization_passed:
            self.arrival_localization_failures += 1
            if (
                self.arrival_localization_failures
                >= self.max_arrival_localization_failures
            ):
                return self.abort("arrival_localization_failed:%s" % reason)
            return self._set_command("REQUEST_FRESH_LOCALIZATION", reason)

        if self.arrival_refinements >= self.max_arrival_refinements:
            return self.abort("arrival_tolerance_not_met:%s" % reason)
        self.arrival_refinements += 1
        self.state = MissionState.NAVIGATING
        self._record(
            "state",
            value=self.state.value,
            arrival_refinement=self.arrival_refinements,
        )
        return self._set_command("REFINE_ARRIVAL", reason)

    def on_recovery_result(
        self,
        success: bool,
        evidence: Optional[LocalizationEvidence] = None,
    ) -> MissionCommand:
        if self.state != MissionState.RECOVERY:
            raise MissionContractError("recovery result received outside RECOVERY")
        if not success:
            return self.abort("recovery_action_failed")
        if evidence is None:
            return self._set_command("REQUEST_FRESH_LOCALIZATION", "recovery_requires_pose")
        self.last_gate = self.gate.evaluate(
            evidence, self.plan.map_id, require_global_search=False
        )
        self._record("post_recovery_localization_gate", **self.last_gate.to_dict())
        if not self.last_gate.passed:
            return self._set_command(
                "REQUEST_FRESH_LOCALIZATION", ";".join(self.last_gate.reasons)
            )
        self.state = MissionState.NAVIGATING
        self._record("state", value=self.state.value)
        return self._set_command("RETRY_NAVIGATION")

    def on_task_result(self, success: bool) -> MissionCommand:
        if self.state != MissionState.TASK:
            raise MissionContractError("task result received outside TASK")
        task = self.catalog.task(self.current_step.task_id or "")
        self.task_attempts += 1
        self._record(
            "task_result",
            task_id=task.task_id,
            status="PASS" if success else "FAIL",
            attempt=self.task_attempts,
        )
        if success:
            return self._advance()
        if self.task_attempts < task.max_attempts:
            return self._set_command("RUN_TASK", "task_retry")
        if task.failure_policy == "continue":
            return self._advance()
        return self.abort("task_failed:%s" % task.task_id)

    def _advance(self) -> MissionCommand:
        self.completed.append(self.current_step.location_id)
        self._record("waypoint_completed", location_id=self.current_step.location_id)
        self.route_index += 1
        self.task_attempts = 0
        self.recovery_attempts = 0
        self.wait_attempts = 0
        self.spin_attempts = 0
        self.arrival_refinements = 0
        self.arrival_localization_failures = 0
        if self.route_index >= len(self.plan.steps):
            self.state = MissionState.COMPLETE
            self._record("state", value=self.state.value)
            self.command = MissionCommand(
                kind="MISSION_COMPLETE",
                mission_id=self.plan.mission_id,
                route_index=len(self.plan.steps),
                requires_safety_gate=False,
                motion_authorized=False,
            )
            self._record("command", **self.command.to_dict())
            return self.command
        self.state = MissionState.NAVIGATING
        self._record("state", value=self.state.value)
        return self._set_command("NAVIGATE")

    def abort(self, reason: str) -> MissionCommand:
        self.state = MissionState.ABORTED
        self._record("state", value=self.state.value, reason=reason)
        self.command = MissionCommand(
            kind="ABORT_AND_ZERO",
            mission_id=self.plan.mission_id,
            route_index=min(self.route_index + 1, len(self.plan.steps)),
            reason=reason,
            requires_safety_gate=False,
            motion_authorized=False,
        )
        self._record("command", **self.command.to_dict())
        return self.command

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "mission": self.plan.to_dict(),
            "route_index": self.route_index,
            "completed": list(self.completed),
            "command": None if self.command is None else self.command.to_dict(),
            "last_gate": None if self.last_gate is None else self.last_gate.to_dict(),
            "last_arrival_gate": (
                None
                if self.last_arrival_gate is None
                else self.last_arrival_gate.to_dict()
            ),
            "last_recovery": self.last_recovery,
            "arrival_refinements": self.arrival_refinements,
            "arrival_localization_failures": self.arrival_localization_failures,
            "events": list(self.events),
            "motion_authorized": False,
        }
