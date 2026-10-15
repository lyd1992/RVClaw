"""Safe translation from human language to a mission request.

The parser returns semantic IDs only.  It never emits a velocity command and
never chooses a map pose outside the loaded, versioned mission catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import re
from typing import Any, Optional
from pathlib import Path

from .mission_contract import (
    MissionCatalog,
    MissionContractError,
    MissionPlan,
    MissionStep,
    load_mission_catalog,
    _normalize_text,
)


@dataclass(frozen=True)
class VoiceRequest:
    original_text: str
    mission_id: Optional[str] = None
    location_id: Optional[str] = None
    task_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "mission_id": self.mission_id,
            "location_id": self.location_id,
            "task_id": self.task_id,
        }


def _locations_in_text(text: str, catalog: MissionCatalog) -> list[str]:
    normalized = _normalize_text(text)
    matches: list[tuple[int, int, str]] = []
    for location_id, entry in catalog.locations.items():
        for alias in entry.all_aliases():
            start = normalized.find(alias)
            while start >= 0:
                matches.append((start, start + len(alias), location_id))
                start = normalized.find(alias, start + 1)
    selected: list[tuple[int, int, str]] = []
    for candidate in sorted(matches, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(candidate[0] < end and start < candidate[1] for start, end, _ in selected):
            continue
        selected.append(candidate)
    return [location_id for _, _, location_id in selected]


def _mission_aliases(catalog: MissionCatalog) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for mission_id in catalog.missions:
        aliases[_normalize_text(mission_id)] = mission_id
        for alias in catalog.mission_aliases.get(mission_id, ()):
            normalized = _normalize_text(alias)
            previous = aliases.get(normalized)
            if previous is not None and previous != mission_id:
                raise MissionContractError("ambiguous mission alias: %s" % alias)
            aliases[normalized] = mission_id
    return aliases


def _resolve_mission(text: str, catalog: MissionCatalog) -> Optional[str]:
    normalized = _normalize_text(text)
    matches = [
        (len(alias), mission_id)
        for alias, mission_id in _mission_aliases(catalog).items()
        if alias and alias in normalized
    ]
    if not matches:
        return None
    longest = max(length for length, _ in matches)
    mission_ids = sorted({mission_id for length, mission_id in matches if length == longest})
    if len(mission_ids) != 1:
        raise MissionContractError("ambiguous mission: %s" % ",".join(mission_ids))
    return mission_ids[0]


def parse_voice_command(text: str, catalog: MissionCatalog) -> VoiceRequest:
    if not isinstance(text, str) or not text.strip():
        raise MissionContractError("voice command must be a non-empty string")
    mission_id = _resolve_mission(text, catalog)
    if mission_id is not None:
        return VoiceRequest(text, mission_id=mission_id)

    locations = _locations_in_text(text, catalog)
    if not locations:
        raise MissionContractError("voice command contains no known location")
    task_id: Optional[str] = None
    try:
        task_id = catalog.resolve_task(text)
    except MissionContractError as exc:
        if "no task found" not in str(exc):
            raise
    if len(locations) > 1 and task_id is None:
        raise MissionContractError(
            "multiple locations require a named mission or task: %s"
            % ",".join(locations)
        )
    return VoiceRequest(text, location_id=locations[0], task_id=task_id)


def compile_voice_request(request: VoiceRequest, catalog: MissionCatalog) -> MissionPlan:
    if request.mission_id is not None:
        return catalog.mission(request.mission_id)
    if request.location_id is None:
        raise MissionContractError("voice request has no location or mission")
    catalog.location(request.location_id)
    if request.task_id is not None:
        catalog.task(request.task_id)
    mission_id = "voice-%s" % re.sub(r"[^a-zA-Z0-9_-]+", "-", request.location_id)
    return MissionPlan(
        mission_id=mission_id,
        map_id=catalog.map_id,
        frame_id=catalog.frame_id,
        steps=(MissionStep(request.location_id, request.task_id),),
        map_yaml=catalog.map_yaml,
    )


def voice_request_to_navigation_goal(
    request: VoiceRequest, catalog: MissionCatalog
) -> dict[str, Any]:
    plan = compile_voice_request(request, catalog)
    step = plan.steps[0]
    location = catalog.location(step.location_id)
    return {
        "map_id": plan.map_id,
        "frame_id": plan.frame_id,
        "location_id": step.location_id,
        "task_id": step.task_id,
        "pose": location.pose.to_dict(),
        "requires_runtime_safety_gate": True,
        "motion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    catalog = load_mission_catalog(args.catalog)
    request = parse_voice_command(args.text, catalog)
    print(json.dumps({"request": request.to_dict(), "goal": voice_request_to_navigation_goal(request, catalog)}, ensure_ascii=False, indent=2))
    return 0
