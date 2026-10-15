"""Run deterministic, hardware-free mission contract scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .dynamic_obstacle_policy import RecoveryContext
from .mission_contract import (
    LocalizationEvidence,
    MissionCatalog,
    MissionRunner,
    Pose2D,
    load_mission_catalog,
)


def _healthy_evidence(
    catalog: MissionCatalog,
    *,
    global_search_complete: bool,
    pose: Pose2D | None = None,
) -> LocalizationEvidence:
    return LocalizationEvidence(
        map_id=catalog.map_id,
        source="amcl_global_scan" if global_search_complete else "amcl",
        pose=pose or Pose2D(0.0, 0.0, 0.0),
        converged=True,
        scan_fresh=True,
        pose_age_s=0.05,
        position_std_m=0.04,
        yaw_std_deg=3.0,
        stable_samples=5,
        global_search_complete=global_search_complete,
    )


def _run_to_completion(runner: MissionRunner) -> dict[str, Any]:
    while runner.state.value not in ("COMPLETE", "ABORTED"):
        if runner.state.value == "NAVIGATING":
            pose = runner.catalog.location(runner.current_step.location_id).pose
            runner.on_navigation_result(
                True,
                arrival_evidence=_healthy_evidence(
                    runner.catalog,
                    global_search_complete=False,
                    pose=pose,
                ),
            )
        elif runner.state.value == "VERIFYING_ARRIVAL":
            pose = runner.catalog.location(runner.current_step.location_id).pose
            runner.on_arrival_localization(
                _healthy_evidence(
                    runner.catalog,
                    global_search_complete=False,
                    pose=pose,
                )
            )
        elif runner.state.value == "TASK":
            runner.on_task_result(True)
        elif runner.state.value == "WAIT_LOCALIZATION":
            runner.start(_healthy_evidence(runner.catalog, global_search_complete=True))
        else:
            raise RuntimeError("unexpected simulation state %s" % runner.state.value)
    return runner.snapshot()


def scenario_happy(catalog: MissionCatalog, mission_id: str) -> dict[str, Any]:
    runner = MissionRunner(catalog, mission_id)
    runner.start(_healthy_evidence(catalog, global_search_complete=True))
    return _run_to_completion(runner)


def scenario_localization_retry(catalog: MissionCatalog, mission_id: str) -> dict[str, Any]:
    runner = MissionRunner(catalog, mission_id)
    bad = LocalizationEvidence(
        map_id=catalog.map_id,
        source="cached_pose",
        pose=Pose2D(0.0, 0.0, 0.0),
        converged=False,
        scan_fresh=False,
        pose_age_s=4.0,
        position_std_m=0.5,
        yaw_std_deg=30.0,
        stable_samples=0,
        global_search_complete=False,
    )
    first = runner.start(bad).to_dict()
    runner.start(_healthy_evidence(catalog, global_search_complete=True))
    return {
        "first_command": first,
        "final": _run_to_completion(runner),
    }


def scenario_transient_obstacle(catalog: MissionCatalog, mission_id: str) -> dict[str, Any]:
    runner = MissionRunner(catalog, mission_id)
    runner.start(_healthy_evidence(catalog, global_search_complete=True))
    context = RecoveryContext(
        obstacle_detected=True,
        obstacle_persistent=False,
        front_stop_active=True,
        path_available=False,
        scan_fresh=True,
        costmap_fresh=True,
        localization_stable=True,
        motion_feedback_fresh=True,
        robot_stopped=True,
        wait_attempts=0,
    )
    recovery_command = runner.on_navigation_result(False, context).to_dict()
    retry_command = runner.on_recovery_result(
        True, _healthy_evidence(catalog, global_search_complete=False)
    ).to_dict()
    pose = catalog.location(runner.current_step.location_id).pose
    runner.on_navigation_result(
        True,
        arrival_evidence=_healthy_evidence(
            catalog, global_search_complete=False, pose=pose
        ),
    )
    final = _run_to_completion(runner)
    return {
        "recovery_command": recovery_command,
        "retry_command": retry_command,
        "final": final,
    }


def scenario_persistent_obstacle(catalog: MissionCatalog, mission_id: str) -> dict[str, Any]:
    runner = MissionRunner(catalog, mission_id)
    runner.start(_healthy_evidence(catalog, global_search_complete=True))
    context = RecoveryContext(
        obstacle_detected=True,
        obstacle_persistent=True,
        front_stop_active=True,
        path_available=False,
        scan_fresh=True,
        costmap_fresh=True,
        localization_stable=True,
        motion_feedback_fresh=True,
        robot_stopped=True,
        spin_clearance_pass=True,
        spin_clearance_m=0.30,
        spin_coverage_deg=350.0,
        wait_attempts=2,
    )
    spin_command = runner.on_navigation_result(False, context).to_dict()
    retry_command = runner.on_recovery_result(
        True, _healthy_evidence(catalog, global_search_complete=False)
    ).to_dict()
    pose = catalog.location(runner.current_step.location_id).pose
    runner.on_navigation_result(
        True,
        arrival_evidence=_healthy_evidence(
            catalog, global_search_complete=False, pose=pose
        ),
    )
    final = _run_to_completion(runner)
    return {
        "spin_command": spin_command,
        "retry_command": retry_command,
        "final": final,
    }


def scenario_arrival_localization_refresh(
    catalog: MissionCatalog, mission_id: str
) -> dict[str, Any]:
    runner = MissionRunner(catalog, mission_id)
    runner.start(_healthy_evidence(catalog, global_search_complete=True))
    request_command = runner.on_navigation_result(True).to_dict()
    target = catalog.location(runner.current_step.location_id).pose
    stale = LocalizationEvidence(
        map_id=catalog.map_id,
        source="amcl",
        pose=target,
        converged=True,
        scan_fresh=False,
        pose_age_s=2.0,
        position_std_m=0.04,
        yaw_std_deg=3.0,
        stable_samples=1,
        global_search_complete=False,
    )
    refresh_command = runner.on_arrival_localization(stale).to_dict()
    task_command = runner.on_arrival_localization(
        _healthy_evidence(
            catalog, global_search_complete=False, pose=target
        )
    ).to_dict()
    final = _run_to_completion(runner)
    return {
        "request_command": request_command,
        "refresh_command": refresh_command,
        "task_command": task_command,
        "final": final,
    }


def scenario_arrival_refinement(
    catalog: MissionCatalog, mission_id: str
) -> dict[str, Any]:
    runner = MissionRunner(catalog, mission_id)
    runner.start(_healthy_evidence(catalog, global_search_complete=True))
    target = catalog.location(runner.current_step.location_id).pose
    offset = Pose2D(target.x_m + 0.35, target.y_m, target.yaw_deg + 25.0)
    refinement_command = runner.on_navigation_result(
        True,
        arrival_evidence=_healthy_evidence(
            catalog, global_search_complete=False, pose=offset
        ),
    ).to_dict()
    task_command = runner.on_navigation_result(
        True,
        arrival_evidence=_healthy_evidence(
            catalog, global_search_complete=False, pose=target
        ),
    ).to_dict()
    final = _run_to_completion(runner)
    return {
        "refinement_command": refinement_command,
        "task_command": task_command,
        "final": final,
    }


SCENARIOS: dict[str, Callable[[MissionCatalog, str], dict[str, Any]]] = {
    "happy": scenario_happy,
    "localization_retry": scenario_localization_retry,
    "transient_obstacle": scenario_transient_obstacle,
    "persistent_obstacle": scenario_persistent_obstacle,
    "arrival_localization_refresh": scenario_arrival_localization_refresh,
    "arrival_refinement": scenario_arrival_refinement,
}


def validate_catalog(catalog: MissionCatalog, mission_id: str) -> dict[str, Any]:
    results = {name: runner(catalog, mission_id) for name, runner in SCENARIOS.items()}
    failures: list[str] = []
    if results["happy"]["state"] != "COMPLETE":
        failures.append("happy scenario did not complete")
    if results["localization_retry"]["final"]["state"] != "COMPLETE":
        failures.append("localization retry did not complete")
    if results["transient_obstacle"]["final"]["state"] != "COMPLETE":
        failures.append("transient obstacle recovery did not complete")
    if results["persistent_obstacle"]["final"]["state"] != "COMPLETE":
        failures.append("persistent obstacle recovery did not complete")
    if results["arrival_localization_refresh"]["final"]["state"] != "COMPLETE":
        failures.append("arrival localization refresh did not complete")
    if results["arrival_refinement"]["final"]["state"] != "COMPLETE":
        failures.append("arrival refinement did not complete")
    if results["transient_obstacle"]["recovery_command"]["kind"] != "STOP_REFRESH_REPLAN":
        failures.append("transient obstacle did not stop and replan")
    if results["persistent_obstacle"]["spin_command"]["kind"] != "SPIN_THEN_RELOCALIZE":
        failures.append("persistent obstacle did not require bounded spin")
    if (
        results["arrival_localization_refresh"]["refresh_command"]["kind"]
        != "REQUEST_FRESH_LOCALIZATION"
    ):
        failures.append("stale arrival pose did not request fresh localization")
    if (
        results["arrival_refinement"]["refinement_command"]["kind"]
        != "REFINE_ARRIVAL"
    ):
        failures.append("off-target arrival did not request pose refinement")
    return {
        "status": "PASS" if not failures else "FAIL",
        "motion_authorized": False,
        "mission_id": mission_id,
        "map_id": catalog.map_id,
        "failures": failures,
        "scenarios": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--mission", default="inspection_route")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_catalog(load_mission_catalog(args.catalog), args.mission)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output exists; use a new evidence path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
