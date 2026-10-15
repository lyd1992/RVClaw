"""Validated speech-event contract for waypoint inspection results.

The navigation route owns motion and publishes atomic JSON events.  The voice
adapter owns audio playback and writes an acknowledgement only after the
playback attempt has returned.  This module has no hardware dependencies so
the contract can be exhaustively tested off-board.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Optional


MAX_INSPECTION_POINTS = 3


@dataclass(frozen=True)
class EventPrompt:
    text: str
    filename: str
    speech_key: str
    silent: bool = False


RESULT_PROMPTS = {
    ("meter_inspection_result", "PASS", "NORMAL"): EventPrompt(
        text="仪表状态正常",
        filename="meter-normal.wav",
        speech_key="meter_normal",
    ),
    ("meter_inspection_result", "PASS", "ABNORMAL"): EventPrompt(
        text="仪表状态异常",
        filename="meter-abnormal.wav",
        speech_key="meter_abnormal",
    ),
    ("meter_inspection_failed", "FAIL", "UNKNOWN"): EventPrompt(
        text="仪表检测失败",
        filename="meter-unreadable.wav",
        speech_key="meter_unreadable",
    ),
    ("meter_inspection_heading_failed", "FAIL", "UNKNOWN"): EventPrompt(
        text="观察方向调整未完成，未执行仪表识别",
        filename="meter-heading-failed.wav",
        speech_key="meter_heading_failed",
    ),
    ("meter_inspection_runtime_failed", "FAIL", "UNKNOWN"): EventPrompt(
        text="仪表巡检系统异常",
        filename="meter-runtime-failed.wav",
        speech_key="meter_runtime_failed",
    ),
}

WAYPOINT_LABELS = {
    "A": "第一个巡检点",
    "B": "第二个巡检点",
    "C": "第三个巡检点",
}

POINT_RESULT_PHRASES = {
    "meter_normal": ("仪表检测完成。结果正常。", "normal"),
    "meter_abnormal": ("仪表检测完成。结果异常。", "abnormal"),
    "meter_unreadable": ("仪表检测失败。将继续后续任务。", "unreadable"),
    "meter_heading_failed": (
        "观察方向调整未完成，未执行仪表识别。将继续后续任务。",
        "heading-failed",
    ),
    "meter_runtime_failed": (
        "仪表巡检系统异常。将继续后续任务。",
        "runtime-failed",
    ),
}

POINT_RESULT_PROMPTS = {
    (waypoint, speech_key): EventPrompt(
        text=f"{label}。{phrase}",
        filename=f"meter-{waypoint.lower()}-{suffix}.wav",
        speech_key=speech_key,
    )
    for waypoint, label in WAYPOINT_LABELS.items()
    for speech_key, (phrase, suffix) in POINT_RESULT_PHRASES.items()
}

NAVIGATION_PROMPTS = {
    "navigation_start_failed_system": EventPrompt("导航启动检查未通过，巡检没有开始。请保持小车不动并检查导航服务，恢复后再次唤醒我，说开始巡检。", "navigation-start-failed-system.wav", "navigation_start_failed_system"),
    "navigation_start_failed_localization": EventPrompt("启动前定位未完成，巡检没有开始。请确认小车位于当前地图内，保持静止，准备好后再次唤醒我，说开始巡检。", "navigation-start-failed-localization.wav", "navigation_start_failed_localization"),
    "navigation_start_failed_obstacle": EventPrompt("启动前路线受阻，巡检没有开始。请清除通行障碍，准备好后再次唤醒我，说开始巡检。", "navigation-start-failed-obstacle.wav", "navigation_start_failed_obstacle"),
    "navigation_start_failed_lidar": EventPrompt("启动前雷达数据异常，巡检没有开始。请检查雷达电源和连接，恢复后再次唤醒我，说开始巡检。", "navigation-start-failed-lidar.wav", "navigation_start_failed_lidar"),
    "navigation_start_failed_chassis": EventPrompt("启动前底盘通信异常，巡检没有开始。请检查总电源和通信线，恢复后再次唤醒我，说开始巡检。", "navigation-start-failed-chassis.wav", "navigation_start_failed_chassis"),
    "navigation_obstacle_recovery_started": EventPrompt("路线受阻，正在等待障碍物移开并重新规划。", "navigation-obstacle-recovery-started.wav", "navigation_obstacle_recovery_started"),
    "navigation_localization_recovery_started": EventPrompt("定位不稳定，正在自动重新定位。", "navigation-localization-recovery-started.wav", "navigation_localization_recovery_started"),
    "navigation_lidar_recovery_started": EventPrompt("雷达数据短暂异常，正在等待雷达恢复。", "navigation-lidar-recovery-started.wav", "navigation_lidar_recovery_started"),
    # The route still waits for this event's acknowledgement, but a successful
    # automatic recovery no longer needs an audible interruption.
    "navigation_recovery_succeeded": EventPrompt(
        "导航已恢复，继续巡检。",
        "navigation-recovery-succeeded.wav",
        "navigation_recovery_succeeded",
        silent=True,
    ),
    "navigation_paused_obstacle": EventPrompt("导航仍然受阻，巡检已暂停。请移开小车周围或通往目标点的障碍物，然后唤醒我，说继续巡检。", "navigation-paused-obstacle.wav", "navigation_paused_obstacle"),
    "navigation_paused_localization": EventPrompt("定位失败，巡检已暂停。请把小车放到当前地图内特征明显且周围空旷的位置，然后唤醒我，说重新定位并继续巡检。", "navigation-paused-localization.wav", "navigation_paused_localization"),
    "navigation_paused_lidar": EventPrompt("雷达数据未恢复，巡检已暂停。请检查雷达电源和USB连接，恢复后唤醒我，说继续巡检。", "navigation-paused-lidar.wav", "navigation_paused_lidar"),
    "navigation_paused_chassis": EventPrompt("底盘通信异常，巡检已暂停。请检查小车总电源、串口三和通信线，恢复后唤醒我，说继续巡检。", "navigation-paused-chassis.wav", "navigation_paused_chassis"),
    "navigation_paused_command_timeout": EventPrompt("导航控制指令超时，巡检已暂停。请保持小车不动，唤醒我，说继续巡检。", "navigation-paused-command-timeout.wav", "navigation_paused_command_timeout"),
    "navigation_paused_deadman_timeout": EventPrompt("导航控制心跳超时，巡检已暂停。请保持小车不动并检查导航服务。", "navigation-paused-deadman-timeout.wav", "navigation_paused_deadman_timeout"),
    "navigation_paused_heading": EventPrompt("已到返航点，车头回正未完成。巡检已暂停。", "navigation-paused-heading.wav", "navigation_paused_heading"),
    "navigation_paused_return_position": EventPrompt("返航位置未达到要求，巡检已暂停。", "navigation-paused-return-position.wav", "navigation_paused_return_position"),
    "navigation_paused_safety": EventPrompt("安全保护已触发，巡检已暂停。请清除周围危险并检查急停，确认安全后人工解除，再唤醒我，说继续巡检。", "navigation-paused-safety.wav", "navigation_paused_safety"),
    "navigation_paused_low_battery": EventPrompt("小车电量过低，巡检已暂停。请关闭运动电源并更换或充电，恢复后唤醒我，说继续巡检。", "navigation-paused-low-battery.wav", "navigation_paused_low_battery"),
    "navigation_paused_system": EventPrompt("导航系统异常，巡检已暂停。请保持小车不动并检查导航服务，恢复后唤醒我，说继续巡检。", "navigation-paused-system.wav", "navigation_paused_system"),
}

NAVIGATION_EVENT_STATUS = {
    "navigation_start_failed": "FAIL",
    "navigation_recovery_started": "STARTED",
    "navigation_recovery_succeeded": "PASS",
    "navigation_paused": "FAIL",
    "navigation_resume_failed": "FAIL",
}

MAPPING_EVENT_PROMPTS = {
    ("mapping_completed", "PASS"): EventPrompt(
        "地图保存成功", "mapping-saved.wav", "mapping_completed"
    ),
    ("mapping_failed", "FAIL"): EventPrompt(
        "地图保存失败", "mapping-save-failed.wav", "mapping_failed"
    ),
    ("mapping_cancelled", "CANCELLED"): EventPrompt(
        "建图已取消，地图没有保存", "mapping-cancelled.wav", "mapping_cancelled"
    ),
    ("mapping_not_stationary", "RETRY"): EventPrompt(
        "小车仍在移动，请关闭手柄并保持静止后再次说建图结束",
        "mapping-not-stationary.wav",
        "mapping_not_stationary",
    ),
}

COMMAND_PROMPTS = {
    "请确认继续巡检": "confirm-resume-inspection.wav",
    "请确认重新定位并继续巡检": "confirm-relocalize-resume-inspection.wav",
    "巡检恢复任务已启动": "inspection-resume-launched.wav",
    "重新定位并恢复巡检任务已启动": "inspection-relocalize-resume-launched.wav",
    "建图已开始，请使用手柄低速巡视环境": "mapping-started.wav",
    "正在保存地图，请保持小车静止": "mapping-saving.wav",
    "当前没有正在进行的建图任务": "mapping-not-running.wav",
    "建图正在进行，请先结束当前建图": "mapping-already-running.wav",
    "当前正在建图": "status-mapping.wav",
    "建图已取消，地图没有保存": "mapping-cancelled.wav",
}


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number != value or number < 0 or number > MAX_INSPECTION_POINTS:
        raise ValueError(f"{label} must be in [0, {MAX_INSPECTION_POINTS}]")
    return number


def normalize_summary(event: Mapping[str, Any]) -> dict[str, int]:
    attempted = _count(event.get("attempted"), "attempted")
    completed = _count(event.get("completed"), "completed")
    failed = _count(event.get("failed"), "failed")
    normal = _count(event.get("normal"), "normal")
    abnormal = _count(event.get("abnormal"), "abnormal")
    has_failure_breakdown = any(
        name in event
        for name in ("heading_failed", "vision_failed", "runtime_failed")
    )
    heading_failed = _count(event.get("heading_failed", 0), "heading_failed")
    vision_failed = _count(
        event.get("vision_failed", failed if not has_failure_breakdown else 0),
        "vision_failed",
    )
    runtime_failed = _count(event.get("runtime_failed", 0), "runtime_failed")
    if attempted != completed + failed:
        raise ValueError("attempted must equal completed plus failed")
    if completed != normal + abnormal:
        raise ValueError("completed must equal normal plus abnormal")
    if failed != heading_failed + vision_failed + runtime_failed:
        raise ValueError("failed must equal heading, vision, and runtime failures")
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "normal": normal,
        "abnormal": abnormal,
        "heading_failed": heading_failed,
        "vision_failed": vision_failed,
        "runtime_failed": runtime_failed,
    }


def _count_word(value: int) -> str:
    return {0: "零", 1: "一", 2: "两", 3: "三"}.get(value, str(value))


def _count_clause(value: int, nonzero_prefix: str, zero_text: str) -> str:
    number = int(value)
    if number == 0:
        return zero_text
    return "%s%s项" % (nonzero_prefix, _count_word(number))


def summary_text(summary: Mapping[str, Any]) -> str:
    counts = normalize_summary(summary)
    clauses = [
        _count_clause(counts["attempted"], "共检查", "没有执行巡检点检查"),
        _count_clause(counts["completed"], "成功完成", "没有成功完成的点位"),
    ]
    for key, prefix in (
        ("heading_failed", "观察方向失败"),
        ("vision_failed", "仪表检测失败"),
        ("runtime_failed", "系统异常"),
    ):
        number = counts[key]
        if number > 0:
            clauses.append("%s%s项" % (prefix, _count_word(number)))
    clauses.extend(
        [
            _count_clause(counts["normal"], "正常结果", "没有正常结果"),
            _count_clause(counts["abnormal"], "异常结果", "没有异常结果"),
        ]
    )
    return "巡检结束。%s。" % "。".join(clauses)


def summary_filename(summary: Mapping[str, Any]) -> str:
    counts = normalize_summary(summary)
    return (
        "inspection-summary-a{attempted}-c{completed}-f{failed}-"
        "h{heading_failed}-v{vision_failed}-r{runtime_failed}-"
        "n{normal}-x{abnormal}.wav"
    ).format(
        **counts
    )


def prompt_for_event(event: Mapping[str, Any]) -> Optional[EventPrompt]:
    if not isinstance(event, Mapping):
        return None
    event_name = str(event.get("event", ""))
    status = str(event.get("status", "")).upper()
    condition = str(event.get("condition", "")).upper()
    result = RESULT_PROMPTS.get((event_name, status, condition))
    if result is not None:
        if event.get("speech_key") not in (None, result.speech_key):
            return None
        waypoint = str(event.get("waypoint", "")).upper()
        return POINT_RESULT_PROMPTS.get((waypoint, result.speech_key), result)
    expected_status = NAVIGATION_EVENT_STATUS.get(event_name)
    if expected_status is not None:
        if status != expected_status:
            return None
        speech_key = str(event.get("speech_key", ""))
        return NAVIGATION_PROMPTS.get(speech_key)
    mapping_prompt = MAPPING_EVENT_PROMPTS.get((event_name, status))
    if mapping_prompt is not None:
        if event.get("speech_key") not in (None, mapping_prompt.speech_key):
            return None
        return mapping_prompt
    if event_name != "inspection_summary" or status != "PASS":
        return None
    try:
        text = summary_text(event)
        filename = summary_filename(event)
    except ValueError:
        return None
    if event.get("speech_key") not in (None, "inspection_summary"):
        return None
    return EventPrompt(
        text=text,
        filename=filename,
        speech_key="inspection_summary",
    )


def speech_ack_path(event_path: str | Path) -> Path:
    return Path(str(event_path) + ".ack")


def write_playback_ack(
    event_path: str | Path,
    event: Mapping[str, Any],
    *,
    played: bool,
    error: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[Path]:
    """Write one atomic ACK for events that explicitly request it."""

    if event.get("requires_ack") is not True:
        return None
    event_id = str(event.get("event_id", "")).strip()
    if not event_id:
        return None
    source_path = Path(event_path)
    ack_path = speech_ack_path(source_path)
    payload = {
        "event_id": event_id,
        "event": event.get("event"),
        "status": "PASS" if played else "FAIL",
        "played": bool(played),
        "timestamp": time.time() if now is None else float(now),
        "error": None if played else (error or "prompt playback failed"),
    }
    ack_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=ack_path.parent,
        prefix=".speech-ack-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        temp_path = Path(stream.name)
    os.replace(temp_path, ack_path)
    return ack_path
