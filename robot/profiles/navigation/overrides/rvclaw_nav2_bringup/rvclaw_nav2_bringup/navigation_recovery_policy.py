"""Pure, fail-closed policy for bounded waypoint-navigation recovery."""

from __future__ import annotations

from typing import Any, Mapping


PROMPTS = {
    "navigation_obstacle_recovery_started": ("路线受阻，正在等待障碍物移开并重新规划。", "navigation-obstacle-recovery-started.wav"),
    "navigation_localization_recovery_started": ("定位不稳定，正在自动重新定位。", "navigation-localization-recovery-started.wav"),
    "navigation_lidar_recovery_started": ("雷达数据短暂异常，正在等待雷达恢复。", "navigation-lidar-recovery-started.wav"),
    "navigation_recovery_succeeded": ("导航已恢复，继续巡检。", "navigation-recovery-succeeded.wav"),
    "navigation_paused_obstacle": ("导航仍然受阻，巡检已暂停。请移开小车周围或通往目标点的障碍物，然后唤醒我，说继续巡检。", "navigation-paused-obstacle.wav"),
    "navigation_paused_localization": ("定位失败，巡检已暂停。请把小车放到当前地图内特征明显且周围空旷的位置，然后唤醒我，说重新定位并继续巡检。", "navigation-paused-localization.wav"),
    "navigation_paused_lidar": ("雷达数据未恢复，巡检已暂停。请检查雷达电源和USB连接，恢复后唤醒我，说继续巡检。", "navigation-paused-lidar.wav"),
    "navigation_paused_chassis": ("底盘通信异常，巡检已暂停。请检查小车总电源、串口三和通信线，恢复后唤醒我，说继续巡检。", "navigation-paused-chassis.wav"),
    "navigation_paused_safety": ("安全保护已触发，巡检已暂停。请清除周围危险并检查急停，确认安全后人工解除，再唤醒我，说继续巡检。", "navigation-paused-safety.wav"),
    "navigation_paused_low_battery": ("小车电量过低，巡检已暂停。请关闭运动电源并更换或充电，恢复后唤醒我，说继续巡检。", "navigation-paused-low-battery.wav"),
    "navigation_paused_system": ("导航系统异常，巡检已暂停。请保持小车不动并检查导航服务，恢复后唤醒我，说继续巡检。", "navigation-paused-system.wav"),
}


def _policy(code, automatic, attempts, delay, started, paused, action, command="继续巡检", relocalize=False):
    return dict(code=code, automatic=automatic, max_attempts=attempts, retry_delay_s=delay,
                started_speech_key=started, paused_speech_key=paused,
                operator_action=action, resume_command=command,
                requires_global_localization=relocalize)


POLICIES = {
    "ROUTE_BLOCKED": _policy("ROUTE_BLOCKED", True, 2, 3.0, "navigation_obstacle_recovery_started", "navigation_paused_obstacle", "clear obstacles without moving the robot"),
    "LOCALIZATION_UNSTABLE": _policy("LOCALIZATION_UNSTABLE", True, 1, 2.0, "navigation_localization_recovery_started", "navigation_paused_localization", "move to a feature-rich open area on the current map", "重新定位并继续巡检", True),
    "LIDAR_DATA_TRANSIENT": _policy("LIDAR_DATA_TRANSIENT", True, 1, 2.0, "navigation_lidar_recovery_started", "navigation_paused_lidar", "check lidar power and USB without moving the robot"),
    "CHASSIS_COMMAND_TIMEOUT": _policy("CHASSIS_COMMAND_TIMEOUT", True, 1, 1.0, "navigation_obstacle_recovery_started", "navigation_paused_chassis", "keep the robot still while the controller is safely reset"),
    "CHASSIS_DEADMAN_TIMEOUT": _policy("CHASSIS_DEADMAN_TIMEOUT", True, 1, 1.0, "navigation_obstacle_recovery_started", "navigation_paused_chassis", "verify the controller stopped before one bounded retry"),
    "LOW_BATTERY": _policy("LOW_BATTERY", False, 0, 0.0, None, "navigation_paused_low_battery", "power off motion and charge or replace the chassis battery"),
    "CHASSIS_COMMUNICATION_FAILED": _policy("CHASSIS_COMMUNICATION_FAILED", False, 0, 0.0, None, "navigation_paused_chassis", "check chassis power, UART3 and the STM32 cable"),
    "SAFETY_STOP_LATCHED": _policy("SAFETY_STOP_LATCHED", False, 0, 0.0, None, "navigation_paused_safety", "clear the hazard and use the manual safety-reset procedure"),
    "NAVIGATION_SERVICE_UNAVAILABLE": _policy("NAVIGATION_SERVICE_UNAVAILABLE", False, 0, 0.0, None, "navigation_paused_system", "keep the robot still and restore the Nav2 runtime"),
    "UNCLASSIFIED_NAVIGATION_FAILURE": _policy("UNCLASSIFIED_NAVIGATION_FAILURE", False, 0, 0.0, None, "navigation_paused_system", "keep the robot still and inspect the route trace"),
}


def _failure_text(result: Mapping[str, Any]) -> str:
    fields = (
        result.get("error"),
        result.get("error_repr"),
        result.get("traceback"),
        result.get("path_check_error"),
        result.get("transition_reason"),
        result.get("final_transition_reason"),
        result.get("final_state"),
    )
    values = [str(value) for value in fields if value is not None]
    events = result.get("events")
    if isinstance(events, (list, tuple)):
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if event.get("name") in {"state", "transition_reason"}:
                values.append(str(event.get("value", "")))
    return " ".join(values).lower()


def _reported_voltage(result: Mapping[str, Any]):
    for container in (result, result.get("metrics")):
        if not isinstance(container, Mapping):
            continue
        for key in ("voltage_v", "battery_voltage_v"):
            try:
                return float(container[key])
            except (KeyError, TypeError, ValueError):
                pass
    return None


def classify_navigation_failure(result: Mapping[str, Any]) -> dict[str, Any]:
    text = _failure_text(result)
    voltage = _reported_voltage(result)
    if "low battery" in text or "voltage below" in text or (
        voltage is not None and voltage < 9.0
    ):
        code = "LOW_BATTERY"
    elif "watchdog_latched" in text and "command_timeout" in text:
        code = "CHASSIS_COMMAND_TIMEOUT"
    elif "deadman_timeout" in text and "estop_latched" not in text:
        code = "CHASSIS_DEADMAN_TIMEOUT"
    elif any(x in text for x in ("estop_latched", "watchdog_latched", "safety stop", "unexpectedly started armed", "did not finish disarmed")):
        code = "SAFETY_STOP_LATCHED"
    elif any(x in text for x in ("hardware transport", "stm32", "serial", "fresh feedback", "feedback_connected")):
        code = "CHASSIS_COMMUNICATION_FAILED"
    elif any(x in text for x in ("static-map localization", "continuous route localization", "amcl", "localization", "pose jump")):
        code = "LOCALIZATION_UNSTABLE"
    elif any(x in text for x in ("fresh scan", "scan was stale", "lidar", "laser")):
        code = "LIDAR_DATA_TRANSIENT"
    elif any(x in text for x in ("action server is unavailable", "goal was rejected", "service is unavailable", "service unavailable")):
        code = "NAVIGATION_SERVICE_UNAVAILABLE"
    elif any(x in text for x in ("computepathtopose", "no valid path", "path_check", "empty or one-pose path", "start clearance", "goal cell cost", "lethal", "obstacle", "collision monitor", "navigatetopose did not report success", "failed to make progress", "progress checker", "action result timed out")):
        code = "ROUTE_BLOCKED"
    else:
        code = "UNCLASSIFIED_NAVIGATION_FAILURE"
    policy = dict(POLICIES[code])
    for field in ("started_speech_key", "paused_speech_key"):
        key = policy.get(field)
        policy[field.replace("key", "text")] = None if key is None else PROMPTS[key][0]
    return policy


def retry_is_safe(result: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
    if decision.get("automatic") is not True or result.get("final_armed") is not False:
        return False
    state = str(result.get("final_state", "")).upper()
    if decision.get("code") in {"CHASSIS_COMMAND_TIMEOUT", "CHASSIS_DEADMAN_TIMEOUT"}:
        text = _failure_text(result)
        command_timeout = decision.get("code") == "CHASSIS_COMMAND_TIMEOUT"
        cause = "command_timeout" if command_timeout else "deadman_timeout"
        return (
            state in {"DISARMED", "WATCHDOG_LATCHED"}
            and cause in text
            and (not command_timeout or "watchdog_latched" in text)
            and "estop_latched" not in text
        )
    return state == "DISARMED"
