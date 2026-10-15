"""Hardware-independent safety state machine for chassis commands."""

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple


ZERO_VELOCITY = (0.0, 0.0, 0.0)
VALID_SOURCES = ("teleop", "autonomy")


@dataclass(frozen=True)
class CommandSample:
    velocity_x_m_s: float
    velocity_y_m_s: float
    angular_z_rad_s: float
    received_monotonic: float


@dataclass(frozen=True)
class ControlDecision:
    velocity_x_m_s: float
    velocity_y_m_s: float
    angular_z_rad_s: float
    state: str
    source: str
    transition_reason: str
    watchdog_active: bool
    saturated: bool
    slew_limited: bool


class SafetySupervisor:
    """Arbitrate commands behind explicit arm, deadman, and estop gates."""

    def __init__(
        self,
        command_timeout_s: float = 0.25,
        deadman_timeout_s: float = 0.15,
        max_linear_x_m_s: float = 0.18,
        max_angular_z_rad_s: float = 0.30,
        max_linear_accel_m_s2: float = 0.20,
        max_angular_accel_rad_s2: float = 0.60,
        require_feedback_connected: bool = False,
    ) -> None:
        positive_values = (
            command_timeout_s,
            deadman_timeout_s,
            max_linear_x_m_s,
            max_angular_z_rad_s,
            max_linear_accel_m_s2,
            max_angular_accel_rad_s2,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_values):
            raise ValueError("timeouts and limits must be finite and positive")

        self.command_timeout_s = command_timeout_s
        self.deadman_timeout_s = deadman_timeout_s
        self.max_linear_x_m_s = max_linear_x_m_s
        self.max_angular_z_rad_s = max_angular_z_rad_s
        self.max_linear_accel_m_s2 = max_linear_accel_m_s2
        self.max_angular_accel_rad_s2 = max_angular_accel_rad_s2
        self.require_feedback_connected = require_feedback_connected

        self.armed = False
        self.estop_latched = True
        self.watchdog_latched = False
        self.armed_monotonic: Optional[float] = None
        self.feedback_connected = False
        self.emergency_stop_asserted = False
        self.deadman_asserted = False
        self.deadman_monotonic: Optional[float] = None
        self.commands: Dict[str, Optional[CommandSample]] = {
            source: None for source in VALID_SOURCES
        }
        self.last_selected_source = ""
        self.autonomy_valid_after = 0.0
        self.last_output = ZERO_VELOCITY
        self.last_tick_monotonic: Optional[float] = None
        self.transition_reason = "startup_estop"

    def _clear_commands(self) -> None:
        for source in VALID_SOURCES:
            self.commands[source] = None
        self.last_selected_source = ""
        self.autonomy_valid_after = 0.0

    def _disarm(self, reason: str) -> None:
        self.armed = False
        self.armed_monotonic = None
        self.last_output = ZERO_VELOCITY
        self._clear_commands()
        self.transition_reason = reason

    def deadman_fresh(self, now_monotonic: float) -> bool:
        return (
            self.deadman_asserted
            and self.deadman_monotonic is not None
            and 0.0 <= now_monotonic - self.deadman_monotonic
            <= self.deadman_timeout_s
        )

    def update_deadman(self, asserted: bool, now_monotonic: float) -> None:
        if asserted:
            self.deadman_asserted = True
            self.deadman_monotonic = now_monotonic
            return
        self.deadman_asserted = False
        self.deadman_monotonic = None
        if self.armed:
            self._disarm("deadman_released")

    def update_feedback_connected(self, connected: bool) -> None:
        self.feedback_connected = connected
        if self.require_feedback_connected and not connected and self.armed:
            self.latch_estop("feedback_disconnected")

    def update_emergency_stop(self, asserted: bool) -> None:
        self.emergency_stop_asserted = asserted
        if asserted:
            self.latch_estop("emergency_stop_topic")

    def latch_estop(self, reason: str) -> None:
        self.estop_latched = True
        self._disarm(reason)

    def latch_watchdog(self, reason: str) -> None:
        self.watchdog_latched = True
        self._disarm(reason)

    def reset_latches(self) -> Tuple[bool, str]:
        if self.armed:
            return False, "cannot reset safety latches while armed"
        if self.emergency_stop_asserted:
            return False, "cannot reset while emergency stop is asserted"
        self.estop_latched = False
        self.watchdog_latched = False
        self._clear_commands()
        self.last_output = ZERO_VELOCITY
        self.transition_reason = "safety_latches_reset"
        return True, "safety latches reset; controller remains disarmed"

    def reset_estop(self) -> Tuple[bool, str]:
        return self.reset_latches()

    def set_armed(self, armed: bool, now_monotonic: float) -> Tuple[bool, str]:
        if not armed:
            self._disarm("operator_disarm")
            return True, "controller disarmed"
        if self.estop_latched:
            return False, "estop is latched"
        if self.watchdog_latched:
            return False, "watchdog is latched"
        if not self.deadman_fresh(now_monotonic):
            return False, "fresh asserted deadman is required before arming"
        if self.require_feedback_connected and not self.feedback_connected:
            return False, "fresh STM32 feedback connection is required"

        self._clear_commands()
        self.last_output = ZERO_VELOCITY
        self.last_tick_monotonic = now_monotonic
        self.armed = True
        self.armed_monotonic = now_monotonic
        self.transition_reason = "operator_arm"
        return True, "controller armed; a new command is required"

    def submit_command(
        self,
        source: str,
        velocity_x_m_s: float,
        velocity_y_m_s: float,
        angular_z_rad_s: float,
        now_monotonic: float,
    ) -> bool:
        if source not in VALID_SOURCES:
            raise ValueError("unknown command source")
        values = (velocity_x_m_s, velocity_y_m_s, angular_z_rad_s, now_monotonic)
        if any(not math.isfinite(value) for value in values):
            self.commands[source] = None
            self.latch_estop("nonfinite_%s_command" % source)
            return False
        if not self.armed:
            return False

        self.commands[source] = CommandSample(
            velocity_x_m_s=velocity_x_m_s,
            velocity_y_m_s=velocity_y_m_s,
            angular_z_rad_s=angular_z_rad_s,
            received_monotonic=now_monotonic,
        )
        return True

    def _sample_fresh(
        self,
        sample: Optional[CommandSample],
        now_monotonic: float,
    ) -> bool:
        return (
            sample is not None
            and 0.0 <= now_monotonic - sample.received_monotonic
            <= self.command_timeout_s
        )

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _approach(current: float, target: float, maximum_delta: float) -> float:
        delta = target - current
        if delta > maximum_delta:
            return current + maximum_delta
        if delta < -maximum_delta:
            return current - maximum_delta
        return target

    def _safe_decision(
        self,
        state: str,
        watchdog_active: bool = False,
    ) -> ControlDecision:
        self.last_output = ZERO_VELOCITY
        return ControlDecision(
            velocity_x_m_s=0.0,
            velocity_y_m_s=0.0,
            angular_z_rad_s=0.0,
            state=state,
            source="none",
            transition_reason=self.transition_reason,
            watchdog_active=watchdog_active,
            saturated=False,
            slew_limited=False,
        )

    def evaluate(self, now_monotonic: float) -> ControlDecision:
        if not math.isfinite(now_monotonic):
            self.latch_estop("nonfinite_clock")
            return self._safe_decision("ESTOP_LATCHED")

        previous_tick = self.last_tick_monotonic
        self.last_tick_monotonic = now_monotonic

        if self.estop_latched:
            return self._safe_decision("ESTOP_LATCHED")
        if self.watchdog_latched:
            return self._safe_decision("WATCHDOG_LATCHED", watchdog_active=True)
        if not self.armed:
            return self._safe_decision("DISARMED")
        if not self.deadman_fresh(now_monotonic):
            self.deadman_asserted = False
            self.deadman_monotonic = None
            self._disarm("deadman_timeout")
            return self._safe_decision("DISARMED")
        if self.require_feedback_connected and not self.feedback_connected:
            self.latch_estop("feedback_disconnected")
            return self._safe_decision("ESTOP_LATCHED")

        teleop = self.commands["teleop"]
        autonomy = self.commands["autonomy"]
        teleop_fresh = self._sample_fresh(teleop, now_monotonic)

        if self.last_selected_source == "teleop" and not teleop_fresh:
            self._disarm("teleop_released")
            return self._safe_decision("DISARMED")

        source = ""
        sample: Optional[CommandSample] = None
        if teleop_fresh:
            source = "teleop"
            sample = teleop
        elif (
            self._sample_fresh(autonomy, now_monotonic)
            and autonomy is not None
            and autonomy.received_monotonic > self.autonomy_valid_after
        ):
            source = "autonomy"
            sample = autonomy

        if sample is None:
            waiting_since = (
                now_monotonic
                if self.armed_monotonic is None
                else self.armed_monotonic
            )
            if (
                self.last_selected_source
                or now_monotonic - waiting_since > self.command_timeout_s
            ):
                self.latch_watchdog("command_timeout")
                return self._safe_decision(
                    "WATCHDOG_LATCHED",
                    watchdog_active=True,
                )
            return self._safe_decision("ARMED_IDLE")

        raw_target = (
            sample.velocity_x_m_s,
            sample.velocity_y_m_s,
            sample.angular_z_rad_s,
        )
        target = (
            self._clamp(raw_target[0], self.max_linear_x_m_s),
            0.0,
            self._clamp(raw_target[2], self.max_angular_z_rad_s),
        )
        saturated = target != raw_target

        dt = 0.0 if previous_tick is None else now_monotonic - previous_tick
        if not math.isfinite(dt) or dt <= 0.0:
            dt = 0.0
        max_linear_delta = self.max_linear_accel_m_s2 * dt
        max_angular_delta = self.max_angular_accel_rad_s2 * dt
        output = (
            self._approach(self.last_output[0], target[0], max_linear_delta),
            0.0,
            self._approach(self.last_output[2], target[2], max_angular_delta),
        )
        slew_limited = output != target
        self.last_output = output
        self.last_selected_source = source
        self.transition_reason = "active_%s" % source

        return ControlDecision(
            velocity_x_m_s=output[0],
            velocity_y_m_s=0.0,
            angular_z_rad_s=output[2],
            state="ACTIVE",
            source=source,
            transition_reason=self.transition_reason,
            watchdog_active=False,
            saturated=saturated,
            slew_limited=slew_limited,
        )
