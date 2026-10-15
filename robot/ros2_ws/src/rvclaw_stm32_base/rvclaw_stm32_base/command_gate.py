"""Timestamp gate for commands crossing the RVClaw arm boundary."""

from dataclasses import dataclass
import math
from typing import Dict, Optional

from .safety import VALID_SOURCES


NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class CommandStampDecision:
    accepted: bool
    reason: str
    fault: bool = False


class StampedCommandGate:
    """Reject pre-arm, stale, future, and replayed command timestamps."""

    def __init__(
        self,
        max_age_s: float = 0.10,
        max_future_skew_s: float = 0.02,
    ) -> None:
        if not math.isfinite(max_age_s) or max_age_s <= 0.0:
            raise ValueError("max command age must be finite and positive")
        if (
            not math.isfinite(max_future_skew_s)
            or max_future_skew_s < 0.0
        ):
            raise ValueError(
                "max future command skew must be finite and nonnegative"
            )

        self.max_age_ns = int(max_age_s * NANOSECONDS_PER_SECOND)
        self.max_future_skew_ns = int(
            max_future_skew_s * NANOSECONDS_PER_SECOND
        )
        self.arm_epoch_ns: Optional[int] = None
        self.last_accepted_stamp_ns: Dict[str, Optional[int]] = {
            source: None for source in VALID_SOURCES
        }

    @staticmethod
    def _valid_nanoseconds(value: int) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )

    def arm(self, epoch_ns: int) -> None:
        if not self._valid_nanoseconds(epoch_ns):
            raise ValueError("arm epoch must be nonnegative integer nanoseconds")
        self.arm_epoch_ns = epoch_ns
        for source in VALID_SOURCES:
            self.last_accepted_stamp_ns[source] = None

    def disarm(self) -> None:
        self.arm_epoch_ns = None
        for source in VALID_SOURCES:
            self.last_accepted_stamp_ns[source] = None

    def validate(
        self,
        source: str,
        stamp_ns: int,
        now_ns: int,
    ) -> CommandStampDecision:
        if source not in VALID_SOURCES:
            raise ValueError("unknown command source")
        if not self._valid_nanoseconds(stamp_ns) or not self._valid_nanoseconds(
            now_ns
        ):
            return CommandStampDecision(
                accepted=False,
                reason="invalid_%s_command_timestamp" % source,
                fault=True,
            )
        if self.arm_epoch_ns is None:
            return CommandStampDecision(
                accepted=False,
                reason="controller_disarmed",
            )
        if stamp_ns <= self.arm_epoch_ns:
            return CommandStampDecision(
                accepted=False,
                reason="prearm_%s_command_timestamp" % source,
            )

        age_ns = now_ns - stamp_ns
        if age_ns < -self.max_future_skew_ns:
            return CommandStampDecision(
                accepted=False,
                reason="future_%s_command_timestamp" % source,
                fault=True,
            )
        if age_ns > self.max_age_ns:
            return CommandStampDecision(
                accepted=False,
                reason="stale_%s_command_timestamp" % source,
            )

        last_stamp_ns = self.last_accepted_stamp_ns[source]
        if last_stamp_ns is not None and stamp_ns <= last_stamp_ns:
            return CommandStampDecision(
                accepted=False,
                reason="replayed_%s_command_timestamp" % source,
            )

        self.last_accepted_stamp_ns[source] = stamp_ns
        return CommandStampDecision(
            accepted=True,
            reason="accepted_%s_command_timestamp" % source,
        )
