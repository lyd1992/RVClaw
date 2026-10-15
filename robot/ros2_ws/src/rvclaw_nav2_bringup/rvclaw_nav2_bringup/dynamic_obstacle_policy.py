"""Fail-closed recovery policy for obstacles discovered during navigation.

This module contains no ROS or hardware I/O.  It is the reviewable decision
contract used before a future runtime recovery supervisor is enabled.  A
recovery action may request replanning or a bounded stationary scan, but only
the normal Nav2 and STM32 safety chain may publish motion commands.
"""

from dataclasses import dataclass
import json
import math
from typing import Optional


POLICY_VERSION = "2026-09-02.1"

ACTION_PROCEED = "PROCEED"
ACTION_SLOW_AND_REPLAN = "SLOW_AND_REPLAN"
ACTION_STOP_WAIT_REPLAN = "STOP_WAIT_REPLAN"
ACTION_REPLAN_ALTERNATE = "REPLAN_ALTERNATE"
ACTION_SPIN_SEARCH = "SPIN_SEARCH_THEN_REPLAN"
ACTION_HOLD_STOP = "HOLD_STOP"
ACTION_ABORT_GOAL = "ABORT_GOAL"

BODY_HALF_WIDTH_M = 0.12725
SIDE_SAFETY_MARGIN_M = 0.05
MIN_SPIN_CLEARANCE_M = 0.25
MIN_SPIN_COVERAGE_DEG = 330.0
SPIN_CHUNK_MAX_RAD = 0.45
SPIN_TOTAL_RAD = 2.0 * math.pi
MAX_SPIN_ATTEMPTS = 1
MAX_REPLAN_ATTEMPTS = 3
MAX_WAIT_ATTEMPTS = 2
WAIT_RECHECK_S = 1.0


@dataclass(frozen=True)
class RecoveryContext:
    """Snapshot of independent evidence at a dynamic-obstacle event.

    ``left_side_gap_m`` and ``right_side_gap_m`` are physical gaps from the
    chassis side to the nearest return.  ``None`` means that side evidence is
    not being used by this decision; freshness and costmap gates still apply.
    """

    obstacle_detected: bool = False
    obstacle_persistent: bool = False
    front_stop_active: bool = False
    path_available: bool = True
    alternate_path_available: bool = False
    scan_fresh: bool = True
    costmap_fresh: bool = True
    localization_stable: bool = True
    motion_feedback_fresh: bool = True
    progress_stalled: bool = False
    robot_stopped: bool = True
    spin_clearance_pass: bool = False
    spin_clearance_m: Optional[float] = None
    spin_coverage_deg: Optional[float] = None
    spin_attempts: int = 0
    replan_attempts: int = 0
    wait_attempts: int = 0
    left_side_gap_m: Optional[float] = None
    right_side_gap_m: Optional[float] = None


def _validate_context(context: RecoveryContext) -> None:
    if not isinstance(context, RecoveryContext):
        raise TypeError("context must be a RecoveryContext")
    for name in ("spin_attempts", "replan_attempts", "wait_attempts"):
        value = getattr(context, name)
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError("%s must be a nonnegative integer" % name)
    for name in (
        "spin_clearance_m",
        "spin_coverage_deg",
        "left_side_gap_m",
        "right_side_gap_m",
    ):
        value = getattr(context, name)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("%s must be finite and nonnegative" % name)


def _side_gap_failure(context: RecoveryContext):
    minimum = SIDE_SAFETY_MARGIN_M
    for side in ("left", "right"):
        gap = getattr(context, "%s_side_gap_m" % side)
        if gap is not None and gap < minimum:
            return (
                "%s side gap %.3f m is inside the %.3f m swept-footprint "
                "margin" % (side, gap, minimum)
            )
    return None


def _spin_is_admitted(context: RecoveryContext) -> bool:
    if not context.robot_stopped or context.spin_attempts >= MAX_SPIN_ATTEMPTS:
        return False
    if not context.spin_clearance_pass:
        return False
    if context.spin_clearance_m is not None and context.spin_clearance_m < MIN_SPIN_CLEARANCE_M:
        return False
    if context.spin_coverage_deg is not None and context.spin_coverage_deg < MIN_SPIN_COVERAGE_DEG:
        return False
    return True


def _evidence_failure(context: RecoveryContext):
    """Return the first invalid recovery input, or ``None`` when usable.

    This deliberately combines independent inputs.  A current scan alone does
    not prove that a costmap has incorporated the obstacle, and a current
    costmap alone cannot prove that the base is still responding to Nav2.
    """
    if not context.scan_fresh:
        return ACTION_HOLD_STOP, "laser scan is stale during recovery", {
            "requires_fresh_scan": True,
        }
    if not context.costmap_fresh:
        return ACTION_HOLD_STOP, "costmap is stale during recovery", {
            "requires_fresh_costmap": True,
        }
    if not context.localization_stable:
        return ACTION_HOLD_STOP, "localization is unstable during recovery", {
            "requires_relocalization": True,
        }
    if not context.motion_feedback_fresh:
        return ACTION_HOLD_STOP, "base motion feedback is stale during recovery", {
            "requires_fresh_motion_feedback": True,
        }
    return None


def decide_recovery(context: RecoveryContext):
    """Choose the next fail-closed action from one obstacle snapshot.

    The result is a command *policy*, not a velocity command.  Every action
    except ``PROCEED`` requires zeroing/cancelling the active goal before a
    new plan is considered.  A spin is never a blind escape: it is allowed
    once, only while stopped, and must be followed by a fresh scan and a new
    path computation.
    """

    _validate_context(context)
    side_failure = _side_gap_failure(context)
    if side_failure:
        return _result(
            ACTION_HOLD_STOP,
            side_failure,
            terminal=False,
            requires_zero_velocity=True,
            requires_fresh_scan=True,
        )

    # A lack of progress is handled as a recovery event even when the laser
    # has not produced a new obstacle return.  This covers a blocked wheel,
    # a low object missed by the scanner, or a transient costmap mismatch;
    # none authorizes another open-loop drive attempt.
    recovery_active = (
        context.obstacle_detected
        or not context.path_available
        or context.progress_stalled
    )
    if recovery_active:
        evidence_failure = _evidence_failure(context)
        if evidence_failure:
            action, reason, fields = evidence_failure
            return _result(
                action,
                reason,
                requires_zero_velocity=True,
                **fields,
            )

    if context.progress_stalled and not context.obstacle_detected:
        if context.replan_attempts < MAX_REPLAN_ATTEMPTS:
            return _result(
                ACTION_SLOW_AND_REPLAN,
                "base made insufficient progress; stop and request a bounded replan",
                requires_zero_velocity=True,
                requires_cancel_goal=True,
            )
        return _result(
            ACTION_ABORT_GOAL,
            "base remained stalled after bounded replans",
            terminal=True,
            requires_zero_velocity=True,
            requires_cancel_goal=True,
        )

    if not context.obstacle_detected:
        if context.path_available:
            return _result(ACTION_PROCEED, "no obstacle in the current corridor")
        if context.replan_attempts < MAX_REPLAN_ATTEMPTS:
            return _result(
                ACTION_SLOW_AND_REPLAN,
                "path is unavailable; request a bounded replan",
                requires_zero_velocity=True,
            )
        return _result(
            ACTION_ABORT_GOAL,
            "path remains unavailable after bounded replans",
            terminal=True,
            requires_zero_velocity=True,
        )

    if not context.front_stop_active:
        if context.replan_attempts < MAX_REPLAN_ATTEMPTS:
            return _result(
                ACTION_SLOW_AND_REPLAN,
                "obstacle is detected outside the hard stop zone; slow and replan",
                requires_zero_velocity=True,
            )
        return _result(
            ACTION_ABORT_GOAL,
            "obstacle persists after bounded non-stop replans",
            terminal=True,
            requires_zero_velocity=True,
        )

    if context.alternate_path_available and context.replan_attempts < MAX_REPLAN_ATTEMPTS:
        return _result(
            ACTION_REPLAN_ALTERNATE,
            "front stop is active and a collision-free alternate path is available",
            requires_zero_velocity=True,
            requires_cancel_goal=True,
        )

    if not context.obstacle_persistent or context.wait_attempts < MAX_WAIT_ATTEMPTS:
        return _result(
            ACTION_STOP_WAIT_REPLAN,
            "front stop is active; wait at zero velocity and recheck the live map",
            wait_recheck_s=WAIT_RECHECK_S,
            requires_zero_velocity=True,
            requires_fresh_scan=True,
            requires_fresh_costmap=True,
        )

    if _spin_is_admitted(context):
        return _result(
            ACTION_SPIN_SEARCH,
            "persistent obstacle has no known alternate path; bounded clear spin may search",
            terminal=False,
            requires_zero_velocity=True,
            requires_cancel_goal=True,
            requires_fresh_scan=True,
            requires_fresh_costmap=True,
            requires_post_spin_replan=True,
            spin_total_yaw_rad=SPIN_TOTAL_RAD,
            spin_chunk_max_rad=SPIN_CHUNK_MAX_RAD,
        )

    return _result(
        ACTION_ABORT_GOAL,
        "persistent obstacle has no safe alternate or spin recovery",
        terminal=True,
        requires_zero_velocity=True,
        requires_cancel_goal=True,
    )


def _result(action, reason, **fields):
    result = {
        "policy_version": POLICY_VERSION,
        "action": action,
        "reason": reason,
        "motion_allowed": action == ACTION_PROCEED,
        "terminal": False,
        "requires_zero_velocity": action != ACTION_PROCEED,
        "requires_cancel_goal": False,
        "requires_fresh_scan": False,
        "requires_fresh_costmap": False,
        "requires_relocalization": False,
        "requires_fresh_motion_feedback": False,
        "requires_post_spin_replan": False,
    }
    result.update(fields)
    return result


def describe_plan():
    return {
        "policy_version": POLICY_VERSION,
        "hardware_io": False,
        "default_action_on_uncertainty": ACTION_HOLD_STOP,
        "obstacle_flow": [
            ACTION_PROCEED,
            ACTION_SLOW_AND_REPLAN,
            ACTION_REPLAN_ALTERNATE,
            ACTION_STOP_WAIT_REPLAN,
            ACTION_SPIN_SEARCH,
            ACTION_ABORT_GOAL,
        ],
        "spin_requires": [
            "robot_stopped",
            "fresh_scan",
            "fresh_costmap",
            "stable_localization",
            "full_swept_clearance",
        ],
        "spin_total_yaw_rad": SPIN_TOTAL_RAD,
        "spin_chunk_max_rad": SPIN_CHUNK_MAX_RAD,
        "minimum_spin_clearance_m": MIN_SPIN_CLEARANCE_M,
        "minimum_spin_coverage_deg": MIN_SPIN_COVERAGE_DEG,
        "max_spin_attempts": MAX_SPIN_ATTEMPTS,
        "max_replan_attempts": MAX_REPLAN_ATTEMPTS,
        "max_wait_attempts": MAX_WAIT_ATTEMPTS,
        "side_safety_margin_m": SIDE_SAFETY_MARGIN_M,
        "side_gap_is_physical_chassis_gap": True,
        "reverse_allowed": False,
        "lateral_allowed": False,
        "unknown_or_stale_evidence_policy": ACTION_HOLD_STOP,
        "progress_stall_policy": "stop_then_bounded_replan_then_abort",
        "requires_fresh_motion_feedback_during_recovery": True,
    }


def main() -> int:
    print(json.dumps(describe_plan(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
