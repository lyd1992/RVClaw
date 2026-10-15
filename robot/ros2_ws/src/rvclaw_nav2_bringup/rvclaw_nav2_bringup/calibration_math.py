"""Pure odometry calibration calculations with no ROS or hardware I/O."""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Iterable, Optional


class CalibrationError(ValueError):
    """Raised when a calibration trace cannot produce a defensible result."""


@dataclass(frozen=True)
class OdomSample:
    time_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    linear_x_m_s: float
    angular_z_rad_s: float


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise CalibrationError(f"{name} must be finite")
    return value


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw after validating and normalizing a quaternion."""
    values = [_finite(value, "quaternion component") for value in (x, y, z, w)]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-9:
        raise CalibrationError("quaternion norm is zero")
    x, y, z, w = (value / norm for value in values)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def normalized_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def summarize_samples(samples: Iterable[OdomSample], motion: str) -> Dict[str, object]:
    """Summarize displacement in the starting base frame and unwrapped yaw."""
    samples = list(samples)
    if motion not in {"straight", "turn"}:
        raise CalibrationError("motion must be 'straight' or 'turn'")
    if len(samples) < 2:
        raise CalibrationError("at least two odometry samples are required")
    for sample in samples:
        for name, value in asdict(sample).items():
            _finite(value, name)

    start = samples[0]
    end = samples[-1]
    duration_s = end.time_s - start.time_s
    if duration_s <= 0.0:
        raise CalibrationError("sample timestamps must increase")

    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    cos_start = math.cos(start.yaw_rad)
    sin_start = math.sin(start.yaw_rad)
    forward_m = cos_start * dx + sin_start * dy
    lateral_m = -sin_start * dx + cos_start * dy

    path_m = 0.0
    yaw_change_rad = 0.0
    previous = start
    for current in samples[1:]:
        if current.time_s <= previous.time_s:
            raise CalibrationError("sample timestamps must increase")
        path_m += math.hypot(current.x_m - previous.x_m, current.y_m - previous.y_m)
        yaw_change_rad += normalized_angle(current.yaw_rad - previous.yaw_rad)
        previous = current

    return {
        "motion": motion,
        "sample_count": len(samples),
        "duration_s": duration_s,
        "forward_displacement_m": forward_m,
        "lateral_displacement_m": lateral_m,
        "path_distance_m": path_m,
        "yaw_change_rad": yaw_change_rad,
        "yaw_change_deg": math.degrees(yaw_change_rad),
        "peak_abs_linear_x_m_s": max(abs(item.linear_x_m_s) for item in samples),
        "peak_abs_angular_z_rad_s": max(
            abs(item.angular_z_rad_s) for item in samples
        ),
        "start": asdict(start),
        "end": asdict(end),
    }


def summarize_dominant_turn(
    samples: Iterable[OdomSample],
    angular_threshold_rad_s: float = 0.05,
    max_inactive_gap_s: float = 3.0,
    padding_s: float = 0.25,
) -> Dict[str, object]:
    """Summarize the largest separated turn while preserving the full trace."""
    samples = list(samples)
    summarize_samples(samples, "turn")
    threshold = _finite(angular_threshold_rad_s, "angular_threshold_rad_s")
    max_gap = _finite(max_inactive_gap_s, "max_inactive_gap_s")
    padding = _finite(padding_s, "padding_s")
    if threshold <= 0.0:
        raise CalibrationError("angular_threshold_rad_s must be greater than zero")
    if max_gap <= 0.0:
        raise CalibrationError("max_inactive_gap_s must be greater than zero")
    if padding < 0.0:
        raise CalibrationError("padding_s must not be negative")

    active = [
        index
        for index, sample in enumerate(samples)
        if abs(sample.angular_z_rad_s) >= threshold
    ]
    if not active:
        raise CalibrationError("turn trace has no active angular-velocity segment")

    groups = []
    first = active[0]
    previous = first
    for index in active[1:]:
        if samples[index].time_s - samples[previous].time_s > max_gap:
            groups.append((first, previous))
            first = index
        previous = index
    groups.append((first, previous))

    candidates = []
    trace_start_s = samples[0].time_s
    for first_active, last_active in groups:
        start_time = samples[first_active].time_s - padding
        end_time = samples[last_active].time_s + padding
        start = first_active
        end = last_active
        while start > 0 and samples[start - 1].time_s >= start_time:
            start -= 1
        while end + 1 < len(samples) and samples[end + 1].time_s <= end_time:
            end += 1
        summary = summarize_samples(samples[start : end + 1], "turn")
        summary.update(
            {
                "trace_start_offset_s": samples[start].time_s - trace_start_s,
                "trace_end_offset_s": samples[end].time_s - trace_start_s,
                "active_sample_count": last_active - first_active + 1,
            }
        )
        candidates.append(summary)

    selected = max(candidates, key=lambda item: abs(float(item["yaw_change_rad"])))
    selected.update(
        {
            "selection": "largest_separated_turn",
            "detected_segment_count": len(candidates),
            "angular_threshold_rad_s": threshold,
            "max_inactive_gap_s": max_gap,
            "padding_s": padding,
        }
    )
    return selected


def calibration_recommendation(
    summary: Dict[str, object],
    actual_distance_m: Optional[float] = None,
    actual_angle_deg: Optional[float] = None,
) -> Dict[str, object]:
    """Calculate a multiplier; the result remains review-only until repeated."""
    motion = summary.get("motion")
    if motion == "straight":
        if actual_angle_deg is not None or actual_distance_m is None:
            raise CalibrationError("straight motion requires only actual_distance_m")
        actual = abs(_finite(actual_distance_m, "actual_distance_m"))
        observed = abs(float(summary["forward_displacement_m"]))
        parameter = "odom_x_scale"
        direction = "forward" if float(summary["forward_displacement_m"]) >= 0 else "reverse"
    elif motion == "turn":
        if actual_distance_m is not None or actual_angle_deg is None:
            raise CalibrationError("turn motion requires only actual_angle_deg")
        actual = abs(math.radians(_finite(actual_angle_deg, "actual_angle_deg")))
        observed_signed = float(summary["yaw_change_rad"])
        observed = abs(observed_signed)
        parameter = (
            "odom_z_scale_positive"
            if observed_signed >= 0.0
            else "odom_z_scale_negative"
        )
        direction = "left" if observed_signed >= 0.0 else "right"
    else:
        raise CalibrationError("summary has an unsupported motion value")

    if actual <= 0.0:
        raise CalibrationError("actual measurement must be greater than zero")
    if observed < 1e-4:
        raise CalibrationError("observed odometry motion is too small")

    multiplier = actual / observed
    return {
        "review_required": True,
        "direction": direction,
        "parameter": parameter,
        "current_value_multiplier": multiplier,
        "observed_magnitude": observed,
        "actual_magnitude": actual,
        "repeat_runs_required": 3,
    }
