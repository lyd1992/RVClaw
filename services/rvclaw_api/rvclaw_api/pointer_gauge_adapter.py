"""Fail-closed bridge from live inspection frames to the resident gauge model."""

import base64
import json
import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_NAME = "NanoDet + YOLOv8s-Pose (ncnn)"
MODEL_URL = "http://127.0.0.1:8018/api/read"
_INFERENCE_DEADLINE = ContextVar("pointer_gauge_inference_deadline", default=None)
_CROP_PAIRS = (
    ((0.25, 0.50, 0.75, 1.00), (0.30, 0.47, 0.80, 1.00)),
    ((0.25, 0.25, 0.75, 0.75), (0.30, 0.22, 0.80, 0.78)),
    ((0.00, 0.25, 0.55, 0.75), (0.05, 0.22, 0.60, 0.78)),
    ((0.45, 0.25, 1.00, 0.75), (0.40, 0.22, 0.95, 0.78)),
)


@contextmanager
def pointer_gauge_inference_budget(timeout_s):
    """Bound all retries together, including an enclosing live API request."""
    seconds = float(timeout_s)
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError("pointer gauge inference budget exhausted")
    deadline = time.monotonic() + seconds
    existing = _INFERENCE_DEADLINE.get()
    token = _INFERENCE_DEADLINE.set(min(existing, deadline) if existing is not None else deadline)
    try:
        yield
    finally:
        _INFERENCE_DEADLINE.reset(token)


def _refusal(reason):
    return {
        "kind": "pointer",
        "status": "unreadable",
        "value": None,
        "raw_text": None,
        "confidence": 0.0,
        "gauge_box": None,
        "reason": reason,
        "model": MODEL_NAME,
        "value_method": "ncnn_gauge_pose",
    }


def translate_gauge_reading(payload, image_size, calibration):
    """Reject unsupported, ambiguous, malformed, or low-confidence readings."""
    if not isinstance(payload, dict):
        return _refusal("gauge service returned a non-object response")
    if payload.get("status") != "readable":
        return _refusal(str(payload.get("status") or "model did not read the gauge"))
    if payload.get("gate_relaxed") or payload.get("n_meters") != 1:
        return _refusal("gauge result is ambiguous or bypassed its geometry gate")
    try:
        value = float(payload["value"])
        ratio = float(payload["ratio"])
        score = float(payload["meter_box"]["score"])
        x, y, box_width, box_height = map(float, payload["meter_box"]["rect"])
        source_width, source_height = map(float, payload["image_size"])
        infer_width, infer_height = map(float, payload["infer_size"])
        lower = float(calibration["min_value"])
        upper = float(calibration["max_value"])
        decimal_places = int(calibration.get("decimal_places", 2))
        width, height = image_size
        numbers = (value, ratio, score, x, y, box_width, box_height,
                   source_width, source_height, infer_width, infer_height, lower, upper)
        if not all(map(math.isfinite, numbers)):
            raise ValueError("nonfinite reading")
        if not (upper > lower and lower <= value <= upper and 0 <= ratio <= 1):
            raise ValueError("reading outside calibrated range")
        if abs(value - (lower + ratio * (upper - lower))) > 0.02:
            raise ValueError("value and ratio disagree")
        if not (0.65 <= score <= 1.0 and 0 <= decimal_places <= 6):
            raise ValueError("confidence or precision outside contract")
        if (abs(source_width - width) > 1 or abs(source_height - height) > 1
                or infer_width <= 0 or infer_height <= 0):
            raise ValueError("model input dimensions mismatch")
        x0 = x * width / infer_width
        y0 = y * height / infer_height
        x1 = (x + box_width) * width / infer_width
        y1 = (y + box_height) * height / infer_height
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("gauge box outside camera frame")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return _refusal("invalid gauge model evidence: %s" % exc)
    return {
        "kind": "pointer",
        "status": "stable",
        "value": value,
        "raw_text": f"{value:.{decimal_places}f}",
        "confidence": score,
        "unit": payload.get("unit", ""),
        "gauge_box": [round(x0), round(y0), round(x1), round(y1)],
        "reason": "",
        "model": MODEL_NAME,
        "value_method": "ncnn_gauge_pose",
        "notes": {
            "ratio": ratio,
            "record_id": payload.get("record_id"),
            "infer_step": payload.get("infer_step"),
            "model_elapsed_ms": payload.get("elapsed_ms"),
            "score_is_dial_detection_not_reading_accuracy": True,
        },
    }


def _post_frame(image, calibration):
    """Submit this frame to the resident model without persisting a result."""
    import cv2

    encoded_ok, jpeg = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not encoded_ok:
        raise ValueError("camera frame JPEG encoding failed")
    body = json.dumps({
        "image_b64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
        "range_min": calibration["min_value"],
        "range_max": calibration["max_value"],
        "unit": "°C",
    }).encode("utf-8")
    request = Request(
        os.getenv("RVCLAW_POINTER_GAUGE_URL", MODEL_URL),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = _INFERENCE_DEADLINE.get()
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("pointer gauge inference budget exhausted")
    with urlopen(request, timeout=5.0 if remaining is None else min(5.0, remaining)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("pointer gauge inference budget exhausted")
    return payload


def _crop_box(window, width, height):
    x0, y0, x1, y1 = window
    return (round(x0 * width), round(y0 * height),
            round(x1 * width), round(y1 * height))


def _intersection_over_union(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _validated_dial_locator(image, calibration, original):
    """Validate only location evidence; this never supplies a measured value."""
    height, width = image.shape[:2]
    try:
        hints = original["detected_meters"]
        if original.get("n_meters") != 1 or len(hints) != 1:
            raise ValueError("ambiguous dial locator")
        x, y, w, h = map(float, hints[0]["rect"])
        score = float(hints[0]["score"])
        iw, ih = map(float, original["infer_size"])
        sw, sh = map(float, original["image_size"])
        lower, upper = float(calibration["min_value"]), float(calibration["max_value"])
        decimals = int(calibration.get("decimal_places", 2))
        if not all(map(math.isfinite, (x, y, w, h, score, iw, ih, sw, sh, lower, upper))):
            raise ValueError("nonfinite dial locator")
        if (iw <= 0 or ih <= 0 or abs(sw - width) > 1 or abs(sh - height) > 1
                or not 0.3 <= score <= 1 or w <= 0 or h <= 0
                or upper <= lower or not 0 <= decimals <= 6):
            raise ValueError("invalid dial locator")
        x, y, w, h = x * width / iw, y * height / ih, w * width / iw, h * height / ih
        hint_box = [x, y, x + w, y + h]
        if not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
            raise ValueError("dial locator outside image")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return hint_box, (x, y, w, h), (lower, upper, decimals)


def _with_visible_dial(reading, image, calibration, original):
    """Keep a validated locator on refusal so the route stays stationary."""
    if reading.get("status") == "stable":
        return reading
    locator = _validated_dial_locator(image, calibration, original)
    if locator is None:
        return reading
    result = dict(reading)
    result["gauge_box"] = [round(v) for v in locator[0]]
    result["notes"] = {**result.get("notes", {}), "locator_only": True,
                       "original_full_frame_status": original.get("status")}
    return result


def _detected_context_crop_consensus(image, calibration, original):
    """Re-detect at bounded context sizes; two strict readings must agree."""
    locator = _validated_dial_locator(image, calibration, original)
    if locator is None:
        return _refusal("invalid or ambiguous gauge crop locator")
    hint_box, (x, y, w, h), (lower, upper, decimals) = locator
    height, width = image.shape[:2]

    cx, cy = x + w / 2, y + h / 2
    # Dimensions and alignment derive solely from the current detection.
    # Coarse alignment supplies a different detector sampling phase.
    grid = max(1, round(min(w, h) / 20))
    contexts = ((2.2, 2.2, 0, 0, 1), (2.25, 2.25, 0, 0, 1),
                (2.0, 2.0, 0, 0, 1), (2.25, 2.25, -0.05, 0, 1),
                (3.0, 2.0, 0, 0, grid), (3.0, 2.0, 0, -0.05, grid))
    accepted, visited = [], set()
    span = upper - lower
    for sx, sy, dx, dy, quantum in contexts:
        deadline = _INFERENCE_DEADLINE.get()
        if deadline is not None and time.monotonic() >= deadline:
            return _refusal("pointer gauge inference budget exhausted")
        xc, yc = cx + dx * w, cy + dy * h
        corners = (xc - sx * w / 2, yc - sy * h / 2,
                   xc + sx * w / 2, yc + sy * h / 2)
        a, b, c, d = (round(v / quantum) * quantum for v in corners)
        window = (max(0, a), max(0, b), min(width, c), min(height, d))
        if window in visited:
            continue
        visited.add(window)
        x0, y0, x1, y1 = window
        if x1 - x0 < 80 or y1 - y0 < 80:
            continue
        crop = image[y0:y1, x0:x1]
        try:
            payload = _post_frame(crop, calibration)
        except (HTTPError, URLError, OSError, TimeoutError, ValueError, KeyError):
            return _refusal("gauge model unavailable during crop verification")
        reading = translate_gauge_reading(payload, (x1 - x0, y1 - y0), calibration)
        if reading["status"] != "stable":
            continue
        local_box = reading["gauge_box"]
        box = [local_box[0] + x0, local_box[1] + y0,
               local_box[2] + x0, local_box[3] + y0]
        if _intersection_over_union(box, hint_box) < 0.5:
            return _refusal("context crop detected a different dial")
        if accepted:
            previous = accepted[0]
            if (abs(float(reading["value"]) - float(previous["reading"]["value"])) > 0.02 * span
                    or _intersection_over_union(box, previous["box"]) < 0.5):
                return _refusal("context crop readings disagree")
        accepted.append({"reading": reading, "box": box, "window": list(window)})
        if len(accepted) == 2:
            a, b = accepted
            value = (float(a["reading"]["value"]) + float(b["reading"]["value"])) / 2
            result = dict(a["reading"])
            result.update(value=value, raw_text=f"{value:.{decimals}f}",
                          confidence=min(a["reading"]["confidence"], b["reading"]["confidence"]),
                          gauge_box=a["box"])
            result["notes"] = {
                "ratio": (a["reading"]["notes"]["ratio"] + b["reading"]["notes"]["ratio"]) / 2,
                "fallback": "two_consistent_detected_context_crops",
                "original_full_frame_status": original.get("status"),
                "locator_box": hint_box,
                "crop_evidence": accepted,
                "score_is_dial_detection_not_reading_accuracy": True,
            }
            return result
    if accepted:
        return _refusal("only one context crop passed; independent crop verification missing")
    return None


def _aspect_gate_crop_consensus(image, calibration):
    """Retry only a rejected aspect-ratio gate, with two matching model crops.

    Never relax the model geometry gate or accept a single plausible-looking
    number.  Both crops must independently pass all normal reading checks,
    agree on the value and detect the same box in original image coordinates.
    The route executor still requires its usual consensus across live frames.
    """
    height, width = image.shape[:2]
    for first_window, second_window in _CROP_PAIRS:
        deadline = _INFERENCE_DEADLINE.get()
        if deadline is not None and time.monotonic() >= deadline:
            return None
        accepted = []
        for window in (first_window, second_window):
            x0, y0, x1, y1 = _crop_box(window, width, height)
            crop = image[y0:y1, x0:x1]
            if crop.shape[0] < 80 or crop.shape[1] < 80:
                break
            try:
                payload = _post_frame(crop, calibration)
            except (HTTPError, URLError, OSError, TimeoutError, ValueError, KeyError):
                break
            reading = translate_gauge_reading(
                payload, (crop.shape[1], crop.shape[0]), calibration
            )
            if reading["status"] != "stable":
                break
            box = reading["gauge_box"]
            accepted.append((
                reading,
                [box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0],
            ))
        if len(accepted) != 2:
            continue
        (a, box_a), (b, box_b) = accepted
        span = float(calibration["max_value"]) - float(calibration["min_value"])
        if abs(float(a["value"]) - float(b["value"])) > 0.02 * span:
            continue
        if _intersection_over_union(box_a, box_b) < 0.5:
            continue
        value = (float(a["value"]) + float(b["value"])) / 2.0
        confidence = min(float(a["confidence"]), float(b["confidence"]))
        decimals = int(calibration.get("decimal_places", 2))
        result = dict(a)
        result.update({
            "value": value,
            "raw_text": f"{value:.{decimals}f}",
            "confidence": confidence,
            "gauge_box": box_a,
            "notes": {
                "ratio": (a["notes"]["ratio"] + b["notes"]["ratio"]) / 2.0,
                "fallback": "two_consistent_overlapping_crops",
                "original_full_frame_status": "pointer_outside_scale",
                "score_is_dial_detection_not_reading_accuracy": True,
            },
        })
        return result
    return None


def _read_pointer_gauge_with_budget(image, calibration):
    """Infer afresh from this exact camera frame; never reuse a prior reading."""
    if not isinstance(calibration, dict):
        return _refusal("pointer gauge calibration is missing")
    try:
        payload = _post_frame(image, calibration)
    except (HTTPError, URLError, OSError, TimeoutError, ValueError, KeyError) as exc:
        return _refusal("gauge model unavailable: %s" % type(exc).__name__)
    reading = translate_gauge_reading(payload, (image.shape[1], image.shape[0]), calibration)
    if reading["status"] == "stable":
        return reading
    aspect_rejected = (isinstance(payload, dict)
                      and payload.get("status") == "pointer_outside_scale"
                      and "ROI aspect ratio out of bounds" in str(payload.get("reason", "")))
    if aspect_rejected or (isinstance(payload, dict) and payload.get("status") == "missing_pose_class"):
        if "detected_meters" in payload:
            recovered = _detected_context_crop_consensus(image, calibration, payload)
            # A located dial that remains unreadable needs another stationary
            # frame, not blind screen windows or further chassis oscillation.
            return _with_visible_dial(recovered or reading, image, calibration, payload)
    if aspect_rejected:
        recovered = _aspect_gate_crop_consensus(image, calibration)
        if recovered is not None:
            return recovered
    return (_with_visible_dial(reading, image, calibration, payload)
            if isinstance(payload, dict) else reading)


def read_pointer_gauge(image, calibration):
    """Infer this exact frame within one shared, bounded retry budget."""
    with pointer_gauge_inference_budget(8.0):
        return _read_pointer_gauge_with_budget(image, calibration)
