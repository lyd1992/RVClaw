"""Fixed-point digital and pointer meter readers.

The digital path treats the configured display as a measurement field.  It
never searches the whole camera image for the highest-confidence number.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import re
import shutil
import subprocess
from typing import Sequence

import cv2
import numpy as np


_DIGIT_SEGMENTS = {
    "0": (1, 1, 1, 1, 1, 1, 0),
    "1": (0, 1, 1, 0, 0, 0, 0),
    "2": (1, 1, 0, 1, 1, 0, 1),
    "3": (1, 1, 1, 1, 0, 0, 1),
    "4": (0, 1, 1, 0, 0, 1, 1),
    "5": (1, 0, 1, 1, 0, 1, 1),
    "6": (1, 0, 1, 1, 1, 1, 1),
    "7": (1, 1, 1, 0, 0, 0, 0),
    "8": (1, 1, 1, 1, 1, 1, 1),
    "9": (1, 1, 1, 1, 0, 1, 1),
}


@dataclass(frozen=True)
class DigitalMeterConfig:
    """Per-point calibration for a digital display.

    ``digit_boxes`` contains normalized ``x, y, width, height`` boxes inside
    the display region.  Requiring these boxes prevents panel labels and serial
    numbers from becoming a reading.  Use ``display_kind='ocr'`` for liquid
    crystal or dot-matrix displays that are not seven-segment glyphs.
    """

    display_kind: str = "seven_segment"
    digit_boxes: tuple[tuple[float, float, float, float], ...] = ()
    decimal_places: int = 0
    expected_digits: int | None = None
    value_range: tuple[float, float] | None = None
    min_sharpness: float = 12.0
    min_contrast: float = 10.0
    max_glare_ratio: float = 0.18
    min_digit_confidence: float = 0.64
    segment_on_threshold: float = 0.30
    allow_minus: bool = False

    @classmethod
    def from_mapping(cls, values: dict | None = None) -> "DigitalMeterConfig":
        """Build a config from YAML-style keys used by the inspection service."""
        values = dict(values or {})
        if "kind" in values and "display_kind" not in values:
            values["display_kind"] = values.pop("kind")
        if "range" in values and "value_range" not in values:
            values["value_range"] = values.pop("range")
        values.pop("unit", None)
        values.pop("roi", None)
        quality = values.pop("quality", {}) or {}
        for key in ("min_sharpness", "min_contrast", "max_glare_ratio"):
            if key in quality and key not in values:
                values[key] = quality[key]
        if "digit_boxes" in values:
            values["digit_boxes"] = tuple(tuple(box) for box in values["digit_boxes"])
        if "value_range" in values and values["value_range"] is not None:
            values["value_range"] = tuple(values["value_range"])
        return cls(**values)

    def __post_init__(self):
        if self.display_kind not in {"seven_segment", "ocr"}:
            raise ValueError("display_kind must be seven_segment or ocr")
        if self.decimal_places < 0:
            raise ValueError("decimal_places must be non-negative")
        if self.expected_digits is not None and self.expected_digits <= 0:
            raise ValueError("expected_digits must be positive")
        if self.value_range is not None and self.value_range[0] > self.value_range[1]:
            raise ValueError("value_range minimum must not exceed maximum")
        for box in self.digit_boxes:
            if len(box) != 4 or not _valid_normalized_box(box):
                raise ValueError(f"invalid normalized digit box: {box}")


@dataclass
class MeterReading:
    meter_type: str
    status: str
    value: float | None
    unit: str | None
    confidence: float
    raw_text: str | None = None
    angle_deg: float | None = None
    bbox: dict | None = None
    quality: dict | None = None
    display_value: dict | None = None
    text_items: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.value is not None and self.status in {"stable", "out_of_range", "alarm"}

    def as_dict(self):
        return {
            "meter_type": self.meter_type,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "confidence": round(float(self.confidence), 4),
            "raw_text": self.raw_text,
            "angle_deg": self.angle_deg,
            "bbox": self.bbox,
            "quality": self.quality or {},
            "display_value": self.display_value,
            "text_items": self.text_items,
        }


def _valid_normalized_box(box: Sequence[float]) -> bool:
    x, y, width, height = map(float, box)
    return (
        0.0 <= x < 1.0
        and 0.0 <= y < 1.0
        and width > 0.0
        and height > 0.0
        and x + width <= 1.0
        and y + height <= 1.0
    )


def _empty_reading(unit: str | None, status: str = "checking", quality=None) -> MeterReading:
    return MeterReading("digital", status, None, unit, 0.0, quality=quality or {})


class DigitalMeterReader:
    """Read one calibrated display region.

    Seven-segment decoding is deterministic and runs without an external
    service.  OCR (Optical Character Recognition) is an explicit fallback for
    non-standard displays and is still restricted to this display crop.
    """

    def __init__(self, command: str = "tesseract"):
        self.command = shutil.which(command)

    def read_from_frame(
        self,
        frame: np.ndarray,
        roi: Sequence[float],
        unit: str | None = None,
        config: DigitalMeterConfig | None = None,
    ) -> MeterReading:
        """Crop a normalized display ROI from a frame and read it."""
        if frame is None or frame.size == 0 or len(roi) != 4 or not _valid_normalized_box(roi):
            return _empty_reading(unit, "invalid_roi")
        height, width = frame.shape[:2]
        x, y, roi_width, roi_height = map(float, roi)
        crop = frame[
            int(round(y * height)) : int(round((y + roi_height) * height)),
            int(round(x * width)) : int(round((x + roi_width) * width)),
        ]
        reading = self.read(crop, unit=unit, config=config)
        reading.bbox = {"x": x, "y": y, "width": roi_width, "height": roi_height}
        if reading.display_value is not None:
            reading.display_value["bbox"] = reading.bbox
        return reading

    def read(
        self,
        image: np.ndarray,
        unit: str | None = None,
        value_range: Sequence[float] | None = None,
        config: DigitalMeterConfig | None = None,
    ) -> MeterReading:
        """Read an already cropped display image.

        ``value_range`` remains supported for callers using the older API and
        overrides the range from ``config`` when supplied.
        """
        config = config or DigitalMeterConfig(
            display_kind="ocr", value_range=tuple(value_range) if value_range else None
        )
        if value_range is not None:
            config = DigitalMeterConfig(**{**config.__dict__, "value_range": tuple(value_range)})
        quality, gray = self._quality(image, config)
        if gray is None:
            reason = quality.get("reason") if quality else "empty"
            status = "insufficient_quality" if reason in {"empty", "display_too_small"} else "unreadable"
            return _empty_reading(unit, status, quality)

        if config.display_kind == "seven_segment":
            # A calibrated seven-segment point fails closed. Falling back to
            # unrestricted OCR would reintroduce label confusion.
            return self._read_seven_segment(gray, unit, config, quality)
        return self._read_ocr(gray, unit, config, quality)

    @staticmethod
    def _quality(image, config: DigitalMeterConfig):
        if image is None or image.size == 0 or image.ndim not in {2, 3}:
            return ({"accepted": False, "reason": "empty"}, None)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if min(gray.shape[:2]) < 32:
            return ({"accepted": False, "reason": "display_too_small"}, None)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        glare_ratio = float(np.mean(gray >= 250))
        quality = {
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0]),
            "sharpness": round(sharpness, 3),
            "contrast": round(contrast, 3),
            "glare_ratio": round(glare_ratio, 4),
            "accepted": True,
        }
        if sharpness < config.min_sharpness:
            quality.update(accepted=False, reason="motion_blur")
        elif contrast < config.min_contrast:
            quality.update(accepted=False, reason="low_contrast")
        elif glare_ratio > config.max_glare_ratio:
            quality.update(accepted=False, reason="glare")
        if not quality["accepted"]:
            return quality, None
        return quality, gray

    def _read_seven_segment(self, gray, unit, config, quality):
        if not config.digit_boxes:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "digit_boxes_required"})
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        normalized_boxes = [
            (
                int(round(x * resized.shape[1])),
                int(round(y * resized.shape[0])),
                int(round(w * resized.shape[1])),
                int(round(h * resized.shape[0])),
            )
            for x, y, w, h in config.digit_boxes
        ]
        _, otsu = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks = (otsu, cv2.bitwise_not(otsu))
        best = None
        for mask in masks:
            chars = []
            confidence = []
            for box in normalized_boxes:
                char, score = self._decode_digit(mask, box, config)
                chars.append(char)
                confidence.append(score)
            if any(char == "?" for char in chars):
                continue
            score = float(np.mean(confidence)) if confidence else 0.0
            if best is None or score > best[0]:
                best = (score, chars, confidence)
        if best is None:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "invalid_segment_pattern"})
        score, chars, confidence = best
        raw = "".join(chars)
        if config.expected_digits is not None and len(raw) != config.expected_digits:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "digit_count_mismatch"})
        if config.decimal_places:
            if config.decimal_places >= len(raw):
                return _empty_reading(unit, "unreadable", {**quality, "reason": "decimal_position_invalid"})
            raw = raw[:-config.decimal_places] + "." + raw[-config.decimal_places :]
        try:
            value = float(raw)
        except ValueError:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "number_parse_failed"})
        status = self._range_status(value, config.value_range)
        if score < config.min_digit_confidence:
            status = "unreadable"
        return MeterReading(
            "digital",
            status,
            value if status != "unreadable" else None,
            unit,
            round(score, 4),
            raw_text=raw,
            quality={**quality, "digit_confidences": [round(float(v), 4) for v in confidence]},
            display_value={"text": raw, "value": value, "confidence": round(score, 4), "method": "seven_segment"},
        )

    @staticmethod
    def _decode_digit(mask, box, config):
        x, y, width, height = box
        clipped = mask[max(0, y) : max(0, y) + max(0, height), max(0, x) : max(0, x) + max(0, width)]
        if clipped.shape[0] < 12 or clipped.shape[1] < 7:
            return "?", 0.0
        regions = (
            # Narrow strips sample the centre of each segment.  Wide boxes
            # dilute a lit segment with its dark background and make 2/3/5
            # indistinguishable on low-resolution captures.
            (0.20, 0.04, 0.60, 0.09),
            (0.77, 0.16, 0.12, 0.28),
            (0.77, 0.56, 0.12, 0.28),
            (0.20, 0.87, 0.60, 0.09),
            (0.11, 0.56, 0.12, 0.28),
            (0.11, 0.16, 0.12, 0.28),
            (0.20, 0.455, 0.60, 0.09),
        )
        occupancy = []
        for rx, ry, rw, rh in regions:
            sx, sy = int(clipped.shape[1] * rx), int(clipped.shape[0] * ry)
            sw = max(1, int(clipped.shape[1] * rw))
            sh = max(1, int(clipped.shape[0] * rh))
            region = clipped[sy : sy + sh, sx : sx + sw]
            occupancy.append(float(np.mean(region > 0)) if region.size else 0.0)
        best_digit, best_score = "?", -1.0
        second_score = -1.0
        for digit, expected in _DIGIT_SEGMENTS.items():
            score = sum((ratio if on else 1.0 - ratio) for ratio, on in zip(occupancy, expected)) / 7.0
            if score > best_score:
                second_score, best_score, best_digit = best_score, score, digit
            elif score > second_score:
                second_score = score
        if best_score < config.min_digit_confidence or best_score - second_score < 0.01:
            return "?", max(0.0, best_score)
        return best_digit, best_score

    def _read_ocr(self, gray, unit, config, quality):
        if not self.command:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "ocr_engine_missing"})
        enlarged = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        encoded = cv2.imencode(".png", binary)
        if not encoded[0]:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "encode_failed"})
        try:
            completed = subprocess.run(
                [
                    self.command,
                    "stdin",
                    "stdout",
                    "--psm",
                    "7",
                    "-c",
                    "tessedit_char_whitelist=0123456789.-",
                ],
                input=encoded[1].tobytes(),
                capture_output=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return _empty_reading(unit, "unreadable", {**quality, "reason": "ocr_failed"})
        raw = completed.stdout.decode("utf-8", errors="ignore").strip()
        normalized = raw.replace(",", ".").replace("O", "0").replace("o", "0")
        normalized = normalized.replace("I", "1").replace("l", "1")
        matches = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)
        if len(matches) != 1:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "ambiguous_ocr", "ocr_text": raw})
        text = matches[0]
        if not config.allow_minus and text.startswith("-"):
            return _empty_reading(unit, "unreadable", {**quality, "reason": "minus_not_allowed", "ocr_text": raw})
        digits = len(re.sub(r"\D", "", text))
        if config.expected_digits is not None and digits != config.expected_digits:
            return _empty_reading(unit, "unreadable", {**quality, "reason": "digit_count_mismatch", "ocr_text": raw})
        value = float(text)
        status = self._range_status(value, config.value_range)
        confidence = min(0.90, 0.55 + min(0.35, quality["contrast"] / 180.0))
        return MeterReading(
            "digital",
            status,
            value,
            unit,
            round(confidence, 4),
            raw_text=text,
            quality={**quality, "ocr_text": raw},
            display_value={"text": text, "value": value, "confidence": round(confidence, 4), "method": "tesseract"},
        )

    @staticmethod
    def _range_status(value, value_range):
        return "out_of_range" if value_range and not value_range[0] <= value <= value_range[1] else "stable"


class DigitalReadingStabilizer:
    """Fuse a short burst without carrying an old value through bad frames."""

    def __init__(self, window: int = 5, min_consistent: int = 3):
        if window <= 0 or min_consistent <= 0 or min_consistent > window:
            raise ValueError("min_consistent must be between 1 and window")
        self.window = window
        self.min_consistent = min_consistent
        self._history: deque[MeterReading] = deque(maxlen=window)

    def reset(self):
        self._history.clear()

    def update(self, reading: MeterReading) -> MeterReading:
        self._history.append(reading)
        candidates = [item for item in self._history if item.valid and item.raw_text]
        if not candidates:
            return _empty_reading(reading.unit, "checking", reading.quality)
        counts = Counter(item.raw_text for item in candidates)
        raw_text, count = counts.most_common(1)[0]
        if count < self.min_consistent:
            status = "unreadable" if len(self._history) == self.window else "checking"
            return _empty_reading(reading.unit, status, reading.quality)
        consensus = [item for item in candidates if item.raw_text == raw_text]
        selected = max(consensus, key=lambda item: item.confidence)
        result = MeterReading(**selected.as_dict())
        result.status = "out_of_range" if any(item.status == "out_of_range" for item in consensus) else "stable"
        result.confidence = round(float(np.mean([item.confidence for item in consensus])), 4)
        result.quality = {**(selected.quality or {}), "consensus_count": count, "window": len(self._history)}
        return result


class PointerMeterReader:
    """OpenCV (Open Source Computer Vision Library) reader for a calibrated gauge."""

    def read(self, image, calibration=None, unit=None):
        calibration = calibration or {}
        if image is None or image.size == 0:
            return MeterReading("pointer", "insufficient_quality", None, unit, 0.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, gray.shape[0] // 4),
            param1=100,
            param2=32,
            minRadius=max(12, gray.shape[0] // 8),
            maxRadius=max(20, gray.shape[0] // 2),
        )
        if circles is None:
            return MeterReading("pointer", "checking", None, unit, 0.0)
        x, y, radius = max(circles[0], key=lambda c: c[2])
        edges = cv2.Canny(gray, 60, 160)
        mask = np.zeros_like(edges)
        cv2.circle(mask, (int(x), int(y)), max(2, int(radius * 0.92)), 255, cv2.FILLED)
        lines = cv2.HoughLinesP(
            cv2.bitwise_and(edges, mask),
            1,
            np.pi / 180,
            threshold=max(12, int(radius * 0.22)),
            minLineLength=max(10, int(radius * 0.35)),
            maxLineGap=8,
        )
        if lines is None:
            return MeterReading("pointer", "checking", None, unit, 0.0)
        best = None
        best_score = -1.0
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(float, line)
            d1, d2 = math.hypot(x1 - x, y1 - y), math.hypot(x2 - x, y2)
            near, tip, length = ((x1, y1), (x2, y2), d2) if d1 < d2 else ((x2, y2), (x1, y1), d1)
            if math.hypot(near[0] - x, near[1] - y) > radius * 0.30:
                continue
            score = length / max(radius, 1.0)
            if score > best_score:
                best_score, best = score, tip
        if best is None:
            return MeterReading("pointer", "checking", None, unit, 0.0)
        angle = (math.degrees(math.atan2(y - best[1], best[0] - x)) + 360.0) % 360.0
        start = float(calibration.get("start_angle_deg", 225.0))
        end = float(calibration.get("end_angle_deg", 315.0))
        minimum, maximum = calibration.get("range", [0.0, 1.0])
        clockwise = calibration.get("direction", "counterclockwise") == "clockwise"
        span = (end - start) % 360.0 if clockwise else (start - end) % 360.0
        offset = (angle - start) % 360.0 if clockwise else (start - angle) % 360.0
        ratio = max(0.0, min(1.0, offset / max(span, 1e-6)))
        value = float(minimum) + ratio * (float(maximum) - float(minimum))
        return MeterReading(
            "pointer", "stable", value, unit, min(0.99, 0.45 + best_score / 4.0), angle_deg=round(angle, 2)
        )


class MeterReader:
    def __init__(self, digital: DigitalMeterReader | None = None):
        self.digital = digital or DigitalMeterReader()
        self.pointer = PointerMeterReader()

    def read(self, image, meter_type, config=None):
        config = config or {}
        if meter_type == "digital":
            if isinstance(config, DigitalMeterConfig):
                return self.digital.read(image, config=config)
            unit = config.get("unit")
            return self.digital.read(image, unit=unit, config=DigitalMeterConfig.from_mapping(config))
        return self.pointer.read(image, config.get("calibration"), config.get("unit"))
