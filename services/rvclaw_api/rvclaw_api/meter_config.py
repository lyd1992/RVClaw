"""Persistent calibration records for fixed digital meters."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
from typing import Any


METER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_REFERENCE_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_UNSET = object()


class MeterConfigError(ValueError):
    """A user-facing meter configuration validation error."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _number(value: Any, field: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise MeterConfigError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise MeterConfigError(f"{field} must be a number") from error
    if not math.isfinite(result):
        raise MeterConfigError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise MeterConfigError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise MeterConfigError(f"{field} must be at most {maximum}")
    return result


def _optional_number(value: Any, field: str, minimum: float | None = None) -> float | None:
    if value is None or value == "":
        return None
    return _number(value, field, minimum=minimum)


def _integer(value: Any, field: str, minimum: int, maximum: int, nullable: bool = False) -> int | None:
    if nullable and (value is None or value == ""):
        return None
    number = _number(value, field, minimum=minimum, maximum=maximum)
    if not number.is_integer():
        raise MeterConfigError(f"{field} must be an integer")
    return int(number)


def _point(value: Any, field: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MeterConfigError(f"{field} must be a point")
    return {
        "x": _number(value.get("x"), f"{field}.x", 0.0, 1.0),
        "y": _number(value.get("y"), f"{field}.y", 0.0, 1.0),
    }


def _region(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MeterConfigError("display_region must be a rectangle")
    x = _number(value.get("x"), "display_region.x", 0.0, 1.0)
    y = _number(value.get("y"), "display_region.y", 0.0, 1.0)
    width = _number(value.get("width"), "display_region.width", 0.001, 1.0)
    height = _number(value.get("height"), "display_region.height", 0.001, 1.0)
    if x + width > 1.0 or y + height > 1.0:
        raise MeterConfigError("display_region must stay inside the reference image")
    return {"x": x, "y": y, "width": width, "height": height}


def _clean_text(value: Any, field: str, maximum: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise MeterConfigError(f"{field} must be text")
    result = value.strip()
    if required and not result:
        raise MeterConfigError(f"{field} is required")
    if len(result) > maximum:
        raise MeterConfigError(f"{field} is too long")
    return result


def normalize_meter_config(payload: Any, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a browser payload and return a stable on-disk record."""
    if not isinstance(payload, dict):
        raise MeterConfigError("Expected a JSON object")

    meter_id = _clean_text(payload.get("meter_id"), "meter_id", 64, required=True)
    if not METER_ID_PATTERN.fullmatch(meter_id):
        raise MeterConfigError("meter_id may contain only letters, numbers, hyphens, and underscores")

    markers = payload.get("alignment_markers", [None, None])
    if not isinstance(markers, list) or len(markers) != 2:
        raise MeterConfigError("alignment_markers must contain exactly two points")

    normal_min = _optional_number(payload.get("normal_min"), "normal_min")
    normal_max = _optional_number(payload.get("normal_max"), "normal_max")
    if normal_min is not None and normal_max is not None and normal_min > normal_max:
        raise MeterConfigError("normal_min must not exceed normal_max")

    record = {
        "meter_id": meter_id,
        "name": _clean_text(payload.get("name"), "name", 120, required=True),
        "display_region": _region(payload.get("display_region")),
        "alignment_markers": [
            _point(markers[0], "alignment_markers[0]"),
            _point(markers[1], "alignment_markers[1]"),
        ],
        "decimal_places": _integer(payload.get("decimal_places", 0), "decimal_places", 0, 6),
        "digit_count": _integer(payload.get("digit_count"), "digit_count", 1, 16, nullable=True),
        "unit": _clean_text(payload.get("unit", ""), "unit", 32),
        "normal_min": normal_min,
        "normal_max": normal_max,
        "max_change": _optional_number(payload.get("max_change"), "max_change", 0.0),
        "min_confidence": _number(payload.get("min_confidence", 0.8), "min_confidence", 0.0, 1.0),
        "consistency_count": _integer(payload.get("consistency_count", 3), "consistency_count", 1, 10),
        "reference_image": (existing or {}).get("reference_image"),
        "updated_at": _now(),
    }
    pointer_scale = payload.get("pointer_scale")
    if pointer_scale is not None:
        if not isinstance(pointer_scale, dict):
            raise MeterConfigError("pointer_scale must be an object")
        if not record['display_region']:
            raise MeterConfigError("pointer_scale requires a configured dial region")
        geometry = {key: _point(pointer_scale.get(key), "pointer_scale."+key)
                    for key in ("center", "start", "end")}
        if not all(geometry.values()):
            raise MeterConfigError("pointer_scale requires center, start and end")
        mask = pointer_scale.get("mutable_region")
        if not isinstance(mask, dict):
            raise MeterConfigError("pointer_scale requires mutable_region")
        geometry["mutable_region"] = {
            key: _number(mask.get(key), "mutable_region."+key, 0.001, 1.)
            for key in ("cx", "cy", "rx", "ry")}
        for axis in ("x", "y"):
            center, radius = geometry["mutable_region"]["c"+axis], geometry["mutable_region"]["r"+axis]
            if center-radius < 0 or center+radius > 1:
                raise MeterConfigError("mutable_region must stay inside reference image")
        geometry["direction"] = pointer_scale.get("direction", "clockwise")
        if geometry["direction"] not in ("clockwise", "counterclockwise"):
            raise MeterConfigError("invalid pointer_scale direction")
        record["pointer_scale"] = geometry
    return record


def meter_is_ready(record: dict[str, Any]) -> bool:
    return bool(
        record.get("reference_image")
        and record.get("display_region")
        and all(record.get("alignment_markers", []))
    )


class MeterConfigStore:
    """Small JSON store shared by the Studio configuration page and inference."""

    def __init__(self, config_path: Path, reference_image_dir: Path):
        self.config_path = config_path
        self.reference_image_dir = reference_image_dir

    def list(self) -> list[dict[str, Any]]:
        records = self._read_records()
        active_id = self.active_meter_id()
        result = []
        for record in sorted(records, key=lambda item: item["meter_id"]):
            public = self.public_record(record)
            public["is_active"] = record["meter_id"] == active_id
            result.append(public)
        return result

    def active_meter_id(self) -> str | None:
        document = self._read_document()
        active_id = document.get("active_meter_id")
        if not isinstance(active_id, str):
            return None
        if not any(item.get("meter_id") == active_id for item in document["meters"]):
            return None
        return active_id

    def get(self, meter_id: str) -> dict[str, Any] | None:
        for record in self._read_records():
            if record["meter_id"] == meter_id:
                return record
        return None

    def upsert(self, payload: Any) -> dict[str, Any]:
        proposed_id = payload.get("meter_id") if isinstance(payload, dict) else None
        existing = self.get(proposed_id) if isinstance(proposed_id, str) else None
        record = normalize_meter_config(payload, existing)
        records = self._read_records()
        records = [item for item in records if item["meter_id"] != record["meter_id"]]
        records.append(record)
        self._write_records(records)
        return self.public_record(record)

    def activate(self, meter_id: str) -> dict[str, Any]:
        if not METER_ID_PATTERN.fullmatch(meter_id):
            raise MeterConfigError("Invalid meter_id")
        records = self._read_records()
        record = next((item for item in records if item["meter_id"] == meter_id), None)
        if record is None:
            raise MeterConfigError("Meter configuration was not found")
        if not meter_is_ready(record):
            raise MeterConfigError("Complete the reference photo, display region, and alignment points first")
        self._write_records(records, active_meter_id=meter_id)
        return self.public_record(record)

    def save_reference_image(self, meter_id: str, content_type: str, data: bytes) -> dict[str, Any]:
        if not METER_ID_PATTERN.fullmatch(meter_id):
            raise MeterConfigError("Invalid meter_id")
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise MeterConfigError("Reference image must be JPEG, PNG, or WebP")
        if not data or len(data) > MAX_REFERENCE_IMAGE_SIZE:
            raise MeterConfigError("Reference image must be between 1 byte and 10 MB")
        records = self._read_records()
        record = next((item for item in records if item["meter_id"] == meter_id), None)
        if record is None:
            raise MeterConfigError("Save the meter configuration before uploading a reference image")

        self.reference_image_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{meter_id}{ALLOWED_IMAGE_TYPES[content_type]}"
        target = self.reference_image_dir / filename
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_bytes(data)
        temporary.replace(target)

        old_filename = record.get("reference_image")
        record["reference_image"] = filename
        record["updated_at"] = _now()
        self._write_records(records)
        if old_filename and old_filename != filename:
            old_path = self.reference_image_dir / old_filename
            if old_path.is_file():
                old_path.unlink()
        return self.public_record(record)

    def delete(self, meter_id: str) -> bool:
        if not METER_ID_PATTERN.fullmatch(meter_id):
            raise MeterConfigError("Invalid meter_id")
        records = self._read_records()
        removed = next((item for item in records if item["meter_id"] == meter_id), None)
        if removed is None:
            return False
        active_id = self.active_meter_id()
        self._write_records(
            [item for item in records if item["meter_id"] != meter_id],
            active_meter_id=None if active_id == meter_id else _UNSET,
        )
        filename = removed.get("reference_image")
        if filename:
            image_path = self.reference_image_dir / filename
            if image_path.is_file():
                image_path.unlink()
        return True

    def public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        result = dict(record)
        filename = result.pop("reference_image", None)
        result["reference_image_url"] = f"/api/meter-reference-images/{filename}" if filename else None
        result["is_ready"] = meter_is_ready(record)
        return result

    def _read_records(self) -> list[dict[str, Any]]:
        return self._read_document()["meters"]

    def _read_document(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"meters": []}
        try:
            document = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MeterConfigError("Could not read meter configuration storage") from error
        records = document.get("meters") if isinstance(document, dict) else None
        if not isinstance(records, list):
            raise MeterConfigError("Meter configuration storage is invalid")
        return {"meters": records, "active_meter_id": document.get("active_meter_id")}

    def _write_records(self, records: list[dict[str, Any]], active_meter_id: Any = _UNSET) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        document = {"version": 1, "meters": sorted(records, key=lambda item: item["meter_id"])}
        if active_meter_id is _UNSET:
            active_meter_id = self._read_document().get("active_meter_id")
        if isinstance(active_meter_id, str) and active_meter_id:
            document["active_meter_id"] = active_meter_id
        temporary = self.config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.config_path)
