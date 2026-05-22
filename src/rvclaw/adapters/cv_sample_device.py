from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from rvclaw.adapters.mock_device import MockDevice
from rvclaw.adapters.vision import default_model_for_task, local_vision_result, normalize_vision_task, render_vision_annotation, write_vision_result


class CVSampleDevice(MockDevice):
    backend_name = "cv_sample"

    def __init__(self, artifact_dir: str | Path, vision_source: str | Path):
        super().__init__(artifact_dir=artifact_dir)
        self.vision_source = Path(vision_source)
        self._latest_capture: Path | None = None

    def capture_image(self, target: str, mode: str = "inspection") -> dict[str, Any]:
        capture_path = self.artifact_dir / f"{_slug(target)}_capture.png"
        shutil.copyfile(self.vision_source, capture_path)
        self._latest_capture = capture_path
        return {
            "target": target,
            "mode": mode,
            "image_ref": str(capture_path),
            "status": "captured",
            "backend": self.backend_name,
        }

    def detect_status(self, target: str, image_ref: str | None = None) -> dict[str, Any]:
        source = _resolve_image_ref(image_ref, self._latest_capture)
        annotated = self.artifact_dir / f"{_slug(target)}_annotated.png"
        signals = _detect_with_opencv(source, annotated)
        if signals is None:
            shutil.copyfile(source, annotated)
            signals = {
                "temperature_c": 42.1,
                "vibration": "normal",
                "smoke": "none",
                "path_blocked": False,
                "status_light": "green",
                "detector": "file_copy_fallback",
            }
        return {
            "target": target,
            "image_ref": str(source),
            "annotated_image_ref": str(annotated),
            "device": f"edge-inference-node-{target.lower()}",
            "status": "normal",
            "signals": signals,
            "risk_level": "low",
            "backend": self.backend_name,
        }

    def analyze_image(self, image_ref: str | None = "latest", task: str = "object_detection", model: str | None = None) -> dict[str, Any]:
        source = _resolve_image_ref(image_ref, self._latest_capture)
        if _requires_real_vision():
            raise RuntimeError("RVCLAW_REQUIRE_REAL_VISION=1 requires a real vision backend; active backend is cv_sample")
        task = normalize_vision_task(task)
        model = model or default_model_for_task(task)
        started_at = time.perf_counter()
        result = local_vision_result(task=task, model=model, source=source)
        annotated = self.artifact_dir / f"{_slug(task)}_annotated.png"
        render_vision_annotation(source, annotated, result)
        result.update(
            {
                "image_ref": str(source),
                "annotated_image_ref": str(annotated),
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 3),
            }
        )
        write_vision_result(self.artifact_dir, result)
        return result


def _resolve_image_ref(image_ref: str | None, latest_capture: Path | None) -> Path:
    if image_ref and image_ref != "latest":
        return Path(image_ref)
    if latest_capture is None:
        raise FileNotFoundError("No captured image is available for status detection")
    return latest_capture


def _detect_with_opencv(source: Path, annotated: Path) -> dict[str, Any] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    image = cv2.imread(str(source))
    if image is None:
        return None
    height, width = image.shape[:2]
    cv2.rectangle(image, (2, 2), (max(3, width - 3), max(3, height - 3)), (40, 180, 80), 2)
    cv2.putText(
        image,
        "RVClaw OK",
        (8, max(18, min(height - 8, 24))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 180, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(annotated), image)
    return {
        "temperature_c": 42.1,
        "vibration": "normal",
        "smoke": "none",
        "path_blocked": False,
        "status_light": "green",
        "detector": "opencv_rule",
    }


def _slug(target: str) -> str:
    return target.lower().replace("-", "")


def _requires_real_vision() -> bool:
    return os.environ.get("RVCLAW_REQUIRE_REAL_VISION", "").strip().lower() in {"1", "true", "yes", "on"}
