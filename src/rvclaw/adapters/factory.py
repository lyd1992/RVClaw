from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rvclaw.adapters.cv_sample_device import CVSampleDevice
from rvclaw.adapters.demozoo_device import DemoZooVisionDevice
from rvclaw.adapters.mock_device import MockDevice


def build_device(artifact_dir: str | Path, vision_source: str | Path | None = None) -> tuple[MockDevice, dict[str, Any]]:
    requested = os.environ.get("RVCLAW_DEVICE_BACKEND", "mock").strip().lower() or "mock"
    vision_backend = os.environ.get("RVCLAW_VISION_BACKEND", "cv_sample").strip().lower() or "cv_sample"
    if vision_source is not None:
        requested = "cv_sample" if requested == "mock" else requested
    if requested in {"cv_sample", "demozoo"} or vision_backend == "demozoo":
        source = Path(vision_source) if vision_source is not None else Path(os.environ.get("RVCLAW_VISION_SOURCE", "examples/vision/a03_normal.png"))
        if source.exists():
            if vision_backend == "demozoo" or requested == "demozoo":
                return DemoZooVisionDevice(artifact_dir=artifact_dir, vision_source=source), {
                    "device_backend": "demozoo",
                    "vision_backend": "demozoo",
                    "demozoo_base_url": os.environ.get("RVCLAW_DEMOZOO_BASE_URL", "http://127.0.0.1:8000"),
                    "require_real_vision": os.environ.get("RVCLAW_REQUIRE_REAL_VISION", "0"),
                    "vision_source": str(source),
                }
            return CVSampleDevice(artifact_dir=artifact_dir, vision_source=source), {
                "device_backend": "cv_sample",
                "vision_backend": "cv_sample",
                "vision_source": str(source),
            }
        return MockDevice(artifact_dir=artifact_dir), {
            "device_backend": "mock_fallback",
            "vision_backend": "mock_fallback",
            "requested_device_backend": requested,
            "requested_vision_backend": vision_backend,
            "vision_source": str(source),
            "device_fallback_reason": "vision_source_missing",
        }
    return MockDevice(artifact_dir=artifact_dir), {"device_backend": "mock", "vision_backend": "mock"}
