from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rvclaw.adapters.cv_sample_device import CVSampleDevice
from rvclaw.adapters.mock_device import MockDevice


def build_device(artifact_dir: str | Path) -> tuple[MockDevice, dict[str, Any]]:
    requested = os.environ.get("RVCLAW_DEVICE_BACKEND", "mock").strip().lower() or "mock"
    if requested == "cv_sample":
        source = Path(os.environ.get("RVCLAW_VISION_SOURCE", "examples/vision/a03_normal.png"))
        if source.exists():
            return CVSampleDevice(artifact_dir=artifact_dir, vision_source=source), {
                "device_backend": "cv_sample",
                "vision_source": str(source),
            }
        return MockDevice(artifact_dir=artifact_dir), {
            "device_backend": "mock_fallback",
            "requested_device_backend": "cv_sample",
            "vision_source": str(source),
            "device_fallback_reason": "vision_source_missing",
        }
    return MockDevice(artifact_dir=artifact_dir), {"device_backend": "mock"}
