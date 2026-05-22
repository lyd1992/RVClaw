from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from rvclaw.adapters.cv_sample_device import CVSampleDevice, _resolve_image_ref, _slug
from rvclaw.adapters.vision import (
    default_model_for_task,
    local_vision_result,
    normalize_demozoo_payload,
    normalize_vision_task,
    render_vision_annotation,
    write_vision_result,
)


class DemoZooClient:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None, endpoint_template: str | None = None):
        self.base_url = (base_url or os.environ.get("RVCLAW_DEMOZOO_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else float(os.environ.get("RVCLAW_VISION_TIMEOUT_S", "30"))
        self.endpoint_template = endpoint_template or os.environ.get("RVCLAW_DEMOZOO_ENDPOINT_TEMPLATE") or "/predict/{model}"

    def predict(self, image_path: Path, task: str, model: str) -> dict[str, Any]:
        url = self.base_url + self.endpoint_template.format(task=task, model=model)
        boundary = f"----rvclaw-{uuid.uuid4().hex}"
        body = _multipart_body(boundary=boundary, image_path=image_path, fields={"task": task, "model": model})
        request = Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))


class DemoZooVisionDevice(CVSampleDevice):
    backend_name = "demozoo"

    def __init__(self, artifact_dir: str | Path, vision_source: str | Path, client: DemoZooClient | None = None):
        super().__init__(artifact_dir=artifact_dir, vision_source=vision_source)
        self.client = client or DemoZooClient()

    def analyze_image(self, image_ref: str | None = "latest", task: str = "object_detection", model: str | None = None) -> dict[str, Any]:
        source = _resolve_image_ref(image_ref, self._latest_capture)
        task = normalize_vision_task(task)
        model = model or default_model_for_task(task)
        started_at = time.perf_counter()
        fallback_reason = None
        try:
            payload = self.client.predict(source, task=task, model=model)
            result = normalize_demozoo_payload(payload, task=task, model=model)
        except (OSError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            fallback_reason = str(exc)
            if _requires_real_vision():
                raise RuntimeError(f"DemoZoo real vision backend is required but unavailable: {fallback_reason}") from exc
            result = local_vision_result(task=task, model=model, source=source)
            result["backend"] = "mock_fallback"
            result["requested_backend"] = "demozoo"
            result["fallback_reason"] = fallback_reason

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


def _multipart_body(boundary: str, image_path: Path, fields: dict[str, str]) -> bytes:
    rows: list[bytes] = []
    for key, value in fields.items():
        rows.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                f"{value}\r\n".encode("utf-8"),
            ]
        )
    rows.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode("utf-8"),
            b"Content-Type: image/png\r\n\r\n",
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(rows)


def _requires_real_vision() -> bool:
    return os.environ.get("RVCLAW_REQUIRE_REAL_VISION", "").strip().lower() in {"1", "true", "yes", "on"}
