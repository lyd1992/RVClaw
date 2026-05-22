from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
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


DEMOZOO_MODEL_FALLBACKS = {
    "classification": ("resnet", "mobilenet_v2", "efficientnet", "swin_tiny"),
    "object_detection": ("yolov8", "yolov5", "yolov11", "yolov6"),
    "segmentation": ("yolov8_seg", "fcn", "unet", "sam"),
    "face_detection": ("yolov5_face",),
}


class DemoZooClient:
    def __init__(self, base_url: str | None = None, timeout_s: float | None = None, endpoint_template: str | None = None):
        self.base_url = (base_url or os.environ.get("RVCLAW_DEMOZOO_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else float(os.environ.get("RVCLAW_VISION_TIMEOUT_S", "30"))
        self.endpoint_template = endpoint_template or os.environ.get("RVCLAW_DEMOZOO_ENDPOINT_TEMPLATE") or "/predict/{model}"
        self._model_endpoint_cache: dict[str, str] | None = None

    def predict(self, image_path: Path, task: str, model: str) -> dict[str, Any]:
        boundary = f"----rvclaw-{uuid.uuid4().hex}"
        body = _multipart_body(boundary=boundary, image_path=image_path, fields={"task": task, "model": model})
        last_error = None
        attempted: list[str] = []
        for url in self._candidate_urls(task=task, model=model):
            attempted.append(url)
            request = Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = _decode_response_payload(response.read(), response.headers.get_content_type())
                    payload["_rvclaw_demozoo_url"] = url
                    return payload
            except HTTPError as exc:
                body_text = _read_http_error_body(exc)
                last_error = f"HTTP Error {exc.code}: {exc.reason}; url={url}; body={body_text[:2000]}"
                if exc.code == 404:
                    continue
                raise RuntimeError(last_error) from exc
            except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}; url={url}"
                raise
        raise RuntimeError(f"DemoZoo predict failed for model={model}; attempted={attempted}; last_error={last_error}")

    def _candidate_urls(self, task: str, model: str) -> list[str]:
        endpoints = []
        registry_endpoint = self._model_endpoints().get(model)
        if registry_endpoint:
            endpoints.append(registry_endpoint)
        endpoints.append(self.endpoint_template.format(task=task, model=model))

        urls: list[str] = []
        seen: set[str] = set()
        for endpoint in endpoints:
            path = "/" + endpoint.strip("/")
            for candidate in (self.base_url + path, self.base_url + path + "/"):
                if candidate not in seen:
                    seen.add(candidate)
                    urls.append(candidate)
        return urls

    def _model_endpoints(self) -> dict[str, str]:
        if self._model_endpoint_cache is not None:
            return self._model_endpoint_cache
        request = Request(self.base_url + "/models", method="GET")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            self._model_endpoint_cache = {}
            return self._model_endpoint_cache

        models = payload.get("models") if isinstance(payload, dict) else None
        endpoints: dict[str, str] = {}
        for row in models if isinstance(models, list) else []:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            endpoint = row.get("endpoint")
            if name and endpoint:
                endpoints[str(name)] = str(endpoint)
        self._model_endpoint_cache = endpoints
        return endpoints


class DemoZooVisionDevice(CVSampleDevice):
    backend_name = "demozoo"

    def __init__(self, artifact_dir: str | Path, vision_source: str | Path, client: DemoZooClient | None = None):
        super().__init__(artifact_dir=artifact_dir, vision_source=vision_source)
        self.client = client or DemoZooClient()

    def analyze_image(self, image_ref: str | None = "latest", task: str = "object_detection", model: str | None = None) -> dict[str, Any]:
        source = _resolve_image_ref(image_ref, self._latest_capture)
        task = normalize_vision_task(task)
        requested_model = model or default_model_for_task(task)
        started_at = time.perf_counter()
        fallback_reason = None
        result: dict[str, Any] | None = None
        model_errors: list[str] = []

        for candidate_model in _candidate_models(task=task, requested_model=requested_model):
            try:
                payload = self.client.predict(source, task=task, model=candidate_model)
                candidate_result = normalize_demozoo_payload(payload, task=task, model=candidate_model)
                if not _is_usable_result(candidate_result):
                    raise RuntimeError(f"{candidate_model} returned no usable {task} result")
                result = candidate_result
                if candidate_model != requested_model:
                    result["requested_model"] = requested_model
                    result["model_fallback_reason"] = "; ".join(model_errors)
                break
            except (OSError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                model_errors.append(f"{candidate_model}: {exc}")

        try:
            if result is None:
                fallback_reason = "; ".join(model_errors)
                raise RuntimeError(fallback_reason)
        except RuntimeError as exc:
            if _requires_real_vision():
                raise RuntimeError(f"DemoZoo real vision backend is required but unavailable: {fallback_reason}") from exc
            result = local_vision_result(task=task, model=requested_model, source=source)
            result["backend"] = "mock_fallback"
            result["requested_backend"] = "demozoo"
            result["fallback_reason"] = fallback_reason

        annotated = self.artifact_dir / f"{_slug(task)}_annotated.png"
        if result.get("image_base64"):
            annotated.write_bytes(base64.b64decode(str(result["image_base64"])))
        else:
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
    image_bytes = image_path.read_bytes()
    for field_name in ("image", "file"):
        rows.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{field_name}"; filename="{image_path.name}"\r\n'.encode("utf-8"),
                b"Content-Type: image/png\r\n\r\n",
                image_bytes,
                b"\r\n",
            ]
        )
    rows.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(rows)


def _decode_response_payload(body: bytes, content_type: str) -> dict[str, Any]:
    if content_type == "application/json" or body.lstrip()[:1] in {b"{", b"["}:
        payload = json.loads(body.decode("utf-8"))
        if isinstance(payload, dict):
            return payload
        return {"results": payload}
    return {
        "image_base64": base64.b64encode(body).decode("ascii"),
        "content_type": content_type,
        "summary": "DemoZoo returned an annotated image result; structured boxes were not provided by the sidecar response.",
    }


def _read_http_error_body(exc: HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _requires_real_vision() -> bool:
    return os.environ.get("RVCLAW_REQUIRE_REAL_VISION", "").strip().lower() in {"1", "true", "yes", "on"}


def _candidate_models(task: str, requested_model: str) -> list[str]:
    ordered = [requested_model, *DEMOZOO_MODEL_FALLBACKS.get(normalize_vision_task(task), ())]
    candidates: list[str] = []
    seen: set[str] = set()
    for model in ordered:
        if model not in seen:
            seen.add(model)
            candidates.append(model)
    return candidates


def _is_usable_result(result: dict[str, Any]) -> bool:
    if result.get("image_base64"):
        return True
    task = normalize_vision_task(str(result.get("task") or "object_detection"))
    if task == "classification":
        return bool(result.get("labels"))
    if task == "segmentation":
        return bool(result.get("segments"))
    if task == "face_detection":
        return True
    return bool(result.get("objects"))
