from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rvclaw.utils import write_json


VISION_TASKS = ("classification", "object_detection", "segmentation", "face_detection")
TASK_DEFAULT_MODELS = {
    "classification": "resnet",
    "object_detection": "yolov8",
    "segmentation": "yolov8_seg",
    "face_detection": "yolov5_face",
}
ALLOWED_VISION_MODELS = (
    "resnet",
    "mobilenet_v2",
    "efficientnet",
    "swin_tiny",
    "yolov5",
    "yolov8",
    "yolov11",
    "yolov8_seg",
    "fcn",
    "unet",
    "sam",
    "yolov5_face",
)


def normalize_vision_task(task: str | None) -> str:
    normalized = (task or "object_detection").strip().lower().replace("-", "_")
    aliases = {
        "classify": "classification",
        "classification": "classification",
        "detect": "object_detection",
        "detection": "object_detection",
        "object_detection": "object_detection",
        "segment": "segmentation",
        "segmentation": "segmentation",
        "face": "face_detection",
        "face_detect": "face_detection",
        "face_detection": "face_detection",
    }
    return aliases.get(normalized, normalized)


def default_model_for_task(task: str) -> str:
    return TASK_DEFAULT_MODELS.get(normalize_vision_task(task), "yolov8")


def local_vision_result(task: str, model: str | None = None, source: str | Path | None = None) -> dict[str, Any]:
    task = normalize_vision_task(task)
    model = model or default_model_for_task(task)
    result = _empty_result(task=task, model=model, backend="cv_sample")
    result["backend_detail"] = "cv_sample_heuristic"
    result["requires_real_model"] = True
    source_hint = _source_hint(source)

    if task == "classification":
        result["labels"] = _classification_labels(source_hint)
        result["summary"] = _classification_summary(source_hint)
    elif task == "segmentation":
        result["segments"] = _segmentation_regions(source_hint)
        result["summary"] = _heuristic_summary("segmentation", source_hint)
    elif task == "face_detection":
        result["faces"] = []
        result["summary"] = "cv_sample only validates the Agent vision workflow; enable DemoZoo/MNN/ONNX for real face detection. No identity recognition is performed."
    else:
        result["objects"] = _object_regions(source_hint)
        result["summary"] = _heuristic_summary("object_detection", source_hint)
    return result


def normalize_demozoo_payload(payload: dict[str, Any], task: str, model: str) -> dict[str, Any]:
    task = normalize_vision_task(task)
    result = _empty_result(task=task, model=model, backend="demozoo")
    labels = _extract_labels(payload)
    objects = _extract_objects(payload)
    segments = _extract_segments(payload)
    faces = _extract_faces(payload)

    if task == "classification":
        result["labels"] = labels or _objects_to_labels(objects)
    elif task == "segmentation":
        result["segments"] = segments or objects
    elif task == "face_detection":
        result["faces"] = faces or objects
    else:
        result["objects"] = objects

    result["raw"] = payload
    result["summary"] = str(payload.get("summary") or _build_summary(result))
    return result


def write_vision_result(artifact_dir: str | Path, result: dict[str, Any]) -> Path:
    return write_json(Path(artifact_dir) / "vision_result.json", result)


def render_vision_annotation(source: Path, annotated: Path, result: dict[str, Any]) -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        shutil.copyfile(source, annotated)
        return

    image = cv2.imread(str(source))
    if image is None:
        shutil.copyfile(source, annotated)
        return
    height, width = image.shape[:2]
    _draw_header(cv2, image, result)

    if result.get("task") == "classification":
        for index, item in enumerate(result.get("labels", [])[:4], start=1):
            label = item.get("label", "unknown")
            confidence = float(item.get("confidence", 0))
            cv2.putText(image, f"{index}. {label} {confidence:.2f}", (20, 42 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (72, 222, 170), 2, cv2.LINE_AA)
    elif result.get("task") == "segmentation":
        overlay = image.copy()
        for item in result.get("segments", []):
            x1, y1, x2, y2 = _bbox_to_pixels(item.get("bbox"), width, height)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (64, 180, 230), -1)
            _draw_box(cv2, image, item, width, height, color=(64, 180, 230))
        cv2.addWeighted(overlay, 0.22, image, 0.78, 0, image)
    elif result.get("task") == "face_detection":
        for item in result.get("faces", []):
            _draw_box(cv2, image, item, width, height, color=(244, 191, 84))
        if not result.get("faces"):
            cv2.putText(image, "No face detected", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (244, 191, 84), 2, cv2.LINE_AA)
    else:
        for item in result.get("objects", []):
            _draw_box(cv2, image, item, width, height, color=(72, 222, 170))
    cv2.imwrite(str(annotated), image)


def _empty_result(task: str, model: str, backend: str) -> dict[str, Any]:
    return {
        "task": normalize_vision_task(task),
        "model": model,
        "backend": backend,
        "summary": "",
        "labels": [],
        "objects": [],
        "segments": [],
        "faces": [],
    }


def _source_hint(source: str | Path | None) -> str:
    if source is None:
        return ""
    return Path(source).name.lower()


def _classification_labels(source_hint: str) -> list[dict[str, Any]]:
    if any(key in source_hint for key in ("robot", "机器人", "humanoid")):
        return [
            {"label": "robot_scene_candidate", "confidence": 0.42},
            {"label": "road_or_transit_scene_candidate", "confidence": 0.31},
            {"label": "uploaded_image_sample", "confidence": 0.27},
        ]
    return [
        {"label": "uploaded_image_sample", "confidence": 1.0},
        {"label": "cv_sample_fallback", "confidence": 1.0},
    ]


def _classification_summary(source_hint: str) -> str:
    if any(key in source_hint for key in ("robot", "机器人", "humanoid")):
        return (
            "cv_sample 根据上传文件名给出低置信度候选标签，用于验证 Agent 流程；"
            "这不是模型分类结果。请启用 DemoZoo/MNN/ONNX 真实视觉后端后再判断图片内容。"
        )
    return (
        "cv_sample 已完成图片分类链路验证，但当前未接入真实分类模型；"
        "请启用 DemoZoo/MNN/ONNX 后端获取实际图像分类结果。"
    )


def _object_regions(source_hint: str) -> list[dict[str, Any]]:
    if any(key in source_hint for key in ("robot", "机器人", "humanoid")):
        return [
            {"label": "robot_region_candidate", "confidence": 0.45, "bbox": [0.34, 0.12, 0.64, 0.92]},
            {"label": "vehicle_region_candidate", "confidence": 0.32, "bbox": [0.66, 0.04, 0.98, 0.72]},
        ]
    return [
        {"label": "sample_region", "confidence": 0.5, "bbox": [0.15, 0.18, 0.82, 0.78]},
        {"label": "highlight_region", "confidence": 0.5, "bbox": [0.76, 0.38, 0.89, 0.63]},
    ]


def _segmentation_regions(source_hint: str) -> list[dict[str, Any]]:
    return [
        {**region, "label": region["label"].replace("_candidate", "_segment")}
        for region in _object_regions(source_hint)
    ]


def _heuristic_summary(task: str, source_hint: str) -> str:
    source_note = "uploaded image" if source_hint else "sample image"
    return (
        f"cv_sample generated heuristic {task} regions for the {source_note} to exercise the Agent workflow; "
        "enable DemoZoo/MNN/ONNX for real model inference."
    )


def _extract_labels(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("labels") or payload.get("topk") or payload.get("classes") or payload.get("predictions")
    rows = candidates if isinstance(candidates, list) else []
    labels = []
    for row in rows:
        if isinstance(row, str):
            labels.append({"label": row, "confidence": 1.0})
        elif isinstance(row, dict):
            label = row.get("label") or row.get("class") or row.get("name") or row.get("category") or row.get("predicted_class")
            if label:
                labels.append({"label": str(label), "confidence": float(row.get("confidence", row.get("score", row.get("prob", 0))))})
    if not labels and payload.get("predicted_class"):
        labels.append({"label": str(payload["predicted_class"]), "confidence": float(payload.get("confidence", payload.get("score", 0)))})
    return labels


def _extract_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("objects") or payload.get("detections") or payload.get("boxes") or payload.get("results") or []
    return [_normalize_region(row) for row in rows if isinstance(row, dict)]


def _extract_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("segments") or payload.get("masks") or []
    return [_normalize_region(row) for row in rows if isinstance(row, dict)]


def _extract_faces(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("faces") or payload.get("face_boxes") or []
    return [_normalize_region(row, default_label="face") for row in rows if isinstance(row, dict)]


def _normalize_region(row: dict[str, Any], default_label: str = "object") -> dict[str, Any]:
    bbox = row.get("bbox") or row.get("box") or row.get("xyxy") or row.get("rect") or []
    return {
        "label": str(row.get("label") or row.get("class") or row.get("name") or default_label),
        "confidence": float(row.get("confidence", row.get("score", row.get("prob", 0)))),
        "bbox": _normalize_bbox(bbox),
    }


def _normalize_bbox(bbox: Any) -> list[float]:
    if isinstance(bbox, dict):
        values = [bbox.get("x1", bbox.get("left", 0)), bbox.get("y1", bbox.get("top", 0)), bbox.get("x2", bbox.get("right", 1)), bbox.get("y2", bbox.get("bottom", 1))]
    elif isinstance(bbox, (list, tuple)):
        values = list(bbox[:4])
    else:
        values = [0, 0, 1, 1]
    return [float(value) for value in values]


def _objects_to_labels(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"label": item.get("label", "object"), "confidence": item.get("confidence", 0)} for item in objects]


def _build_summary(result: dict[str, Any]) -> str:
    task = result.get("task")
    if task == "classification":
        labels = result.get("labels", [])
        if labels:
            top = labels[0]
            return f"分类结果为 {top.get('label')}，置信度 {float(top.get('confidence', 0)):.2f}。"
        return "未返回有效分类结果。"
    if task == "segmentation":
        return f"分割得到 {len(result.get('segments', []))} 个区域。"
    if task == "face_detection":
        return f"检测到 {len(result.get('faces', []))} 张人脸，仅记录位置，不做身份识别。"
    return f"检测到 {len(result.get('objects', []))} 个目标。"


def _draw_header(cv2: Any, image: Any, result: dict[str, Any]) -> None:
    text = f"{result.get('task')} | {result.get('model')} | {result.get('backend')}"
    cv2.rectangle(image, (10, 10), (min(image.shape[1] - 10, 460), 44), (18, 27, 31), -1)
    cv2.putText(image, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 246, 242), 2, cv2.LINE_AA)


def _draw_box(cv2: Any, image: Any, item: dict[str, Any], width: int, height: int, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = _bbox_to_pixels(item.get("bbox"), width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{item.get('label', 'object')} {float(item.get('confidence', 0)):.2f}"
    cv2.putText(image, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def _bbox_to_pixels(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
    values = _normalize_bbox(bbox)
    if max(values) <= 1.5:
        x1, y1, x2, y2 = values[0] * width, values[1] * height, values[2] * width, values[3] * height
    else:
        x1, y1, x2, y2 = values
    return (
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
        max(0, min(width - 1, int(x2))),
        max(0, min(height - 1, int(y2))),
    )
