import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_PATH = Path(os.getenv("RVCLAW_YOLO_MODEL", "models/yolov8n.pt"))
DEFAULT_CONFIDENCE = float(os.getenv("RVCLAW_YOLO_CONF", "0.35"))


class YoloV8nDetector:
    """YOLOv8n adapter for K3 local video inspection.

    The adapter uses ultralytics when it is installed and a local model file is
    present. It returns a deterministic offline payload otherwise, so F5/F7 can
    still boot on a fresh K3 and show exactly what is missing.
    """

    def __init__(self, model_path=None, confidence=DEFAULT_CONFIDENCE):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.confidence = confidence
        self._model = None
        self._load_error = None

    def runtime(self):
        available = self._ensure_model()
        return {
            "target": "K3 CoM260 local",
            "backend": "ultralytics-yolov8" if available else "offline-yolov8n-contract",
            "model": "yolov8n",
            "model_path": str(self.model_path),
            "confidence": self.confidence,
            "offline": not available,
            "available": available,
            "status": "ready" if available else "model-adapter-fallback",
            "message": None if available else self._load_error,
        }

    def detect_image(self, image_path):
        source = Path(image_path)
        if source.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png", ".webp"}:
            return []
        if self._ensure_model():
            try:
                return self._detect_with_ultralytics(source)
            except FileNotFoundError:
                return []
        return []

    def detect_frame(self, frame):
        if self._ensure_model():
            return self._detect_with_ultralytics(frame)
        return []

    def _ensure_model(self):
        if self._model is not None:
            return True
        try:
            from ultralytics import YOLO

            if self.model_path.exists():
                self._model = YOLO(str(self.model_path))
            elif self.model_path.name == "yolov8n.pt":
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self._model = YOLO("yolov8n.pt")
                downloaded = Path("yolov8n.pt")
                if downloaded.exists() and downloaded.resolve() != self.model_path.resolve():
                    shutil.copy2(downloaded, self.model_path)
            else:
                self._load_error = f"YOLOv8n model file not found: {self.model_path}"
                return False
            return True
        except Exception as exc:
            self._load_error = f"YOLOv8n runtime unavailable: {exc}"
            return False

    def _detect_with_ultralytics(self, source):
        predict_source = str(source) if isinstance(source, Path) else source
        results = self._model.predict(source=predict_source, conf=self.confidence, verbose=False)
        detections = []
        names = getattr(self._model, "names", {}) or {}
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = names.get(cls_id, str(cls_id))
                detections.append(
                    {
                        "track_id": f"Y8N-{len(detections) + 1:02d}",
                        "label": label,
                        "value": label,
                        "confidence": round(confidence, 4),
                        "safety_state": _safety_state(label),
                        "bbox": _xyxy_to_bbox(xyxy),
                        "model": "yolov8n",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
        return detections

def _xyxy_to_bbox(xyxy):
    left, top, right, bottom = xyxy
    return {
        "x": round(float(left), 2),
        "y": round(float(top), 2),
        "width": round(float(right - left), 2),
        "height": round(float(bottom - top), 2),
    }


def _safety_state(label):
    normalized = str(label).lower()
    if normalized in {"person", "no_helmet", "helmet", "hardhat"}:
        return "restricted-zone-watch" if normalized in {"person", "no_helmet"} else "normal"
    return "normal"
