import base64
import html
from datetime import datetime, timezone
from pathlib import Path


class LocalInspectionRecognizer:
    """Deterministic local baseline for the K3 sample inspection flow."""

    def __init__(self, target="K3 CoM260 local", backend="rule-baseline"):
        self.target = target
        self.backend = backend

    def inspect(self, image_path, task_id, point_id, category):
        source = Path(image_path)
        self._ensure_supported_source(source)

        detection = self._build_detection(category)
        is_anomaly = self._is_anomaly(category, detection["value"])
        return {
            "task_id": task_id,
            "point_id": point_id,
            "category": category,
            "value": detection["value"],
            "confidence": detection["confidence"],
            "is_anomaly": is_anomaly,
            "anomaly_reason": None if not is_anomaly else "sample threshold exceeded",
            "evidence_uri": source.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "target": self.target,
                "backend": self.backend,
                "model": "sample-rule-v0",
                "offline": True,
            },
            "detections": [detection],
        }

    def save_annotated_svg(self, image_path, result, output_path):
        source = Path(image_path)
        target = Path(output_path)
        encoded = base64.b64encode(source.read_bytes()).decode("ascii")
        mime = "image/svg+xml" if source.suffix.lower() == ".svg" else "image/png"
        detection = result["detections"][0]
        bbox = detection["bbox"]
        label = f"{result['category']} {result['value']} ({result['confidence']:.2f})"
        status = "ANOMALY" if result["is_anomaly"] else "OK"
        color = "#dc2626" if result["is_anomaly"] else "#16a34a"

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="620" viewBox="0 0 960 620">
  <rect width="960" height="620" fill="#111827"/>
  <image x="40" y="40" width="880" height="495" preserveAspectRatio="xMidYMid meet" href="data:{mime};base64,{encoded}"/>
  <rect x="{bbox['x']}" y="{bbox['y']}" width="{bbox['width']}" height="{bbox['height']}" fill="none" stroke="{color}" stroke-width="6"/>
  <rect x="{bbox['x']}" y="{bbox['y'] - 44}" width="440" height="38" rx="4" fill="{color}"/>
  <text x="{bbox['x'] + 16}" y="{bbox['y'] - 18}" fill="#ffffff" font-family="monospace" font-size="20">{html.escape(label)}</text>
  <text x="40" y="585" fill="#f9fafb" font-family="monospace" font-size="24">{html.escape(result['point_id'])} / {html.escape(status)}</text>
</svg>
"""
        target.write_text(svg, encoding="utf-8")
        return target

    def _ensure_supported_source(self, source):
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix.lower() not in {".svg", ".png", ".jpg", ".jpeg"}:
            raise ValueError(f"unsupported image format: {source.suffix}")

    def _build_detection(self, category):
        if category == "meter_reading":
            return {
                "label": "pressure_gauge",
                "value": 0.42,
                "confidence": 0.91,
                "bbox": {"x": 310, "y": 145, "width": 340, "height": 260},
            }
        if category == "indicator_status":
            return {
                "label": "green_indicator",
                "value": "green",
                "confidence": 0.88,
                "bbox": {"x": 330, "y": 150, "width": 300, "height": 220},
            }
        if category == "personnel_safety":
            return {
                "label": "safe_area_clear",
                "value": "clear",
                "confidence": 0.86,
                "bbox": {"x": 280, "y": 130, "width": 390, "height": 300},
            }
        return {
            "label": category,
            "value": "observed",
            "confidence": 0.8,
            "bbox": {"x": 300, "y": 140, "width": 360, "height": 260},
        }

    def _is_anomaly(self, category, value):
        if category == "meter_reading":
            return float(value) > 0.8
        if category == "indicator_status":
            return value == "red"
        if category == "personnel_safety":
            return value != "clear"
        return False
