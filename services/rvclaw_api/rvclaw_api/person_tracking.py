from datetime import datetime, timezone

from rvclaw_algo import YoloV8nDetector


def get_person_tracking_payload():
    detector = YoloV8nDetector()
    yolo_runtime = detector.runtime()
    yolo_detections = detector.detect_image("samples/sample_meter.svg")
    return {
        "source": {
            "type": "demo-video",
            "name": "people-detection-sample",
            "replaceable_with": "ros2_camera_topic",
            "topic": "/camera/color/image_raw",
            "adapter": "rvclaw_perception.camera_stream",
            "transport": "mp4-file",
            "uri": "/assets/videos/people-detection.mp4",
        },
        "stream": {
            "width": 960,
            "height": 540,
            "fps": 15,
            "duration_sec": 12,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_id": "camera_color_optical_frame",
            "encoding": "rgb8",
        },
        "runtime": {
            "target": "K3 CoM260 local",
            "backend": yolo_runtime["backend"],
            "model": "yolov8n",
            "offline": True,
            "available": yolo_runtime["available"],
            "status": yolo_runtime["status"],
            "message": yolo_runtime["message"],
        },
        "detections": yolo_detections,
        "tracks": [
            {
                "track_id": "P-01",
                "label": "person",
                "confidence": 0.93,
                "safety_state": "normal",
                "keyframes": [
                    {"t": 0.00, "bbox": {"x": 112, "y": 168, "width": 72, "height": 196}},
                    {"t": 0.25, "bbox": {"x": 178, "y": 158, "width": 74, "height": 202}},
                    {"t": 0.50, "bbox": {"x": 262, "y": 152, "width": 78, "height": 206}},
                    {"t": 0.75, "bbox": {"x": 348, "y": 160, "width": 76, "height": 200}},
                    {"t": 1.00, "bbox": {"x": 424, "y": 176, "width": 74, "height": 192}},
                ],
            },
            {
                "track_id": "P-02",
                "label": "person",
                "confidence": 0.89,
                "safety_state": "restricted-zone-watch",
                "keyframes": [
                    {"t": 0.00, "bbox": {"x": 760, "y": 178, "width": 64, "height": 184}},
                    {"t": 0.25, "bbox": {"x": 704, "y": 170, "width": 68, "height": 190}},
                    {"t": 0.50, "bbox": {"x": 648, "y": 165, "width": 70, "height": 196}},
                    {"t": 0.75, "bbox": {"x": 594, "y": 172, "width": 68, "height": 188}},
                    {"t": 1.00, "bbox": {"x": 544, "y": 184, "width": 66, "height": 180}},
                ],
            },
        ],
    }
