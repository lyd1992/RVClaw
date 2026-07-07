from datetime import datetime, timezone

from rvclaw_algo import YoloV8nDetector


def get_vision_state():
    detector = YoloV8nDetector()
    runtime = detector.runtime()
    detections = detector.detect_image("samples/sample_meter.svg")
    alerts = [
        {
            "level": "WARN",
            "source": "F7",
            "message": f"{item['label']} requires safety review",
            "track_id": item["track_id"],
            "timestamp": item["timestamp"],
        }
        for item in detections
        if item.get("safety_state") != "normal"
    ]
    return {
        "features": {
            "F5": "video_capture",
            "F7": "studio_task_display",
        },
        "video": {
            "camera": "C70 RGB",
            "source": {
                "type": "ros2-camera",
                "topic": "/camera/color/image_raw",
                "adapter": "rvclaw_perception.camera_stream",
                "transport": "ros2-image-topic",
            },
            "stream": {
                "width": 960,
                "height": 540,
                "fps": 15,
                "frame_id": "camera_color_optical_frame",
                "encoding": "rgb8",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "status": "ready-for-camera",
        },
        "model": runtime,
        "detections": detections,
        "alerts": alerts,
        "device": {
            "host": "K3 CoM260 Kit",
            "camera": "C70 RGB",
            "camera_status": "configured",
            "radar_status": "reserved",
            "base_status": "reserved",
        },
        "studio": {
            "show_video": True,
            "show_detection_boxes": True,
            "show_alerts": True,
            "show_device_status": True,
            "report_entry": True,
        },
    }
