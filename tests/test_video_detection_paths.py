import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))

from rvclaw_api import video_detection


class VideoDetectionPathTest(unittest.TestCase):
    def test_display_video_can_differ_from_inference_video(self):
        captured = {}

        class FakeDetector:
            def runtime(self):
                return {
                    "available": True,
                    "model": "yolov8n",
                    "model_path": "models/yolov8n.pt",
                    "confidence": 0.35,
                    "offline": False,
                    "status": "ready",
                    "message": None,
                }

        def fake_stream(video_path):
            captured["stream_path"] = Path(video_path)
            return {"width": 960, "height": 540, "fps": 15, "duration_sec": 3}

        def fake_loader(video_path, detector, runtime, stream):
            captured["inference_path"] = Path(video_path)
            return [{"time_sec": 0, "frame_index": 0, "detections": []}]

        previous_detector = video_detection.YoloV8nDetector
        previous_stream = video_detection._video_stream
        previous_loader = video_detection._load_or_build_frame_detections

        try:
            video_detection.YoloV8nDetector = lambda: FakeDetector()
            video_detection._video_stream = fake_stream
            video_detection._load_or_build_frame_detections = fake_loader

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                display_video = tmp / "factory_people_demo.mp4"
                inference_video = tmp / "factory_people_demo_cv2.mp4"
                display_video.write_bytes(b"display")
                inference_video.write_bytes(b"inference")

                payload = video_detection.get_video_detection_payload(
                    video_path=display_video,
                    inference_video_path=inference_video,
                )

            self.assertEqual(payload["source"]["name"], "factory_people_demo.mp4")
            self.assertEqual(captured["stream_path"], inference_video)
            self.assertEqual(captured["inference_path"], inference_video)
            self.assertEqual(payload["runtime"]["display_video"], str(display_video))
            self.assertEqual(payload["runtime"]["inference_video"], str(inference_video))
        finally:
            video_detection.YoloV8nDetector = previous_detector
            video_detection._video_stream = previous_stream
            video_detection._load_or_build_frame_detections = previous_loader

    def test_default_inference_video_uses_cv2_sidecar_when_present(self):
        captured = {}

        class FakeDetector:
            def runtime(self):
                return {"available": False, "model": "yolov8n"}

        def fake_stream(video_path):
            captured["stream_path"] = Path(video_path)
            return {"width": 960, "height": 540, "fps": 15, "duration_sec": 3}

        previous_detector = video_detection.YoloV8nDetector
        previous_stream = video_detection._video_stream

        try:
            video_detection.YoloV8nDetector = lambda: FakeDetector()
            video_detection._video_stream = fake_stream

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                display_video = tmp / "factory_people_demo.mp4"
                sidecar_video = tmp / "factory_people_demo_cv2.mp4"
                display_video.write_bytes(b"display")
                sidecar_video.write_bytes(b"inference")

                payload = video_detection.get_video_detection_payload(video_path=display_video)

            self.assertEqual(captured["stream_path"], sidecar_video)
            self.assertEqual(payload["runtime"]["display_video"], str(display_video))
            self.assertEqual(payload["runtime"]["inference_video"], str(sidecar_video))
        finally:
            video_detection.YoloV8nDetector = previous_detector
            video_detection._video_stream = previous_stream

    def test_ffmpeg_fallback_samples_frames_when_cv2_video_capture_cannot_open(self):
        previous_cv2 = sys.modules.get("cv2")
        previous_run = video_detection.subprocess.run
        cv2_stub = types.ModuleType("cv2")

        class ClosedCapture:
            def __init__(self, path):
                self.path = path

            def isOpened(self):
                return False

            def release(self):
                return None

        class FakeCompletedProcess:
            returncode = 0
            stderr = ""

        class FakeDetector:
            def detect_image(self, image_path):
                return [
                    {
                        "track_id": "Y8N-01",
                        "label": "person",
                        "confidence": 0.88,
                        "bbox": {"x": 10, "y": 12, "width": 40, "height": 80},
                    }
                ]

        def fake_run(command, capture_output, text):
            output_pattern = Path(command[-1])
            output_pattern.parent.mkdir(parents=True, exist_ok=True)
            (output_pattern.parent / "frame_000001.jpg").write_bytes(b"jpg-1")
            (output_pattern.parent / "frame_000002.jpg").write_bytes(b"jpg-2")
            return FakeCompletedProcess()

        try:
            cv2_stub.VideoCapture = ClosedCapture
            sys.modules["cv2"] = cv2_stub
            video_detection.subprocess.run = fake_run

            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = Path(tmpdir) / "factory_people_demo_cv2.mp4"
                video_path.write_bytes(b"video")

                frames = video_detection._build_frame_detections(
                    video_path,
                    FakeDetector(),
                    {"fps": 15, "width": 960, "height": 540, "duration_sec": 3},
                )

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0]["decode_backend"], "ffmpeg")
            self.assertEqual(frames[0]["detections"][0]["label"], "person")
        finally:
            video_detection.subprocess.run = previous_run
            if previous_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = previous_cv2
if __name__ == "__main__":
    unittest.main()
