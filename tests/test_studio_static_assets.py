import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class StudioStaticAssetsTest(unittest.TestCase):
    def test_studio_page_contains_sample_flow_view(self):
        page = (REPO_ROOT / "studio" / "web" / "index.html").read_text(encoding="utf-8")

        self.assertIn("/api/sample-flow", page)
        self.assertIn("annotated-image", page)
        self.assertIn("json-output", page)
        self.assertIn("report-link", page)
        self.assertIn("RVClaw EdgeOne Studio", page)
        self.assertIn("EdgeOne-01", page)
        self.assertIn("Run RVBench", page)
        self.assertIn("Camera Streams", page)
        self.assertIn("Model Runtime", page)
        self.assertIn("Device Health", page)
        self.assertIn("RVBench Report", page)
        self.assertIn("performance-chart", page)
        self.assertIn("log-stream", page)
        self.assertIn("/api/person-tracking", page)
        self.assertIn("/api/vision-state", page)
        self.assertIn("/api/video-detections", page)
        self.assertIn("person-video", page)
        self.assertIn("person-overlay", page)
        self.assertIn("Person Tracking", page)
        self.assertIn("person-count", page)
        self.assertIn("YOLOv8n", page)
        self.assertIn("Accuracy", page)
        self.assertIn("model-accuracy", page)
        self.assertNotIn("model-precision", page)
        self.assertIn("f5-video-status", page)
        self.assertIn("f7-alert-count", page)
        self.assertIn("yolo-detection-count", page)
        self.assertIn("videoDetectionPayload.runtime.flow", page)
        self.assertIn("formatOverlayMode", page)
        self.assertIn("video-decode-unavailable", page)
        self.assertIn("video_inference_error", page)
        self.assertIn("inference_video", page)
        self.assertNotIn("person-demo-canvas", page)
        self.assertNotIn("captureStream", page)

    def test_studio_contains_real_person_tracking_video_asset(self):
        video = REPO_ROOT / "studio" / "web" / "assets" / "videos" / "people-detection.mp4"

        self.assertTrue(video.exists())
        self.assertGreater(video.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
