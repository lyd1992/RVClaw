import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))

from rvclaw_api.video_detection import get_video_detection_payload


class VideoDetectionFlowTest(unittest.TestCase):
    def test_video_detection_payload_describes_realtime_yolov8n_flow(self):
        payload = get_video_detection_payload()

        self.assertEqual(payload["runtime"]["model"], "yolov8n")
        self.assertEqual(payload["runtime"]["flow"], "video-frame -> yolov8n -> per-frame-boxes -> studio-overlay")
        self.assertEqual(payload["source"]["topic"], "/camera/color/image_raw")
        self.assertIn(payload["source"]["type"], {"sample-video", "browser-canvas"})
        self.assertIn(payload["runtime"]["overlay_mode"], {"model-output", "model-unavailable"})
        self.assertIn("frames", payload)
        self.assertIn("detections", payload)
        if payload["frames"]:
            self.assertIn("time_sec", payload["frames"][0])
            self.assertIn("detections", payload["frames"][0])


if __name__ == "__main__":
    unittest.main()
