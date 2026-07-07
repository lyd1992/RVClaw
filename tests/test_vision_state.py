import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))

from rvclaw_api.vision_state import get_vision_state


class VisionStateTest(unittest.TestCase):
    def test_f5_f7_state_declares_yolov8n_video_and_studio_contract(self):
        payload = get_vision_state()

        self.assertEqual(payload["features"]["F5"], "video_capture")
        self.assertEqual(payload["features"]["F7"], "studio_task_display")
        self.assertEqual(payload["video"]["camera"], "C70 RGB")
        self.assertEqual(payload["video"]["source"]["topic"], "/camera/color/image_raw")
        self.assertEqual(payload["video"]["stream"]["encoding"], "rgb8")
        self.assertEqual(payload["model"]["model"], "yolov8n")
        self.assertIn("offline", payload["model"])
        self.assertIn("available", payload["model"])
        for detection in payload["detections"]:
            self.assertEqual(detection["model"], "yolov8n")
        self.assertTrue(payload["studio"]["show_video"])
        self.assertTrue(payload["studio"]["show_detection_boxes"])

    def test_f5_f7_contract_is_present(self):
        schema_path = REPO_ROOT / "contracts" / "vision_state.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertIn("video", schema["required"])
        self.assertIn("model", schema["required"])
        self.assertEqual(schema["properties"]["model"]["properties"]["model"]["enum"], ["yolov8n"])


if __name__ == "__main__":
    unittest.main()
