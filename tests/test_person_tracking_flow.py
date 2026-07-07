import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))

from rvclaw_api.person_tracking import get_person_tracking_payload


class PersonTrackingFlowTest(unittest.TestCase):
    def test_demo_video_payload_describes_replaceable_camera_source_and_tracks(self):
        payload = get_person_tracking_payload()

        self.assertEqual(payload["source"]["type"], "demo-video")
        self.assertEqual(payload["source"]["replaceable_with"], "ros2_camera_topic")
        self.assertEqual(payload["source"]["topic"], "/camera/color/image_raw")
        self.assertEqual(payload["source"]["transport"], "mp4-file")
        self.assertEqual(payload["source"]["uri"], "/assets/videos/people-detection.mp4")
        self.assertEqual(payload["stream"]["width"], 960)
        self.assertEqual(payload["stream"]["height"], 540)
        self.assertEqual(payload["stream"]["frame_id"], "camera_color_optical_frame")
        self.assertEqual(payload["stream"]["encoding"], "rgb8")
        self.assertGreaterEqual(payload["stream"]["fps"], 12)

        tracks = payload["tracks"]
        self.assertGreaterEqual(len(tracks), 2)
        self.assertEqual(tracks[0]["label"], "person")
        self.assertIn("track_id", tracks[0])
        self.assertGreaterEqual(len(tracks[0]["keyframes"]), 4)
        self.assertIn("bbox", tracks[0]["keyframes"][0])

    def test_person_tracking_contract_keeps_camera_source_replaceable(self):
        schema_path = REPO_ROOT / "contracts" / "person_tracking.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertIn("source", schema["required"])
        self.assertIn("stream", schema["required"])
        self.assertIn("tracks", schema["required"])
        self.assertIn("demo-video", schema["properties"]["source"]["properties"]["type"]["enum"])
        self.assertIn("ros2-camera", schema["properties"]["source"]["properties"]["type"]["enum"])
        self.assertIn("transport", schema["properties"]["source"]["properties"])
        self.assertIn("mp4-file", schema["properties"]["source"]["properties"]["transport"]["enum"])
        self.assertIn("uri", schema["properties"]["source"]["properties"])
        self.assertIn("frame_id", schema["properties"]["stream"]["properties"])
        self.assertIn("encoding", schema["properties"]["stream"]["properties"])


if __name__ == "__main__":
    unittest.main()
