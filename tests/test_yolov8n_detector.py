import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))

from rvclaw_algo import YoloV8nDetector


class YoloV8nDetectorTest(unittest.TestCase):
    def test_runtime_supports_headless_cv2_without_imshow(self):
        previous_cv2 = sys.modules.get("cv2")
        previous_ultralytics = sys.modules.get("ultralytics")
        cv2_stub = types.ModuleType("cv2")

        class FakeYOLO:
            def __init__(self, model_path):
                import cv2

                if not hasattr(cv2, "imshow"):
                    raise AttributeError("module 'cv2' has no attribute 'imshow'")
                self.model_path = model_path

        ultralytics_stub = types.ModuleType("ultralytics")
        ultralytics_stub.YOLO = FakeYOLO

        try:
            sys.modules["cv2"] = cv2_stub
            sys.modules["ultralytics"] = ultralytics_stub
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "yolov8n.pt"
                model_path.write_bytes(b"fake-yolo")

                runtime = YoloV8nDetector(model_path=model_path).runtime()

            self.assertTrue(runtime["available"])
            self.assertEqual(runtime["backend"], "ultralytics-yolov8")
            self.assertTrue(hasattr(cv2_stub, "imshow"))
            self.assertTrue(hasattr(cv2_stub, "waitKey"))
            self.assertTrue(hasattr(cv2_stub, "destroyAllWindows"))
        finally:
            if previous_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = previous_cv2
            if previous_ultralytics is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = previous_ultralytics


if __name__ == "__main__":
    unittest.main()