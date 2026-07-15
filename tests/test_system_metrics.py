import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_api"))
sys.path.insert(0, str(REPO_ROOT / "robot" / "ros2_ws" / "src" / "rvclaw_algo"))
sys.path.insert(0, str(REPO_ROOT / "services" / "rvclaw_report"))

from rvclaw_api.system_metrics import get_system_metrics


class SystemMetricsTest(unittest.TestCase):
    def test_payload_has_stable_shape_without_optional_sensors(self):
        payload = get_system_metrics()

        self.assertIn("device", payload)
        self.assertIn("usage", payload)
        self.assertIn("thermal", payload)
        self.assertIn("power", payload)
        self.assertIn("hostname", payload["device"])
        self.assertIn("cpu_percent", payload["usage"])
        self.assertIsInstance(payload["thermal"]["sensors"], list)
        self.assertIsInstance(payload["power"]["available"], bool)


if __name__ == "__main__":
    unittest.main()
