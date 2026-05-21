from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.utils import platform_profile


class PlatformProfileTest(unittest.TestCase):
    def test_platform_profile_uses_k3_environment_metadata(self) -> None:
        old_env = {
            "RVCLAW_PLATFORM_SOC": os.environ.get("RVCLAW_PLATFORM_SOC"),
            "RVCLAW_PLATFORM_ISA": os.environ.get("RVCLAW_PLATFORM_ISA"),
            "RVCLAW_RVV_VLEN": os.environ.get("RVCLAW_RVV_VLEN"),
        }
        try:
            os.environ["RVCLAW_PLATFORM_SOC"] = "K3-Pico-ITX-32GB"
            os.environ["RVCLAW_PLATFORM_ISA"] = "rv64gcv"
            os.environ["RVCLAW_RVV_VLEN"] = "unknown"

            profile = platform_profile()

            self.assertEqual(profile["soc"], "K3-Pico-ITX-32GB")
            self.assertEqual(profile["isa"], "rv64gcv")
            self.assertEqual(profile["rvv_vlen"], "unknown")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
