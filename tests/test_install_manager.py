from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.install.manager import build_install_script, get_backend_spec, plan_backend_install


class InstallManagerTest(unittest.TestCase):
    def test_backend_aliases_and_script_generation(self) -> None:
        llama = get_backend_spec("llama.cpp")
        self.assertEqual(llama.key, "llama_cpp")

        script = build_install_script(["llama.cpp"], deps_dir="third_party")
        self.assertIn("https://github.com/ggml-org/llama.cpp.git", script)
        self.assertIn("cmake -S . -B build", script)
        self.assertIn("RVCLAW_DEPS_DIR", script)

    def test_plan_can_pin_refs(self) -> None:
        plan = plan_backend_install(["mnn"], refs={"mnn": "3.5.0"})
        commands = "\n".join(plan[0][1])
        self.assertIn("git checkout 3.5.0", commands)


if __name__ == "__main__":
    unittest.main()
