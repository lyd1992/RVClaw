from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rvclaw.memory.memory_api import MemoryManager


def main() -> int:
    memory = MemoryManager.from_path(ROOT / "runs" / "memory.sqlite3")
    matches = memory.query("A-03 device inspection", limit=5)
    for match in matches:
        print(match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
