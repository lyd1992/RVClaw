"""Keep native speech synthesis and its initialization outside capture threads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
from threading import Lock
import time
from typing import Optional


# MATCHA_ZH (Chinese-only) silently drops Latin letters such as A/B/C, and its
# ~19s warm-up acquires a /dev/tcm buffer that the camera and the chassis adapter
# also need. MATCHA_ZH_EN reads Latin letters through the CPU pinyin front-end,
# warms up in ~0.3s instead of ~19s, and never touches /dev/tcm. Keep the default
# and the override in one place so the service, the prompt generator and the
# health report can never disagree about which backend is in use.
DEFAULT_TTS_BACKEND = os.getenv("RVCLAW_TTS_BACKEND", "MATCHA_ZH_EN")


class SpeechSynthesizer:
    def __init__(self, model_dir: str, backend: Optional[str] = None):
        self._lock = Lock()
        self._buffer = b""
        self.process = subprocess.Popen(
            [sys.executable, "-m", "rvclaw_voice.synthesis", model_dir,
             backend or DEFAULT_TTS_BACKEND],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        try:
            if not self._read(60).get("ready"):
                raise RuntimeError("speech synthesizer did not initialize")
        except BaseException:
            self.close()
            raise

    def _read(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError("speech synthesis timed out")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError("speech synthesizer exited")
            self._buffer += data
            if len(self._buffer) > 65536:
                raise RuntimeError("invalid speech synthesis response")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def synthesize(self, text: str, path: Path, timeout: float) -> bool:
        if not self._lock.acquire(timeout=timeout):
            raise TimeoutError("speech synthesizer busy")
        try:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("speech synthesizer is not running")
            self.process.stdin.write((json.dumps({"text": text, "path": str(path)}) + "\n").encode())
            result = self._read(timeout)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "speech synthesis failed"))
            return True
        except Exception:
            self.close()
            raise
        finally:
            self._lock.release()

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        process.stdin.close()
        process.stdout.close()


def main():
    # Native libraries print to stdout; reserve a separate descriptor for replies.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import spacemit_tts

    config = spacemit_tts.Config(
        backend=getattr(spacemit_tts.BackendType, sys.argv[2]), model_dir=sys.argv[1],
    )
    engine = spacemit_tts.Engine(config)
    print(json.dumps({"ready": True}), file=protocol)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not engine.synthesize_to_file(request["text"], request["path"]):
                raise RuntimeError("model did not produce audio")
            print(json.dumps({"ok": True}), file=protocol)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), file=protocol)


if __name__ == "__main__":
    main()
