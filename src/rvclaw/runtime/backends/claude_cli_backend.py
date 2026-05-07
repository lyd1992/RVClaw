from __future__ import annotations

import subprocess

from ..runtime_api import RuntimeRequest, RuntimeResponse


class ClaudeCliRuntimeBackend:
    name = "claude_cli"

    def __init__(self, command: str = "claude", timeout_s: int = 60):
        self.command = command
        self.timeout_s = timeout_s

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        completed = subprocess.run(
            [self.command],
            input=request.prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Claude CLI failed")
        return RuntimeResponse(
            text=completed.stdout.strip(),
            metrics={"engine": self.name, "model": request.model},
        )
