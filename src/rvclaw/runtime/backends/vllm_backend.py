from __future__ import annotations

from ..runtime_api import RuntimeRequest, RuntimeResponse


class VLLMBackend:
    name = "vllm"

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        raise NotImplementedError("vLLM backend adapter is a post-v0.1 service plugin.")
