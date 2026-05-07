from __future__ import annotations

from ..runtime_api import RuntimeRequest, RuntimeResponse


class LlamaCppBackend:
    name = "llama_cpp"

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        raise NotImplementedError("llama.cpp backend adapter is planned after Demo Claw v0.1.")
