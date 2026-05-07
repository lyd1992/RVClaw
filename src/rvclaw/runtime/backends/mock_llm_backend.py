from __future__ import annotations

from ..runtime_api import RuntimeRequest, RuntimeResponse


class MockLLMBackend:
    name = "mock_llm"

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(
            text=f"[mock] {request.prompt[:160]}",
            metrics={
                "engine": self.name,
                "model": request.model,
                "tokens_s": None,
                "ttft_ms": 0,
            },
        )
