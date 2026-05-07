from __future__ import annotations

from ..runtime_api import RuntimeRequest, RuntimeResponse


class MNNBackend:
    name = "mnn"

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        raise NotImplementedError("MNN backend adapter is a post-v0.1 optimization plugin.")
