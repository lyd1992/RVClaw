from __future__ import annotations

from ..runtime_api import RuntimeRequest, RuntimeResponse


class OnnxRuntimeBackend:
    name = "onnx_runtime"

    def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        raise NotImplementedError("ONNX Runtime backend adapter is planned for model/CV plugins.")
