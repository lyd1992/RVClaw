# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Call the C++ :cpp:class:`VisionService` from Python via pybind11 module ``_vision_service_cpp``.

Build the extension::

    cd src/python
    ./build_wheel.sh --build-ext

Example::

    import cv2
    from spacemit_vision import VisionServiceNative, VisionServiceStatus

    svc = VisionServiceNative.create("examples/yolov8/config/yolov8.yaml")
    img = cv2.imread("test.jpg")
    status, results = svc.infer_image(img)
    assert status == VisionServiceStatus.OK
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


def _preload_libvision() -> None:
    """Load bundled libvision.so before the extension so ldd NEEDED libvision resolves."""
    pkg_dir = Path(__file__).resolve().parent
    lib = pkg_dir / "libvision.so"
    if not lib.is_file():
        return
    mode = getattr(os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(lib), mode=mode)


_preload_libvision()

try:
    from . import _vision_service_cpp as _ext
except ImportError:
    try:
        import _vision_service_cpp as _ext
    except ImportError as _import_error:  # pragma: no cover - extension optional at import time
        _ext = None
        _IMPORT_ERROR = _import_error
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None

__all__ = [
    "VisionServiceNative",
    "VisionServiceResult",
    "VisionServiceKeypoint",
    "VisionServiceInferParams",
    "VisionServiceResponse",
    "VisionServiceTiming",
    "VisionServiceTimingOptions",
    "VisionServiceStatus",
    "extension_available",
    "extension_import_error",
]


def extension_available() -> bool:
    return _ext is not None


def extension_import_error() -> ImportError | None:
    return _IMPORT_ERROR


def _require_ext() -> None:
    if _ext is None:
        raise ImportError(
            "Cannot load native extension. After `pip install spacemit_vision`, the wheel "
            "should contain spacemit_vision/_vision_service_cpp*.so and libvision.so. "
            "Reinstall/rebuild the wheel; if import still fails, check ldd on the .so for "
            "missing system libraries (OpenCV, onnxruntime, etc.)."
        ) from _IMPORT_ERROR


VisionServiceStatus = None  # type: ignore[assignment,misc]
VisionServiceResult = None  # type: ignore[assignment,misc]
VisionServiceKeypoint = None  # type: ignore[assignment,misc]
VisionServiceInferParams = None  # type: ignore[assignment,misc]
VisionServiceResponse = None  # type: ignore[assignment,misc]
VisionServiceTiming = None  # type: ignore[assignment,misc]
VisionServiceTimingOptions = None  # type: ignore[assignment,misc]

if _ext is not None:
    VisionServiceStatus = _ext.VisionServiceStatus
    VisionServiceResult = _ext.VisionServiceResult
    VisionServiceKeypoint = _ext.VisionServiceKeypoint
    VisionServiceInferParams = _ext.VisionServiceInferParams
    VisionServiceResponse = _ext.VisionServiceResponse
    VisionServiceTiming = _ext.VisionServiceTiming
    VisionServiceTimingOptions = _ext.VisionServiceTimingOptions


class VisionServiceNative:
    """
    Thin wrapper around :cpp:class:`VisionService` with typed hints for NumPy images.

    Images must be ``uint8``, shape ``(H, W, 3)``, BGR, C-contiguous (OpenCV default).

    The last successful ``infer_*`` response is cached so ``draw(image)`` works without
    passing a response explicitly (C++ ``Draw`` is stateless and requires one).
    """

    __slots__ = ("_svc", "_last_response")

    def __init__(self, impl: object) -> None:
        _require_ext()
        self._svc = impl
        self._last_response: Optional[object] = None

    @staticmethod
    def create(
        config_path: str,
        model_path_override: str = "",
        lazy_load: bool = False,
        timing_enabled: bool = False,
        timing_print_to_stdout: bool = False,
    ) -> VisionServiceNative:
        # Examples do not warm up; cold-start timings are misleading, so keep
        # stdout printing off by default (same as C++ VisionServiceTimingOptions).
        _require_ext()
        impl = _ext.VisionService.create(config_path, model_path_override, lazy_load)
        svc = VisionServiceNative(impl)
        options = VisionServiceTimingOptions()
        options.enabled = timing_enabled
        options.print_to_stdout = timing_print_to_stdout
        svc.set_timing_options(options)
        return svc

    @staticmethod
    def last_create_error() -> str:
        _require_ext()
        return _ext.VisionService.last_create_error()

    def _make_params(
        self,
        conf: float = -1.0,
        iou: float = -1.0,
        top_k: int = -1,
        kp_threshold: float = -1.0,
        mask_threshold: float = -1.0,
        max_det: int = -1,
    ) -> object:
        params = VisionServiceInferParams()
        params.conf_threshold = conf
        params.iou_threshold = iou
        params.top_k = top_k
        params.kp_threshold = kp_threshold
        params.mask_threshold = mask_threshold
        params.max_det = max_det
        return params

    def infer(
        self,
        image_or_path: Union[str, npt.NDArray[np.uint8]],
        params: Optional[object] = None,
    ) -> Tuple[object, object]:
        """Unified inference entry point; returns ``(status, VisionServiceResponse)``."""
        if params is None:
            params = VisionServiceInferParams()
        if isinstance(image_or_path, str):
            status, response = self._svc.infer(image_path=image_or_path, params=params)
        else:
            arr = np.ascontiguousarray(image_or_path)
            if arr.dtype != np.uint8:
                raise TypeError("image must be uint8 BGR (HxWx3)")
            status, response = self._svc.infer(image_bgr_uint8=arr, params=params)
        if status == VisionServiceStatus.OK:
            self._last_response = response
        return status, response

    def infer_image(
        self,
        image_or_path: Union[str, npt.NDArray[np.uint8]],
        conf: float = -1.0,
        iou: float = -1.0,
        prompts: Optional[List[str]] = None,
    ) -> Tuple[object, List]:
        """Run image inference; returns flat ``VisionServiceResult`` list for compatibility.

        ``conf`` / ``iou`` <= 0 use each model's configured defaults. Positive values apply
        only to this call.

        ``prompts`` is only used by open-vocabulary models (e.g. YOLO-World): pass a list of
        text labels to set/override the vocabulary for this call. Empty/None -> use the
        model's configured default vocabulary. Ignored by other models. Requires a numpy
        image (not a path).
        """
        if prompts:
            arr = np.ascontiguousarray(image_or_path)
            if arr.dtype != np.uint8:
                raise TypeError("image must be uint8 BGR (HxWx3)")
            status, results, response = self._svc.infer_image_prompts(
                image_bgr_uint8=arr, prompts=list(prompts), conf=conf, iou=iou
            )
        elif isinstance(image_or_path, str):
            status, results, response = self._svc.infer_image(
                image_path=image_or_path, conf=conf, iou=iou
            )
        else:
            arr = np.ascontiguousarray(image_or_path)
            if arr.dtype != np.uint8:
                raise TypeError("image must be uint8 BGR (HxWx3)")
            status, results, response = self._svc.infer_image(
                image_bgr_uint8=arr, conf=conf, iou=iou
            )
        if status == VisionServiceStatus.OK:
            self._last_response = response
        return status, results

    def infer_embedding(
        self,
        image_or_path: Union[str, npt.NDArray[np.uint8]],
    ) -> Tuple[object, List[float]]:
        if isinstance(image_or_path, str):
            status, embedding, response = self._svc.infer_embedding(image_path=image_or_path)
        else:
            arr = np.ascontiguousarray(image_or_path)
            if arr.dtype != np.uint8:
                raise TypeError("image must be uint8 BGR (HxWx3)")
            status, embedding, response = self._svc.infer_embedding(image_bgr_uint8=arr)
        if status == VisionServiceStatus.OK:
            self._last_response = response
        return status, embedding

    def encode_text(self, text: str) -> Tuple[object, List[float]]:
        _require_ext()
        status, embedding = self._svc.encode_text(text=text)
        return status, embedding

    @staticmethod
    def embedding_similarity(a: List[float], b: List[float]) -> float:
        _require_ext()
        return _ext.VisionService.embedding_similarity(embedding_a=a, embedding_b=b)

    def infer_sequence(
        self,
        pts: npt.NDArray[np.float32],
        image_width: int,
        image_height: int,
    ) -> Tuple[object, List[float]]:
        arr = np.ascontiguousarray(pts, dtype=np.float32)
        status, scores, response = self._svc.infer_sequence(
            pts=arr, image_width=image_width, image_height=image_height
        )
        if status == VisionServiceStatus.OK:
            self._last_response = response
        return status, scores

    def get_class_names(self) -> List[str]:
        return self._svc.get_class_names()

    def get_sequence_class_names(self) -> List[str]:
        """Deprecated alias for :meth:`get_class_names`."""
        return self.get_class_names()

    def get_fall_down_class_index(self) -> int:
        """Read ``fall_down_index`` from model/app yaml when present; otherwise ``-1``."""
        return self._svc.get_fall_down_class_index()

    def draw(
        self,
        image_bgr_uint8: npt.NDArray[np.uint8],
        response: Optional[object] = None,
    ) -> Tuple[object, npt.NDArray[np.uint8]]:
        arr = np.ascontiguousarray(image_bgr_uint8)
        if arr.dtype != np.uint8:
            raise TypeError("image must be uint8 BGR (HxWx3)")
        resp = response if response is not None else self._last_response
        if resp is None:
            raise RuntimeError(
                "draw() requires a prior infer_* call or an explicit response argument"
            )
        return self._svc.draw(image_bgr_uint8=arr, response=resp)

    def supports_draw(self) -> bool:
        return self._svc.supports_draw()

    def release(self) -> None:
        self._svc.release()

    def set_timing_options(self, options: object) -> None:
        self._svc.set_timing_options(options)

    def get_last_timing(self) -> object:
        return self._svc.get_last_timing()

    def get_default_image(self) -> str:
        return self._svc.get_default_image()

    def get_config_path_value(self, config_key: str) -> str:
        return self._svc.get_config_path_value(config_key)

    def last_error(self) -> str:
        return self._svc.last_error()
