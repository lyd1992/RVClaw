"""
spacemit_tts - SpacemiT TTS Python Package

A Pythonic interface to the SpacemiT text-to-speech engine.

Basic Usage:
    >>> import spacemit_tts
    >>>
    >>> # Quick synthesis
    >>> engine = spacemit_tts.Engine()
    >>> result = engine.synthesize("你好世界")
    >>> result.save("output.wav")
    >>>
    >>> # With configuration
    >>> config = spacemit_tts.Config.preset("matcha_zh_en")
    >>> engine = spacemit_tts.Engine(config)
    >>>
    >>> # Streaming synthesis
    >>> class MyCallback(spacemit_tts.TtsCallback):
    ...     def on_event(self, result):
    ...         print(f"Got audio: {result.duration_ms}ms")
    >>>
    >>> engine.synthesize_streaming("你好", MyCallback())

Advanced Features:
    - Multi-language support (Chinese, English, bilingual)
    - Streaming synthesis with callbacks
    - Adjustable speed, volume, speaker
    - NumPy audio array output
    - Real-time factor (RTF) monitoring
"""

from .engine import Engine, Config, Result, BackendType, AudioFormat
from .callback import TtsCallback, PrintCallback, SaveCallback, CollectCallback
from .utils import synthesize, synthesize_to_file

__version__ = "1.0.4"
__author__ = "muggle"
__all__ = [
    "Engine",
    "Config",
    "Result",
    "BackendType",
    "AudioFormat",
    "TtsCallback",
    "PrintCallback",
    "SaveCallback",
    "CollectCallback",
    "synthesize",
    "synthesize_to_file",
]
