"""
TTS Callback - Base class for streaming callbacks

Provides callback interface for streaming TTS synthesis.
"""

from abc import ABC
from pathlib import Path
from typing import Any

try:
    from . import _spacemit_tts as _tts
except ImportError:
    _tts = None


# =============================================================================
# TtsCallback - Base callback class
# =============================================================================

class TtsCallback(ABC):
    """
    Base class for TTS streaming callbacks

    Callback chain:
        on_open() → on_event() [multiple] → on_complete() → on_close()

    Error chain:
        on_open() → ... → on_error() → on_close()

    Example:
        >>> class MyCallback(TtsCallback):
        ...     def on_event(self, result):
        ...         print(f"Got {result.duration_ms}ms audio")
        ...         result.save(f"chunk_{self.count}.wav")
        ...         self.count += 1
        >>>
        >>> callback = MyCallback()
        >>> engine.synthesize_streaming("你好世界", callback)
    """

    def __init__(self):
        self._native_callback = None

    def on_open(self) -> None:
        """
        Called when synthesis session starts

        Override this to initialize resources.
        """
        pass

    def on_event(self, result: Any) -> None:
        """
        Called when audio chunk is ready

        This is the main callback where you receive synthesized audio.

        Args:
            result: TtsEngineResult with audio data

        Note:
            - In streaming mode, this is called for each completed sentence
            - Don't perform blocking operations here as it may slow down synthesis
        """
        pass

    def on_complete(self) -> None:
        """
        Called when synthesis completes successfully

        Override this to finalize processing.
        """
        pass

    def on_error(self, message: str) -> None:
        """
        Called on error

        Args:
            message: Error description
        """
        pass

    def on_close(self) -> None:
        """
        Called when session closes

        This is always called last, regardless of success or error.
        Override this to clean up resources.
        """
        pass

    def _get_native_callback(self):
        """
        Get native C++ callback wrapper (internal use)

        Returns:
            Native callback object for C++ engine
        """
        if _tts is None:
            raise ImportError("_spacemit_tts module not found")

        if self._native_callback is None:
            self._native_callback = _tts.TtsCallback()
            self._native_callback.on_open(self.on_open)
            self._native_callback.on_event(self.on_event)
            self._native_callback.on_complete(self.on_complete)
            self._native_callback.on_error(self.on_error)
            self._native_callback.on_close(self.on_close)

        return self._native_callback


# =============================================================================
# PrintCallback - Debug callback
# =============================================================================

class PrintCallback(TtsCallback):
    """
    Debug callback that prints events to console

    Useful for monitoring synthesis progress.

    Example:
        >>> callback = PrintCallback()
        >>> engine.synthesize_streaming("你好世界", callback)
        [TTS] Session opened
        [TTS] Audio: 2304ms, RTF=0.023
        [TTS] Audio: 1856ms, RTF=0.021
        [TTS] Synthesis completed
        [TTS] Session closed
    """

    def __init__(self, prefix: str = "[TTS]"):
        """
        Create print callback

        Args:
            prefix: Prefix for log messages
        """
        super().__init__()
        self.prefix = prefix

    def on_open(self):
        print(f"{self.prefix} Session opened")

    def on_event(self, result):
        print(f"{self.prefix} Audio: {result.get_duration_ms()}ms, "
              f"RTF={result.get_rtf():.3f}")

    def on_complete(self):
        print(f"{self.prefix} Synthesis completed")

    def on_error(self, message: str):
        print(f"{self.prefix} Error: {message}")

    def on_close(self):
        print(f"{self.prefix} Session closed")


# =============================================================================
# SaveCallback - File saving callback
# =============================================================================

class SaveCallback(TtsCallback):
    """
    Callback that saves audio chunks to files

    Each chunk is saved as a separate WAV file.

    Example:
        >>> callback = SaveCallback(output_dir="output")
        >>> engine.synthesize_streaming("你好世界，今天天气很好", callback)
        # Creates output/chunk_000.wav, output/chunk_001.wav, etc.
    """

    def __init__(self, output_dir: str = ".", prefix: str = "chunk"):
        """
        Create save callback

        Args:
            output_dir: Directory to save chunks
            prefix: Filename prefix
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.chunk_count = 0

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_event(self, result):
        """Save audio chunk to file"""
        filename = self.output_dir / f"{self.prefix}_{self.chunk_count:03d}.wav"
        success = result.save_to_file(str(filename))
        if success:
            print(f"Saved: {filename} ({result.get_duration_ms()}ms)")
        else:
            print(f"Failed to save: {filename}")
        self.chunk_count += 1

    def on_complete(self):
        print(f"Saved {self.chunk_count} chunks to {self.output_dir}")


# =============================================================================
# CollectCallback - Collect all results
# =============================================================================

class CollectCallback(TtsCallback):
    """
    Callback that collects all results in a list

    Useful for gathering all audio chunks for further processing.

    Example:
        >>> callback = CollectCallback()
        >>> engine.synthesize_streaming("你好世界", callback)
        >>> for i, result in enumerate(callback.results):
        ...     print(f"Chunk {i}: {result.duration_ms}ms")
    """

    def __init__(self):
        """Create collect callback"""
        super().__init__()
        self.results = []
        self.error_message = None

    def on_event(self, result):
        """Collect result"""
        self.results.append(result)

    def on_error(self, message: str):
        """Store error message"""
        self.error_message = message

    def get_total_duration_ms(self) -> int:
        """
        Get total duration of all collected audio

        Returns:
            Total duration in milliseconds
        """
        return sum(r.get_duration_ms() for r in self.results)

    def __len__(self) -> int:
        """Number of collected results"""
        return len(self.results)

    def __repr__(self) -> str:
        return f"<CollectCallback results={len(self.results)}>"
