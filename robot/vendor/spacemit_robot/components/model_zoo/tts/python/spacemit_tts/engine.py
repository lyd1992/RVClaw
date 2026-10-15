"""
TTS Engine - Main synthesis interface

Provides Pythonic wrappers for the C++ TTS engine.
"""

from enum import Enum
from typing import Optional, Union
from pathlib import Path
import numpy as np

try:
    from . import _spacemit_tts as _tts
except ImportError:
    raise ImportError(
        "_spacemit_tts module not found. Build Python bindings with:\n"
        "  cd tts/build && cmake .. -DBUILD_PYTHON_BINDINGS=ON && make -j"
    ) from None


# =============================================================================
# Enumerations
# =============================================================================

class BackendType(Enum):
    """TTS backend types"""
    MATCHA_ZH = _tts.BackendType.MATCHA_ZH
    """Matcha Chinese (22050Hz)"""

    MATCHA_EN = _tts.BackendType.MATCHA_EN
    """Matcha English (22050Hz)"""

    MATCHA_ZH_EN = _tts.BackendType.MATCHA_ZH_EN
    """Matcha Chinese-English bilingual (16000Hz)"""

    COSYVOICE = _tts.BackendType.COSYVOICE
    """CosyVoice (reserved)"""

    VITS = _tts.BackendType.VITS
    """VITS (reserved)"""

    PIPER = _tts.BackendType.PIPER
    """Piper TTS (reserved)"""

    KOKORO = _tts.BackendType.KOKORO
    """Kokoro TTS (reserved)"""

    def to_native(self):
        """Convert to native C++ enum value"""
        return self.value


class AudioFormat(Enum):
    """Audio output formats"""
    PCM = _tts.AudioFormat.PCM
    """Raw PCM data"""

    WAV = _tts.AudioFormat.WAV
    """WAV file format"""

    MP3 = _tts.AudioFormat.MP3
    """MP3 format (reserved)"""

    OGG = _tts.AudioFormat.OGG
    """OGG format (reserved)"""

    def to_native(self):
        """Convert to native C++ enum value"""
        return self.value


# =============================================================================
# Config - Configuration wrapper
# =============================================================================

class Config:
    """
    TTS Configuration

    Example:
        >>> config = Config.preset("matcha_zh_en")
        >>> config.speech_rate = 1.2
        >>> config.volume = 80
    """

    def __init__(self, backend: BackendType = BackendType.MATCHA_ZH_EN,
                 model_dir: str = "~/.cache/models/tts/matcha-tts",
                 provider: str = "auto"):
        """
        Create TTS configuration

        Args:
            backend: Backend type (default: MATCHA_ZH_EN)
            model_dir: Model directory path
            provider: Inference provider policy: auto, cpu, spacemit
        """
        self._config = _tts.TtsConfig()
        self._config.backend = backend.to_native()
        self._config.model_dir = str(Path(model_dir).expanduser())
        self._config.provider = provider

    @staticmethod
    def preset(name: str) -> "Config":
        """Create configuration from preset name (e.g. 'matcha_zh', 'matcha_en', 'matcha_zh_en', 'kokoro')"""
        config = Config.__new__(Config)
        config._config = _tts.TtsConfig.preset(name)
        return config

    @staticmethod
    def available_presets() -> list:
        """Get list of available preset names"""
        return _tts.TtsConfig.available_presets()

    # Properties for convenient access
    @property
    def speech_rate(self) -> float:
        """Speech rate (>1.0 fast, <1.0 slow)"""
        return self._config.speech_rate

    @speech_rate.setter
    def speech_rate(self, value: float):
        self._config.speech_rate = value

    @property
    def volume(self) -> int:
        """Volume [0, 100]"""
        return self._config.volume

    @volume.setter
    def volume(self, value: int):
        self._config.volume = value

    @property
    def speaker_id(self) -> int:
        """Speaker ID (multi-speaker models)"""
        return self._config.speaker_id

    @speaker_id.setter
    def speaker_id(self, value: int):
        self._config.speaker_id = value

    @property
    def sample_rate(self) -> int:
        """Output sample rate (Hz)"""
        return self._config.sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        self._config.sample_rate = value

    @property
    def pitch(self) -> float:
        """Pitch adjustment"""
        return self._config.pitch

    @pitch.setter
    def pitch(self, value: float):
        self._config.pitch = value

    @property
    def provider(self) -> str:
        """Provider for all models: auto, cpu, spacemit"""
        return self._config.provider

    @provider.setter
    def provider(self, value: str):
        self._config.provider = value

    # Builder methods (chainable)
    def with_speed(self, speed: float) -> "Config":
        """
        Set speech rate (chainable)

        Args:
            speed: Speech rate (>1.0 fast, <1.0 slow)

        Returns:
            Self for chaining
        """
        self._config = self._config.withSpeed(speed)
        return self

    def with_speaker(self, speaker_id: int) -> "Config":
        """
        Set speaker ID (chainable)

        Args:
            speaker_id: Speaker ID

        Returns:
            Self for chaining
        """
        self._config = self._config.withSpeaker(speaker_id)
        return self

    def with_volume(self, volume: int) -> "Config":
        """
        Set volume (chainable)

        Args:
            volume: Volume [0, 100]

        Returns:
            Self for chaining
        """
        self._config = self._config.withVolume(volume)
        return self

    def with_provider(self, provider: str) -> "Config":
        """Set inference provider policy: auto, cpu, spacemit"""
        self._config = self._config.withProvider(provider)
        return self

    def __repr__(self) -> str:
        return (f"<Config backend={self._config.backend} "
                f"provider={self.provider} "
                f"sample_rate={self.sample_rate}Hz "
                f"speech_rate={self.speech_rate:.2f}>")


# =============================================================================
# Result - Synthesis result wrapper
# =============================================================================

class Result:
    """
    Synthesis result wrapper

    Provides convenient access to audio data and metadata.

    Example:
        >>> result = engine.synthesize("Hello")
        >>> print(f"Duration: {result.duration_ms}ms, RTF: {result.rtf:.3f}")
        >>> result.save("output.wav")
    """

    def __init__(self, native_result):
        """
        Wrap native C++ result

        Args:
            native_result: C++ TtsEngineResult object
        """
        self._result = native_result

    @property
    def audio_float(self) -> np.ndarray:
        """
        Audio as float32 numpy array

        Returns:
            NumPy array with values in [-1.0, 1.0]
        """
        return np.array(self._result.get_audio_float(), dtype=np.float32)

    @property
    def audio_int16(self) -> np.ndarray:
        """
        Audio as int16 numpy array

        Returns:
            NumPy array with 16-bit PCM values
        """
        return np.array(self._result.get_audio_int16(), dtype=np.int16)

    @property
    def audio_bytes(self) -> bytes:
        """
        Audio as raw bytes

        Returns:
            Raw audio data as bytes
        """
        return bytes(self._result.get_audio_data())

    @property
    def sample_rate(self) -> int:
        """Sample rate (Hz)"""
        return self._result.get_sample_rate()

    @property
    def duration_ms(self) -> int:
        """Audio duration (milliseconds)"""
        return self._result.get_duration_ms()

    @property
    def processing_time_ms(self) -> int:
        """Processing time (milliseconds)"""
        return self._result.get_processing_time_ms()

    @property
    def rtf(self) -> float:
        """
        Real-Time Factor

        Ratio of processing time to audio duration.
        Values < 1.0 indicate faster than real-time.

        Returns:
            RTF value (processing_time / audio_duration)
        """
        return self._result.get_rtf()

    @property
    def is_success(self) -> bool:
        """Check if synthesis succeeded"""
        return self._result.is_success()

    @property
    def message(self) -> str:
        """Error message (if any)"""
        return self._result.get_message()

    @property
    def request_id(self) -> str:
        """Request ID"""
        return self._result.get_request_id()

    def save(self, file_path: Union[str, Path]) -> bool:
        """
        Save audio to WAV file

        Args:
            file_path: Output file path

        Returns:
            True if successful
        """
        return self._result.save_to_file(str(file_path))

    def __bool__(self) -> bool:
        """Check if result has audio content"""
        return not self._result.is_empty()

    def __repr__(self) -> str:
        status = "success" if self.is_success else "failed"
        return (f"<Result {status} duration={self.duration_ms}ms "
                f"rtf={self.rtf:.3f}>")


# =============================================================================
# Engine - Main TTS engine
# =============================================================================

class Engine:
    """
    TTS Engine - Main text-to-speech interface

    Example:
        >>> engine = Engine()
        >>> result = engine.synthesize("你好世界")
        >>> result.save("output.wav")

    Context manager:
        >>> with Engine() as engine:
        ...     result = engine.synthesize("你好")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Create TTS engine

        Args:
            config: Configuration object (default: Chinese Matcha)
        """
        if config is None:
            config = Config()
        self._engine = _tts.TtsEngine(config._config)
        self._config = config

    def _require_engine(self):
        engine = getattr(self, "_engine", None)
        if engine is None:
            raise RuntimeError("TTS engine is closed")
        return engine

    def close(self) -> None:
        """Release native backend resources held by this engine."""
        engine = getattr(self, "_engine", None)
        if engine is None:
            return
        engine.shutdown()
        self._engine = None

    shutdown = close

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def synthesize(self, text: str) -> Result:
        """
        Synthesize text (blocking)

        Releases Python GIL during synthesis, allowing other threads to run.

        Args:
            text: Text to synthesize

        Returns:
            Synthesis result

        Example:
            >>> result = engine.synthesize("你好世界")
            >>> print(f"Generated {result.duration_ms}ms audio")
        """
        native_result = self._require_engine().call(text)
        return Result(native_result)

    def synthesize_to_file(self, text: str, file_path: Union[str, Path]) -> bool:
        """
        Synthesize text and save to file

        Args:
            text: Text to synthesize
            file_path: Output file path

        Returns:
            True if successful

        Example:
            >>> engine.synthesize_to_file("你好", "output.wav")
        """
        return self._require_engine().call_to_file(text, str(file_path))

    def synthesize_streaming(self, text: str, callback):
        """
        Streaming synthesis with callback

        Args:
            text: Text to synthesize
            callback: TtsCallback instance

        Example:
            >>> class MyCallback(TtsCallback):
            ...     def on_event(self, result):
            ...         print(f"Got audio: {result.duration_ms}ms")
            >>>
            >>> engine.synthesize_streaming("你好", MyCallback())
        """
        from .callback import TtsCallback
        if isinstance(callback, TtsCallback):
            native_cb = callback._get_native_callback()
            self._require_engine().streaming_call(text, native_cb)
        else:
            raise TypeError("callback must be an instance of TtsCallback")

    def set_speed(self, speed: float):
        """
        Set speech rate

        Args:
            speed: Speech rate (>1.0 = faster, <1.0 = slower)

        Example:
            >>> engine.set_speed(1.5)  # 50% faster
        """
        self._require_engine().set_speed(speed)

    def set_speaker(self, speaker_id: int):
        """
        Set speaker ID (for multi-speaker models)

        Args:
            speaker_id: Speaker ID
        """
        self._require_engine().set_speaker(speaker_id)

    def set_volume(self, volume: int):
        """
        Set volume

        Args:
            volume: Volume [0, 100]
        """
        self._require_engine().set_volume(volume)

    def update_lexicon(self, entries: list):
        """
        Update custom pronunciation lexicon

        Args:
            entries: List of dicts with 'word', 'phoneme', and optional 'locale' keys,
                     or list of PronunciationEntry objects.

                     locale='zh' (default): phoneme is space-separated pinyin with
                         tone numbers 1-4, e.g. "wei4 ni3".
                     locale='en' (matcha:en or matcha:zh-en): phoneme is
                         English word(s), rendered by espeak-ng, e.g. "space meet".

        Example:
            >>> # Chinese polyphone correction: force '为你' to wei4 ni3
            >>> engine.update_lexicon([
            ...     {"word": "为你", "phoneme": "wei4 ni3"},
            ... ])
            >>> # English word (matcha:en or matcha:zh-en): render via espeak
            >>> engine.update_lexicon([
            ...     {"word": "SpaceMIT", "phoneme": "space meet", "locale": "en"},
            ... ])
        """
        native_entries = []
        for e in entries:
            if isinstance(e, dict):
                entry = _tts.PronunciationEntry(
                    e["word"], e["phoneme"], e.get("locale", "zh"))
            else:
                entry = e
            native_entries.append(entry)
        self._require_engine().update_lexicon(native_entries)

    @property
    def config(self) -> Config:
        """Get current configuration"""
        native_config = self._require_engine().get_config()
        config = Config.__new__(Config)
        config._config = native_config
        return config

    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        engine = getattr(self, "_engine", None)
        if engine is None:
            return False
        return engine.is_initialized()

    @property
    def engine_name(self) -> str:
        """Get engine type name"""
        return self._require_engine().get_engine_name()

    @property
    def sample_rate(self) -> int:
        """Get output sample rate (Hz)"""
        return self._require_engine().get_sample_rate()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        if getattr(self, "_engine", None) is None:
            return "<Engine closed>"
        return (f"<Engine backend={self.engine_name} "
                f"sample_rate={self.sample_rate}Hz "
                f"initialized={self.is_initialized}>")
