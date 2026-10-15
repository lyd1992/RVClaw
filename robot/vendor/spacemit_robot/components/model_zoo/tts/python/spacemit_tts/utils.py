"""
Utility functions for quick TTS operations

Provides convenient one-line functions for common tasks.
"""

from typing import Union
from pathlib import Path

from .engine import Engine, Config, BackendType, Result


def synthesize(text: str,
               backend: BackendType = BackendType.MATCHA_ZH_EN,
               model_dir: str = "~/.cache/models/tts/matcha-tts") -> Result:
    """
    Quick text-to-speech synthesis

    Creates an engine, synthesizes text, and returns the result.

    Args:
        text: Text to synthesize
        backend: TTS backend type (default: MATCHA_ZH_EN)
        model_dir: Model directory path

    Returns:
        Synthesis result

    Example:
        >>> result = synthesize("你好世界")
        >>> result.save("output.wav")
        >>>
        >>> # English synthesis
        >>> result = synthesize("Hello world", backend=BackendType.MATCHA_EN)
    """
    config = Config(backend=backend, model_dir=model_dir)
    engine = Engine(config)
    return engine.synthesize(text)


def synthesize_to_file(text: str,
                       file_path: Union[str, Path],
                       backend: BackendType = BackendType.MATCHA_ZH_EN,
                       model_dir: str = "~/.cache/models/tts/matcha-tts") -> bool:
    """
    Quick synthesis directly to file

    Creates an engine, synthesizes text, and saves to file.

    Args:
        text: Text to synthesize
        file_path: Output file path
        backend: TTS backend type (default: MATCHA_ZH_EN)
        model_dir: Model directory path

    Returns:
        True if successful

    Example:
        >>> synthesize_to_file("你好世界", "output.wav")
        True
        >>>
        >>> # Bilingual synthesis
        >>> synthesize_to_file("你好 Hello", "output.wav",
        ...                    backend=BackendType.MATCHA_ZH_EN)
    """
    config = Config(backend=backend, model_dir=model_dir)
    engine = Engine(config)
    return engine.synthesize_to_file(text, file_path)


def synthesize_batch(texts: list[str],
                     output_dir: Union[str, Path],
                     backend: BackendType = BackendType.MATCHA_ZH_EN,
                     model_dir: str = "~/.cache/models/tts/matcha-tts",
                     prefix: str = "output") -> list[Path]:
    """
    Synthesize multiple texts to files

    Args:
        texts: List of texts to synthesize
        output_dir: Output directory
        backend: TTS backend type
        model_dir: Model directory path
        prefix: Filename prefix

    Returns:
        List of output file paths

    Example:
        >>> texts = ["你好", "世界", "测试"]
        >>> files = synthesize_batch(texts, "output")
        >>> print(files)
        [Path('output/output_000.wav'), Path('output/output_001.wav'), ...]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(backend=backend, model_dir=model_dir)
    engine = Engine(config)

    output_files = []
    for i, text in enumerate(texts):
        filename = output_dir / f"{prefix}_{i:03d}.wav"
        success = engine.synthesize_to_file(text, filename)
        if success:
            output_files.append(filename)
        else:
            print(f"Warning: Failed to synthesize text {i}: {text[:50]}...")

    return output_files
