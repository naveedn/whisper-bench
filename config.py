"""
Global configuration for Whisper benchmark to ensure apples-to-apples comparison.

This module defines consistent settings across all Whisper implementations
to ensure fair performance and quality comparisons.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Global configuration for all benchmark implementations."""

    # Model settings
    model_size: str = "base"

    # Audio processing settings
    # These should be consistent across all implementations where possible
    language: Optional[str] = None  # Auto-detect language
    task: str = "transcribe"  # vs "translate"

    # Transcription quality settings
    beam_size: int = 5  # For beam search decoding
    best_of: int = 5  # Number of candidates to consider

    # Audio preprocessing
    # Note: Some implementations may not support all of these
    temperature: float = 0.0  # Deterministic output (0.0) vs creative (1.0)
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # Performance settings
    batch_size: int = 1  # Keep small for memory consistency

    # Output settings
    return_timestamps: bool = True
    word_timestamps: bool = False  # May not be supported by all implementations

    # Device settings (where applicable)
    device: str = "auto"  # Let each implementation choose optimal device
    compute_type: str = "int8"  # For faster-whisper

    # Silence detection (for implementations that support it)
    vad_filter: bool = False  # Voice Activity Detection
    vad_threshold: float = 0.5

    def get_openai_whisper_kwargs(self) -> dict:
        """Get kwargs specific to OpenAI Whisper."""
        return {
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "word_timestamps": self.word_timestamps,
        }

    def get_faster_whisper_kwargs(self) -> dict:
        """Get kwargs specific to faster-whisper."""
        return {
            "language": self.language,
            "task": self.task,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "vad_threshold": self.vad_threshold,
        }

    def get_mlx_whisper_kwargs(self) -> dict:
        """Get kwargs specific to mlx-whisper."""
        return {
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "word_timestamps": self.word_timestamps,
        }

    def get_lightning_whisper_mlx_kwargs(self) -> dict:
        """Get kwargs specific to lightning-whisper-mlx."""
        # Note: API may be different, adjust based on actual implementation
        return {
            "language": self.language,
            "task": self.task,
            "temperature": self.temperature,
            # Add other parameters as supported by lightning-whisper-mlx
        }


# Default global configuration
DEFAULT_CONFIG = BenchmarkConfig()


def get_config() -> BenchmarkConfig:
    """Get the default benchmark configuration."""
    return DEFAULT_CONFIG


def update_config(**kwargs) -> BenchmarkConfig:
    """Update the global configuration with new values."""
    global DEFAULT_CONFIG
    for key, value in kwargs.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
    return DEFAULT_CONFIG
