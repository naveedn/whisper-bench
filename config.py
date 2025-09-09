"""
Global configuration for Whisper benchmark to ensure apples-to-apples comparison.

This module defines consistent settings across all Whisper implementations
to ensure fair performance and quality comparisons.

Settings based on optimal configuration for each implementation:
- OpenAI Whisper: beam_size=5, temperature=0, word_timestamps=True, condition_on_previous_text=False
- MLX Whisper: beam_size=5, temperature=0, word_timestamps=True, condition_on_previous_text=False, initial_prompt=None
- Lightning Whisper MLX: beam_size=5, temperature=0, word_timestamps=True, condition_on_previous_text=False, batch_size=8, quant=8bit
- faster-whisper: beam_size=5, temperature=0, word_timestamps=True, condition_on_previous_text=False, vad_filter=False
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Global configuration for all benchmark implementations."""

    # Model settings
    model_size: str = "base"  # Will be mapped to implementation-specific models

    # CRITICAL: Core transcription settings for TRUE apples-to-apples comparison
    beam_size: int = 5  # Beam search width - MUST be consistent across ALL models
    temperature: int = 0  # Deterministic output (0 = deterministic) - MUST be consistent
    word_timestamps: bool = True  # Word-level timing - MUST be consistent across ALL models
    condition_on_previous_text: bool = False  # Don't use previous context - MUST be consistent

    # Model-specific settings that should be None/disabled for fair comparison
    initial_prompt: Optional[str] = None  # For mlx-whisper (no rolling prompt)

    # Audio processing settings (less critical but should be consistent)
    language: Optional[str] = None  # Auto-detect language
    task: str = "transcribe"  # vs "translate"

    # Performance settings - implementation-specific optimizations allowed
    batch_size: int = 8  # Lightning Whisper MLX optimal setting

    # Device settings - standardized for fair comparison
    device: str = "cpu"  # Use CPU for consistent comparison across implementations
    compute_type: str = "int8"  # For faster-whisper quantization
    quant: str = "8bit"  # For lightning-whisper-mlx quantization

    # VAD settings - DISABLED for fair comparison
    vad_filter: bool = False  # Voice Activity Detection - OFF for consistency

    # Output settings
    return_timestamps: bool = True

    # Model repository mappings for MLX implementations
    mlx_model_mapping = {
        "base": "mlx-community/whisper-base-mlx-q4",
        "small": "mlx-community/whisper-small-mlx-q4",
        "medium": "mlx-community/whisper-medium-mlx-q4",
        "large": "mlx-community/whisper-large-v2-mlx-q4",
        "large-v3": "mlx-community/whisper-large-v3-mlx-q4"
    }

    # Lightning model mapping
    lightning_model_mapping = {
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v2",
        "large-v3": "distil-large-v3"  # As shown in your config
    }

    def get_openai_whisper_kwargs(self) -> dict:
        """Get kwargs for OpenAI Whisper - matches your provided config exactly."""
        return {
            "beam_size": self.beam_size,  # 5
            "temperature": self.temperature,  # 0
            "word_timestamps": self.word_timestamps,  # True
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            # Optional additional settings
            "language": self.language,
            "task": self.task,
        }

    def get_faster_whisper_kwargs(self) -> dict:
        """Get kwargs for faster-whisper - matches your provided config exactly."""
        return {
            "beam_size": self.beam_size,  # 5
            "temperature": self.temperature,  # 0
            "word_timestamps": self.word_timestamps,  # True
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            "vad_filter": self.vad_filter,  # False - use your own VAD + chunking
            # Optional additional settings
            "language": self.language,
            "task": self.task,
        }

    def get_mlx_whisper_kwargs(self) -> dict:
        """Get kwargs for mlx-whisper - matches your provided config exactly."""
        return {
            "word_timestamps": self.word_timestamps,  # True
            "beam_size": self.beam_size,  # 5
            "temperature": self.temperature,  # 0
            "initial_prompt": self.initial_prompt,  # None - add rolling prompt if needed
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            # Optional additional settings
            "language": self.language,
            "task": self.task,
        }

    def get_lightning_whisper_mlx_kwargs(self) -> dict:
        """Get kwargs for lightning-whisper-mlx - matches your provided config exactly."""
        return {
            "beam_size": self.beam_size,  # 5
            "temperature": self.temperature,  # 0
            "word_timestamps": self.word_timestamps,  # True
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            # Optional additional settings
            "language": self.language,
        }

    def get_lightning_whisper_mlx_init_kwargs(self) -> dict:
        """Get initialization kwargs for lightning-whisper-mlx."""
        return {
            "model": self.lightning_model_mapping.get(self.model_size, "base"),
            "batch_size": self.batch_size,  # 8 - adjust for memory
            "quant": self.quant  # "8bit"
        }

    def get_mlx_model_repo(self) -> str:
        """Get the MLX model repository path for the current model size."""
        return self.mlx_model_mapping.get(self.model_size, "mlx-community/whisper-base-mlx-q4")


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


def print_config_summary():
    """Print a summary of the current configuration for verification."""
    config = get_config()
    print("🔧 Benchmark Configuration Summary:")
    print("=" * 50)
    print("CRITICAL CONSISTENCY SETTINGS:")
    print(f"   beam_size: {config.beam_size}")
    print(f"   temperature: {config.temperature}")
    print(f"   word_timestamps: {config.word_timestamps}")
    print(f"   condition_on_previous_text: {config.condition_on_previous_text}")
    print(f"   vad_filter: {config.vad_filter}")
    print()
    print("MODEL-SPECIFIC SETTINGS:")
    print(f"   device: {config.device}")
    print(f"   compute_type: {config.compute_type}")
    print(f"   batch_size: {config.batch_size}")
    print(f"   quant: {config.quant}")
    print()
    print("MODEL MAPPINGS:")
    print(f"   MLX model: {config.get_mlx_model_repo()}")
    print(f"   Lightning model: {config.lightning_model_mapping.get(config.model_size, 'base')}")
    print("=" * 50)
