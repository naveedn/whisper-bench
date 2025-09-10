"""
Global configuration for Whisper benchmark to ensure apples-to-apples comparison.

This module defines consistent settings across all Whisper implementations
to ensure fair performance and quality comparisons.

CRITICAL: All models use greedy decoding (beam_size=1) for true apples-to-apples comparison:
- OpenAI Whisper: beam_size=1, temperature=0, word_timestamps=True, condition_on_previous_text=False
- faster-whisper: beam_size=1, temperature=0, word_timestamps=True, condition_on_previous_text=False, vad_filter=False
- MLX Whisper: greedy decoding only (no beam search support), temperature=0, word_timestamps=True, condition_on_previous_text=False
- Lightning Whisper MLX: greedy decoding only (no beam search support), batch_size=8, no quant
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Global configuration for all benchmark implementations."""

    # Model settings
    model_size: str = "base"  # Will be mapped to implementation-specific models

    # CRITICAL: Core transcription settings for TRUE apples-to-apples comparison
    beam_size: int = 1  # Greedy decoding for consistency - mlx-whisper and lightning-whisper-mlx only support greedy
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

    # Lightning model mapping - CORRECTED: Proper size progression for performance
    lightning_model_mapping = {
        "tiny": "tiny",                    # 39M params - Maximum speed option
        "base": "base",                    # 74M params - FIXED: Actually smaller than large!
        "small": "small",                  # 244M params - Proper small model
        "medium": "medium",                # 769M params - Standard medium
        "large": "large-v2",               # 1.55B params - Stable large model
        "large-v3": "large-v3",            # 1.55B params - Latest large model
        "turbo": "large-v3-turbo",         # 809M params - NEW: 8x speed improvement
    }

    # VAD timestamp processing settings
    use_vad_timestamps: bool = False  # Enable VAD-based segment processing
    vad_timestamps_file: Optional[str] = None  # Path to VAD timestamps JSON file
    max_segment_duration: float = 30.0  # Maximum segment length in seconds (for chunking long segments)
    segment_padding: float = 0.1  # Padding around VAD segments in seconds

    def get_openai_whisper_kwargs(self) -> dict:
        """Get kwargs for OpenAI Whisper - using greedy decoding for consistency with MLX models."""
        return {
            "beam_size": self.beam_size,  # 1 - Greedy decoding for apples-to-apples comparison
            "temperature": self.temperature,  # 0
            "word_timestamps": self.word_timestamps,  # True
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            # Optional additional settings
            "language": self.language,
            "task": self.task,
        }

    def get_faster_whisper_kwargs(self) -> dict:
        """Get kwargs for faster-whisper - using greedy decoding for consistency with MLX models."""
        return {
            "beam_size": self.beam_size,  # 1 - Greedy decoding for apples-to-apples comparison
            "temperature": self.temperature,  # 0
            "word_timestamps": self.word_timestamps,  # True
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            "vad_filter": self.vad_filter,  # False - use your own VAD + chunking
            # Optional additional settings
            "language": self.language,
            "task": self.task,
        }

    def get_mlx_whisper_kwargs(self) -> dict:
        """Get kwargs for mlx-whisper - beam search not supported, uses greedy decoding."""
        return {
            "word_timestamps": self.word_timestamps,  # True
            "temperature": self.temperature,  # 0
            "initial_prompt": self.initial_prompt,  # None - add rolling prompt if needed
            "condition_on_previous_text": self.condition_on_previous_text,  # False
            # Optional additional settings
            "language": self.language,
            "task": self.task,
            # beam_size NOT supported - mlx-whisper uses greedy decoding only
        }

    def get_lightning_whisper_mlx_kwargs(self) -> dict:
        """Get kwargs for lightning-whisper-mlx - only supports language parameter in transcribe()."""
        return {
            # lightning-whisper-mlx.transcribe() only accepts: audio_path, language=None
            # All other parameters (beam_size, temperature, word_timestamps, etc.) are NOT supported
            "language": self.language,
        }

    def get_lightning_whisper_mlx_init_kwargs(self) -> dict:
        """Get initialization kwargs for lightning-whisper-mlx."""
        return {
            "model": self.lightning_model_mapping.get(self.model_size, "base"),
            "batch_size": self.batch_size,  # 8 - adjust for memory
            # "quant": self.quant  # "8bit" - Disabled due to QuantizedLinear compatibility issue
        }

    def get_mlx_model_repo(self) -> str:
        """Get the MLX model repository path for the current model size."""
        return self.mlx_model_mapping.get(self.model_size, "mlx-community/whisper-base-mlx-q4")

    def load_vad_timestamps(self) -> list:
        """Load VAD timestamps from JSON file."""
        if not self.use_vad_timestamps or not self.vad_timestamps_file:
            return []

        import json
        from pathlib import Path

        vad_file = Path(self.vad_timestamps_file)
        if not vad_file.exists():
            print(f"Warning: VAD timestamps file not found: {vad_file}")
            return []

        with open(vad_file, 'r') as f:
            timestamps = json.load(f)

        # Add padding to segments
        for segment in timestamps:
            segment['start_sec'] = max(0, segment['start_sec'] - self.segment_padding)
            segment['end_sec'] = segment['end_sec'] + self.segment_padding
            segment['duration_sec'] = segment['end_sec'] - segment['start_sec']

        return timestamps


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
