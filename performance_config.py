"""
Performance optimization configuration overlay system.

This module provides a non-destructive way to apply performance optimizations
to benchmark implementations while preserving the ability to run standard
apples-to-apples comparisons.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from config import BenchmarkConfig


@dataclass
class PerformanceOptimizations:
    """Container for performance-only optimization settings."""

    # Global performance flags
    enable_performance_mode: bool = False
    use_gpu_if_available: bool = False

    # Faster-whisper specific optimizations (using actual constructor parameters)
    faster_whisper_cpu_threads: int = 4  # Actual parameter: cpu_threads
    faster_whisper_num_workers: int = 2  # Actual parameter: num_workers
    faster_whisper_chunk_length: int = 30  # Actual parameter: chunk_length (transcribe method)
    faster_whisper_gpu_compute_type: str = "float16"

    # Lightning-whisper-mlx specific optimizations
    lightning_batch_size: int = 12
    lightning_enable_quantization: bool = True  # Disabled due to QuantizedLinear compatibility issues on Apple Hardware
    lightning_quantization_type: str = "8bit"
    lightning_use_distilled_models: bool = True

    # MLX-whisper specific optimizations
    mlx_enable_memory_management: bool = True
    mlx_compression_ratio_threshold: float = 2.4
    mlx_no_speech_threshold: float = 0.6
    mlx_explicit_float16: bool = True


class PerformanceConfigOverlay:
    """
    Configuration overlay system that applies performance optimizations
    without modifying the original configuration.
    """

    def __init__(self, base_config: BenchmarkConfig, performance_opts: PerformanceOptimizations):
        self.base_config = base_config
        self.performance_opts = performance_opts
        self._original_values = {}

    def __enter__(self):
        """Enter context - apply performance optimizations."""
        if not self.performance_opts.enable_performance_mode:
            return self.base_config

        # Store original values for restoration
        self._store_original_values()

        # Apply performance optimizations
        self._apply_performance_optimizations()

        return self.base_config

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original configuration."""
        self._restore_original_values()

    def _store_original_values(self):
        """Store original configuration values for restoration."""
        self._original_values = {
            'batch_size': self.base_config.batch_size,
            'device': self.base_config.device,
            'compute_type': self.base_config.compute_type,
            'quant': self.base_config.quant,
        }

    def _restore_original_values(self):
        """Restore original configuration values."""
        for key, value in self._original_values.items():
            setattr(self.base_config, key, value)
        self._original_values.clear()

    def _apply_performance_optimizations(self):
        """Apply performance optimizations to the base configuration."""
        # Apply Lightning-whisper-mlx optimizations
        if hasattr(self.base_config, 'batch_size'):
            self.base_config.batch_size = self.performance_opts.lightning_batch_size

        # Apply GPU optimizations if available and enabled
        if self.performance_opts.use_gpu_if_available:
            try:
                import torch
                if torch.cuda.is_available():
                    self.base_config.device = "cuda"
                    self.base_config.compute_type = self.performance_opts.faster_whisper_gpu_compute_type
            except ImportError:
                # Torch not available, keep CPU settings
                pass

    def get_faster_whisper_performance_kwargs(self) -> Dict[str, Any]:
        """Get performance-optimized kwargs for faster-whisper model loading."""
        if not self.performance_opts.enable_performance_mode:
            return {}

        kwargs = {}

        # These ARE valid WhisperModel constructor parameters in faster-whisper/CTranslate2
        # The error was likely due to parameter conflicts, not invalid parameters

        # Use actual valid WhisperModel constructor parameters
        kwargs.update({
            'cpu_threads': self.performance_opts.faster_whisper_cpu_threads,
            'num_workers': self.performance_opts.faster_whisper_num_workers,
        })

        return kwargs

    def get_faster_whisper_transcribe_kwargs(self) -> Dict[str, Any]:
        """Get performance-optimized kwargs for faster-whisper transcription."""
        if not self.performance_opts.enable_performance_mode:
            return {}

        return {
            # Note: batch_size is not a valid parameter for WhisperModel.transcribe()
            # Only chunk_length is supported for performance optimization
            'chunk_length': self.performance_opts.faster_whisper_chunk_length,
        }

    def get_lightning_whisper_mlx_performance_kwargs(self) -> Dict[str, Any]:
        """Get performance-optimized kwargs for lightning-whisper-mlx initialization."""
        if not self.performance_opts.enable_performance_mode:
            return {}

        kwargs = {'batch_size': self.performance_opts.lightning_batch_size}

        # Add quantization if enabled
        if self.performance_opts.lightning_enable_quantization:
            kwargs['quant'] = self.performance_opts.lightning_quantization_type

        return kwargs

    def get_lightning_model_override(self, model_size: str) -> Optional[str]:
        """Get distilled model override for lightning-whisper-mlx if enabled."""
        if not self.performance_opts.enable_performance_mode:
            return None

        if not self.performance_opts.lightning_use_distilled_models:
            return None

        # CORRECTED: Speed-optimized model mappings that actually improve performance
        speed_mapping = {
            "tiny": "tiny",                    # Already fastest, no override needed
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large-v3-turbo",         # Use turbo (809M, 8x faster) instead of large (1.55B)
            "large-v3": "large-v3-turbo",      # Use turbo for best speed/accuracy trade-off
            "turbo": "large-v3-turbo"          # Direct turbo request
        }

        return speed_mapping.get(model_size)

    def get_mlx_whisper_performance_kwargs(self) -> Dict[str, Any]:
        """Get performance-optimized kwargs for mlx-whisper transcription."""
        if not self.performance_opts.enable_performance_mode:
            return {}

        kwargs = {}

        # Add performance thresholds
        kwargs['compression_ratio_threshold'] = self.performance_opts.mlx_compression_ratio_threshold
        kwargs['no_speech_threshold'] = self.performance_opts.mlx_no_speech_threshold

        return kwargs

    def should_enable_mlx_memory_management(self) -> bool:
        """Check if MLX memory management should be enabled."""
        return (self.performance_opts.enable_performance_mode and
                self.performance_opts.mlx_enable_memory_management)

    def should_use_mlx_float16(self) -> bool:
        """Check if explicit float16 should be used for MLX."""
        return (self.performance_opts.enable_performance_mode and
                self.performance_opts.mlx_explicit_float16)


# Predefined performance optimization profiles
PERFORMANCE_PROFILES = {
    "conservative": PerformanceOptimizations(
        enable_performance_mode=True,
        use_gpu_if_available=False,
        lightning_batch_size=10,
        lightning_enable_quantization=False,
        faster_whisper_cpu_threads=2,
        faster_whisper_num_workers=1,
        faster_whisper_chunk_length=20,
        mlx_enable_memory_management=True,
    ),

    "aggressive": PerformanceOptimizations(
        enable_performance_mode=True,
        use_gpu_if_available=True,
        lightning_batch_size=16,
        lightning_enable_quantization=False,  # Disabled due to Apple Hardware compatibility
        lightning_quantization_type="4bit",
        lightning_use_distilled_models=True,
        faster_whisper_cpu_threads=8,
        faster_whisper_num_workers=4,
        faster_whisper_chunk_length=15,  # Smaller chunks for aggressive processing
        mlx_enable_memory_management=True,
        mlx_explicit_float16=True,
    ),

    "balanced": PerformanceOptimizations(
        enable_performance_mode=True,
        use_gpu_if_available=True,
        lightning_batch_size=12,
        lightning_enable_quantization=False,  # Disabled due to Apple Hardware compatibility
        lightning_quantization_type="8bit",
        faster_whisper_cpu_threads=4,
        faster_whisper_num_workers=2,
        faster_whisper_chunk_length=25,
        mlx_enable_memory_management=True,
    ),

    "baseline": PerformanceOptimizations(
        enable_performance_mode=False,
    )
}


def create_performance_overlay(base_config: BenchmarkConfig,
                             profile_name: str = "baseline") -> PerformanceConfigOverlay:
    """
    Create a performance configuration overlay.

    Args:
        base_config: The base benchmark configuration
        profile_name: Name of the performance profile to use

    Returns:
        PerformanceConfigOverlay instance
    """
    if profile_name not in PERFORMANCE_PROFILES:
        raise ValueError(f"Unknown performance profile: {profile_name}. "
                        f"Available profiles: {list(PERFORMANCE_PROFILES.keys())}")

    performance_opts = PERFORMANCE_PROFILES[profile_name]
    return PerformanceConfigOverlay(base_config, performance_opts)


def list_performance_profiles() -> Dict[str, str]:
    """List available performance profiles with descriptions."""
    return {
        "baseline": "Standard configuration for apples-to-apples comparison",
        "conservative": "Minimal performance improvements, maximum compatibility",
        "balanced": "Good balance of performance and stability",
        "aggressive": "Maximum performance optimizations, may require more resources"
    }
