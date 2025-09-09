"""Benchmark implementations for different Whisper models."""

from .base import BenchmarkResult, BaseBenchmark
from .openai_whisper import OpenAIWhisperBenchmark
from .faster_whisper import FasterWhisperBenchmark
from .mlx_whisper import MLXWhisperBenchmark
from .lightning_whisper_mlx import LightningWhisperMLXBenchmark

__all__ = [
    "BenchmarkResult",
    "BaseBenchmark",
    "OpenAIWhisperBenchmark",
    "FasterWhisperBenchmark",
    "MLXWhisperBenchmark",
    "LightningWhisperMLXBenchmark"
]
