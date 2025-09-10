"""
OpenAI Whisper benchmark implementation.
"""

import time
from .base import BaseBenchmark


class OpenAIWhisperBenchmark(BaseBenchmark):
    """Benchmark implementation for OpenAI Whisper."""

    def __init__(self, config, output_dir, performance_overlay=None):
        super().__init__(config, output_dir, performance_overlay)
        # OpenAI Whisper doesn't have performance optimizations in this system
        # but we maintain the interface for consistency

    def get_model_name(self) -> str:
        return "whisper"

    def get_directory_name(self) -> str:
        return "whisper"

    def _load_model(self) -> tuple[object, float]:
        """Load OpenAI Whisper model."""
        import whisper

        start_time = time.time()
        model = whisper.load_model(self.config.model_size)
        load_time = time.time() - start_time

        return model, load_time

    def _transcribe(self, model: object, audio_path: str) -> tuple[str, float]:
        """Transcribe audio using OpenAI Whisper."""
        start_time = time.time()

        # Use consistent config settings for fair comparison
        kwargs = self.config.get_openai_whisper_kwargs()
        result = model.transcribe(audio_path, **kwargs)

        transcribe_time = time.time() - start_time
        transcript = result["text"].strip()

        return transcript, transcribe_time
