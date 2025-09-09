"""
mlx-whisper benchmark implementation.
"""

import time
from .base import BaseBenchmark


class MLXWhisperBenchmark(BaseBenchmark):
    """Benchmark implementation for mlx-whisper."""

    def get_model_name(self) -> str:
        return "mlx-whisper"

    def get_directory_name(self) -> str:
        return "mlx_whisper"

    def _load_model(self) -> tuple[object, float]:
        """MLX Whisper doesn't have separate model loading - return placeholder."""
        import mlx_whisper
        # MLX Whisper handles model loading internally during transcription
        return mlx_whisper, -1  # -1 indicates no separate load time

    def _transcribe(self, model: object, audio_path: str) -> tuple[str, float]:
        """Transcribe audio using mlx-whisper."""
        start_time = time.time()

        # Use consistent config settings for fair comparison
        kwargs = self.config.get_mlx_whisper_kwargs()

        # MLX Whisper API - model loading happens during transcribe
        result = model.transcribe(
            audio_path,
            path_or_hf_repo=self.config.model_size,
            **kwargs
        )

        transcribe_time = time.time() - start_time
        transcript = result["text"].strip()

        return transcript, transcribe_time
