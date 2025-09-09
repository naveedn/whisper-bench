"""
lightning-whisper-mlx benchmark implementation.
"""

import time
from .base import BaseBenchmark


class LightningWhisperMLXBenchmark(BaseBenchmark):
    """Benchmark implementation for lightning-whisper-mlx."""

    def get_model_name(self) -> str:
        return "lightning-whisper-mlx"

    def get_directory_name(self) -> str:
        return "lightning_whisper_mlx"

    def _load_model(self) -> tuple[object, float]:
        """Load lightning-whisper-mlx model."""
        import lightning_whisper_mlx

        start_time = time.time()

        # Lightning Whisper MLX initialization
        transcriber = lightning_whisper_mlx.LightningWhisperMLX(
            model=self.config.model_size
        )

        load_time = time.time() - start_time

        return transcriber, load_time

    def _transcribe(self, model: object, audio_path: str) -> tuple[str, float]:
        """Transcribe audio using lightning-whisper-mlx."""
        start_time = time.time()

        # Use consistent config settings for fair comparison
        kwargs = self.config.get_lightning_whisper_mlx_kwargs()

        # Lightning Whisper MLX transcription
        # Note: API may vary - adjust based on actual implementation
        result = model.transcribe(audio_path, **kwargs)

        transcribe_time = time.time() - start_time

        # Extract text from result (format may vary)
        if isinstance(result, dict) and "text" in result:
            transcript = result["text"].strip()
        elif isinstance(result, str):
            transcript = result.strip()
        else:
            transcript = str(result).strip()

        return transcript, transcribe_time
