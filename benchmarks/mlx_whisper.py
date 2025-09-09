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
        """Transcribe audio using mlx-whisper.

        Matches provided config exactly:
        path_or_hf_repo="mlx-community/whisper-large-v2-mlx-q4", word_timestamps=True,
        beam_size=5, temperature=0, initial_prompt=None, condition_on_previous_text=False
        """
        start_time = time.time()

        # Use EXACT config settings from provided example for fair comparison
        kwargs = self.config.get_mlx_whisper_kwargs()

        # Use the proper MLX model repository path from config mapping
        model_repo = self.config.get_mlx_model_repo()

        # MLX Whisper API - model loading happens during transcribe
        result = model.transcribe(
            audio_path,
            path_or_hf_repo=model_repo,
            **kwargs
        )

        transcribe_time = time.time() - start_time
        transcript = result["text"].strip()

        return transcript, transcribe_time
