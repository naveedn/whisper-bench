"""
faster-whisper benchmark implementation.
"""

import time
from .base import BaseBenchmark


class FasterWhisperBenchmark(BaseBenchmark):
    """Benchmark implementation for faster-whisper."""

    def __init__(self, config, output_dir, performance_overlay=None):
        super().__init__(config, output_dir)
        self.performance_overlay = performance_overlay

    def get_model_name(self) -> str:
        return "faster-whisper"

    def get_directory_name(self) -> str:
        return "faster_whisper"

    def _load_model(self) -> tuple[object, float]:
        """Load faster-whisper model.

        Matches provided config exactly:
        WhisperModel("large-v3", device="cpu", compute_type="int8")
        """
        from faster_whisper import WhisperModel
        import os

        start_time = time.time()

        # Environment variables can still be useful for OpenMP optimization
        if self.performance_overlay and self.performance_overlay.performance_opts.enable_performance_mode:
            os.environ['OMP_NUM_THREADS'] = str(self.performance_overlay.performance_opts.faster_whisper_cpu_threads)

        # Base model parameters
        model_kwargs = {
            'device': self.config.device,
            'compute_type': self.config.compute_type
        }

        # Apply performance optimizations if available
        if self.performance_overlay:
            perf_kwargs = self.performance_overlay.get_faster_whisper_performance_kwargs()
            model_kwargs.update(perf_kwargs)

        # Use EXACT settings from provided example for fair comparison
        model = WhisperModel(
            self.config.model_size,
            **model_kwargs
        )

        load_time = time.time() - start_time

        return model, load_time

    def _transcribe(self, model: object, audio_path: str) -> tuple[str, float]:
        """Transcribe audio using faster-whisper.

        Matches provided config exactly:
        beam_size=5, temperature=0, word_timestamps=True, condition_on_previous_text=False, vad_filter=False
        """
        start_time = time.time()

        # Use EXACT config settings from provided example for fair comparison
        kwargs = self.config.get_faster_whisper_kwargs()

        # Apply performance optimizations if available
        if self.performance_overlay:
            perf_kwargs = self.performance_overlay.get_faster_whisper_transcribe_kwargs()
            kwargs.update(perf_kwargs)

        segments, info = model.transcribe(audio_path, **kwargs)

        # Collect all segments (as shown in provided example)
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text)

        transcript = " ".join(transcript_parts).strip()
        transcribe_time = time.time() - start_time

        return transcript, transcribe_time
