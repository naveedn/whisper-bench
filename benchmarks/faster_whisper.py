"""
faster-whisper benchmark implementation.
"""

import time
from .base import BaseBenchmark


class FasterWhisperBenchmark(BaseBenchmark):
    """Benchmark implementation for faster-whisper."""

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

        start_time = time.time()

        # Use EXACT settings from provided example for fair comparison
        model = WhisperModel(
            self.config.model_size,
            device=self.config.device,  # "cpu"
            compute_type=self.config.compute_type  # "int8"
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
        segments, info = model.transcribe(audio_path, **kwargs)

        # Collect all segments (as shown in provided example)
        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text)

        transcript = " ".join(transcript_parts).strip()
        transcribe_time = time.time() - start_time

        return transcript, transcribe_time
