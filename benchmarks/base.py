"""
Base classes for Whisper benchmark implementations.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import BenchmarkConfig


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single model/file combination."""
    model_name: str
    audio_file: str
    file_size_mb: float
    duration_seconds: Optional[float]
    load_time: float
    transcribe_time: float
    total_time: float
    success: bool
    error: Optional[str]
    transcript: str
    processing_rate_mb_per_sec: float
    processing_rate_realtime: Optional[float]  # how many times faster than real-time


class BaseBenchmark(ABC):
    """Base class for all Whisper benchmark implementations."""

    def __init__(self, config: BenchmarkConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.model_name = self.get_model_name()

        # Create output directory for this model
        self.transcripts_dir = output_dir / "transcripts" / self.get_directory_name()
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the display name for this benchmark model."""
        pass

    @abstractmethod
    def get_directory_name(self) -> str:
        """Return the directory name for storing transcripts."""
        pass

    @abstractmethod
    def _load_model(self) -> tuple[object, float]:
        """Load the model and return (model, load_time)."""
        pass

    @abstractmethod
    def _transcribe(self, model: object, audio_path: str) -> tuple[str, float]:
        """Transcribe audio and return (transcript, transcribe_time)."""
        pass

    def get_audio_duration(self, audio_path: str) -> Optional[float]:
        """Get audio duration in seconds using librosa."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except ImportError:
            print("Warning: librosa not available, cannot calculate audio duration")
            return None
        except Exception as e:
            print(f"Warning: Could not get duration for {audio_path}: {e}")
            return None

    def benchmark(self, audio_path: str) -> BenchmarkResult:
        """Run benchmark on a single audio file."""
        audio_file = Path(audio_path)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        duration = self.get_audio_duration(audio_path)

        try:
            # Load model
            start_load = time.time()
            model, explicit_load_time = self._load_model()
            load_time = explicit_load_time if explicit_load_time >= 0 else time.time() - start_load

            # Transcribe
            start_transcribe = time.time()
            transcript, explicit_transcribe_time = self._transcribe(model, audio_path)
            transcribe_time = explicit_transcribe_time if explicit_transcribe_time >= 0 else time.time() - start_transcribe

            total_time = load_time + transcribe_time

            # Save transcript
            model_size = self.config.model_size
            output_file = self.transcripts_dir / f"{audio_file.stem}_{model_size}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript.strip())

            processing_rate_mb = file_size_mb / total_time if total_time > 0 else 0
            processing_rate_rt = duration / total_time if duration and total_time > 0 else None

            return BenchmarkResult(
                model_name=f"{self.model_name}-{model_size}",
                audio_file=audio_file.name,
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                load_time=load_time,
                transcribe_time=transcribe_time,
                total_time=total_time,
                success=True,
                error=None,
                transcript=transcript.strip(),
                processing_rate_mb_per_sec=processing_rate_mb,
                processing_rate_realtime=processing_rate_rt
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=f"{self.model_name}-{self.config.model_size}",
                audio_file=audio_file.name,
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                load_time=0,
                transcribe_time=0,
                total_time=0,
                success=False,
                error=str(e),
                transcript="",
                processing_rate_mb_per_sec=0,
                processing_rate_realtime=None
            )
