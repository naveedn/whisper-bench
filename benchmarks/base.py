"""
Base classes for Whisper benchmark implementations.
"""

import time
import tempfile
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

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

    def __init__(self, config: BenchmarkConfig, output_dir: Path, performance_overlay=None):
        self.config = config
        self.output_dir = output_dir
        self.performance_overlay = performance_overlay
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

    def extract_audio_segment(self, audio_path: str, start_sec: float, end_sec: float) -> str:
        """Extract a segment from audio file and return path to temporary file."""
        try:
            import librosa
            import soundfile as sf

            # Load the specific segment
            y, sr = librosa.load(audio_path, sr=None, offset=start_sec, duration=end_sec - start_sec)

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Write segment to temporary file
            sf.write(temp_path, y, sr)

            return temp_path
        except Exception as e:
            print(f"Error extracting audio segment: {e}")
            return None

    def process_vad_segments(self, audio_path: str) -> List[Dict]:
        """Process audio using VAD timestamps if enabled."""
        vad_segments = self.config.load_vad_timestamps()
        if not vad_segments:
            raise ValueError("No VAD timestamps available - VAD mode requires timestamps file")

        segment_results = []

        for i, segment in enumerate(vad_segments):
            start_sec = segment['start_sec']
            end_sec = segment['end_sec']

            print(f"Processing VAD segment {i+1}/{len(vad_segments)}: {start_sec:.2f}s - {end_sec:.2f}s")

            # Extract audio segment
            segment_path = self.extract_audio_segment(audio_path, start_sec, end_sec)
            if not segment_path:
                raise ValueError(f"Failed to extract audio segment {i+1}")

            try:
                # Load model for this segment
                model, _ = self._load_model()

                # Transcribe the segment
                transcript, transcribe_time = self._transcribe(model, segment_path)

                segment_results.append({
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'duration_sec': end_sec - start_sec,
                    'transcript': transcript,
                    'transcribe_time': transcribe_time
                })

            except Exception as e:
                raise ValueError(f"Error processing segment {i+1}: {e}")
            finally:
                # Clean up temporary file
                try:
                    Path(segment_path).unlink()
                except:
                    pass

        return segment_results



    def benchmark(self, audio_path: str) -> BenchmarkResult:
        """Run benchmark on a single audio file."""
        audio_file = Path(audio_path)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        duration = self.get_audio_duration(audio_path)

        # Check if VAD processing is enabled
        if self.config.use_vad_timestamps:
            return self._benchmark_with_vad_segments(audio_path, audio_file, file_size_mb, duration)

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

    def _benchmark_with_vad_segments(self, audio_path: str, audio_file: Path, file_size_mb: float, duration: Optional[float]) -> BenchmarkResult:
        """Benchmark processing using VAD segments."""
        start_total = time.time()

        try:
            # Process VAD segments
            segment_results = self.process_vad_segments(audio_path)

            total_time = time.time() - start_total
            total_transcribe_time = sum(seg['transcribe_time'] for seg in segment_results)

            # Combine all segment transcripts
            combined_transcript = ""
            for i, seg in enumerate(segment_results):
                if seg['transcript'] and not seg['transcript'].startswith('ERROR:'):
                    combined_transcript += f"[{seg['start_sec']:.1f}s-{seg['end_sec']:.1f}s] {seg['transcript']}\n"

            # Calculate VAD-based duration (only speech segments)
            vad_duration = sum(seg['duration_sec'] for seg in segment_results)

            # Save transcript with VAD segments
            model_size = self.config.model_size
            output_file = self.transcripts_dir / f"{audio_file.stem}_{model_size}_vad.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(combined_transcript.strip())

            # Save detailed segment results
            segments_file = self.transcripts_dir / f"{audio_file.stem}_{model_size}_vad_segments.json"
            with open(segments_file, 'w', encoding='utf-8') as f:
                json.dump(segment_results, f, indent=2)

            processing_rate_mb = file_size_mb / total_time if total_time > 0 else 0
            processing_rate_rt = vad_duration / total_transcribe_time if vad_duration and total_transcribe_time > 0 else None

            return BenchmarkResult(
                model_name=f"{self.model_name}-{model_size}-VAD",
                audio_file=audio_file.name,
                file_size_mb=file_size_mb,
                duration_seconds=vad_duration,  # Use VAD duration instead of full audio
                load_time=0,  # Model loading distributed across segments
                transcribe_time=total_transcribe_time,
                total_time=total_time,
                success=True,
                error=None,
                transcript=combined_transcript.strip(),
                processing_rate_mb_per_sec=processing_rate_mb,
                processing_rate_realtime=processing_rate_rt
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=f"{self.model_name}-{self.config.model_size}-VAD",
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
