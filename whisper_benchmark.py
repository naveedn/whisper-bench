#!/usr/bin/env python3
"""
Whisper Model Benchmark Script

Compares performance and quality of:
- OpenAI Whisper
- faster-whisper
- mlx-whisper

Outputs timing data and saves transcriptions for manual quality inspection.
"""

import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import traceback

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


class WhisperBenchmark:
    """Benchmark different Whisper implementations."""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create subdirectories for each model's outputs
        self.transcripts_dir = self.output_dir / "transcripts"
        self.transcripts_dir.mkdir(exist_ok=True)

        for model in ["whisper", "faster_whisper", "mlx_whisper"]:
            (self.transcripts_dir / model).mkdir(exist_ok=True)

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

    def benchmark_openai_whisper(self, audio_path: str, model_size: str = "base") -> BenchmarkResult:
        """Benchmark OpenAI Whisper."""
        audio_file = Path(audio_path)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        duration = self.get_audio_duration(audio_path)

        try:
            import whisper

            # Load model
            start_load = time.time()
            model = whisper.load_model(model_size)
            load_time = time.time() - start_load

            # Transcribe
            start_transcribe = time.time()
            result = model.transcribe(audio_path)
            transcribe_time = time.time() - start_transcribe

            total_time = load_time + transcribe_time
            transcript = result["text"].strip()

            # Save transcript
            output_file = self.transcripts_dir / "whisper" / f"{audio_file.stem}_{model_size}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)

            processing_rate_mb = file_size_mb / total_time if total_time > 0 else 0
            processing_rate_rt = duration / total_time if duration and total_time > 0 else None

            return BenchmarkResult(
                model_name=f"whisper-{model_size}",
                audio_file=audio_file.name,
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                load_time=load_time,
                transcribe_time=transcribe_time,
                total_time=total_time,
                success=True,
                error=None,
                transcript=transcript,
                processing_rate_mb_per_sec=processing_rate_mb,
                processing_rate_realtime=processing_rate_rt
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=f"whisper-{model_size}",
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

    def benchmark_faster_whisper(self, audio_path: str, model_size: str = "base") -> BenchmarkResult:
        """Benchmark faster-whisper."""
        audio_file = Path(audio_path)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        duration = self.get_audio_duration(audio_path)

        try:
            from faster_whisper import WhisperModel

            # Load model
            start_load = time.time()
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            load_time = time.time() - start_load

            # Transcribe
            start_transcribe = time.time()
            segments, info = model.transcribe(audio_path, beam_size=5)
            transcript = " ".join([segment.text for segment in segments])
            transcribe_time = time.time() - start_transcribe

            total_time = load_time + transcribe_time

            # Save transcript
            output_file = self.transcripts_dir / "faster_whisper" / f"{audio_file.stem}_{model_size}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript.strip())

            processing_rate_mb = file_size_mb / total_time if total_time > 0 else 0
            processing_rate_rt = duration / total_time if duration and total_time > 0 else None

            return BenchmarkResult(
                model_name=f"faster-whisper-{model_size}",
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
                model_name=f"faster-whisper-{model_size}",
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

    def benchmark_mlx_whisper(self, audio_path: str, model_size: str = "base") -> BenchmarkResult:
        """Benchmark mlx-whisper."""
        audio_file = Path(audio_path)
        file_size_mb = audio_file.stat().st_size / (1024 * 1024)
        duration = self.get_audio_duration(audio_path)

        try:
            import mlx_whisper

            # MLX Whisper doesn't have separate load/transcribe phases in the API
            start_total = time.time()
            # Use openai/whisper-base format to avoid HF auth issues
            model_name = f"openai/whisper-{model_size}" if model_size != "base" else "openai/whisper-base"
            result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_name)
            total_time = time.time() - start_total

            transcript = result["text"].strip()

            # Save transcript
            output_file = self.transcripts_dir / "mlx_whisper" / f"{audio_file.stem}_{model_size}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)

            processing_rate_mb = file_size_mb / total_time if total_time > 0 else 0
            processing_rate_rt = duration / total_time if duration and total_time > 0 else None

            return BenchmarkResult(
                model_name=f"mlx-whisper-{model_size}",
                audio_file=audio_file.name,
                file_size_mb=file_size_mb,
                duration_seconds=duration,
                load_time=0,  # Not separately measurable
                transcribe_time=total_time,
                total_time=total_time,
                success=True,
                error=None,
                transcript=transcript,
                processing_rate_mb_per_sec=processing_rate_mb,
                processing_rate_realtime=processing_rate_rt
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=f"mlx-whisper-{model_size}",
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

    def run_benchmark(self, audio_files: list, model_sizes: list = ["base"]) -> Dict[str, Any]:
        """Run complete benchmark across all models and files."""
        results = []

        models = [
            ("whisper", self.benchmark_openai_whisper),
            ("faster-whisper", self.benchmark_faster_whisper),
            ("mlx-whisper", self.benchmark_mlx_whisper)
        ]

        # Option to skip problematic models
        skip_models = []  # Add model names here to skip, e.g., ["mlx-whisper"]
        models = [(name, func) for name, func in models if name not in skip_models]

        total_tests = len(audio_files) * len(models) * len(model_sizes)
        current_test = 0

        print(f"Starting benchmark with {total_tests} total tests...")
        print(f"Audio files: {[Path(f).name for f in audio_files]}")
        print(f"Models: {[name for name, _ in models]}")
        print(f"Model sizes: {model_sizes}")
        print("-" * 80)

        for audio_file in audio_files:
            if not Path(audio_file).exists():
                print(f"⚠️  Audio file not found: {audio_file}")
                continue

            print(f"\n🎵 Processing: {Path(audio_file).name}")

            for model_size in model_sizes:
                print(f"  📏 Model size: {model_size}")

                for model_name, benchmark_func in models:
                    current_test += 1
                    print(f"    🔄 [{current_test}/{total_tests}] {model_name}...", end=" ", flush=True)

                    try:
                        result = benchmark_func(audio_file, model_size)
                        results.append(result)

                        if result.success:
                            print(f"✅ {result.total_time:.1f}s", end="")
                            if result.processing_rate_realtime:
                                print(f" ({result.processing_rate_realtime:.1f}x realtime)")
                            else:
                                print()
                        else:
                            print(f"❌ {result.error}")

                    except Exception as e:
                        print(f"💥 Unexpected error: {e}")
                        traceback.print_exc()

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

        # Generate summary
        summary = self.generate_summary(results)
        summary_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"\n📊 Results saved to: {results_file}")
        print(f"📋 Summary saved to: {summary_file}")
        print(f"📝 Transcripts saved to: {self.transcripts_dir}")

        return {
            "results": results,
            "summary": summary,
            "files": {
                "results": str(results_file),
                "summary": str(summary_file),
                "transcripts": str(self.transcripts_dir)
            }
        }

    def generate_summary(self, results: list) -> str:
        """Generate a human-readable summary of benchmark results."""
        summary = ["Whisper Models Benchmark Summary"]
        summary.append("=" * 50)
        summary.append("")

        # Group by model
        by_model = {}
        for result in results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)

        # Overall stats
        successful_results = [r for r in results if r.success]
        if successful_results:
            summary.append("📈 Overall Performance:")
            summary.append(f"   Successful tests: {len(successful_results)}/{len(results)}")

            avg_times = {}
            for model_name in by_model:
                model_results = [r for r in by_model[model_name] if r.success]
                if model_results:
                    avg_time = sum(r.total_time for r in model_results) / len(model_results)
                    avg_times[model_name] = avg_time

            if avg_times:
                fastest = min(avg_times.items(), key=lambda x: x[1])
                summary.append(f"   Fastest model: {fastest[0]} (avg: {fastest[1]:.1f}s)")
                summary.append("")

        # Per-model breakdown
        for model_name, model_results in by_model.items():
            summary.append(f"🤖 {model_name}")
            summary.append("-" * 30)

            successful = [r for r in model_results if r.success]
            failed = [r for r in model_results if not r.success]

            if successful:
                avg_time = sum(r.total_time for r in successful) / len(successful)
                avg_mb_rate = sum(r.processing_rate_mb_per_sec for r in successful) / len(successful)

                realtime_rates = [r.processing_rate_realtime for r in successful if r.processing_rate_realtime]
                avg_realtime = sum(realtime_rates) / len(realtime_rates) if realtime_rates else None

                summary.append(f"   ✅ Successful: {len(successful)}")
                summary.append(f"   ⏱️  Average time: {avg_time:.1f}s")
                summary.append(f"   📊 Processing rate: {avg_mb_rate:.1f} MB/s")
                if avg_realtime:
                    summary.append(f"   🚀 Realtime factor: {avg_realtime:.1f}x")

                # Per-file breakdown
                for result in successful:
                    summary.append(f"      {result.audio_file}: {result.total_time:.1f}s")

            if failed:
                summary.append(f"   ❌ Failed: {len(failed)}")
                for result in failed:
                    summary.append(f"      {result.audio_file}: {result.error}")

            summary.append("")

        # Quality comparison note
        summary.append("🔍 Quality Comparison:")
        summary.append("   Check the transcripts/ directory to manually compare")
        summary.append("   transcription quality between models.")
        summary.append("")

        return "\n".join(summary)


def main():
    """Main entry point for benchmark script."""
    # Configuration
    audio_dir = Path("../audio-files-wav")

    # Get available audio files
    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        print("Please ensure audio files are available or update the audio_dir path.")
        return

    # Limit to smaller files for initial testing
    # Remove this filter to test all files
    audio_files = [f for f in audio_files if f.stat().st_size < 500 * 1024 * 1024]  # < 500MB

    # For testing, just use the first file
    if audio_files:
        audio_files = audio_files[:1]

    if not audio_files:
        print("No suitable audio files found (all files may be too large)")
        return

    # Model sizes to test
    model_sizes = ["base"]  # Start with base, add "small", "medium", "large" as needed

    # Run benchmark
    benchmark = WhisperBenchmark()
    results = benchmark.run_benchmark([str(f) for f in audio_files], model_sizes)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(results["summary"])


if __name__ == "__main__":
    main()
