#!/usr/bin/env python3
"""
Whisper Model Benchmark Script

Compares performance and quality of:
- OpenAI Whisper
- faster-whisper
- mlx-whisper
- lightning-whisper-mlx

Outputs timing data and saves transcriptions for manual quality inspection.
"""

import json
import time
import traceback
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

from config import get_config, update_config
from performance_config import create_performance_overlay, list_performance_profiles
from benchmarks import (
    BenchmarkResult,
    OpenAIWhisperBenchmark,
    FasterWhisperBenchmark,
    MLXWhisperBenchmark,
    LightningWhisperMLXBenchmark
)


class WhisperBenchmarkRunner:
    """Main benchmark runner that coordinates all model benchmarks."""

    def __init__(self, output_dir: str = "outputs", performance_profile: str = "baseline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.config = get_config()
        self.performance_profile = performance_profile

        # Create performance overlay
        self.performance_overlay = create_performance_overlay(self.config, performance_profile)

        # Initialize all benchmark implementations with performance overlay
        self.benchmarks = {
            "whisper": OpenAIWhisperBenchmark(self.config, self.output_dir, self.performance_overlay),
            "faster-whisper": FasterWhisperBenchmark(self.config, self.output_dir, self.performance_overlay),
            "mlx-whisper": MLXWhisperBenchmark(self.config, self.output_dir, self.performance_overlay),
            "lightning-whisper-mlx": LightningWhisperMLXBenchmark(self.config, self.output_dir, self.performance_overlay),
        }

        # Option to skip problematic models
        self.skip_models = []  # Test all models with consistent beam_size=1 (greedy decoding)

    def run_benchmark(self, audio_files: List[str], model_sizes: List[str] = ["base"]) -> Dict[str, Any]:
        """Run complete benchmark across all models and files."""
        results = []

        # Filter models based on skip list
        active_benchmarks = {
            name: benchmark for name, benchmark in self.benchmarks.items()
            if name not in self.skip_models
        }

        total_tests = len(audio_files) * len(active_benchmarks) * len(model_sizes)
        current_test = 0

        print(f"Starting benchmark with {total_tests} total tests...")
        print(f"Performance profile: {self.performance_profile}")
        print(f"Audio files: {[Path(f).name for f in audio_files]}")
        print(f"Models: {list(active_benchmarks.keys())}")
        print(f"Model sizes: {model_sizes}")
        print("-" * 80)

        for audio_file in audio_files:
            if not Path(audio_file).exists():
                print(f"⚠️  Audio file not found: {audio_file}")
                continue

            print(f"\n🎵 Processing: {Path(audio_file).name}")

            for model_size in model_sizes:
                print(f"  📏 Model size: {model_size}")

                # Update config for this model size
                update_config(model_size=model_size)

                for model_name, benchmark in active_benchmarks.items():
                    current_test += 1
                    print(f"    🔄 [{current_test}/{total_tests}] {model_name}...", end=" ", flush=True)

                    try:
                        # Update benchmark config and apply performance overlay
                        benchmark.config = get_config()

                        # Apply performance optimizations using context manager
                        with self.performance_overlay:
                            result = benchmark.benchmark(audio_file)

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

                        # Create failed result
                        failed_result = BenchmarkResult(
                            model_name=f"{model_name}-{model_size}",
                            audio_file=Path(audio_file).name,
                            file_size_mb=Path(audio_file).stat().st_size / (1024 * 1024),
                            duration_seconds=None,
                            load_time=0,
                            transcribe_time=0,
                            total_time=0,
                            success=False,
                            error=str(e),
                            transcript="",
                            processing_rate_mb_per_sec=0,
                            processing_rate_realtime=None
                        )
                        results.append(failed_result)

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

        transcripts_dir = self.output_dir / "transcripts"

        print(f"\n📊 Results saved to: {results_file}")
        print(f"📋 Summary saved to: {summary_file}")
        print(f"📝 Transcripts saved to: {transcripts_dir}")

        return {
            "results": results,
            "summary": summary,
            "files": {
                "results": str(results_file),
                "summary": str(summary_file),
                "transcripts": str(transcripts_dir)
            }
        }

    def generate_summary(self, results: List[BenchmarkResult]) -> str:
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

        # Configuration used
        summary.append("⚙️  Configuration Used:")
        config_dict = asdict(self.config)
        for key, value in config_dict.items():
            summary.append(f"   {key}: {value}")

        return "\n".join(summary)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Whisper Model Benchmark Tool")

    parser.add_argument(
        "--performance-profile", "-p",
        choices=list(list_performance_profiles().keys()),
        default="baseline",
        help="Performance optimization profile to use"
    )

    parser.add_argument(
        "--list-profiles", "-l",
        action="store_true",
        help="List available performance profiles and exit"
    )

    parser.add_argument(
        "--model-sizes", "-m",
        nargs="+",
        default=["base"],
        choices=["tiny", "base", "small", "medium", "large", "large-v3", "turbo"],
        help="Model sizes to benchmark (turbo = large-v3-turbo for 8x speed)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Output directory for benchmark results"
    )

    return parser.parse_args()


def main():
    """Main entry point for benchmark script."""
    # Parse command line arguments
    args = parse_arguments()

    # Handle --list-profiles
    if args.list_profiles:
        print("Available Performance Profiles:")
        print("=" * 40)
        profiles = list_performance_profiles()
        for profile, description in profiles.items():
            print(f"  {profile:12} - {description}")
        return

    # Check for inputs folder
    inputs_dir = Path("inputs")

    if inputs_dir.exists():
        # Look for audio files in inputs folder
        audio_files = list(inputs_dir.glob("*.wav")) + list(inputs_dir.glob("*.mp3")) + list(inputs_dir.glob("*.flac"))

        if audio_files:
            print(f"🎯 Found audio file in inputs folder: {audio_files[0].name}")

            # Look for VAD timestamps file
            vad_files = list(inputs_dir.glob("*VAD_timestamps.json")) + list(inputs_dir.glob("*_vad.json"))

            if vad_files:
                print(f"🔍 Found VAD timestamps file: {vad_files[0].name}")
                print("🔍 Enabling VAD timestamp processing to avoid hallucinations on silent sections")

                # Update config to use VAD timestamps
                update_config(
                    use_vad_timestamps=True,
                    vad_timestamps_file=str(vad_files[0])
                )
            else:
                print("No VAD timestamps found, using regular processing")
                update_config(use_vad_timestamps=False)

            audio_files = audio_files[:1]  # Use just the first audio file
        else:
            print("No audio files found in inputs folder")
            return
    else:
        # Fallback to legacy approach
        print("No inputs folder found, checking ../audio-files-wav")
        audio_dir = Path("../audio-files-wav")

        # Get available audio files
        audio_files = list(audio_dir.glob("*.wav"))
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            print("Please create an 'inputs' folder with audio file and VAD timestamps, or ensure audio files are available in ../audio-files-wav")
            return

        # Limit to smaller files for initial testing
        audio_files = [f for f in audio_files if f.stat().st_size < 500 * 1024 * 1024]  # < 500MB

        # For testing, just use the first file
        if audio_files:
            audio_files = audio_files[:1]

        # No VAD processing for legacy mode
        update_config(use_vad_timestamps=False)

    if not audio_files:
        print("No suitable audio files found")
        return

    # Use command line arguments
    model_sizes = args.model_sizes

    # Run benchmark with performance profile
    runner = WhisperBenchmarkRunner(
        output_dir=args.output_dir,
        performance_profile=args.performance_profile
    )
    results = runner.run_benchmark([str(f) for f in audio_files], model_sizes)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(results["summary"])


if __name__ == "__main__":
    main()
