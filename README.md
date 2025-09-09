# Whisper Model Benchmark

This project benchmarks and compares the performance of three popular Whisper implementations:

- **OpenAI Whisper** - The original implementation
- **faster-whisper** - Optimized implementation using CTranslate2
- **mlx-whisper** - Apple Silicon optimized implementation using MLX

## Features

- ⏱️ **Performance timing** - Measures load time, transcription time, and total processing time
- 📊 **Processing rates** - Calculates MB/s and realtime factor (how many times faster than audio duration)
- 💾 **Quality comparison** - Saves transcriptions from each model for manual quality inspection
- 📈 **Comprehensive reporting** - Generates detailed JSON results and human-readable summaries

## Installation

1. Install most dependencies:
```bash
uv sync
```

2. Install mlx-whisper separately (due to low-level dependency issues):
```bash
uv pip install mlx-whisper
```

3. Or install everything manually:
```bash
pip install mlx-whisper openai-whisper faster-whisper librosa numpy
```

## Usage

### Quick Start

Run the benchmark on available audio files:

```bash
python whisper_benchmark.py
```

This will:
- Process all `.wav` files in `../audio-files-wav/`
- Test with the "base" model size for all three implementations
- Save results to `benchmark_results/`

### Output Structure

```
benchmark_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json     # Detailed JSON results
├── benchmark_summary_YYYYMMDD_HHMMSS.txt      # Human-readable summary
└── transcripts/                                # Individual transcriptions
    ├── whisper/
    │   ├── audio1_base.txt
    │   └── audio2_base.txt
    ├── faster_whisper/
    │   ├── audio1_base.txt
    │   └── audio2_base.txt
    └── mlx_whisper/
        ├── audio1_base.txt
        └── audio2_base.txt
```

### Customization

Edit the `main()` function in `whisper_benchmark.py` to customize:

```python
# Change audio directory
audio_dir = Path("../audio-files-wav")

# Test different model sizes
model_sizes = ["tiny", "base", "small", "medium", "large"]

# Filter files by size (useful for testing)
audio_files = [f for f in audio_files if f.stat().st_size < 100 * 1024 * 1024]  # < 100MB
```

### Advanced Usage

You can also use the benchmark class programmatically:

```python
from whisper_benchmark import WhisperBenchmark

benchmark = WhisperBenchmark(output_dir="my_results")

# Test specific audio file with specific model
result = benchmark.benchmark_mlx_whisper("audio.wav", "base")
print(f"Processing took: {result.total_time:.2f}s")

# Run full benchmark
results = benchmark.run_benchmark(
    audio_files=["file1.wav", "file2.wav"],
    model_sizes=["base", "small"]
)
```

## Performance Metrics

The benchmark tracks several key metrics:

- **Load Time**: Time to load the model (where applicable)
- **Transcribe Time**: Time for actual transcription
- **Total Time**: End-to-end processing time
- **Processing Rate (MB/s)**: Throughput in megabytes per second
- **Realtime Factor**: How many times faster than audio duration (e.g., 5x = processes 1 hour of audio in 12 minutes)

## Model Comparison

### Expected Performance Characteristics

- **OpenAI Whisper**: Most compatible, moderate speed
- **faster-whisper**: Fastest CPU performance, good accuracy
- **mlx-whisper**: Optimized for Apple Silicon, can leverage GPU acceleration

### Quality vs Speed Trade-offs

- **tiny/base**: Fastest but lower accuracy
- **small/medium**: Good balance of speed and accuracy  
- **large**: Best accuracy but slowest

## Troubleshooting

### Common Issues

1. **MLX Whisper fails**: Only works on Apple Silicon Macs
2. **Out of memory**: Try smaller model sizes or shorter audio files
3. **Missing dependencies**: Run `uv sync` or install packages manually

### Performance Tips

- Use shorter audio clips for initial testing
- Start with "base" model size before trying larger models
- Monitor system resources during benchmarking
- For MLX Whisper, ensure you're running on Apple Silicon

## Sample Output

```
🎵 Processing: 2-zaboombafool.wav
  📏 Model size: base
    🔄 [1/3] whisper... ✅ 12.3s (2.1x realtime)
    🔄 [2/3] faster-whisper... ✅ 8.7s (2.9x realtime)  
    🔄 [3/3] mlx-whisper... ✅ 5.2s (4.9x realtime)

📊 Results saved to: benchmark_results/benchmark_results_20240908_203045.json
📋 Summary saved to: benchmark_results/benchmark_summary_20240908_203045.txt
📝 Transcripts saved to: benchmark_results/transcripts
```

## Contributing

To add support for additional Whisper implementations:

1. Add a new `benchmark_<model_name>` method to the `WhisperBenchmark` class
2. Follow the existing pattern for error handling and result structure
3. Add the new model to the `models` list in `run_benchmark()`