# Whisper Model Benchmark

This project benchmarks and compares the performance of four popular Whisper implementations:

- **OpenAI Whisper** - The original implementation
- **faster-whisper** - Optimized implementation using CTranslate2
- **mlx-whisper** - Apple Silicon optimized implementation using MLX
- **lightning-whisper-mlx** - Another Apple Silicon optimized implementation

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

2. Install MLX-based models separately (due to low-level dependency issues):
```bash
uv pip install mlx-whisper lightning-whisper-mlx
```

3. Or install everything manually:
```bash
pip install mlx-whisper lightning-whisper-mlx openai-whisper faster-whisper librosa numpy
```

## Usage

### Quick Start

Run the benchmark on available audio files:

```bash
python main.py
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
    ├── mlx_whisper/
    │   ├── audio1_base.txt
    │   └── audio2_base.txt
    └── lightning_whisper_mlx/
        ├── audio1_base.txt
        └── audio2_base.txt
```

## Architecture

The app now uses a clean modular architecture:

```
mlx-whisper-test/
├── main.py                    # Main entry point
├── config.py                  # Global configuration for fair comparisons  
├── benchmarks/               # Individual benchmark implementations
│   ├── __init__.py
│   ├── base.py              # Abstract base class
│   ├── openai_whisper.py    # OpenAI Whisper implementation
│   ├── faster_whisper.py    # faster-whisper implementation
│   ├── mlx_whisper.py       # mlx-whisper implementation
│   └── lightning_whisper_mlx.py  # lightning-whisper-mlx implementation
└── benchmark_results/        # Output directory
```

### Customization

Edit the `main()` function in `main.py` to customize:

```python
# Change audio directory
audio_dir = Path("../audio-files-wav")

# Test different model sizes
model_sizes = ["tiny", "base", "small", "medium", "large"]

# Filter files by size (useful for testing)
audio_files = [f for f in audio_files if f.stat().st_size < 100 * 1024 * 1024]  # < 100MB
```

### Advanced Usage

### Configuration

The app uses a global configuration system to ensure fair comparisons. Edit `config.py` to adjust:

```python
from config import update_config

# Update global settings  
update_config(
    temperature=0.0,        # Deterministic output
    beam_size=5,           # Beam search width
    language=None,         # Auto-detect language
    word_timestamps=True   # Include word-level timing
)
```

You can also use individual benchmark classes programmatically:

```python
from config import get_config
from benchmarks import OpenAIWhisperBenchmark
from pathlib import Path

config = get_config()
benchmark = OpenAIWhisperBenchmark(config, Path("results"))

# Test specific audio file
result = benchmark.benchmark("audio.wav")
print(f"Processing took: {result.total_time:.2f}s")
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
    🔄 [1/4] whisper... ✅ 12.3s (2.1x realtime)
    🔄 [2/4] faster-whisper... ✅ 8.7s (2.9x realtime)  
    🔄 [3/4] mlx-whisper... ✅ 5.2s (4.9x realtime)
    🔄 [4/4] lightning-whisper-mlx... ✅ 4.1s (6.2x realtime)

📊 Results saved to: benchmark_results/benchmark_results_20240908_203045.json
📋 Summary saved to: benchmark_results/benchmark_summary_20240908_203045.txt
📝 Transcripts saved to: benchmark_results/transcripts
```

## Global Configuration Features

The new configuration system ensures apples-to-apples comparisons by:

- **Consistent parameters**: All models use the same beam_size, temperature, etc. where supported
- **Model-specific kwargs**: Each implementation gets appropriate parameters via `get_*_kwargs()` methods
- **Fair comparisons**: Settings like device selection and compute precision are standardized
- **Reproducible results**: Deterministic settings by default (temperature=0.0)

## Contributing

To add support for additional Whisper implementations:

1. Create a new file in `benchmarks/` (e.g., `benchmarks/new_model.py`)
2. Inherit from `BaseBenchmark` and implement required methods:
   - `get_model_name()`: Display name
   - `get_directory_name()`: Output directory name  
   - `_load_model()`: Model loading logic
   - `_transcribe()`: Transcription logic
3. Add kwargs support to `config.py` via `get_new_model_kwargs()`
4. Register in `benchmarks/__init__.py` and `main.py`