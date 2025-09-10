# Whisper Bench

A comprehensive benchmarking tool for comparing Whisper speech-to-text implementations with VAD (Voice Activity Detection) support.

## Features

- 🎯 **4 Whisper Implementations**: OpenAI Whisper, faster-whisper, mlx-whisper, lightning-whisper-mlx
- 🔍 **VAD Support**: Process only speech segments to avoid hallucinations on silent audio
- ⏱️ **Performance Metrics**: Timing, throughput, and realtime factor analysis
- 📊 **Fair Comparisons**: Consistent configuration across all models (greedy decoding, temperature=0)
- 📝 **Quality Analysis**: Save individual transcripts for manual comparison
- 🎛️ **Simple Setup**: Drop audio files in `inputs/` folder and run

## Quick Start

### 1. Installation
```bash
git clone <repo-url>
cd whisper-bench

# Basic installation
uv sync

# Or install with performance optimizations (includes GPU support)
uv sync --extra performance

# Or install everything
uv sync --extra all
```

### 2. Add Your Audio
```bash
# Place your files in the inputs folder:
inputs/
├── my_audio.wav                    # Your audio file
└── my_audio_VAD_timestamps.json    # Optional: VAD timestamps for better quality
```

### 3. Run Benchmark
```bash
# Standard benchmark (apples-to-apples comparison)
uv run python main.py

# Or use performance optimizations for speed testing
uv run python main.py --performance-profile aggressive
```

## Input Files

### Audio Files
Supported formats: `.wav`, `.mp3`, `.flac`

### VAD Timestamps (Optional)
JSON file with speech segments to avoid processing silent sections:
```json
[
  {
    "start_sec": 10.5,
    "end_sec": 15.8, 
    "duration_sec": 5.3
  }
]
```

Benefits of VAD:
- ✅ Avoids Whisper hallucinations on long silent sections
- ✅ More accurate quality assessment 
- ✅ Focus transcription on actual speech content

## Configuration

### Standard Configuration (Fair Comparison)

All models use **identical settings** for fair comparison:
- **Greedy decoding** (beam_size=1) - most restrictive common denominator
- **Temperature=0** - deterministic output
- **Word timestamps=True** - detailed timing information
- **No conditioning** on previous text

### Performance Optimization Profiles

For speed testing, use performance profiles that maintain quality while optimizing computational efficiency:

```bash
# List available profiles
uv run python main.py --list-profiles

# Performance profiles (quality-preserving optimizations only):
uv run python main.py --performance-profile baseline      # Standard comparison
uv run python main.py --performance-profile conservative  # Minimal optimizations
uv run python main.py --performance-profile balanced      # Good balance
uv run python main.py --performance-profile aggressive    # Maximum performance
```

**Performance optimizations include:**
- GPU acceleration (when available)
- Optimized batch processing
- Memory management improvements  
- Threading optimization (`cpu_threads`, `num_workers`)
- Model quantization (limited by Apple Hardware compatibility)

**Quality parameters remain unchanged** - only computational efficiency is improved.

**Apple Hardware Limitations:**
- Lightning-whisper-mlx quantization disabled due to QuantizedLinear compatibility issues
- MLX models automatically use float16 precision without explicit dtype settings

**Performance Model Selection:**
When using aggressive performance profiles, models are automatically optimized for speed:
- `base` → `tiny` (74M → 39M parameters, 47% smaller)
- `small` → `base` (244M → 74M parameters, 70% smaller)  
- `medium` → `small` (769M → 244M parameters, 68% smaller)
- `large` → `large-v3-turbo` (1.55B → 809M parameters, 8x faster)
- `turbo` → `large-v3-turbo` (new option for maximum speed)

**Threading Optimization Details:**
- **faster-whisper**: Uses `cpu_threads` (OpenMP threads), `num_workers` (parallel workers), and `chunk_length` (audio segmentation)
- **Environment variables**: `OMP_NUM_THREADS` automatically set for additional optimization
- **Note**: `batch_size` is not supported by faster-whisper's transcribe() method

## Command Line Options

```bash
uv run python main.py [OPTIONS]

Options:
  -p, --performance-profile {baseline,conservative,balanced,aggressive}
                        Performance optimization profile to use (default: baseline)
  -l, --list-profiles   List available performance profiles and exit
  -m, --model-sizes {tiny,base,small,medium,large,large-v3,turbo} [...]
                        Model sizes to benchmark (turbo = large-v3-turbo for 8x speed) (default: base)
  -o, --output-dir DIR  Output directory for benchmark results (default: benchmark_results)
  -h, --help           Show help message and exit

Examples:
  uv run python main.py                                    # Standard benchmark
  uv run python main.py -p aggressive -m base small        # Performance test with multiple sizes
  uv run python main.py -p balanced -m turbo               # Test new turbo model (8x faster)
  uv run python main.py -p balanced -o my_results          # Custom output directory
  uv run python main.py --list-profiles                    # Show available profiles
```

## Output

Results are saved to `benchmark_results/`:
```
benchmark_results/
├── benchmark_results_YYYYMMDD_HHMMSS.json  # Detailed metrics
├── benchmark_summary_YYYYMMDD_HHMMSS.txt   # Human-readable summary
└── transcripts/                            # Individual model outputs
    ├── whisper/
    ├── faster_whisper/ 
    ├── mlx_whisper/
    └── lightning_whisper_mlx/
```

## Model Comparison

| Model | Speed | Platform | Beam Search |
|-------|-------|----------|-------------|
| **OpenAI Whisper** | Moderate | All | ✅ |
| **faster-whisper** | Fast | All | ✅ |
| **mlx-whisper** | Fast | Apple Silicon | ❌ (greedy only) |
| **lightning-whisper-mlx** | Very Fast | Apple Silicon | ❌ (greedy only) |

**Note**: All models now use greedy decoding (beam_size=1) to ensure fair comparison.

## Example Output

```
🎯 Found audio file in inputs folder: podcast.wav
🔍 Found VAD timestamps file: podcast_VAD_timestamps.json
🔍 Enabling VAD timestamp processing to avoid hallucinations on silent sections

Starting benchmark with 4 total tests...
Models: ['whisper', 'faster-whisper', 'mlx-whisper', 'lightning-whisper-mlx']
--------------------------------------------------------------------------------

🎵 Processing: podcast.wav
  📏 Model size: base
    🔄 [1/4] whisper... ✅ 45.2s (12.3x realtime)
    🔄 [2/4] faster-whisper... ✅ 32.1s (17.3x realtime)
    🔄 [3/4] mlx-whisper... ✅ 18.7s (29.7x realtime)
    🔄 [4/4] lightning-whisper-mlx... ✅ 15.3s (36.4x realtime)

================================================================================
BENCHMARK COMPLETE!
================================================================================
```

## Requirements

- Python 3.11+
- Audio files in supported formats (.wav, .mp3, .flac)
- For MLX models: Apple Silicon Mac
- For VAD processing: `librosa` and `soundfile` (auto-installed)
- For GPU acceleration: PyTorch (install with `--extra performance`)

### Dependencies

**Core dependencies (always installed):**
- `mlx-whisper` - MLX-optimized Whisper for Apple Silicon
- `openai-whisper` - Original OpenAI implementation  
- `faster-whisper` - CTranslate2-based acceleration
- `lightning-whisper-mlx` - High-performance MLX implementation
- `librosa` - Audio processing
- `soundfile` - Audio I/O
- `numpy` - Numerical computing

**Optional dependencies:**
- `torch>=2.0.0` - GPU detection and acceleration (install with `--extra performance`)

**Installation options:**
```bash
uv sync                    # Basic installation (CPU only)
uv sync --extra performance # Include GPU support  
uv sync --extra all        # Everything
```

## Architecture

```
whisper-bench/
├── main.py              # Entry point with inputs/ folder detection
├── config.py            # Global settings for fair comparison
├── benchmarks/          # Model implementations
│   ├── base.py         # Abstract base class with VAD support
│   ├── openai_whisper.py
│   ├── faster_whisper.py  
│   ├── mlx_whisper.py
│   └── lightning_whisper_mlx.py
├── inputs/              # Place your audio + VAD files here
└── benchmark_results/   # Generated results and transcripts
```

## Performance Comparison Workflow

### 1. Quality Baseline (Apples-to-Apples)
```bash
# Standard comparison for quality assessment
uv run python main.py --performance-profile baseline -m base
```

### 2. Performance Testing
```bash
# Test with performance optimizations
uv run python main.py --performance-profile aggressive -m base

# Compare multiple model sizes with optimizations
uv run python main.py --performance-profile balanced -m base small medium
```

### 3. GPU vs CPU Performance
```bash
# Install GPU support first
uv sync --extra performance

# Run with GPU optimizations
uv run python main.py --performance-profile aggressive
```

The performance profiles ensure **identical quality** while optimizing computational efficiency, allowing you to measure pure performance improvements without quality trade-offs.

## Troubleshooting

- **MLX models fail**: Requires Apple Silicon Mac
- **No audio found**: Place files in `inputs/` folder  
- **VAD not working**: Ensure `librosa` installed with `uv sync`
- **Out of memory**: Use shorter audio files or smaller model sizes
- **GPU not detected**: Install with `uv sync --extra performance` and ensure CUDA available
- **Performance profiles not working**: Check that performance_config.py imports correctly
- **Command not found**: Use `uv run python main.py` instead of just `python main.py`