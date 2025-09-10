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
uv sync  # Install dependencies
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
uv run python main.py
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

All models use **identical settings** for fair comparison:
- **Greedy decoding** (beam_size=1) - most restrictive common denominator
- **Temperature=0** - deterministic output
- **Word timestamps=True** - detailed timing information
- **No conditioning** on previous text

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
- Audio files in supported formats
- For MLX models: Apple Silicon Mac
- For VAD processing: `librosa` and `soundfile` (auto-installed)

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

## Troubleshooting

- **MLX models fail**: Requires Apple Silicon Mac
- **No audio found**: Place files in `inputs/` folder
- **VAD not working**: Ensure `librosa` installed with `uv sync`
- **Out of memory**: Use shorter audio files or smaller model sizes