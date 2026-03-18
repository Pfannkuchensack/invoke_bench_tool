# invokeai-bench

Benchmark tool for [InvokeAI](https://github.com/invoke-ai/InvokeAI) — measures generation performance across all supported model types.

## Features

- Benchmarks **txt2img** and **img2img** generation
- Supports all InvokeAI model types: SD 1.5, SD 2, SDXL, FLUX.1, FLUX.2 Klein, SD3, Z-Image, CogView4
- Auto-discovery mode to benchmark all installed models at once
- Detailed timing statistics (mean, median, std, min, max)
- JSON result export with system info (GPU, driver, VRAM)
- Result comparison between two benchmark runs

## Requirements

- Python >= 3.10
- A running InvokeAI instance (default: `http://127.0.0.1:9090`)

> **Important:** Enable **Low VRAM Mode** in InvokeAI settings for more consistent and representative benchmark results. This prevents model caching from skewing timing measurements between runs.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### List available models

```bash
invokeai-bench list-models
invokeai-bench list-models --base flux
```

### Run benchmarks from a config file

```bash
invokeai-bench run --config examples/benchmark.toml
```

Options:

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to benchmark TOML config (required) |
| `--output PATH` | Output JSON path (default: `benchmark_results_<timestamp>.json`) |
| `--host URL` | Override InvokeAI host |
| `--dry-run` | Print graph JSON without submitting |
| `--verbose` | Show per-iteration timing |

### Auto-benchmark all installed models

```bash
invokeai-bench run-all
```

Options:

| Flag | Description |
|------|-------------|
| `--host URL` | InvokeAI host (default: `http://127.0.0.1:9090`) |
| `--output PATH` | Output JSON path |
| `--iterations N` | Iterations per model (default: 3) |
| `--warmup N` | Warmup runs per model (default: 1) |
| `--seed N` | Base seed (default: 42) |
| `--dry-run` | Print graph JSON without submitting |
| `--verbose` | Show per-iteration timing |

### Compare two benchmark runs

```bash
invokeai-bench compare baseline.json current.json
```

Positive change % = slower (regression), negative = faster (improvement).

## Configuration

Benchmark scenarios are defined in TOML files. See the [examples/](examples/) directory for ready-to-use configs for each model type:

| File | Model Type |
|------|------------|
| [sd1.toml](examples/sd1.toml) | Stable Diffusion 1.5 |
| [sd2.toml](examples/sd2.toml) | Stable Diffusion 2 |
| [sdxl.toml](examples/sdxl.toml) | Stable Diffusion XL |
| [flux.toml](examples/flux.toml) | FLUX.1 |
| [flux2.toml](examples/flux2.toml) | FLUX.2 Klein |
| [sd3.toml](examples/sd3.toml) | Stable Diffusion 3 |
| [z-image.toml](examples/z-image.toml) | Z-Image |
| [cogview4.toml](examples/cogview4.toml) | CogView4 |
| [benchmark.toml](examples/benchmark.toml) | Multi-model example |

### Config structure

```toml
[connection]
host = "http://127.0.0.1:9090"

[defaults]
warmup_runs = 1
iterations = 3
seed = 42
positive_prompt = "a photo of a mountain landscape at golden hour, highly detailed"
negative_prompt = "blurry, bad quality, worst quality"

[[scenario]]
name = "my_benchmark"
type = "txt2img"           # or "img2img"
base = "sdxl"              # sd-1, sd-2, sdxl, flux, flux2, sd-3, z-image, cogview4
model_name = "my-model"    # must match installed model name
width = 1024
height = 1024
steps = 30
cfg_scale = 7.0
```

Adjust `model_name` values to match your installed models. Use `invokeai-bench list-models` to see what's available.

## Supported model types

| Base | Default Resolution | Default Steps | Default CFG |
|------|--------------------|---------------|-------------|
| sd-1 | 512x512 | 20 | 7.5 |
| sd-2 | 768x768 | 20 | 7.5 |
| sdxl | 1024x1024 | 30 | 7.0 |
| flux | 1024x1024 | 20 | 1.0 |
| flux2 | 1024x1024 | 20 | 1.0 |
| sd-3 | 1024x1024 | 28 | 3.5 |
| z-image | 1024x1024 | 8 | 1.0 |
| cogview4 | 1024x1024 | 25 | 3.5 |

## License

MIT
