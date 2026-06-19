# Model compression

Seiso integrates three vendored compression pipelines. Code Llama and RL quant also have `seiso` CLI subcommands; image compression is **Forge-only** (or vendored `python -m sd_compress`).

| Pipeline | Forge page | API prefix | CLI |
|----------|------------|------------|-----|
| Code Llama (LLM) | `/compress` | `/api/compress` | `seiso compress run` |
| Stable Diffusion (image) | `/image-compress` | `/api/image-compress` | Forge only* |
| Adaptive RL quant (GGUF) | `/rl-quant` | `/api/rl-quant` | `seiso rl-quant run` |

\* Or vendored `python -m sd_compress` in `third_party/sd-distill-prune-quant/`.

Vendored sources: `third_party/codellama-compress/`, `third_party/sd-distill-prune-quant/`, `third_party/adaptive-rl-quant/`.

## Install extras

Base Forge + training stack:

```bash
pip install -e ".[forge,train,dev]"
```

Optional per pipeline:

| Extra | Purpose |
|-------|---------|
| `.[compress-quant]` | GPTQ / AWQ for Code Llama pipeline (Linux NVIDIA; needs train stack) |
| `.[compress-eval]` | lm-eval harness for Code Llama evaluate stage |
| `.[image-compress]` | SD distill / prune / finetune (PyTorch, diffusers) |
| `.[image-compress-onnx]` | ONNX export for image pipeline |

Install example:

```bash
pip install -e ".[compress-quant,compress-eval,image-compress,image-compress-onnx]"
```

If GPTQ build fails on Linux NVIDIA:

```bash
pip install auto-gptq autoawq --no-build-isolation
```

Vendored packages are bootstrapped at runtime from `third_party/` — no separate clone required.

## Code Llama compression (LLM)

Stages: distill → prune → finetune → evaluate → export (configurable). Supports GPTQ/AWQ quant and speculative decoding.

### Forge

1. `seiso forge` → **Compress** (`/compress`)
2. Pick a preset, adjust teacher/student models, start job
3. Logs stream over SSE; outputs under `{SEISO_DATA_DIR}/compress/{user_id}/`

### CLI

```bash
seiso compress run --preset smoke
seiso compress manifest-verify --run-dir <run_dir>
seiso compress speculative --target-model <path> --draft-model <path>
```

Example config reference: `configs/example_compress.json`.

Presets: `smoke`, `full`, `distill_only`, `prune_recover`, `quantize`.

## Image compression (Stable Diffusion)

Stages: progressive distillation, pruning, fine-tune, quantize, ONNX/shard export.

### Forge

1. `seiso forge` → **Image Compress** (`/image-compress`)
2. Select preset and model; start job
3. Outputs under `{SEISO_DATA_DIR}/image_compress/{user_id}/`

Requires `.[image-compress]` (and `.[image-compress-onnx]` for ONNX export).

**No `seiso` subcommand** — use Forge only, or the vendored `python -m sd_compress` CLI under `third_party/sd-distill-prune-quant/`.

## Adaptive RL quantization

Trains a reinforcement-learning policy for adaptive GGUF quantization levels. Optionally co-trains CUDA kernel launch profiles (`kernel_rl_enabled`).

### Forge

1. `seiso forge` → **RL Quant** (`/rl-quant`)
2. Enable **CUDA kernel RL** in the experiment config (optional)
3. Choose preset (`minimal`, `reproducible`, `post_train`, etc.) and optional checkpoint/GGUF paths
4. Outputs under `{SEISO_DATA_DIR}/rl_quant/{user_id}/{job_id}/`

### CLI

```bash
seiso rl-quant run --preset minimal --training-episodes 256
seiso rl-quant run --preset reproducible --kernel-rl --training-episodes 512
seiso rl-quant run --kernel-rl --kernel-live-benchmark   # NVIDIA CUDA micro-bench
seiso rl-quant profiles                                  # list kernel launch profiles
```

Smoke config reference: `configs/rl_quant_smoke.json`.

Integrated pipeline only — upstream `adaptive-rl-quant*` CLIs in `third_party/adaptive-rl-quant/` are optional for advanced research; use `seiso rl-quant run` for the Forge-equivalent path.

## Platform notes

| Platform | LLM compress | Image compress | RL quant |
|----------|--------------|----------------|----------|
| Linux NVIDIA | ✓ (CUDA kernels in training stages) | ✓ | ✓ |
| Linux AMD ROCm | ✓ (Triton fallback) | ✓ | ✓ |
| Windows NVIDIA | ✓ (CUDA JIT) | ✓ | ✓ (simulator; live CUDA bench if GPU available) |
| macOS | CPU/MPS (slow for large models) | MPS/CPU | ✓ (simulator / analytic kernel metrics) |

For upstream pipeline details, see vendored READMEs:

- `third_party/codellama-compress/README.seiso.md`
- `third_party/sd-distill-prune-quant/README.seiso.md`
- `third_party/adaptive-rl-quant/README.md`
