# Model compression

Seiso integrates three vendored compression pipelines, each available in Forge and (where noted) via CLI.

| Pipeline | Forge page | API prefix | Vendored tree |
|----------|------------|------------|---------------|
| Code Llama (LLM) | `/compress` | `/api/compress` | `third_party/codellama-compress` |
| Stable Diffusion (image) | `/image-compress` | `/api/image-compress` | `third_party/sd-distill-prune-quant` |
| Adaptive RL quant (GGUF) | `/rl-quant` | `/api/rl-quant` | `third_party/adaptive-rl-quant` |

## Install extras

Base Forge + training stack:

```bash
pip install -e ".[forge,train,dev]"
```

Optional per pipeline:

| Extra | Purpose |
|-------|---------|
| `.[compress-quant]` | GPTQ / AWQ for Code Llama pipeline |
| `.[compress-eval]` | lm-eval harness for Code Llama evaluate stage |
| `.[image-compress]` | SD distill / prune / finetune (PyTorch, diffusers) |
| `.[image-compress-onnx]` | ONNX export for image pipeline |

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

## Adaptive RL quantization

Trains a reinforcement-learning policy for adaptive GGUF quantization levels.

### Forge

1. `seiso forge` → **RL Quant** (`/rl-quant`)
2. Choose preset (`minimal`, `post_train`, etc.) and optional checkpoint/GGUF paths
3. Outputs under `{SEISO_DATA_DIR}/rl_quant/{user_id}/`

Smoke config reference: `configs/rl_quant_smoke.json`.

## Platform notes

| Platform | LLM compress | Image compress | RL quant |
|----------|--------------|----------------|----------|
| Linux NVIDIA | ✓ (CUDA kernels in training stages) | ✓ | ✓ |
| Linux AMD ROCm | ✓ (Triton fallback) | ✓ | ✓ |
| Windows NVIDIA | ✓ (CUDA JIT) | ✓ | limited* |
| macOS | CPU/MPS (slow for large models) | MPS/CPU | limited* |

\* RL quant depends on vendored Rust binaries; Linux is the primary target.

For upstream pipeline details, see vendored READMEs:

- `third_party/codellama-compress/README.seiso.md`
- `third_party/sd-distill-prune-quant/README.seiso.md`
- `third_party/adaptive-rl-quant/README.md`
