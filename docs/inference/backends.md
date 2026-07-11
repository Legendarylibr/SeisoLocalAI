# Inference backends

Seiso routes chat/inference by platform and model format.

| Backend | Platform | Install | Use case |
|---------|----------|---------|----------|
| **llama-swap (GGUF sidecar)** | NVIDIA Linux / optional elsewhere | external `llama-swap` service | Required default for GGUF on native Linux NVIDIA; isolates crashes from Forge |
| **llama.cpp (GGUF)** | CPU / macOS / explicit override | `.[llamacpp]` | In-process GGUF backend; blocked by default on native Linux NVIDIA |
| **MLX** | macOS Apple Silicon | `.[mlx]` | Fast local chat on M-series |
| **PyTorch** | CUDA / MPS / CPU | `.[train]` | HF weights, 4-bit via bitsandbytes |

## Detection

- `seiso.models.loader.detect_backend()` — inference
- `forge/services/hardware.py` — `preferred_inference_backend` in UI
- Forge Chat model picker shows backend per model

## llama-swap setup

On native Linux NVIDIA, GGUF chat defaults to **Ollama-first** isolation (direct
`/v1/chat/completions` at `SEISO_OLLAMA_URL`). llama-swap is an optional fallback
when Ollama is unavailable. Install with:

```bash
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/scripts/bootstrap/linux-nvidia.sh | bash
```

For manual setup, point Forge at Ollama:

```bash
export SEISO_LLAMASWAP_ENABLED=true
export SEISO_LLAMASWAP_URL=http://127.0.0.1:8080
```

The `start` launcher will try to start the right local sidecars before Forge:
on native Linux NVIDIA it starts Ollama when available, chooses `ollama` only
when healthy, then starts `llama-swap` if the binary is installed. On macOS,
llama-swap uses its `llamacpp` engine when explicitly enabled. Disable launcher
sidecar startup with `SEISO_SIDECAR_AUTOSTART=0`.

Seiso chooses `llamacpp` as the llama-swap engine on macOS. On native Linux
NVIDIA, it prefers `ollama` only when Ollama's local API is healthy, then falls
back to `llamacpp` as a llama-swap-managed subprocess engine. Override with
`SEISO_LLAMASWAP_ENGINE=llamacpp` or `SEISO_LLAMASWAP_ENGINE=ollama`. If your
llama-swap config uses a model key rather than the GGUF file path, set
`SEISO_LLAMASWAP_MODEL`. If your llama-swap config is not in the default
location, set `SEISO_LLAMASWAP_CONFIG=/path/to/config.yaml`.

Forge does not silently fall back to in-process llama.cpp on native Linux
NVIDIA. To accept that risk explicitly, set:

```bash
export SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1
```

### Native Linux throughput (Ollama planner)

On consumer NVIDIA (GeForce / RTX), the sidecar planner aims for **full GPU
offload** when residual VRAM after weights + KV is still ≥ ~4 GB (display /
activation slack). Prefill `num_batch` scales with free VRAM: 128 when tight,
256 mid, **512 when ≥16 GB free** (consumer and workstation). Hard VRAM
reserves and the context clamp remain in place to avoid driver hangs.

**Near-max weight files** (e.g. 12B BF16 GGUF ≈ 23 GB on a 24 GB card) cannot
full-offload. For small contexts (≤4k) the planner automatically **packs more
layers** onto the GPU (higher free-VRAM fraction, lower fixed reserve) and caps
`num_batch` at 256 so compute scratch does not OOM. Smaller models keep the
safe default clamps unchanged. Optional overrides:
`SEISO_SIDECAR_LARGE_WEIGHT_PACK_RATIO`, `SEISO_SIDECAR_LARGE_WEIGHT_PACK_RESERVE_MB`.

| Knob | Safe default behavior | Opt-in higher util |
|------|----------------------|--------------------|
| Layer offload | Full when residual ≥4 GB; footprint throttle only when residual is tight; auto pack for near-max BF16 at ≤4k ctx | `SEISO_OLLAMA_NUM_GPU=-1` or `SEISO_OLLAMA_GPU_LAYER_RATIO=1` |
| Prefill batch | 128 / 256 / 512 by free VRAM (256 cap on large-weight pack) | `SEISO_OLLAMA_NUM_BATCH=512` or `SEISO_SIDECAR_PERF_MODE=1` |
| Free-VRAM ratio | ~0.62 of free (plus hard reserve); ~0.94 when packing near-max weights | `SEISO_SIDECAR_PERF_MODE=1` (~0.70) or `SEISO_SIDECAR_VRAM_BUDGET_RATIO` |
| Profile | `SEISO_INFERENCE_PROFILE=interactive` | `throughput` (perf mode + longer keep-alive) |

Do **not** disable `SEISO_SIDECAR_VRAM_CLAMP` on a display-attached GPU.

## MLX setup (macOS)

```bash
pip install -e ".[mlx]"
```

Models must be MLX-compatible (or converted). Chat route `mlx` in inference API.

## OpenAI-compatible endpoint

With Forge running (`seiso forge`), external tools can call:

```text
POST http://127.0.0.1:8765/v1/chat/completions
```

Set `SEISO_ALLOW_OPENAI_TOOLS=true` to enable tool calling on this endpoint. See [forge.md](../forge.md).

## Training vs inference

Training **never** uses MLX — `load_model(..., for_training=True)` forces PyTorch (CUDA, ROCm, MPS, or CPU).

## Memory management

- **Free Memory** (Chat or Model Hub) unloads llama.cpp, MLX, and PyTorch models from RAM/VRAM. Disk cache under `{SEISO_DATA_DIR}/hf_cache/` is unchanged.
- With llama-swap, **Free Memory** also calls `POST /api/models/unload` on the sidecar so Ollama/llama.cpp subprocesses can release external VRAM before training/export jobs. Set `SEISO_LLAMASWAP_UNLOAD_SCOPE=model` to request model-specific unload first, or `SEISO_LLAMASWAP_UNLOAD_SCOPE=none` to disable sidecar unload calls.
- Seiso keeps **one inference model** loaded at a time in the local pool; switching models unloads the previous one.
- **Headroom refresh:** after Free memory, hardware fit labels update immediately so the next model is not falsely blocked.
- **API:** `GET /api/models/vram` · `POST /api/models/vram/unload` (alias: `POST /api/inference/cancel`)
- **Stream abort only:** `POST /api/inference/cancel-generation` stops generation without unloading.

### Mac RAM tiers (Apple Silicon + Intel)

Sizing uses installed RAM + free headroom, not chip model. Loads are estimated as **GGUF file size + ~0.8 GB**.

| RAM | Comfortable chat (Q4) | Notes |
|-----|----------------------|-------|
| 16 GB | ≤9B (Phi-4 Mini, Gemma 3 4B) | Intel Mac: CPU-only, prefer ≤3B |
| 24 GB | up to ~24B with Free memory first | Close other apps before large loads |
| 32 GB+ | 27B class, some MoE if file fits | MoE still needs full GGUF in RAM (mmap) |

On ≤24 GB unified Macs, Forge prefers **llama.cpp + GGUF** over MLX unless headroom is ample. Tune with `SEISO_LLAMA_*` in `.env.example`.
