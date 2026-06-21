# Inference backends

Seiso routes chat/inference by platform and model format.

| Backend | Platform | Install | Use case |
|---------|----------|---------|----------|
| **llama.cpp (GGUF)** | All | `.[llamacpp]` | Default on NVIDIA when GGUF on disk |
| **MLX** | macOS Apple Silicon | `.[mlx]` | Fast local chat on M-series |
| **PyTorch** | CUDA / MPS / CPU | `.[train]` | HF weights, 4-bit via bitsandbytes |
| **Ollama** | All | External | CPU-only tier fallback |

## Detection

- `seiso.models.loader.detect_backend()` — inference
- `forge/services/hardware.py` — `preferred_inference_backend` in UI
- Forge Chat model picker shows backend per model

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

- **Free memory** (Chat or Model Hub) unloads llama.cpp, MLX, PyTorch, and Ollama models from RAM/VRAM. Disk cache under `{SEISO_DATA_DIR}/hf_cache/` is unchanged.
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
