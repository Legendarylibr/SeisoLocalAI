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

## Training vs inference

Training **never** uses MLX — `load_model(..., for_training=True)` forces PyTorch (CUDA, ROCm, MPS, or CPU).
