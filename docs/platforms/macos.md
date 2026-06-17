# macOS

## Summary

| Feature | Apple Silicon | Intel Mac |
|---------|---------------|-----------|
| Forge UI | ✓ | ✓ |
| MLX chat inference | ✓ (with `[mlx]`) | — |
| GGUF / Ollama chat | ✓ | ✓ |
| QLoRA 4-bit training | ✗ (no bitsandbytes) | ✗ |
| 16-bit LoRA training | ✓ (MPS, small models) | ✓ (CPU, tiny models) |
| Fused GPU kernels | ✗ | ✗ |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[forge,train,mlx,dev]"
```

## Start Forge

```bash
seiso forge
```

## MLX inference (chat)

Requires `mlx-lm` from the `[mlx]` extra:

```bash
pip install -e ".[mlx]"
```

In Forge Chat, pick a model with MLX backend or let hardware detection prefer MLX on Apple Silicon.

## Training on macOS

Training **always uses PyTorch** (never MLX), even when MLX is installed for chat.

- Use **16-bit LoRA** — Forge pre-fills `quant: 16bit` when bitsandbytes is unavailable
- **MPS** is used automatically when available (`torch.backends.mps`)
- Fused kernel checkboxes are **disabled** in Training Studio (no CUDA GPU)
- Keep models small (1–3B), `max_seq_length` 1024–2048, gradient checkpointing on

```bash
seiso train --config configs/example_lora.yaml
```

Set in YAML:

```yaml
quant: 16bit
use_triton: false
use_fused_ce: false
```

## Build Forge UI

```bash
cd forge-ui && npm install && npm run build
```
