# Training quickstart

## Forge (web UI)

```bash
cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

1. Open **http://127.0.0.1:8765** and sign in
2. Go to **Training Studio** (`/train`)
3. Settings are pre-filled from local hardware detection
4. Click **Start training** — logs stream over SSE

## CLI

```bash
seiso train --config configs/example_lora.yaml
```

## Example config

```yaml
model_id: meta-llama/Llama-3.2-3B-Instruct
dataset: ./data/sample.jsonl
output_dir: ./outputs/lora-run
method: lora
quant: 4bit
epochs: 1
batch_size: 2
max_seq_length: 2048
gradient_checkpointing: true
use_triton: true       # fused RMSNorm + SwiGLU (GPU)
use_fused_ce: true     # fused cross-entropy loss
```

## Outputs

Checkpoints land in `output_dir/checkpoint-<timestamp>/` with `seiso_manifest.json` including kernel metadata.

## Platform notes

| Platform | Quant | Fused kernels |
|----------|-------|---------------|
| Linux NVIDIA | 4-bit QLoRA | ✓ CUDA |
| Linux AMD ROCm | 4-bit* | ✓ Triton |
| Windows NVIDIA | 4-bit QLoRA | ✓ CUDA JIT |
| macOS | 16-bit LoRA only | ✗ |

\* bitsandbytes on ROCm depends on your PyTorch build.

See [kernels.md](kernels.md) and the [platform guides](../README.md#platform-guides).
