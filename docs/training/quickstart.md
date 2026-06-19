# Training quickstart

Fine-tune open models with QLoRA, LoRA, or full fine-tuning using Forge Training Studio or the CLI.

**Prerequisites:** Seiso installed with `[train]` extra. See [install.md](../install.md).

---

## Forge (web UI)

```bash
start
# or: seiso forge
```

1. Open **http://127.0.0.1:8765** and sign in
2. Download a base model from **Model Hub** (`/hub`) if you haven't already
3. Go to **Training Studio** (`/train`)
4. Review settings — pre-filled from hardware detection:
   - **Quant method:** 4-bit QLoRA (NVIDIA), 16-bit LoRA (macOS), auto-detected
   - **Fused kernels:** enabled on CUDA/ROCm when available
   - **Batch size / seq length:** tuned to your VRAM
5. Upload a JSONL dataset or use the bundled sample (`data/sample.jsonl`)
6. Click **Start training** — logs stream over SSE in real time
7. Checkpoints appear under `{SEISO_DATA_DIR}/checkpoints/{user_id}/{job_id}/`

### Multi-GPU

Enable **Multi-GPU** in Training Studio or set `multi_gpu: true` in YAML. Forge launches:

```bash
torchrun --nproc_per_node=N -m seiso.training.worker --config <yaml>
```

See [multi-gpu.md](multi-gpu.md).

---

## CLI

```bash
source .venv/bin/activate
seiso train --config configs/example_lora.yaml
```

On Linux NVIDIA bare metal, approve native CUDA kernel JIT:

```bash
export SEISO_NVIDIA_HOST_VENV_ACK=1
seiso train --config configs/example_lora.yaml
```

---

## Example config

`configs/example_lora.yaml`:

```yaml
model_id: meta-llama/Llama-3.2-3B-Instruct
dataset: ./data/sample.jsonl
output_dir: ./outputs/lora-run
method: lora
quant: 4bit              # use 16bit on macOS
dataset_format: auto
epochs: 1
batch_size: 2
learning_rate: 0.0002
max_seq_length: 2048
lora_r: 16
lora_alpha: 32
gradient_accumulation_steps: 4
gradient_checkpointing: true
train_on_responses_only: true
use_triton: true         # fused RMSNorm + SwiGLU (GPU)
use_fused_ce: true       # fused cross-entropy loss
use_fused_lora: true     # fused LoRA delta (CUDA, rank ≤ 64)
seed: 42
save_steps: 50
```

### Key config fields

| Field | Description |
|-------|-------------|
| `model_id` | Hugging Face model ID or local path |
| `dataset` | JSONL path (chat format with `messages` array) |
| `method` | `lora`, `full`, or `embedding` |
| `quant` | `4bit`, `8bit`, or `16bit` |
| `use_triton` | Enable fused RMSNorm + SwiGLU MLP |
| `use_fused_ce` | Fused cross-entropy in SFTTrainer |
| `use_fused_lora` | Fused LoRA delta kernel (CUDA) |
| `multi_gpu` | Enable distributed training |

---

## Dataset format

JSONL with one object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Sample file: `data/sample.jsonl`

---

## Outputs

Checkpoints land in `output_dir/checkpoint-<timestamp>/` with:

- LoRA adapter weights (`adapter_model.safetensors`)
- `seiso_manifest.json` — kernel metadata, quant settings, training config
- Tokenizer files

Export after training: [getting-started.md § Step 6](../getting-started.md#step-6--export-and-deploy) or **Export** page in Forge.

---

## Platform notes

| Platform | Quant | Fused kernels | Notes |
|----------|-------|---------------|-------|
| Linux NVIDIA | 4-bit QLoRA | ✓ CUDA native | Set `SEISO_NVIDIA_HOST_VENV_ACK=1` on bare metal |
| Linux AMD ROCm | 4-bit* | ✓ Triton | Install Triton manually |
| Windows NVIDIA | 4-bit QLoRA | ✓ CUDA JIT | VS Build Tools for kernel compile |
| WSL2 + NVIDIA | 4-bit QLoRA | ✓ CUDA + Triton | Set `SEISO_NVIDIA_WSL_ACK=1` |
| macOS | 16-bit LoRA only | ✗ | MPS for Apple Silicon; keep models 1–3B |

\* bitsandbytes on ROCm depends on your PyTorch build.

### macOS training tips

```yaml
quant: 16bit
use_triton: false
use_fused_ce: false
max_seq_length: 1024
gradient_checkpointing: true
```

See [kernels.md](kernels.md) and [platform guides](../README.md#platform-guides).
