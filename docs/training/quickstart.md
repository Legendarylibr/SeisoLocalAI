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
2. Download a **safetensors** base model from **Model Hub** (`/hub`) — GGUF mirrors are inference-only and cannot be used for LoRA/QLoRA training
3. Go to **Training Studio** (`/train`)
4. Pick a **base model** and **dataset** (Hugging Face hub ID, uploaded JSONL, or local path under your sandbox)
5. Wait for **dataset analysis** — Seiso scans the **entire** dataset, detects schema, normalizes rows, and suggests format/hyperparameters from the data (not from chat defaults)
6. Review or override settings — hardware caps still apply (batch size, VRAM, fused kernels)
7. Click **Start training** — logs stream over SSE in real time
8. Checkpoints appear under `{SEISO_DATA_DIR}/checkpoints/{user_id}/{job_id}/`

### Dataset analysis (Training Studio)

When you select a dataset, Forge calls `POST /api/training/analyze-dataset`. The report includes:

- Detected **format** (`auto`, `chat`, `alpaca`, `sharegpt`, `preference`, `text`)
- **Domain** label (instruction tuning, Q&A, conversational, code corpus, plain text, …)
- Row retention after normalization and deduplication
- Suggested `max_seq_length`, `epochs`, `warmup_ratio`, and response-only loss
- Preview of normalized rows

Training also writes `dataset_analysis.json` beside each checkpoint for reproducibility.

Preflight validation (`POST /api/training/validate-dataset`) and job start use the same full-corpus analysis — not a partial sample.

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
preprocess_dataset: true
deduplicate_dataset: true
use_triton: true          # fused RMSNorm + SwiGLU (GPU)
use_fused_ce: true        # fused cross-entropy loss
use_fused_lora: true      # fused LoRA delta (CUDA, rank ≤ 64)
neftune_noise_alpha: 5.0  # instruction-tuning noise (null to disable)
seed: 42
save_steps: 50
```

### Key config fields

| Field | Description |
|-------|-------------|
| `model_id` | Hugging Face model ID or local safetensors path |
| `dataset` | Hub ID, JSONL/JSON path, or directory |
| `dataset_format` | `auto`, `chat`, `alpaca`, `sharegpt`, `preference`, or `text` |
| `method` | `lora`, `full`, or `embedding` |
| `quant` | `4bit`, `8bit`, `16bit`, or `none` |
| `preprocess_dataset` | Normalize and clean rows before training |
| `deduplicate_dataset` | Drop exact duplicate rows after normalization |
| `train_on_responses_only` | Mask loss to assistant/output tokens (non-text formats) |
| `assistant_only_loss` | TRL-native masking when the trainer tokenizes chat rows (`null` = auto) |
| `dataset_num_proc` | Parallel workers for dataset map (`null` = auto, `0` = off) |
| `pad_to_multiple_of` | Batch padding multiple for tensor cores (`null` = 8 on CUDA) |
| `warmup_ratio` | Linear warmup fraction (analysis may suggest 0.03–0.1 by corpus size) |
| `use_triton` | Enable fused RMSNorm + SwiGLU MLP |
| `use_fused_ce` | Fused cross-entropy in SFTTrainer |
| `use_fused_lora` | Fused LoRA delta kernel (CUDA) |
| `neftune_noise_alpha` | NEFTune noise for instruction tuning (`null` disables) |
| `packing` | Sequence packing (large plain-text corpora) |
| `padding_free` | Padding-free packing with flash attention (CUDA + packing) |
| `multi_gpu` | Enable distributed training (or Forge checkbox) |

Modern training defaults (bf16 compute on CUDA when supported, paged AdamW 8-bit for 4/8-bit quant, non-reentrant gradient checkpointing, cosine LR schedule) are applied automatically in `seiso/training/practices.py`.

---

## Dataset formats

Seiso accepts JSONL, JSON, local dataset directories, and Hugging Face hub IDs. Set `dataset_format: auto` to detect schema from stratified samples across the full corpus.

### Chat / messages

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Instruction (Alpaca-style)

```json
{"instruction": "Summarize this.", "input": "optional context", "output": "..."}
{"prompt": "Write hello world", "completion": "print('hello')"}
{"query": "What is 2+2?", "response": "4"}
{"question": "...", "answer": "..."}
```

### ShareGPT

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### Preference (chosen side used for SFT)

```json
{"chosen": "...", "rejected": "..."}
```

### Plain text / code

```json
{"text": "..."}
```

Sample file: `data/sample.jsonl` (chat format, four rows).

---

## Training API (Forge)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/training/recommendations` | Hardware + dataset-aware config hints |
| `POST` | `/api/training/analyze-dataset` | Full-corpus schema analysis |
| `POST` | `/api/training/validate-dataset` | Preflight before job start |
| `POST` | `/api/training/jobs` | Start training job |
| `GET` | `/api/training/jobs/{id}/stream` | SSE logs and metrics |

---

## Outputs

Checkpoints land in `output_dir/checkpoint-<timestamp>/` with:

- LoRA adapter weights (`adapter_model.safetensors`) or full weights for `method: full`
- `seiso_manifest.json` — kernel metadata, quant settings, training config snapshot
- `dataset_analysis.json` — corpus analysis used for the run (when preprocessing ran)
- `train_config_snapshot.json` — resolved `TrainConfig` at job start
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