# Training quickstart

Fine-tune open models with QLoRA, LoRA, full fine-tuning, or slime-style post-training using Forge Training Studio or the CLI.

**Prerequisites:** Seiso installed with `[train]` extra. See [install.md](../install.md).

For a step-by-step runbook covering supervised training, slime, compression, Distill-RL, RL quant, and quant regression studies, see [pipelines.md](pipelines.md).

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

Enable **Multi-GPU** in Training Studio or set `multi_gpu: true` in YAML. Forge and
the CLI launch:

```bash
accelerate launch --multi_gpu --num_processes=N --module seiso.training.worker --config <yaml>
```

Distributed launches use [huggingface/accelerate](https://github.com/huggingface/accelerate).
Single-GPU training is unchanged when distributed mode is disabled.

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

For release-style single-GPU post-training with rollout rewards, verifier data, best checkpoints, and auto-stop:

```bash
seiso train --config configs/example_training_slime.yaml
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
| `method` | `lora`, `full`, `embedding`, or `slime` |
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
| `use_rslora` | Rank-stabilized LoRA (RSLoRA) |
| `neftune_noise_alpha` | NEFTune noise for instruction tuning (`null` disables) |
| `early_stopping` | Stop when eval loss plateaus (uses `eval_split_ratio` / `max_eval_samples`) |
| `deterministic` | Reproducible seeds and CUDA settings (`false` enables TF32 + cuDNN benchmark on CUDA) |
| `torch_compile` | Opt-in `torch.compile` on CUDA |
| `extra` | Extension dict — e.g. `use_fused_lora_qkv: true` for batched attention LoRA (see [kernels.md](kernels.md)) |
| `packing` | Sequence packing (large plain-text corpora) |
| `padding_free` | Padding-free packing with flash attention (CUDA + packing) |
| `multi_gpu` | Enable distributed training (or Forge checkbox) |
| `distributed_strategy` | `auto`, `none`, or `ddp` high-level launch policy |
| `distributed_nproc_per_node` | Local Accelerate workers (`null` = all visible GPUs) |
| `distributed_num_nodes` | Total machines for multi-machine Accelerate |
| `distributed_node_rank` | Rank of this node in a multi-node run |
| `distributed_master_addr` / `distributed_master_port` | Accelerate rendezvous endpoint |
| `ddp_backend` | Optional DDP backend (`null`, `nccl`, `gloo`, etc.) |

Modern training defaults (bf16 compute on CUDA when supported, paged AdamW 8-bit for 4/8-bit quant, non-reentrant gradient checkpointing, cosine LR schedule) are applied automatically in `seiso/training/practices.py`.

---

## Slime Post-Training

Use `method: slime` for **slime-style GRPO** post-training (Hugging Face generate + policy update, not the full THUDM/slime Megatron+SGLang Ray stack) when you want rollout generation, reward/verifier traces, checkpointing, and automatic stopping around one local causal LM. Single-process runs keep CPU work bounded while the GPU does rollout and policy updates; multi-GPU runs use **data-parallel DDP** (Accelerate) and shard prompt groups across ranks — see `configs/example_training_slime_ddp.yaml`.

### High-level data generation (required for meaningful signal)

Tiny hand-written smoke JSONL (tens of easy arithmetic items) does **not**
produce useful GRPO: outcome rewards are nearly uniform, dynamic sampling drops
all groups, and training ends with `no_trainable_groups`.

Seiso ships a **high-level data generator** that builds large, deterministic,
checkable prompt corpora (**prompts + labels/tests only** — completions always
come from online rollouts):

```bash
# Standalone (inspect before training)
python -m seiso.rl_verify.data_gen \
  --out data/slime_generated.jsonl \
  --count 500 \
  --mix numeric:0.5,choice:0.2,code:0.3 \
  --difficulty easy:0.35,medium:0.45,hard:0.20 \
  --seed 17 --print-summary
```

Or enable generation inside the slime config (`data_gen: true`) so training
materializes `output_dir/slime_generated.jsonl` automatically:

| Field | Meaning |
|-------|---------|
| `data_gen` | Turn on high-level corpus generation before the first rollout |
| `data_gen_count` | Prompt count (prefer **200+**; 400–2000 for real runs) |
| `data_gen_mix` | Stream mix: `numeric` / `choice` / `code` |
| `data_gen_difficulty` | `easy` / `medium` / `hard` weights |
| `data_gen_seed` | Deterministic seed (same seed ⇒ same corpus) |
| `reward: auto` | Per-row checker from generated `reward` / `benchmark` fields |
| `rollout_backend` | `hf` (default, colocated generate) \| `sglang` \| `auto` |
| `sglang_base_url` | Required for `sglang` (e.g. `http://127.0.0.1:30000`) |

**Single-GPU:** `rollout_backend: hf`. **Multi-GPU example:** `sglang` + DDP policy. Logprobs always recomputed on the actor.

Streams:

- **numeric** — multi-step arithmetic / word problems with exact answers  
- **choice** — multiple-choice with letter labels  
- **code** — unit-test-grounded programs (sandbox-verified goldens)

Start from `configs/example_training_slime.yaml`:

```yaml
method: slime
model_id: Qwen/Qwen2.5-0.5B-Instruct
dataset: data/slime_sample.jsonl   # placeholder when data_gen is true
reward: auto
max_vram_gb: 16
rollouts_per_prompt: 4
rollout_batch_size: 4
dynamic_sampling_filter: reward_nonzero_std
over_sampling_batch_size: 8
balance_data: false
policy_micro_batch_size: 2
batch_size: 1
learning_rate: 0.000005
# 0 saves VRAM; use ~0.02–0.05 for multi-epoch runs (loads a frozen ref model).
kl_coef: 0.0
require_thinking_trace: true
format_reward_weight: 0.1
process_reward_weight: 0.0
missing_thinking_penalty: 0.5
slime_use_lora: true
auto_stop: true
auto_stop_metric: reward_mean
write_verifier_data: true
data_gen: true
data_gen_count: 400
data_gen_mix: "numeric:0.55,choice:0.15,code:0.30"
```

Bundled smoke datasets (expand for real training):

| Dataset | Checker | Config |
|---------|---------|--------|
| `data/slime_sample.jsonl` | `numeric` | `configs/example_slime_single_gpu.yaml`, `configs/example_training_slime.yaml` |
| `data/slime_code_sample.jsonl` | `code` (unit-test pass fraction) | `configs/example_slime_code.yaml` |
| `data/slime_choice_sample.jsonl` | `choice` | `configs/example_slime_choice.yaml` |

Important fields:

| Field | Description |
|-------|-------------|
| `max_vram_gb` | Upper VRAM cap used to fail before out-of-memory conditions |
| `prompt_field`, `answer_field` | Dataset columns for prompts and target answers |
| `metadata_field` | Optional upstream-style metadata column, default `metadata`; JSON strings are parsed and carried into reward samples and bounded verifier records |
| `reward` | Verifier checker: `exact_match`, `numeric`, `choice`, `contains_answer`, `field`, `code` (unit-test pass fraction; 1.0 = all tests pass), or `auto` |
| `reward_field` | Dataset reward column when `reward: field` |
| `require_thinking_trace` | When true, rollout prompts may end with open `<think>`. Format is OK if the **generation** closes thinking: either a full `<think>...</think>` block or a continuation that only emits `</think>` then the answer |
| `outcome_reward_weight` | Weight for hard outcome (correctness) from the shared verifier |
| `format_reward_weight` | Small bonus when the completion contains a closed thinking block (side channel; keep below outcome weight) |
| `process_reward_weight` | Experimental lexical process score; keep `0` for verifiable outcome-first RL |
| `missing_thinking_penalty` | Penalty when format is required but the model omits a closed think block |
| `min_thinking_tokens` | Only used when `process_reward_weight > 0` |
| `kl_coef` | Coefficient on KL to a frozen reference model; `0` skips loading the ref (lower VRAM). Prefer `0.01`–`0.05` for longer post-training runs |
| `rollouts_per_prompt` | slime `--n-samples-per-prompt` |
| `rollout_batch_size` | slime `--rollout-batch-size` (**prompts**, not sequences) |
| `train_batch_size` | Target prompts after dynamic filter; `null` → same as `rollout_batch_size` |
| `over_sampling_batch_size` | slime oversample; when set under filtering must be **≥ `rollout_batch_size`** (prompts) |
| `answer_field` | slime `--label-key` (default `label`; also accepts `answer`) |
| `apply_chat_template` | slime `--apply-chat-template` (default true) |
| `rollout_backend` | `hf` (colocated generate) \| `sglang` \| `auto` (`data_gen` is an alias of `hf`) |
| `dynamic_sampling_filter` | slime-style nonzero-std filter on **outcome** reward |
| `clip_ratio` / `clip_ratio_high` | slime `eps_clip` / `eps_clip_high` |
| `grpo_std_normalization` | slime group mean/std advantages |
| `calculate_per_token_loss` | slime per-token vs per-sample loss |
| `balance_data` | Distributed prompt-length balancing |
| `policy_micro_batch_size` | Policy update microbatch size to control VRAM |
| `shuffle_buffer_size` | Bounded CPU shuffle buffer for long datasets |
| `max_samples_per_epoch` | Optional per-epoch cap for smoke runs or data-efficient loops |
| `slime_use_lora` | Train LoRA adapters instead of full model weights |
| `auto_stop_*` | Plateau detection; defaults monitor `reward_mean` (also logs `group_pass_rate`) |
| `best_checkpoint_dir` | Directory under `output_dir` for the best observed metric checkpoint |
| `write_verifier_data` | Writes JSONL with outcome, format, checker, extracted answer, proof fields, and status per rollout |
| `verifier_max_text_chars` | Per-field text cap to keep verifier JSONL bounded |

### Code rewards (sandboxed proofs)

Use `reward: code` with dataset rows that include unit tests. The shared verifier
extracts Python from the completion (fenced blocks preferred) and runs tests in a
restricted subprocess (`seiso.codellama_compress.code_exec`). Outcome score is the
**pass fraction** of tests (1.0 only if all pass).

Example config: `configs/example_slime_code.yaml` with `data/slime_code_sample.jsonl`.

```json
{
  "prompt": "Write add(a, b).",
  "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
  "solution": "def add(a, b):\n    return a + b\n",
  "timeout_s": 3,
  "benchmark": "code",
  "synth": true
}
```

- `tests` / `test`: assert lines (list or string) or a full check harness  
- `solution`: optional known-good program (for SFT / synthetic DPO; **ignored by the slime reward**, which only scores model completions)  
- `prompt_code` / `code_prefix`: optional HumanEval-style prefix prepended before the solution  
- `setup`: optional imports/helpers before the solution  
- `timeout_s`: wall budget for the sample (split across test units)

This is a **checkable proof**, not lexical process reward. Do not run untrusted
code on sensitive hosts; the sandbox is best-effort, not a full VM.

#### Deterministic synthetic code (guaranteed passers)

Do **not** rely on an LLM to invent solutions for the dataset. Seiso synthesizes
code tasks **deterministically** so every row has a solution that already passes
its unit tests (fail-closed via the same sandbox verifier):

1. Hand-authored pure-function catalog + seeded I/O variants  
2. **Tests derived** from I/O cases (same source of truth as the solution)  
3. Sandbox check: drop any task whose golden solution fails  
4. **Hard negatives** = deterministic mutants of the golden (must fail ≥1 test)

```bash
# Rewrite data/slime_code_sample.jsonl, data/distill_code_synth.jsonl,
# and data/synthetic_code_preferences.jsonl
python -m seiso.rl_verify --data-dir data --seed 0
```

| Artifact | Use |
|----------|-----|
| `data/slime_code_sample.jsonl` | Slime `reward: code` prompts + tests (+ `solution` metadata) |
| `data/distill_code_synth.jsonl` | Distill prompt library for verifiable code rollouts |
| `data/synthetic_code_preferences.jsonl` | Offline DPO pairs (golden chosen, mutant rejected) — no model rollouts required |

Same `--seed` ⇒ same catalog order and mutants. Online slime GRPO still samples
the policy; the golden `solution` is not injected into the reward path.

**Hard negatives (DPO / distill-RL):** when a group of rollouts for the same prompt
contains both a verifier pass and fails, Distill-RL keeps:

- `chosen` = a completion that **passes** (all unit tests for `code`; outcome score > 0.5 for math/choice)  
- `rejected` = a **fail**, preferring near-miss / partial pass (hard negative)

Empty/syntax-only fails are weaker negatives and only used if no stronger fail exists.
Pairs with no pass in the group are dropped. This is appropriate for **offline
preference** learning; online slime GRPO already demotes fails via group rewards
and does not need a separate hard-negative loss.

For Distill-RL preference rollouts on verifiable tasks, start from
`data/distill_verifiable_prompts.jsonl` (math + choice + code) rather than the
alignment-style post-train library.

Slime checkpoints are exportable like other Seiso checkpoints. LoRA slime runs are treated as adapter checkpoints; non-LoRA slime runs are treated like full checkpoints. In distributed SLIME runs, rank 0 writes shared checkpoints and metrics, while verifier JSONL is rank-scoped to avoid concurrent writes.

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

For `method: slime`, the final checkpoint lands in `output_dir` by default, or `output_dir/final_checkpoint_dir` when configured. Slime also writes:

- `checkpoint-best/` — best metric checkpoint by `auto_stop_metric`
- `slime_single_gpu_metrics.jsonl` — per-step loss, reward, best metric, and stop state
- `slime_training_state.json` — final step, stop reason, and best checkpoint metadata
- `slime_verifier_data.jsonl` — prompt/answer/completion/reward rows when `write_verifier_data: true`

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
