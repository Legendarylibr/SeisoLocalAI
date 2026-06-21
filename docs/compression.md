# Model compression

Seiso integrates three compression / alignment pipelines. LLM compression, Distill-RL, and RL quant also have `seiso` CLI subcommands.

| Pipeline | Forge page | API prefix | CLI |
|----------|------------|------------|-----|
| LLM compression | `/compress` | `/api/compress` | `seiso compress run` |
| Distill-RL (teacher → DPO) | `/distill-rl` | `/api/distill-rl` | `seiso distill-rl run` |
| Adaptive RL quant (GGUF) | `/rl-quant` | `/api/rl-quant` | `seiso rl-quant run` |

Vendored sources: `third_party/codellama-compress/`, `third_party/adaptive-rl-quant/`.

## Install extras

Base Forge + training stack:

```bash
pip install -e ".[forge,train,dev]"
```

Optional per pipeline:

| Extra | Purpose |
|-------|---------|
| `.[compress-quant]` | GPTQ / AWQ for LLM compression pipeline (Linux NVIDIA; needs train stack) |
| `.[compress-eval]` | lm-eval harness for LLM compression evaluate stage |
| `.[rl-quant]` | Integrated adaptive RL quant (stdlib; no extra deps) |

Install example:

```bash
pip install -e ".[compress-quant,compress-eval,rl-quant]"
```

If GPTQ build fails on Linux NVIDIA:

```bash
pip install auto-gptq autoawq --no-build-isolation
```

Vendored packages are bootstrapped at runtime from `third_party/` — no separate clone required.

---

## LLM compression

Compress HuggingFace causal LMs via teacher → student distillation, optional MLP pruning, recovery fine-tuning, evaluation, export, and optional GPTQ/AWQ quantization.

**Default models:** `codellama/CodeLlama-13b-hf` (teacher) and `codellama/CodeLlama-7b-hf` (student). Override with any compatible HF model ID or local checkpoint path via CLI, API, or Forge.

### Model compatibility

| Stage | Requirement |
|-------|-------------|
| `distill` | Any `AutoModelForCausalLM`. Teacher and student should share tokenizer vocabulary (same model family is safest). |
| `prune` | **Llama-family only** — targets `layers[i].mlp.{gate_proj, up_proj, down_proj}`. Skip this stage for GPT-2, etc. |
| `finetune`, `evaluate`, `export`, `quantize_*` | Any causal LM loaded by Transformers. |

Default training data is `bigcode/starcoderdata` (Python code). Override via `configs/example_compress.json` or a custom `config_file`.

### Stages

`distill` → `prune` → `finetune` → `evaluate` → `export` → `quantize_gptq` / `quantize_awq` (all configurable).

| Stage | Description |
|-------|-------------|
| `distill` | Teacher → student KL distillation |
| `prune` | Shape-preserving MLP neuron masking (Llama-family) |
| `finetune` | Post-prune recovery fine-tuning |
| `evaluate` | Perplexity + speed smoke check |
| `export` | vLLM / Docker / GGUF helper scripts |
| `quantize_gptq` | GPTQ 4-bit (requires `.[compress-quant]`, Linux NVIDIA) |
| `quantize_awq` | AWQ 4-bit (requires `.[compress-quant]`, Linux NVIDIA) |

### Presets

| Preset | Stages | Notes |
|--------|--------|-------|
| `smoke` | distill, prune, finetune, evaluate, export | 2 distill/finetune steps; tiny sample cap |
| `full` | distill, prune, finetune, evaluate, export | Production step counts |
| `distill_only` | distill, evaluate | Skip prune/finetune |
| `prune_recover` | prune, finetune, evaluate, export | Requires `--model-dir` starting checkpoint |
| `quantize` | quantize_gptq, evaluate, export | Requires `--model-dir` |

### Output layout

```
{SEISO_DATA_DIR}/compress/{user_id}/{job_id}/runs/<run_id>/
├── distilled/          # after distill
├── pruned/             # after prune
├── finetuned/          # after finetune
├── manifest.json       # hash-chained reproducibility record
└── …
```

Forge jobs use your session `user_id` and a UUID `job_id`. CLI `seiso compress run` writes under `{SEISO_DATA_DIR}/compress/local/cli/runs/<run_id>/`.

### Forge

1. `seiso forge` → **Compress** (`/compress`)
2. Pick a preset, set teacher/student models (any HF repo or local path), toggle stages
3. Logs stream over SSE; link a training job to pre-fill `model_dir` for prune-recover runs

### CLI

```bash
seiso compress run --preset smoke
seiso compress run --preset full \
  --teacher-model meta-llama/Llama-2-13b-hf \
  --student-model meta-llama/Llama-2-7b-hf
seiso compress run --preset distill_only \
  --teacher-model mistralai/Mistral-7B-v0.1 \
  --student-model mistralai/Mistral-7B-v0.1
seiso compress run --preset prune_recover --model-dir ~/.seiso/checkpoints/<user>/<job>/

seiso compress manifest-verify --run-dir ~/.seiso/compress/local/cli/runs/<run_id>
seiso compress speculative --target-model ./finetuned --draft-model ./distilled
```

Config reference: `configs/example_compress.json`.

Requires `.[train]` for GPU distillation/finetune stages. Optional `.[compress-quant]` for GPTQ/AWQ, `.[compress-eval]` for lm-eval.

---

## Distill-RL (teacher → DPO)

Teacher KL distillation into a smaller student, preference rollouts (teacher completions preferred over student), DPO alignment, and evaluation. Produces hash-chained manifests and optional multi-seed aggregation.

**Auto-sweep (default on):** Before the final DPO stage, runs a compact grid search over DPO hyperparameters (`dpo_beta`, `dpo_learning_rate`, preset-dependent). Disable with `--no-auto-sweep` (CLI) or `auto_sweep: false` (API). Custom grids via `sweep_config` path or `sweep_grid` in JSON config.

### Presets

| Preset | Default models | Purpose |
|--------|----------------|---------|
| `smoke` | `openai-community/gpt2` | Fast CI-style run (2 distill steps, tiny prompts) |
| `reproducible` | `openai-community/gpt2` | Multi-seed (`13, 42, 99`) research preset |
| `full` | CodeLlama 13B → 7B | Production-scale code alignment |

Stages: `distill`, `rollout`, `dpo`, `evaluate`.

### Output layout

```
{SEISO_DATA_DIR}/distill_rl/{user_id}/{job_id}/
├── distilled/
├── preferences/
├── dpo/
├── evaluation/
├── sweep/              # when auto_sweep runs
└── manifest.json
```

Multi-seed runs aggregate under `{job_id}-multiseed/`.

### Forge

1. `seiso forge` → **Distill-RL** (`/distill-rl`)
2. Pick preset, set teacher/student models, toggle stages and auto-sweep
3. Logs stream over SSE

### CLI

```bash
seiso distill-rl presets
seiso distill-rl run --preset smoke
seiso distill-rl run --preset full \
  --teacher-model codellama/CodeLlama-13b-hf \
  --student-model codellama/CodeLlama-7b-hf
seiso distill-rl run --preset reproducible --seeds 13,42,99
seiso distill-rl run --preset smoke --no-auto-sweep
seiso distill-rl run --preset smoke --distilled-path ~/.seiso/distill_rl/cli/<job>/distilled
```

Config references: `configs/distill_rl_smoke.json`, `configs/distill_rl_reproducible.json`.

Requires `.[train]` for GPU stages (same stack as LLM compression distillation).

---

## Adaptive RL quantization

Trains a reinforcement-learning policy for adaptive GGUF quantization levels. Optionally co-trains CUDA kernel launch profiles (`kernel_rl_enabled`).

**Auto-sweep (default on):** Grid-searches learning rates (and reward weights on `post_train`) before the full training run. Disable with `--no-auto-sweep` or `auto_sweep: false`. Custom grid via `--sweep-config` or API `sweep_config`.

### Presets

| Preset | Backend | Notes |
|--------|---------|-------|
| `minimal` | simulator | Fast smoke (256 episodes) |
| `reproducible` | simulator | Fixed seeds, logged artifacts (Forge default) |
| `post_train` | simulator | Links fine-tune checkpoint for quality sidecar |

Backends: `simulator` (default) or `llama_cpp` (requires `--gguf-path`). Training backend: `stdlib` (default) or `pytorch`.

### Output layout

```
{SEISO_DATA_DIR}/rl_quant/{user_id}/{job_id}/
├── sweep/              # when auto_sweep runs
├── recommendation.json
└── …
```

CLI writes under `{SEISO_DATA_DIR}/rl_quant/cli/<job_id>/`.

### Forge

1. `seiso forge` → **RL Quant** (`/rl-quant`)
2. Enable **CUDA kernel RL** optionally; choose preset and checkpoint/GGUF paths
3. Toggle auto-sweep in experiment config

### CLI

```bash
seiso rl-quant run --preset minimal --training-episodes 256
seiso rl-quant run --preset reproducible --kernel-rl --training-episodes 512
seiso rl-quant run --kernel-rl --kernel-live-benchmark   # NVIDIA CUDA micro-bench
seiso rl-quant run --preset minimal --no-auto-sweep
seiso rl-quant profiles                                  # list kernel launch profiles
```

Smoke config reference: `configs/rl_quant_smoke.json`.

Integrated pipeline only — upstream `adaptive-rl-quant*` CLIs in `third_party/adaptive-rl-quant/` are optional for advanced research; use `seiso rl-quant run` for the Forge-equivalent path.

---

## Platform notes

| Platform | LLM compress | Distill-RL | RL quant |
|----------|--------------|------------|----------|
| Linux NVIDIA | ✓ (CUDA kernels in training stages) | ✓ | ✓ |
| Linux AMD ROCm | ✓ (Triton fallback) | ✓ | ✓ |
| Windows NVIDIA | ✓ (CUDA JIT) | ✓ | ✓ (simulator; live CUDA bench if GPU available) |
| macOS | CPU/MPS (slow for large models) | CPU/MPS (slow for large models) | ✓ (simulator / analytic kernel metrics) |

For upstream pipeline details, see vendored READMEs:

- `third_party/codellama-compress/README.seiso.md`
- `third_party/adaptive-rl-quant/README.md`
