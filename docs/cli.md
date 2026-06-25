# CLI reference

Run commands from the **repository root** with your virtualenv active:

- Linux / macOS / WSL: `source .venv/bin/activate`
- Windows: `.\.venv\Scripts\Activate.ps1`

Install entry points:

| Command | Installed as |
|---------|----------------|
| `seiso` | `pip install -e ".[forge,train,...]"` |
| `seiso-bench-kernels` | same (script entry point) |
| `seiso-train-worker` | same (multi-GPU worker; used via `torchrun`) |

Helper scripts (repo `scripts/`, not on `PATH`):

| Command / script | Purpose |
|------------------|---------|
| `start` | Install or launch Forge — on `PATH` via `~/.local/bin` after install |
| `./scripts/install.sh` | Lower-level installer (system deps, venv, pip extras, UI build) |
| `./scripts/start.sh` | Lower-level launcher (`seiso forge --open`; used by `start`) |
| `./scripts/doctor.sh` | Diagnose install, HF, GPU stack (runs automatically on install/start failure) |
| `./scripts/precheck.sh` | Fast local CI gate (`make precheck`) |
| `./scripts/install_flash_attn.sh` | Optional Flash Attention (Linux NVIDIA) |
| `start-router-vllm` | vLLM Smart Router Docker stack — on `PATH` after install |
| `./scripts/start-router-vllm.sh` | Lower-level vLLM router launcher (used by `start-router-vllm`) |

---

## `seiso forge`

Launch the Forge web server (API + built UI).

```bash
seiso forge
seiso forge --open             # open browser when /health is ready (default via start.sh)
seiso forge --reload          # auto-reload Python on code changes
seiso forge --port 8766       # custom port
```

Requires `forge-ui/dist` — build with `cd forge-ui && npm run build` or use `start`.

Open **http://127.0.0.1:8765**. OpenAI-compatible chat: **http://127.0.0.1:8765/v1/chat/completions** (no `/api` prefix).

## `seiso doctor`

Diagnose Python, Node, HF, and optional GPU packages.

```bash
seiso doctor
seiso doctor --network   # also probe huggingface.co
```

Delegates to `./scripts/doctor.sh` when run from a clone.

## `seiso train`

Fine-tune from a YAML config.

```bash
seiso train --config configs/example_lora.yaml
```

Example config: `configs/example_lora.yaml` (dataset: `data/sample.jsonl`).

Forge Training Studio runs the same training stack but adds full-dataset analysis, live recommendations, and SSE job streaming via `/api/training/*` (see [training/quickstart.md](training/quickstart.md)).

**Checkpoints (CLI):** written under the YAML `output_dir` (example: `./outputs/lora-run/checkpoint-<timestamp>/`), including `seiso_manifest.json` and `dataset_analysis.json`.

**Checkpoints (Forge UI):** `{SEISO_DATA_DIR}/checkpoints/{user_id}/{job_id}/`

**Default data dir:** `$HOME/.seiso` (Linux/macOS/WSL) or `%USERPROFILE%\.seiso` (Windows)

## `seiso chat`

Terminal chat with a local model.

```bash
seiso chat --model meta-llama/Llama-3.2-3B-Instruct --prompt "Hello"
seiso chat --model /path/to/model.gguf   # interactive mode (omit --prompt)
```

## `seiso export`

Export a training checkpoint to merged weights, LoRA, full fine-tune, or GGUF.

```bash
# CLI training output (example_lora.yaml → ./outputs/lora-run/)
seiso export --checkpoint ./outputs/lora-run/checkpoint-<timestamp> --formats merged,gguf

# Forge training output (Linux/macOS/WSL)
seiso export --checkpoint "$HOME/.seiso/checkpoints/<user>/<job_id>/checkpoint-<timestamp>" --formats merged,gguf

# Forge training output (Windows)
seiso export --checkpoint "$env:USERPROFILE\.seiso\checkpoints\<user>\<job_id>\checkpoint-<timestamp>" --formats merged,gguf

seiso export --checkpoint <path> --profile inference
seiso export --checkpoint <path> --hub-repo user/my-model
seiso export --checkpoint <path> --hub-repo user/my-model --precheck-only
seiso export --checkpoint <path> --profile list
```

Exports land under `{SEISO_DATA_DIR}/exports/` by default.

## `seiso inference`

One-shot inference (alias for single-turn `seiso chat`).

```bash
seiso inference --model meta-llama/Llama-3.2-3B-Instruct --prompt "Summarize Seiso in one sentence."
```

## `seiso bench-inference`

Measure load time, time-to-first-token, and generation throughput.

```bash
seiso bench-inference --model /path/to/model.gguf --max-tokens 128
seiso bench-inference --model <path> --compare    # baseline vs optimized
seiso bench-inference --model <path> --json
```

## `seiso compress`

LLM compression pipeline (vendored `third_party/codellama-compress`). Accepts any HuggingFace causal LM; the `prune` stage requires Llama-family architecture (Llama, CodeLlama, Mistral, etc.).

```bash
# Presets: smoke | full | distill_only | prune_recover | quantize
seiso compress run --preset smoke
seiso compress run --preset full \
  --teacher-model codellama/CodeLlama-13b-hf \
  --student-model codellama/CodeLlama-7b-hf
seiso compress run --preset distill_only \
  --teacher-model meta-llama/Llama-2-13b-hf \
  --student-model meta-llama/Llama-2-7b-hf
seiso compress run --preset prune_recover --model-dir ~/.seiso/checkpoints/<user>/<job>/

# Verify hash-chained manifest (run_dir is under …/runs/<run_id>/)
seiso compress manifest-verify --run-dir "$HOME/.seiso/compress/local/cli/runs/<run_id>"
seiso compress speculative --target-model ./finetuned --draft-model ./distilled --prompt "def fib(n):"
```

CLI output: `{SEISO_DATA_DIR}/compress/local/cli/runs/<run_id>/`.

Requires `.[train]` for GPU stages. Optional `.[compress-quant]` for GPTQ/AWQ, `.[compress-eval]` for lm-eval.

Config reference: `configs/example_compress.json`.

See [compression.md](compression.md).

## `seiso distill-rl`

Teacher-to-student KL distillation, preference rollouts (teacher chosen / student rejected), and DPO fine-tuning with research artifacts. **Auto-sweep** (default on) grid-searches DPO hyperparameters before the final alignment run.

```bash
# List presets (smoke | reproducible | full) and stage order
seiso distill-rl presets

# Fast smoke (uses gpt2 by default — no GPU download required for tiny runs)
seiso distill-rl run --preset smoke

# Full teacher → student with all stages (example: CodeLlama)
seiso distill-rl run --preset full \
  --teacher-model codellama/CodeLlama-13b-hf \
  --student-model codellama/CodeLlama-7b-hf

# Skip distill when a checkpoint already exists
seiso distill-rl run --preset smoke --distilled-path ~/.seiso/distill_rl/cli/<job>/distilled

# Multi-seed reproducibility
seiso distill-rl run --preset reproducible --seeds 13,42,99 --json

# Disable hyperparameter sweep
seiso distill-rl run --preset smoke --no-auto-sweep
```

Requires `.[train]` for GPU stages. Outputs: `{SEISO_DATA_DIR}/distill_rl/cli/<job_id>/` (CLI) or `{SEISO_DATA_DIR}/distill_rl/{user_id}/{job_id}/` (Forge).

Forge equivalent: **Distill-RL** page (`/distill-rl`) or `POST /api/distill-rl/jobs`.

Config references: `configs/distill_rl_smoke.json`, `configs/distill_rl_reproducible.json`.

See [compression.md](compression.md).

## `seiso rl-quant`

Adaptive RL quantization + optional CUDA kernel profile co-training (vendored `third_party/adaptive-rl-quant`). **Auto-sweep** (default on) grid-searches learning rates before the full run.

```bash
# Fast smoke (simulator backend, analytic kernel metrics)
seiso rl-quant run --preset minimal --training-episodes 256

# Kernel RL — joint quant policy + CUDA launch profiles
seiso rl-quant run --preset reproducible --kernel-rl --training-episodes 512

# Live CUDA micro-benchmarks (NVIDIA GPU; slower, ground-truth)
seiso rl-quant run --kernel-rl --kernel-live-benchmark

# Disable hyperparameter sweep
seiso rl-quant run --preset minimal --no-auto-sweep

# Custom sweep grid (JSON/TOML)
seiso rl-quant run --preset minimal --sweep-config configs/my_sweep.json

# List tunable kernel profiles
seiso rl-quant profiles

# Machine-readable summary
seiso rl-quant run --preset minimal --kernel-rl --json
```

Presets: `minimal` | `reproducible` | `post_train`. Backends: `simulator` (default) | `llama_cpp`.

Outputs: `{SEISO_DATA_DIR}/rl_quant/cli/<job_id>/` (CLI user `cli`).

Forge equivalent: **RL Quant** page (`/rl-quant`) or `POST /api/rl-quant/jobs`.

Config reference: `configs/rl_quant_smoke.json`.

## `seiso experiment`

Research benchmarks and regression studies (headless; no Forge server required).

### `seiso experiment quant-regression`

Train one model at several QLoRA quants, export GGUFs, and measure deployment-quant regression (HF merged-weight eval and/or llama.cpp route eval).

```bash
# Default study config (Qwen 3B + MetaMathQA)
seiso experiment quant-regression

# Custom base training YAML (quant overridden per run)
seiso experiment quant-regression -c configs/examples/quant_regression_study.yaml

# Compare training quants and GGUF export variants
seiso experiment quant-regression \
  --quants 4bit,8bit,16bit \
  --gguf-quants q4_k_m,q8_0,f16 \
  --measurement both

# Reuse checkpoints from a prior study
seiso experiment quant-regression --study-dir ~/.seiso/experiments/my-study --skip-training

# Machine-readable report
seiso experiment quant-regression --json
```

Requires `.[train]` and `llama.cpp` (`LLAMA_CPP_DIR` or system `convert_hf_to_gguf`) for GGUF export / route eval. Outputs land under the study `output_dir` from the base YAML (default example: `~/.seiso/experiments/quant-regression-qwen3b-metamath/`).

Config reference: `configs/examples/quant_regression_study.yaml`.

## `seiso router`

Smart Router gateway — routes chat to specialists (llama.cpp or vLLM). vLLM stacks execute completions through **LiteLLM**.

```bash
pip install -e ".[router]"

# Full vLLM Docker stack (Nemotron + vLLM + llama-swap + router)
seiso router --stack vllm
start-router-vllm              # same stack; registered on PATH after install
start-router-vllm -d           # detached

# Router process only (backend stack must already be running)
seiso router                   # llama.cpp config (default)
seiso router --vllm            # vLLM + LiteLLM config

# llama.cpp Docker stack
seiso router --stack llamacpp
```

Forge integration: set `SEISO_MODEL_ROUTER_ENABLED=true` and `SEISO_MODEL_ROUTER_URL=http://127.0.0.1:8780` in `.env`. Chat model picker shows **Smart Router (auto-route)**.

Endpoint: `http://127.0.0.1:8780/v1/chat/completions`. Full stack docs: [deploy/model-router/README.md](../deploy/model-router/README.md).

## `seiso-bench-kernels`

Benchmark fused training kernels (NVIDIA CUDA or AMD Triton).

```bash
seiso-bench-kernels --op all --rows 4096 --hidden 4096 --vocab 32000
seiso-bench-kernels --op rms --dtype bfloat16
```

## Multi-GPU training (`seiso-train-worker`)

Distributed training uses the worker entry point via `torchrun` (not a top-level `seiso` subcommand):

```bash
torchrun --nproc_per_node=2 -m seiso.training.worker --config configs/example_lora.yaml
# equivalent installed script:
torchrun --nproc_per_node=2 seiso-train-worker --config configs/example_lora.yaml
```

See [training/multi-gpu.md](training/multi-gpu.md).

---

## Forge-only workflows (no `seiso` subcommand)

| Workflow | Forge page | API prefix |
|----------|------------|------------|
| Training with SSE job UI | `/train` | `/api/training` |
| Export jobs | `/export` | `/api/export` |
| Knowledge ingest / retrieve | `/knowledge` | `/api/knowledge` |
| Recipe graph jobs | `/recipes` | `/api/recipes` |

Compression and distill-RL pipelines also have CLI equivalents (`seiso compress run`, `seiso distill-rl run`, `seiso rl-quant run`).

Upstream vendor CLIs (optional, not installed by default): `adaptive-rl-quant`, `adaptive-rl-quant-pytorch`, etc. in `third_party/adaptive-rl-quant/`. Prefer `seiso rl-quant run` for the integrated pipeline.
