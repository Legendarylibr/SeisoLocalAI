# CLI reference

Run commands from the **repository root** with your virtualenv active:

- Linux / macOS / WSL: `source .venv/bin/activate`
- Windows: `.\.venv\Scripts\Activate.ps1`

Install entry points:

| Command | Installed as |
|---------|----------------|
| `seiso` | `pip install -e ".[forge,train,...]"` |
| `seiso-bench-kernels` | same (script entry point) |
| `seiso-train-worker` | same (distributed worker; used via Accelerate) |

Helper scripts (repo `scripts/`, not on `PATH`):

| Command / script | Purpose |
|------------------|---------|
| `start` | Install or launch Forge — on `PATH` via `~/.local/bin` after install |
| `./scripts/install.sh` | Lower-level installer (system deps, venv, pip extras, UI build) |
| `./scripts/start.sh` | Lower-level launcher (`seiso forge --open`; used by `start`) |
| `./scripts/doctor.sh` | Diagnose install, HF, GPU stack (runs automatically on install/start failure) |
| `./scripts/precheck.sh` | Fast local CI gate (`make precheck`) |
| `./scripts/install_flash_attn.sh` | Optional Flash Attention (Linux NVIDIA) |

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

Open **http://127.0.0.1:8765**. Compat API chat: **http://127.0.0.1:8765/v1/chat/completions** (no `/api` prefix).

## `seiso doctor`

Diagnose Python, Node, HF, and optional GPU packages.

```bash
seiso doctor
seiso doctor --network   # also probe huggingface.co
```

Delegates to `./scripts/doctor.sh` when run from a clone.

## `seiso train`

Fine-tune or post-train from a YAML config.

```bash
seiso train --config configs/example_lora.yaml
```

Example config: `configs/example_lora.yaml` (dataset: `HuggingFaceH4/no_robots`). CI/format smoke uses `data/sample.jsonl` via `configs/smoke_train_*.yaml`.

Single-GPU slime-style post-training is also a first-class training method:

```bash
seiso train --config configs/example_training_slime.yaml
```

Use `method: slime` for local rollout/reward policy updates with LoRA adapters, verifier JSONL, best/final checkpoints, and plateau auto-stop. See [training/quickstart.md § Slime Post-Training](training/quickstart.md#slime-post-training).

NVIDIA [NeMo RL](https://github.com/NVIDIA-NeMo/RL) is a separate first-class method that shells out to an **external** checkout (Apache 2.0; not vendored into Seiso). Seiso maps YAML → Hydra overrides and runs `uv run python examples/run_*.py` inside that tree:

```bash
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git ~/nemo-rl
export SEISO_NEMO_RL_ROOT=~/nemo-rl   # uv required: https://docs.astral.sh/uv/
seiso train --config configs/example_training_nemo_rl.yaml
```

Use `method: nemo_rl` with `nemo_rl_recipe: grpo|dpo|distillation|smoke`. Dry-run preview: `configs/smoke_nemo_rl.yaml` (`nemo_rl_dry_run: true`). Cite NeMo RL when publishing results — BibTeX and slime-vs-NeMo guidance in [training/quickstart.md § NeMo RL](training/quickstart.md#nemo-rl).

## `seiso slime`

Dedicated single-process slime CLI (same core as `seiso train -c … method: slime`):

```bash
seiso slime --config configs/example_training_slime.yaml
```

## `seiso nemo-rl`

Dedicated NeMo RL launcher (same core as `seiso train -c … method: nemo_rl`). Requires a recursive [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) clone (`SEISO_NEMO_RL_ROOT` or `nemo_rl_root`) and `uv` on PATH (or `SEISO_UV`):

```bash
seiso nemo-rl --config configs/example_training_nemo_rl.yaml
```

Writes `{output_dir}/nemo_rl_launch.yaml` + `seiso_manifest.json` before (and even during dry-run of) the external process. Checkpoints land under NeMo RL’s `checkpointing.checkpoint_dir` (Seiso defaults that to `output_dir`).

For multi-GPU / vLLM / SGLang rollouts, prefer `seiso train` with a slime YAML (or `scripts/run_slime_vllm_ddp.sh`). Forge Training Studio runs the same training stack with full-dataset analysis, live recommendations, and SSE job streaming via `/api/training/*` (see [training/quickstart.md](training/quickstart.md)).

**Checkpoints (CLI):** written under the YAML `output_dir` (example: `./outputs/lora-run/checkpoint-<timestamp>/` for SFT, or `./outputs/slime-train-method/` for slime), including `seiso_manifest.json`. SFT runs also write `dataset_analysis.json`; slime runs write `slime_single_gpu_metrics.jsonl` (stable filename), `slime_training_state.json`, and optional `slime_verifier_data.jsonl`. Implementation lives in `seiso.slime` (legacy import path `seiso.slime_single_gpu` still works). NeMo RL runs write `nemo_rl_launch.yaml` plus `seiso_manifest.json` under `output_dir` (checkpoints themselves land in NeMo RL’s `checkpointing.checkpoint_dir`).

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

LLM compression pipeline (`seiso.codellama_compress`). Accepts any HuggingFace causal LM; the `prune` stage requires Llama-family architecture (Llama, CodeLlama, Mistral, etc.).

```bash
# Default preset is full. smoke is CI-only.
# Presets: full | prune_recover | distill_only | quantize | smoke
seiso compress run --preset full \
  --teacher-model codellama/CodeLlama-13b-hf \
  --student-model codellama/CodeLlama-7b-hf
seiso compress run --preset smoke
seiso compress run --preset distill_only \
  --teacher-model meta-llama/Llama-2-13b-hf \
  --student-model meta-llama/Llama-2-7b-hf
seiso compress run --preset prune_recover --model-dir ~/.seiso/checkpoints/<user>/<job>/

# Verify hash-chained manifest (run_dir is under …/cli-<job_id>/runs/<run_id>/)
seiso compress manifest-verify --run-dir "$HOME/.seiso/compress/local/cli-<job_id>/runs/<run_id>"
seiso compress speculative --target-model ./finetuned --draft-model ./distilled --prompt "def fib(n):"
```

CLI output: `{SEISO_DATA_DIR}/compress/local/cli-<job_id>/runs/<run_id>/`.

Requires `.[train]` for GPU stages. Optional `.[compress-quant]` for GPTQ/AWQ, `.[compress-eval]` for lm-eval.

Config reference: `configs/example_compress.json`.

See [compression.md](compression.md).

## `seiso distill-rl`

Teacher-to-student KL distillation, preference rollouts (teacher chosen / student rejected), and DPO fine-tuning with research artifacts. **Auto-sweep** (default on) grid-searches DPO hyperparameters before the final alignment run.

```bash
# List presets (smoke | reproducible | full) and stage order
seiso distill-rl presets

# Default preset is reproducible (needs dataset_ref for product runs).
seiso distill-rl run --preset reproducible --seeds 13,42,99 --json

# Full teacher → student with all stages (example: CodeLlama)
seiso distill-rl run --preset full \
  --teacher-model codellama/CodeLlama-13b-hf \
  --student-model codellama/CodeLlama-7b-hf

# CI fixture only (gpt2 + bundled prompts)
seiso distill-rl run --preset smoke

# Skip distill when a checkpoint already exists
seiso distill-rl run --preset smoke --distilled-path ~/.seiso/distill_rl/cli/<job>/distilled

# Disable hyperparameter sweep
seiso distill-rl run --preset smoke --no-auto-sweep
```

Requires `.[train]` for GPU stages. Outputs: `{SEISO_DATA_DIR}/distill_rl/cli/<job_id>/` (CLI) or `{SEISO_DATA_DIR}/distill_rl/{user_id}/{job_id}/` (Forge).

Forge equivalent: **Distill-RL** page (`/distill-rl`) or `POST /api/distill-rl/jobs`.

Config references: `configs/distill_rl_smoke.json`, `configs/distill_rl_reproducible.json`.

See [compression.md](compression.md).

## `seiso rl-quant`

Adaptive RL quantization + optional CUDA kernel profile co-training (`seiso.adaptive_quant`). **Auto-sweep** (default on) grid-searches learning rates before the full run.

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

## `seiso provenance`

Nostr attestation of local run-manifest digests (not weights). Relay I/O is
included with `[forge]` (default install); the `[nostr]` extra remains an alias.
Outbound is on by default (`SEISO_ALLOW_NOSTR=0` to disable); relays default to
public digests-only endpoints.

Training runs (default) also write `dataset_merkle.json` and seal
`dataset_merkle_root` in attestation **v2**. Use `dataset-prove` /
`dataset-verify-proof` so a holder of a row can prove corpus membership without
publishing the row — see
[provenance-nostr.md § Membership](provenance-nostr.md#training-data-membership-private-examples).

```bash
seiso provenance keygen
seiso provenance attest path/to/manifest.json --relay wss://relay.example.com
seiso provenance verify path/to/manifest.json
seiso provenance show path/to/manifest.json
seiso provenance dataset-prove path/to/seiso_manifest.json --row row.json -o proof.json
seiso provenance dataset-verify-proof proof.json --manifest path/to/seiso_manifest.json
```

See [provenance-nostr.md](provenance-nostr.md) for Forge UI settings, auto-attest
(`SEISO_NOSTR_ATTEST=1`), privacy limits / non-goals, and how this relates to
`seiso compress manifest-verify`.

## `seiso pay` (opt-in marketplace)

Remote sats marketplace for inference / finetune / RL. **Self-hosted stays free** —
do not enable this for local-only use. Requires `SEISO_ALLOW_PAY=1`.

Settlement uses **opt-in Ark** addresses (`SEISO_OPERATOR_ARK`, `SEISO_PROTOCOL_TREASURY_ARK`).
Without a treasury Ark and without `SEISO_PAY_FAUCET=1`, paid settles fail closed.
**Ark chain settlement is not functional currently** — `SEISO_ARK_BACKEND=bark|second` is reserved for a future client wire; leave unset or use faucet for smoke tests.

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PAY_FAUCET=1   # dev only — never on a public market
# export SEISO_PROTOCOL_TREASURY_ARK=ark1…
# export SEISO_OPERATOR_ARK=ark1…
seiso pay quote --type finetune --preset smoke
seiso pay session create --sats 20000 --scopes inference,finetune,rl
seiso pay job start --type finetune --preset smoke --dry-run
seiso pay serve --host 127.0.0.1 --port 8787   # operator sidecar
```

Default protocol fee is 5% (`SEISO_PROTOCOL_FEE_BPS=500`) added on top of compute.
See [pay/marketplace.md](pay/marketplace.md).

## `seiso mesh` (experimental)

Buzz-coordinated multi-node / shared training. Opt-in (`SEISO_ALLOW_MESH=1`); **no** protocol fee.
Share `SEISO_MESH_TOKEN` out-of-band (never post to Buzz). Post announce/plan/worker receipts to the channel.

```bash
export SEISO_ALLOW_MESH=1
export SEISO_MESH_TOKEN=…   # out-of-band; never post to Buzz
seiso mesh announce --channel "$CHANNEL" --gpus 2
seiso mesh plan --channel "$CHANNEL" --type finetune --nodes 2
seiso mesh worker --plan plan.json --rank 0
```

Prefer local → mesh → paid marketplace when orchestrating from Buzz
([seiso-orchestrate skill](../.agents/skills/seiso-orchestrate/SKILL.md)).
See [training/mesh.md](training/mesh.md).

## External Smart Router

The router backend service now lives in [Legendarylibr/SeisoModelRouter](https://github.com/Legendarylibr/SeisoModelRouter). Run that service separately when you want multi-specialist routing.

Forge integration: set `SEISO_MODEL_ROUTER_ENABLED=true` and `SEISO_MODEL_ROUTER_URL=http://127.0.0.1:8780` in `.env`. Chat model picker shows **Smart Router (auto-route)**.

Expected endpoint: `http://127.0.0.1:8780/v1/chat/completions`.

## `seiso-bench-kernels`

Benchmark fused training kernels (NVIDIA CUDA or AMD Triton), or print
shape → FLOP/byte intensity estimates for those kernels.

```bash
# Timed CUDA/ROCm benches (requires GPU)
seiso-bench-kernels --op all --rows 4096 --hidden 4096 --vocab 32000
seiso-bench-kernels --op rms --dtype bfloat16

# Shape-math intensity (CPU-ok; never gates training)
seiso-bench-kernels --roofline-only --rows 4096 --hidden 4096 --vocab 32000
seiso-bench-kernels --roofline-only --json
# Optional: text roofline then timed benches (no --json)
seiso-bench-kernels --roofline --op all --rows 4096 --hidden 4096
```

**Roofline / SoT bar:** for FP16/BF16 **GEMM-family** ops only, intensity
**I ≥ 300 FLOP/byte** is marked `performance_truth=true` (H100-class dense
TC/HBM reference ridge as a shape-math bar for a strong compute-bound
*candidate* — not a measured device roofline). Elementwise/CE and float32 stay
heuristic. Pass `--intermediate` for real SwiGLU widths (~8/3×hidden often);
default intermediate is `4×hidden`. See [training/kernel-shape.md](training/kernel-shape.md).

## Distributed training (`seiso-train-worker`)

Distributed training uses Hugging Face Accelerate with the worker entry point.
For most runs, set `multi_gpu: true` and use `seiso train --config ...`; the CLI
launches Accelerate for you.

```bash
accelerate launch --multi_gpu --num_processes=2 --module seiso.training.worker --config configs/example_lora.yaml
# equivalent installed script:
accelerate launch --multi_gpu --num_processes=2 seiso-train-worker --config configs/example_lora.yaml
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

Prefer `seiso rl-quant run` for the integrated pipeline; the bundled `seiso.adaptive_quant` package provides the research internals.
