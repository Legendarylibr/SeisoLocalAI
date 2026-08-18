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
| `start` | Install or launch the Seiso TUI — on `PATH` via `~/.local/bin` after install |
| `./scripts/install.sh` | Lower-level installer (system deps, venv, pip extras, UI build) |
| `./scripts/start.sh` | Lower-level launcher (`seiso tui`; `SEISO_UI=forge` for the web API) |
| `./scripts/doctor.sh` | Diagnose install, HF, GPU stack (runs automatically on install/start failure) |
| `./scripts/precheck.sh` | Fast local CI gate (`make precheck`) |
| `./scripts/install_flash_attn.sh` | Optional Flash Attention (Linux NVIDIA) |

---

## `seiso forge`

Launch the Forge web server (API + built UI).

```bash
seiso forge
seiso forge --open             # open browser when /health is ready (opt-in; start defaults to TUI)
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

## `seiso tui`

Default **workspace UI**. Copies the Forge sidebar, Chat, Hub, Dashboard, and studio pages — no browser. Hub searches Hugging Face live (not only files already on disk). `start` launches this.

```bash
seiso tui                 # terminal UI (also what `start` runs)
seiso tui --list          # local GGUF inventory, smallest first
seiso tui --model 1       # pick by index / path / name substring
```

**Move with the keyboard, then press Enter.** `↑`/`↓` (or the mouse wheel) scroll the highlighted `▸` row. `←`/`→` or Tab switch the sidebar and the page. Enter opens the highlighted item (a page, a local model, a Hub download, or a studio config). Type to chat, or start a `/command`.

Same **Nostr account** as Forge: first launch creates a recovery key (`nsec`) or restores one; later sessions unlock from the saved session (24h, same JWT secret as the web UI) or by pasting the key / NIP-49 backup. Settings can rotate, import, sign out, or start a new session (`RESET`). Integrations toggles auto-attest and relays (`/relays wss://…`).

Hub: `/search qwen`, or scroll to a row and press Enter to open (on disk) or download. Chat loads weights on the first message; `/unload` frees RAM/VRAM. Studio pages (`/train`, `/compress`, …) show the CLI — scroll a config and press Enter, or `/run configs/example_lora.yaml`. Optional web API: `SEISO_UI=forge start` or `seiso forge`.

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


## `seiso experiment`

Adaptive RL quantization and quant-regression studies moved to the standalone [Adaptive-RL-Quantization](https://github.com/Legendarylibr/Adaptive-RL-Quantization) research repo. The product CLI keeps an `experiment` group that prints a pointer when invoked without subcommands.

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

> **Not functional yet — do not use.** Scaffolding only. Do not run for production or real funds.

Remote sats marketplace for inference / finetune / RL. **Self-hosted stays free** —
do not enable this for local-only use. Requires `SEISO_ALLOW_PAY=1`.

Settlement payment methods (all live rails **not functional yet — do not use**):
**Ark** addresses (`SEISO_OPERATOR_ARK`, `SEISO_PROTOCOL_TREASURY_ARK`) and
**L402** (Lightning HTTP 402 — see [marketplace.md](pay/marketplace.md) and
[L402 explained](https://lightningfaucet.com/learn/l402-payments-explained/)).
Without a treasury Ark and without `SEISO_PAY_FAUCET=1`, paid settles fail closed.
**Ark chain settlement is not functional currently** — `SEISO_ARK_BACKEND=bark|second`
is reserved for a future client wire; leave unset or use faucet for smoke tests.
**L402 is not functional for live Lightning** — use `SEISO_PAY_L402_SIM=1`
(or faucet) for simulated fund/exchange; see marketplace docs.

```bash
export SEISO_ALLOW_PAY=1
export SEISO_PAY_FAUCET=1   # dev only — never on a public market
# export SEISO_PROTOCOL_TREASURY_ARK=ark1…
# export SEISO_OPERATOR_ARK=ark1…
seiso pay quote --type finetune --preset smoke
seiso pay session create --sats 20000 --scopes inference,finetune,rl
# seiso pay session fund --session ID --sats 20000 --l402   # sim L402
seiso pay job start --type finetune --preset smoke --dry-run
seiso pay serve --host 127.0.0.1 --port 8787   # operator sidecar
```

Default protocol fee is 5% (`SEISO_PROTOCOL_FEE_BPS=500`) added on top of compute.
Failed/cancelled jobs refund escrow to the prepaid session balance (not Lightning).
See [pay/marketplace.md](pay/marketplace.md).

## `seiso mesh` (experimental secondary)

> **Secondary / opt-in.** Local single-node Forge/CLI stays primary. Mesh is
> Buzz-agent-only multi-node coordination — not available from the Forge UI.

Buzz-**agent**-only multi-node / shared training. Opt-in (`SEISO_ALLOW_MESH=1` +
valid `BUZZ_PRIVATE_KEY` nsec); plans are **NIP-01 / BIP-340** signed. **No**
protocol fee. Forge UI keeps full local training config (`nnodes=1`) and refuses
mesh — see `GET /api/training/surface`. Share `SEISO_MESH_TOKEN` (≥16 chars)
out-of-band. **Relay only with signing:** the signed `nostr_event` (NIP-01 +
BIP-340) is channel authority — unsigned receipts are local pointers.
Seiso does not NIP-98 to the relay. On Buzz, embed `nostr_event` JSON in a
kind-9 `buzz messages send` (Buzz rejects `--kind 31251–31254`).

```bash
export SEISO_ALLOW_MESH=1
export SEISO_MESH_TOKEN=…   # ≥16 chars; out-of-band; never post to Buzz
export BUZZ_PRIVATE_KEY=nsec1…   # must be a valid Nostr secret (signing key)
# optional but recommended: export SEISO_MESH_TRUSTED_NPUBS=npub1planner…
# (required unless SEISO_MESH_ALLOW_ANY_PLANNER=1 for single-operator smoke)
seiso mesh announce --channel "$CHANNEL" --gpus 2 >announce.json
jq -c .nostr_event announce.json | buzz messages send --channel "$CHANNEL" --content -
seiso mesh plan --channel "$CHANNEL" --type finetune --nodes 2 --master-addr 10.0.0.1 --gpus-per-node 2 >plan.json
jq -c .nostr_event plan.json | buzz messages send --channel "$CHANNEL" --content -
# peers:
seiso mesh import-plan --event plan_event.json
seiso mesh worker --plan "$JOB_ID" --rank 0 -c configs/smoke_train_gpu.yaml --dry-run
seiso mesh worker --plan "$JOB_ID" --rank 1 -c configs/smoke_train_gpu.yaml --launch --confirm-launch
```

## `seiso route`

Model-aware picker. Prints a `RouteDecision` JSON from local inventory (no GPU).

```bash
seiso route --task chat --context 8192 --vram-mb 8192 \
  --inventory-json '[{"model_id":"qwen-7b","backend":"llamacpp","role":"chat","context_tokens":8192,"vram_mb":5000,"downloaded":true,"params_b":7}]'
# Optional localhost external router when nothing local fits:
seiso route --task chat --external --router-url http://127.0.0.1:8780 --inventory-json '[]'
```

See [ROADMAP.md](../ROADMAP.md) pillar 3.

## `seiso agent` (decide / plan / Buzz-facing signed status)

```bash
# Where should this job run? (local → mesh → pay → ask_human)
seiso agent decide --job finetune --local-healthy
seiso agent decide --job slime --no-local-healthy --mesh-peers --route-class local_then_mesh

# One-step harness plan (default --dry-run: no jobs)
seiso agent plan --dry-run --task chat --goal "smoke chat"
```

Prefer these over ad-hoc scripts; they call `decide_compute` and `run_harness`.

## `seiso agent status` (Buzz-facing signed status)

Generic agent milestones use the same **relay only with signing** policy as mesh:

```bash
export BUZZ_PRIVATE_KEY=nsec1…
seiso agent status --role train --status started --channel "$CHANNEL" --job-id "$JOB" >status.json
jq -c .nostr_event status.json | buzz messages send --channel "$CHANNEL" --content -
# buzz_receipt is a local pointer only — not channel authority
```

Prefer local → mesh → paid marketplace when orchestrating from a Buzz agent
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

Compression and distill-RL pipelines also have CLI equivalents (`seiso compress run`, `seiso distill-rl run`).
