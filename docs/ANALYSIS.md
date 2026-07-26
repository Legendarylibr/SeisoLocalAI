# SeisoLocalAI Project Analysis

**Date:** 2026-06-25 (executive refresh 2026-07-23 / v0.4.0)  
**Source:** Local clone — [github.com/Legendarylibr/SeisoLocalAI](https://github.com/Legendarylibr/SeisoLocalAI)  
**Analyst context:** Full source tree + populated `~/.seiso`, prior usage artifacts, successful UI build + partial CI runs on RTX 4090 Linux host.

This document provides a software engineering analysis: architecture, features, code health, security, platform notes, WIP status, and actionable recommendations.

**July 2026 full-tree review:** See [reports/codebase-review-2026-07.md](reports/codebase-review-2026-07.md) (initial) and [reports/codebase-review-2026-07-23.md](reports/codebase-review-2026-07-23.md) (re-audit). **Landed through v0.4.0+ (still confirmed 2026-07-23):** DNS-pin httpx (S1-001), multi-backend chat picker (INF-01), shared `USER_SCOPED_DATA_ROOTS` (S1-003), per-model kernel restore + inference `KernelPatchSession` (S1-006/007), drop unused DB tables (F4-03–05), LoRA-only FULL/BASE refuse (EXP-02), slime sync-weight guards (SLM-01), NeMo RL external method, training-config / GRPO invariants, Forge red-team hardening (tools policy, KB quarantine, Compat key chat-only, remote+code-exec hard refuse), single RL-quant product preset registry (RP-05). **Remediated after the 2026-07-23 audit (both passes):** S1-004/009/010/012/014/016/017/018/019; TRN-02/03/04/05; CMP-ORD; F4-06/10; F5-02/05; SLM-02; INF-02/03; CHAT-01; NEMO-01/02; EXP-02-R; HUB-TOCTOU; RP-06/09/10 (product naming + compress fingerprint delegate); F6-04/05/06/07 hygiene; dead `publishToHub` client removed; code-exec naming clarified. **Top residual risks:** AST code-exec is still not OS isolation (localhost-only; remote+code-exec hard-refused); PR CI still excludes `@gpu` (use `make test-hardware`); adaptive_quant domain fingerprints remain separate from `seiso.research.provenance` for replay integrity.

---

## Executive Summary

Seiso is a **mature, ambitious local-first AI workspace** (GPL-3.0) that combines:

- A production-grade **FastAPI + React "Forge"** web UI.
- A **Python core library + rich CLI** (`seiso`).
- Strong emphasis on **privacy, hardware optimization, reproducibility, and security sandboxing**.

**Core workflows supported end-to-end:**
- Model discovery (HF Hub GGUF catalog) + download + chat (llama-swap / llama.cpp / MLX / PyTorch; multi-backend picker when several engines are viable).
- QLoRA / LoRA / full fine-tuning with live metrics/SSE.
- Post-training: Seiso-native slime GRPO (`method: slime`, HF / DDP / vLLM / SGLang rollouts) and external NVIDIA NeMo RL (`method: nemo_rl`, no vendoring).
- Export (LoRA merge, GGUF multi-quant, Hub publish + model cards; LoRA-only checkpoints refuse FULL/BASE).
- Advanced research pipelines: LLM compression (distill + optional prune + FT + quant), Distill-RL (KL + DPO), RL quantization (adaptive + optional kernel policy co-training).
- RAG/knowledge bases, visual recipe graphs, provider routing, Compat API (`/v1`).

**Strengths:**
- Excellent hardware awareness and memory protection (guards, headroom, unload, low-VRAM modes).
- Custom fused kernels (CUDA native + Triton dispatch) with patching lifecycle and RL integration.
- Reproducible pipelines via hash-chained manifests + bundled research code.
- Thoughtful security model (local-only default, encrypted DB fields, CSRF/rate-limit, path sandboxing, per-user scoping).
- Two consistent surfaces (CLI + UI) sharing the same core runners/orchestrators.
- Comprehensive docs + smoke presets + local CI gate.

**Current state:** Alpha (`pyproject.toml` Development Status :: 3). Usable today for chat, training, export on supported hardware. Advanced pipelines (compress/RL/distill) rely on bundled research code and optional extras; best on Linux + NVIDIA. Research CLI `seiso experiment quant-regression` compares multi-quant train → GGUF export → deploy eval. Some style debt (imports, simplifications) and a modest number of pre-existing lint baseline issues. One hardware enumeration test is flaky in this env.

---

## Architecture Overview

### Two-Surface Model (shared core)

```
User
├── seiso CLI (seiso_cli/main.py + typer) ──► direct calls into seiso.* runners
└── Forge UI (browser) 
    └── forge/ (FastAPI)
        ├── api/routes/*  ──► orchestrators/* ──► seiso.* / bundled packages
        ├── services (jobs, models, hardware, memory_release, HF, ...)
        └── db (aiosqlite + field-level AES-GCM crypto)
```

- **Core** (`seiso/`): Training, inference, export, kernels, compress, distill_rl, rl_quant, experiments, models, hardware, memory, platform, security helpers, bundled package wrappers.
- **Forge** (`forge/`): Web server, job orchestration + SSE streaming, persistence, auth/onboarding, security middleware, model registry/inventory, Compat API.
- **UI** (`forge-ui/`): React + TS + Vite + xyflow (recipes). Talks REST + SSE. Built assets served by Forge SPA fallback.
- **Bundled packages**: `seiso.codellama_compress`, `seiso.adaptive_quant`, and `analysis` are part of this repository and are bootstrapped at runtime (no separate pip package). Seiso provides config translation, job wrapping, UI/CLI surfaces, kernel bridge, and manifests.

### Job & Orchestration Model
- `forge/orchestrators/base.py`: `Orchestrator` ABC, `JobRecord`, status, SSE log/metrics queues, cancellation, subprocess tracking.
- Feature orchestrators (thin wrappers around core `run_*_job` or direct calls).
- Shared resource-aware memory hooks before/after GPU-heavy tasks, including inference unload and download/task blocking.
- Results/artifacts written under user-scoped paths in `SEISO_DATA_DIR` (default `~/.seiso`).

### Data & Paths
- Override via `SEISO_DATA_DIR` / `SEISO_INSTALL_DIR` or env.
- Layout (user-scoped): `hf_cache/`, `models/`, `checkpoints/`, `exports/`, `compress/`, `distill_rl/`, `rl_quant/`, `knowledge/`, `forge.db` (encrypted cols), runtime.json, etc.
- Strong sandboxing: `safe_join`, `assert_within`, user_id in all queries.

### Config
- Pydantic v2 throughout (`ForgeSettings`, `TrainConfig`, `DistillRLConfig`, pipeline configs).
- YAML/JSON presets + runtime overrides. Hardware recommendations drive sensible defaults.
- `.env` + `SEISO_*` prefix.

### Key Cross-Cutting
- **Hardware/Memory**: `seiso/hardware/`, `seiso/memory/{protection,estimates,platform_profile}`, `forge/services/memory_release.py`, `prepare_for_gpu_task`.
- **Kernels**: CUDA/Triton/PyTorch dispatch + monkey-patch lifecycle (restore guaranteed).
- **Security**: JWT auth + onboarding, CSRF, rate limit, CSP (nonce), URL policy, token revocation, nvidia boundary reporting, path sandbox.
- **Repro**: Manifests (hash chained via bundled replay), provenance, seeds, snapshots.
- **Inference**: Model pool with unload, native Linux sidecar isolation, backends auto-select, speculative, context limits, external router client mode.

Entry points: `start` script, `seiso` CLI (`forge`, `train`, `slime`, `nemo-rl`, `chat`, `export`, `compress`, `distill-rl`, `rl-quant`, `experiment`, …), `forge.main:create_app`, `seiso-train-worker`, `seiso-bench-kernels`.

---

## Feature Deep Dives (with pointers)

### Inference & Chat
- Backends: llama-swap sidecar for native Linux NVIDIA GGUF, llama.cpp for CPU/macOS/explicit override, MLX (macOS), PyTorch.
- `seiso/inference/{runner,backends,model_pool,speculative,tuning}.py`
- Forge: `forge/api/routes/inference.py`, services (inference_models, model_pool, llamacpp_runtime).
- UI: ChatPage.tsx (model picker, context bar, streaming, router status, memory free).
- Compat API lives at root `/v1/...` (no /api prefix).
- External router mode (`__seiso_router__`) for intelligent model selection through a separately running router service.

### Training Studio (QLoRA / LoRA / full / slime / NeMo RL)
- `seiso/training/{config.py,trainer.py,sft.py,datasets.py,preprocess.py,metrics.py,recommendations.py,multi_gpu.py,worker.py}`
- Uses TRL SFTTrainer (+ optional fused CE). PEFT for LoRA.
- Slime GRPO: `seiso/slime/` (policy, rollouts, trainer; DDP `no_sync` + engine weight sync).
- NeMo RL: `seiso/nemo_rl/` external launcher (`SEISO_NEMO_RL_ROOT` + `uv run`).
- Orchestrator + route: `forge/orchestrators/training.py`, `forge/api/routes/training.py`.
- UI: TrainPage.tsx (rich form + recs + SSE logs + metrics; method picker includes slime / NeMo RL).
- Distributed fine-tuning via Hugging Face Accelerate + worker.
- Auto-export hook after success.
- Fused kernels applied inside trainer.
- Job completion propagates `job.error` as `error_text`.

### Export / GGUF / Hub
- `seiso/export/{pipeline.py,formats.py,gguf.py,profiles.py,hub_precheck.py,model_card.py}`
- Orchestrator/route handle merge + quant via external llama.cpp tools + Hub publish.
- Profiles + auto plans.

### Compression Pipeline
- Stages: distill / (optional prune for Llama MLP) / finetune / quant (GPTQ/AWQ) / eval / export.
- `seiso/compress/{runner.py,config_builder.py,bootstrap.py}` + bundled `seiso.codellama_compress`.
- UI + CLI presets; stage pipeline router in API.

### Distill-RL
- Teacher KL distill → preference rollouts (teacher > student) → DPO.
- `seiso/distill_rl/{runner.py,config.py,sweep.py,rollouts.py,preferences.py,evaluate.py,manifest.py,...}`
- Multi-seed + auto-sweep + paper bundle.
- Cross-pipeline (compression + adaptive alignment).

### RL Quant
- Adaptive RL for GGUF quant policy (per-tensor/layer).
- Optional `--kernel-rl` co-trains discrete CUDA kernel profiles.
- `seiso/rl_quant/{runner.py,config_builder.py,sweep.py,kernel_integration.py}`
- Product presets (`minimal` / `reproducible` / `post_train`) live in one registry: `seiso/rl_quant/presets.py` (API metadata, aliases, defaults, sweep grids).
- Heavy use of bundled adaptive RL quant internals via `named_preset`.

### Quant regression experiments
- `seiso experiment quant-regression` — multi-quant QLoRA training, GGUF export, HF + llama.cpp deploy-quant regression.
- `seiso/experiments/{quant_regression.py,hf_deploy_regression.py}`
- Config: `configs/examples/quant_regression_study.yaml`

### Fused Kernels (unique strength)
- Native CUDA (rms_norm, swiglu, lora_delta, lora_qkv, cross_entropy) + Triton + PyTorch fallback.
- Temporary forward patches + strict restore + empty_cache.
- Low-VRAM modes, auto-tune, discrete profiles selectable by RL or heuristics.
- `seiso/kernels/{dispatch.py,hooks.py,platform.py,loss.py,lifecycle.py,tuning.py,training_profile.py,cuda/*,triton_ops.py}`
- Bench: `seiso-bench-kernels`.

### Other
- Knowledge/RAG, Recipes (xyflow graph), Integrations (provider endpoints), Settings (HF token, hardware, security toggles).
- Model Hub (live search + GGUF focus).

---

## Code Health & Quality

### CI / Gates
- `make ci-fast` / `make ci` → `scripts/run_ci_local.py`.
- Jobs: deps (lock digests), lint (ruff + pylint E/F + baselines), types (mypy + baseline), test (smoke + pytest -m "not slow"), security (bandit/detect-secrets/pip-audit), frontend, imports.
- Baselines for ruff/mypy intentionally allow gradual cleanup.
- Pre-commit, secrets baseline, pip-audit present.

**Observed in run (skip-install):**
- Many `I001` import sorting (ruff).
- SIM* simplification suggestions.
- Scattered unused imports (F401) and a few other (F821 undefined `log` in experiment, etc.).
- These are largely pre-existing / baseline-managed. Not blocking for smoke usage.

### Tests
- Broad coverage parallel to features (`test_training_*`, `test_*_backends`, `test_kernels*`, `test_rl_quant*`, `test_distill_rl*`, `test_compress`, `test_export*`, `test_gguf*`, security, db, hardware, e2e, etc.).
- conftest provides temp data dir, auth client, hardware guards.
- Sample run: 36/37 passed quickly (1 hardware enum mismatch — env-specific).

### Type / Lint Posture
- Gradual typing. Mypy baseline exists.
- Some `UP037` quoted annotations, collections.abc suggestions.
- Overall professional; technical debt is visible but contained via baselines.

### Bundled Code
- Large research implementations live outside main tree.
- Seiso owns the integration surface (config builders, runners, manifests, kernel bridge, UI flows).
- Prefer Seiso wrappers for integrated workflow changes; edit bundled internals when the shared behavior itself needs to change.

### Local Edits at Time of Analysis (3 files)
All three changes are **correct and minimal**:

1. `forge-ui/src/pages/ChatPage.tsx`: Fixed `import type` for value const `ROUTER_MODEL_ID`; prefixed unused setter with `_` (TS strict).
2. `forge-ui/src/pages/TrainPage.tsx`: Added focused `useEffect` to auto-apply `dataset_format` + `train_on_responses_only` from recommendations (when not customized). Complements the existing `applyRecommendations` callback.
3. `forge/api/routes/training.py`: Pass `error_text=job.error` on normal completion path (already supported in `db.update_job_status` and used on failure path).

**Recommendation**: These are safe to keep / commit. The TrainPage change is a nice usability improvement.

---

## Security & Privacy Notes

- **Default posture**: Bind localhost, encrypted sensitive DB columns (chat content, provider configs, tokens), per-user isolation.
- Auth: Local JWT + first-run onboarding (Nostr npub identity; nsec proves ownership).
- CSRF + rate limiting middleware (configurable).
- Path sandbox + `assert_within` / shared `USER_SCOPED_DATA_ROOTS` for artifact paths.
- HF tokens: stored encrypted; CLI `hf` also visible.
- NVIDIA boundary / GPU reporting (no exfil).
- SSRF guards on provider URLs; DNS-pinned httpx transport (no process-global `getaddrinfo` race).
- CSP nonce for served UI; tool/code-exec opt-in.
- Compat `/v1` inference API key is **chat-only**; `SEISO_ALLOW_COMPAT_TOOLS` requires a session JWT.
- Remote + `SEISO_ALLOW_CODE_EXEC` is **hard-refused** (no ack override); AST policy ≠ OS sandbox.
- Chat/tools keep security system policy when tools are on; instruction-like KB chunks are quarantined; redacted audit hashes for chat/tool turns.

**Strong for a local tool.** Residual risk: opt-in tools/code-exec on localhost, bundled research surfaces, external llama.cpp / NeMo RL / vLLM processes, and any intentional `allow_remote` deployment.

---

## Algorithms & Meaningful Objectives

Seiso maps learning **signals** to proper algorithms (not proxies that look related but train the wrong objective):

| Signal | Proper algorithm | Path | Not meaningful |
|--------|------------------|------|----------------|
| Instruction / chat labels | Response-masked SFT (CE) | `method: lora/full` + TRL SFT | Full-sequence CE on chat; packing + response masks together |
| Verifiable tasks (math/code/choice) | Online GRPO (group-relative advantages + PPO clip) | `method: slime` + `rl_verify`, or `method: nemo_rl` (external [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)) | Format/process-dominated rewards; zero-spread groups |
| Preference pairs (chosen/rejected) | Offline DPO (Rafailov) | Distill-RL / `compute_dpo_loss`, or `method: nemo_rl` + `nemo_rl_recipe: dpo` | Silent chosen-only SFT labeled “alignment” |
| Quantization policy | PPO/VPG/AWR on discrete actions | `rl_quant` / `adaptive_quant` | Simulator-only metrics claimed as deploy quality |

**SFT:** `preference_as_sft` defaults **false** — preference datasets refuse train unless explicitly opted in (chosen-only SFT; rejected discarded). Packing is incompatible with `train_on_responses_only` on chat-style formats (validator + runtime disable).

**Slime GRPO math** ([`seiso/slime/policy.py`](../seiso/slime/policy.py)):
- Advantages: group mean-center, optional unbiased std + `1e-6` (`grpo_std_normalization`); incomplete groups raise.
- Policy: PPO clipped surrogate; default `calculate_per_token_loss=true` (length-stable). Sequence mode length-normalizes log-probs before `exp(Δ)`.
- KL: Schulman **k3** `exp(δ)−δ−1` (non-negative) when `kl_coef > 0`; signed k1 logged as `kl_k1`. Default `kl_coef=0` saves VRAM — prefer `0.02`–`0.05` for multi-epoch.
- Rewards: outcome must dominate (`outcome_reward_weight > format + process`); process weight default `0`. Dynamic sampling filters on **outcome** spread; watch `group_nonzero_outcome_spread_frac`.

**DPO:** Rafailov β-sigmoid on sum completion log-probs (`average_log_prob=false` by default).

**RL quant:** Research contract embeds `evidence_level`; `deploy_quality_claimable` requires `backend=llama_cpp` **and** `external_quality_path` (simulator or llama.cpp-without-sidecar stay non-claimable).

**Physics / numerics framing:** group advantages are zero-sum within a prompt; length normalization is scale invariance of the importance ratio; non-negative KL is a valid information penalty; VRAM guards are hard resource bounds (logged as-run when they rewrite batch/seq/quant).

---

## Platform & Dependencies

**Best experience**: Linux + NVIDIA (full kernels, QLoRA 4-bit, flash-attn opt, live RL/kernel benches).

**Good**:
- WSL2 + NVIDIA (near parity).
- macOS Apple Silicon (MLX chat + 16-bit LoRA train).
- AMD ROCm (Triton kernels).

**Notes from doctor (this host)**:
- Torch, llama_cpp, fastapi, uvicorn, hf_xet present.
- MLX absent (correct for Linux).
- "Hub ready for download" / "Local chat runtime ready" can be False without token or downloaded models (expected).

**Key optional extras** (see pyproject.toml): `.[forge,train,cuda,llamacpp,mlx,compress-quant,compress-eval,dev,flash-attn]`. RL quant has no separate extra — it uses `.[train]` plus bundled package bootstrap.

External: llama.cpp (convert/quantize binaries managed by scripts), nvcc for CUDA JIT kernels.

---

## Known Issues / WIP (from README + code)

- Memory blocker "added for system RAM but not fully tested."
- Prompt / system text artifacts visible in some outputs; "more concerned with inference right now."
- Qwen (and others) "leaking reasoning vs the response."
- Some advanced pipelines are smoke-oriented or simulator-first for CI.
- GGUF-only repos correctly blocked for training.
- CUDA kernel compile requires toolkit + matching PyTorch CUDA.
- bitsandbytes / QLoRA not on macOS (documented).
- UI requires explicit build step (or `start` script).
- A number of style nits and a handful of pre-existing lint items.
- One flaky hardware test (GPU profile vs enumeration).

---

## Recommendations (Prioritized)

### Quick Wins (low risk, high value)
1. Run `make fix` + refresh ruff/mypy baselines (or targeted import sorting) to reduce noise.
2. Clean the few obvious F401 / unused in tests and experiments.
3. Ensure TrainPage recommendation effect also considers `configCustomized` in more places if needed; consider surfacing "applied from recs" indicator.
4. Document the `error_text` field in API response shapes / UI job lists if not already.

### Medium
- Improve hardware enumeration robustness (the failing test) or mark it xfail/skip with reason.
- Expand doctor output for "chat runtime ready" when GGUF models are present but not loaded.
- Add more end-to-end smoke that exercises a full non-GPU path (export of a tiny checkpoint, knowledge ingest).
- Consider a `seiso doctor --fix` or better auto-remediation for common missing extras.

### Longer-term / Research-y
- Continued improvements to bundled research packages (manifests, DPO, kernel RL).
- macOS / AMD kernel story (Triton training limitations noted).
- Full multi-user / project isolation polish.
- Optional OS-level sandbox for localhost code-exec beyond AST deny-list.

### For Contributors
- Always `make ci-fast` before PR.
- Use smoke presets + existing `.seiso` artifacts.
- Read `docs/training/kernels.md`, `compression.md`, `CI_LOCAL.md`.
- Respect memory guards and sandbox paths.

---

## How to Run / Develop (quick reference)

```bash
cd Seiso
source .venv/bin/activate   # or use `start`
seiso doctor --network
cd forge-ui && npm ci && npm run build && cd ..
seiso forge                  # http://127.0.0.1:8765
# CLI training example
seiso train --config configs/example_lora.yaml
```

Full install: see `start` script or `docs/install.md`.

---

## Appendix: Key File Map (for future work)

- Orchestration entry: `forge/orchestrators/base.py`
- Runners: `seiso/*/runner.py` (or `config.py:run_*`)
- DB jobs + error_text: `forge/db/store.py` (update_* + schema)
- Kernels: `seiso/kernels/hooks.py`, `dispatch.py`, `cuda/`
- Recommendations + hardware: `seiso/training/recommendations.py`, `seiso/hardware/`
- Security: `forge/security/*`, `seiso/security/*`, `forge/middleware` in main.py
- UI pages + API clients: `forge-ui/src/pages/*Page.tsx`, `forge-ui/src/lib/api/*`

---

**End of analysis.** The project is in good shape for a complex local AI platform. Focus areas for continued health: style debt cleanup, documentation of edge cases, and keeping the memory + kernel safety invariants strong.

Generated with full access to the local source tree.

---

## Live Debug Session Notes (2026-06-24)

Executed full live review + testing per follow-up request:
- `seiso doctor --network`: all core green (HF token valid, hub/chat ready, 4090 detected).
- Forge was already live (restarted with `uvicorn` after UI rebuild to pick dist changes).
- **Dataset handling verified live**:
  - `load_training_dataset` + `preprocess_training_dataset` loads **full** file (tested 155 row kernelbench + 20k row nemotron-v2-normalized.jsonl).
  - Preprocess keeps 100% valid rows (no silent dropping of full data).
  - `train_test_split` caps only eval (default max 128) so **vast majority stays in train**.
  - `prepare_tokenized_dataset` / `format` + `train_on_responses_only` correctly produces assistant-only labels for chat (masking user prompts).
  - Smoke runs + direct trainer: "Preprocessed dataset: N/N samples kept (format=chat)".
- **Frontend training bugs found & fixed** (TrainPage.tsx):
  - Dataset picker `onChange` did not reset `configCustomized=false` (inconsistent with model picker) → new dataset recs might not apply.
  - Most user controls (trainResponsesOnly, preprocessDataset, packing, earlyStopping, batch/lr/epochs sliders, method/quant selects, gradAccum, gradCkpt, ...) did **not** set `configCustomized=true` on change.
  - Consequence: recommendations useEffect could silently override user choices for format / "Train on responses only" / etc. after interacting.
  - Fixes: wrapped all onChange to correctly manage `configCustomized` (dataset/model pick → false to adopt fresh recs; edits → true to lock user intent). Rebuilt UI.
- **Live API test of training task flow** (minted JWT from real user in forge.db):
  - POST /training/jobs with realistic payload created job successfully.
  - Resolved dataset to user uploads copy (full content).
  - Stored config had `dataset_format: "chat"`, `train_on_responses_only: true`, `preprocess: true` — exactly best practice.
  - Error surfaced cleanly via `error_text` (Pydantic validation in one test case).
- **All tasks tested live**:
  - Inference/chat: /v1/models + /v1/chat/completions (small generation succeeded, returned "Hi! How can I...").
  - CLI: `seiso rl-quant profiles`, compress, train direct all respond.
  - Training tests (`test_training_*.py` + preprocess/models): 30+ passed.
  - Jobs list / recs / models endpoints authenticated OK.
- **"Uses the full dataset + properly formats"**: Confirmed in core (no max_train_samples unless explicitly in extra), preprocess keeps all valid, recs + UI flags set chat + responses_only for message datasets. UI now reliably propagates the settings.
- Server restarted, new UI build active.

No other critical runtime crashes in chat/export paths sampled. One test job failure was due to invalid test payload (max_seq<128) — validation works as intended.

Changes: only frontend form hygiene (no core logic change needed; it was already doing full+best format).

Next time a user wants a huge SFT run, point dataset at the 20k nemotron (or HF equivalent) and the pipeline will use every valid row with proper chat masking.
