# Codebase Review — 2026-07-23

Full-tree correctness audit of SeisoLocalAI (product + research). Refreshes [codebase-review-2026-07.md](codebase-review-2026-07.md) and the residual-risk paragraph in [docs/ANALYSIS.md](../ANALYSIS.md).

**Scope:** entire tree — `seiso/` (including `adaptive_quant`, `codellama_compress`, `analysis`), `forge/`, `forge-ui/`, `seiso_cli/`, `configs/`, `scripts/`, `tests/`, docs accuracy touchpoints.

**Remediation status:** Correctness fixes from the priority queue landed in-tree after this report (see Status columns and “Remediation landed” below).

**Correctness definition:** defects that make the system wrong, unsafe, or dishonest — silent wrong learning, lifecycle leaks, sandbox/auth/tools, claim honesty, job/API invariants, CI blind spots. Style-only noise stays P3 or is dropped.

---

## Snapshot (Phase 0)

| Metric | Value |
|--------|-------|
| Python files (`seiso/` + `forge/` + `seiso_cli/`) | 498 |
| TS/TSX under `forge-ui/src/` | 100 |
| Test modules (`tests/test_*.py`) | 141 |
| Test LOC (approx) | ~37k |
| Mypy baseline lines | 399 |
| Ruff baseline lines | 0 |
| Largest `seiso/` package | `adaptive_quant` (~27k LOC / 123 files) |

### Subsystem map (LOC approx)

| Layer | Paths | LOC / files | Role |
|-------|-------|-------------|------|
| CLI | `seiso_cli/` | ~1k / 13 | Typer entry → core runners |
| Forge API | `forge/` | ~20.5k / 123 | Jobs + SSE + security + DB |
| UI | `forge-ui/src/` | ~14.4k / 100 | React + typed API clients |
| Core product | `seiso/{inference,training,slime,export,chat,kernels,memory,security}` | ~33k | Shared runners |
| Research wrappers | `seiso/{compress,distill_rl,rl_quant,experiments,nemo_rl}` | ~3k | Product integration |
| Bundled research | `seiso/{adaptive_quant,codellama_compress,analysis,rl_verify}` | ~36k | Research internals |
| Compat shim | `seiso/slime_single_gpu/` | thin | Re-exports → `seiso.slime` |

### Entry points

- `start` → Forge by default
- `seiso` → `seiso_cli/main.py` (`forge`, `doctor`, `train`, `slime`, `nemo-rl`, `chat`, `export`, `inference`, `bench-inference`, `rl-quant`, `compress`, `distill-rl`, `experiment`)
- `forge.main:create_app`
- `seiso-train-worker` → `seiso.training.worker:main`
- `seiso-bench-kernels` → `seiso.kernels.benchmark:main`

### Smoke configs vs default CI

| Config | Referenced outside `configs/`? |
|--------|--------------------------------|
| `smoke_train_cpu.yaml` | Yes (AGENTS, tests, docs) |
| `smoke_slime_cpu.yaml` | Docs + slime error string |
| `smoke_nemo_rl.yaml` | Docs + `test_docs_accuracy` existence assert |
| `smoke_train_gpu.yaml` | **None** |
| `smoke_train_moe_cpu.yaml` | **None** |
| `smoke_train_gpu_e2e.yaml` | Docs/report only |
| `rl_quant_smoke.json` / `distill_rl_smoke.json` | Docs only |

Default CI: `pytest -m "not slow and not gpu"` (~3 `@slow` files, ~5 `@gpu` files). Local `make ci` frontend = typecheck+build; GitHub Actions also runs vitest. `make live-check` is manual only.

---

## Severity rubric

| Sev | Meaning |
|-----|---------|
| P0 | Security, data loss, wrong claim/objective, host RCE when feature enabled |
| P1 | Incorrect common-path behavior; cancel/cleanup; product honesty |
| P2 | Redundant path / drift / maintainability |
| P3 | Style, docs, baseline noise |

---

## July open-item re-verification

| ID | Prior | 2026-07-23 | Evidence |
|----|-------|------------|----------|
| S1-004 | Open | **Still open** | Relative datasets resolve into `uploads/` (`forge/services/user_paths.py`); Forge training sandboxes the full user tree (`forge/api/routes/training.py`) |
| S1-006 | Open | **Fixed** | Per-model restore uses `id(model)` + orphan filter (`seiso/kernels/lifecycle.py`) |
| S1-007 | Open | **Fixed** | Inference uses `KernelPatchSession` + `commit()` (`seiso/inference/tuning.py`) |
| S1-009 | Open | **Still open** | Triple(+) GPU serialize: orchestrator `_ACTIVE_RESOURCES`, `_ACTIVE_GPU_TASKS`, file `gpu_resource_lock`, `GPU_EXECUTOR` |
| S1-010 | Open | **Still open (Keep)** | Non-empty Bearer skips CSRF (`forge/security/csrf.py`); empty Bearer hole closed |
| S1-012 | Open | **Still open** | Raw `Path / user_id` joins remain (hf_auth, bundled `job_output_root`, export/training routes) |
| S1-014 | Open | **Still open (partial)** | Train/export await executor before release; base `cancel()` still returns without awaiting finally; bundled path has 2s timeout → **S1-017** |
| S1-016 | Open | **Still open** | `debug=True` adds CSP `unsafe-inline`; no refuse of debug+remote |
| SLM-01 | Fixed | **Still fixed** | HTTP backends require `*_sync_weights` (`seiso/slime/rollout_resolve.py` + invariants tests) |
| TRN-02 | Open | **Still open (expanded)** | Preference gates only when `dataset_format == PREFERENCE`; AUTO bypass → **TRN-04** |
| TRN-03 | Open | **Still open** | Hard-fail at validate vs soft-disable at runtime |
| SLM-02 | Open | **Still open** | `reward_nonzero_std` ≡ `outcome_nonzero_std` (`seiso/slime/policy.py`) |
| INF-02 | Open | **Still open** | `InferenceBackend.ROUTER` falls through to generic ValueError in local resolver |
| INF-03 | Open | **Still open** | CLI chat prints device-class `detect_backend()`, not inference backend |
| EXP-02 | Fixed | **Still fixed** | LoRA-only FULL/BASE refuse + tests; edge **EXP-02-R** |
| RP-01/02/05/07/08 | Fixed | **Still fixed** | Claim gates + single product preset registry hold |
| RP-06 | Partial | **Partial** | Product CLI is `seiso rl-quant`; package/UI still advertise `adaptive-rl-quant-*` |
| RP-09–21 | Open | **Still open** | Orphan research presets + provenance helper drift (**RP-09**, **RP-10**) |
| F4-06 | Open | **Still open** | Hub publish jobs memory-only; no DB table / reconcile |
| F4-07–09 | Partial | **Partial** | Live vs durable SSE status shape mismatch → **F4-10** |
| F5-02 | Open | **Still open** | RL Quant page not on `useStagePipelinePage` |
| F6-03 | Partial | **Partial** | Slime/compress/NeMo covered; smoke/route matrix still thin |
| F6-04 | Open | **Still open** | Several smoke YAMLs unreferenced |
| F6-05 | Partial | **Partial** | nvcc soft-skip landed; `@gpu`/`@slow` still sparse |

---

## Findings

### Phase 1 — Safety / lifecycle

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| S1-004 | P2 | Training sandbox | uploads-only resolve vs full-tree Forge assert | Align resolve + assert scopes | Open |
| S1-009 | P2 | GPU serialize | Triple mechanisms | Consolidate under one owner | Open |
| S1-010 | P2 | CSRF | Any non-empty Bearer skips CSRF | Keep for API clients; optionally require valid JWT before skip | Open (Keep) |
| S1-012 | P3 | Paths | Raw `Path/user_id` joins after shared roots | Route all joins through `safe_join` / helpers | Open |
| S1-014 | P3 | Cancel | Base cancel does not await GPU finally | Optional await; train/export already wait | Open (partial) |
| S1-016 | P3 | CSP | `unsafe-inline` when debug; no debug+remote refuse | Refuse debug+remote at startup | Open |
| **S1-017** | **P1** | Bundled cancel | `_bundled_job.py:139-151` waits ≤2s then `release_after_task` while executor may still run | Match train/export: await shielded future with no short timeout before release | **New / Open** |
| **S1-018** | **P2** | Kernel hooks | `hooks.py` residual RMSNorm/decoder exception path can strip shared `_seiso_orig_forward` / sticky-skip | Clear residual markers; never delete pre-existing orig | **New / Open** |
| **S1-019** | **P2** | Training orch | Default `output_dir` omits `user_id` (`training.py` orch); HTTP route always supplies scoped path | Default `checkpoints/{user_id}/{job_id}` or refuse missing `output_dir` | **New / Open** |

**Verified healthy:** `KernelPatchSession` restore/`commit`; trainer `finally` release; pinned httpx + no redirects; AES-GCM field crypto; remote+code-exec hard refuse; PENDING cancel; S1-006/007 fixed.

### Phase 2 — Learning objectives

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| TRN-02 | P1 | Preference gates | Gates only for explicit `PREFERENCE` (`config.py:517+`) | Probe AUTO preference-shaped rows at validate | Open |
| **TRN-04** | **P1** | Slime + AUTO | AUTO + `{prompt,chosen,rejected}` validates; slime `_reward_sample` leaves empty answer → vacuous/filtered GRPO | Refuse preference-shaped rows for slime (and SFT without `preference_as_sft`) under AUTO | **New / Open** |
| TRN-03 | P2 | Packing | Hard-fail validate vs soft-disable runtime | Unify to one policy | Open |
| **TRN-05** | **P2** | Response mask | `train_on_responses_only` silently ignored for TEXT (`datasets.py`) | Warn or refuse TEXT + responses_only | **New / Open** |
| SLM-02 | P2 | Filter names | `reward_nonzero_std` filters on `outcome_reward` only (`policy.py:397-404`) | Alias/document or filter on composite reward | Open |
| **NEMO-01** | **P2** | NeMo sandbox | Only `output_dir` asserted; cancel/killpg untested; `SEISO_NEMO_RL_ROOT` can point outside Forge sandbox | Sandbox path-like overrides; cancel tests; document recipe corpus vs `TrainConfig.dataset` | **New / Open** |
| **NEMO-02** | **P2** | NeMo honesty | AUTO preference + non-dpo NeMo validates while Seiso dataset unused by Hydra | Early preference probe + explicit “recipe corpus” ack | **New / Open** |
| SLM-std | P3 | Std convention | Filter uses population std; advantages use sample std | Align or document | Open |
| CODE-score | P3 | rl_verify | `code_outcome_score` binary-only vs dense `code_outcome_value` | Route helper through value API or document | Open |

**Verified healthy:** SLM-01 sync-weight contract; GRPO advantage grouping / KL k3 / packing+AUTO+responses_only hard-fail (TRN-01); explicit preference refuse without `preference_as_sft`.

### Phase 3 — Inference / export / chat

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| INF-02 | P2 | Router enum | Local resolve has no router branch (`backends.py`) | Explicit “not a local backend” error | Open |
| INF-03 | P2 | CLI chat | Prints device class not inference backend (`seiso_cli/commands/chat.py`) | Print resolved inference backend | Open |
| **CHAT-01** | **P2** | Sanitize | `strip_tool_calls=False` returns raw content — no reasoning strip (`sanitize.py` + routes) | Always strip reasoning unless opted out; tools flag only for tool syntax | **New / Open** |
| CHAT-02 | P3 | Tags | Only `<think>` / `<redacted_thinking>` | Extend if product models emit other wrappers | Open |
| EXP-02 | — | Export | LoRA-only FULL/BASE refuse | — | Fixed |
| **EXP-02-R** | **P3** | Export kind | `config.json` ⇒ `kind=full` even with adapter weights and no `adapter_config.json` | Refuse when adapter weights present without merge | **New / Open** |
| HUB-TOCTOU | P3 | Hub publish | Route precheck then push with `skip_precheck=True` | Optional re-precheck at push | Open |

**Verified healthy:** Compat inference-key chat-only; pool unload / `prepare_for_gpu_task` eject; LoRA FULL/BASE refuse tests.

### Phase 4 — Research pipelines

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| **CMP-ORD** | **P1** | Compress stages | `validate_stages` membership-only (`bundled/config_builder.py:30-35`); UI `FALLBACK_STAGES` puts evaluate/export **before** quantize (`CompressPage.tsx:9-17`) vs backend `STAGE_ORDER` | Sort by `STAGE_ORDER` on build; fix FALLBACK; reorder on toggle | **New / Open** |
| RP-06 | P2 | Naming | `adaptive-rl-quant-*` still in package/UI; not in `[project.scripts]` | Scrub to `seiso rl-quant` / `python -m` | Partial |
| RP-09 | P2 | Presets | Orphan research presets (`CONFIG_4090`, etc.) not in product registry | Document research-only or wire aliases | Open |
| RP-10 | P3 | Provenance | Triplicated fingerprint helpers (research / compress / distill) | Consolidate on `seiso.research.provenance` | Open |

**Verified healthy:** `deploy_quality_claimable` clamp (RP-01/02); single product RL-quant preset registry (RP-05); product stage application / checkpoint_path wiring from prior remediations.

**CI-thin residual (no concrete bug proven):** `adaptive_quant` (~27k LOC, high mypy debt), deep `codellama_compress` internals, multi-GPU slime / real NeMo `uv` launch, GPU e2e.

### Phase 5 — Forge API / DB / tools

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| F4-06 | P1 | Hub publish | `export.py` create_job only; no DB; restart loses job; token in memory payload | Persist publish jobs (+ redact token) | Open |
| **F4-06b** | **P2** | Memory jobs | Recipes + knowledge ingest also memory-only | Persist or document class | **New / Open** |
| F4-07–09 | P2 | Jobs UI | Train/Export/RL Quant history still weak on `error_text` display | Render `error_text` in all job tables | Partial |
| **F4-10** | **P2** | SSE status | Live training yields plain `j.status.value`; durable replay yields `json.dumps({"status":…})`; TrainPage compares plain strings (`TrainPage.tsx:738-740`) | Unify status payload shape | **New / Open** |
| **F5-05** | **P1** | Train cancel | `POST …/jobs/{id}/cancel` exists; **no** `cancelTraining` in `training.ts`; TrainPage never cancels | Wire cancel client + UI control | **New / Open** |

**Verified healthy:** Provider URL policy + DNS pin; knowledge quarantine; CSRF middleware; rate limit; DB-backed `error_text` write for train/export/compress/distill/rl-quant.

### Phase 6 — UI / CLI / docs / tests / dead paths

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| F5-02 | P1 | RL Quant UI | Not on `useStagePipelinePage` / `StagePipelineShell` | Migrate page to shared hook | Open |
| F6-03 | P2 | Docs tests | Smoke/route matrix thin | Expand `test_docs_accuracy` | Partial |
| F6-04 | P2 | Smoke configs | Unreferenced GPU/MoE/e2e/rl_quant/distill smokes | Adopt in CI/docs tests or archive under `configs/archive/` | Open |
| F6-05 | P2 | GPU tests | Sparse markers; PR CI excludes gpu/slow | Soft-skip + optional hardware job (partial) | Partial |
| **F6-06** | **P2** | Local CI | `job_frontend` skips vitest; GHA runs `npm test` | Add vitest to local frontend job | **New / Open** |
| **F6-07** | **P3** | Live check | `make live-check` never in CI | Optional scheduled live smoke | **New / Open** |

---

## Keep / deprecate / delete (structure cleanup inventory)

These are structural/dead-path recommendations for a follow-on cleanup PR (not executed in this audit pass).

| Candidate | Verdict | Evidence |
|-----------|---------|----------|
| `seiso/slime_single_gpu/` | **Deprecate** (keep thin shim) | Re-exports only; no production imports; legacy metrics filename `slime_single_gpu_metrics.jsonl` remains |
| `ExportFormat.BASE` | **Deprecate** | Same branch as FULL; test/registry only |
| Sync `POST /export/publish` + UI `publishToHub` | **Deprecate → delete UI dead client after grace** | Jobs path canonical; `publishToHub` has zero UI callers |
| Orphan `adaptive-rl-quant-*` names | **Delete from docs/UI strings** | Not in `[project.scripts]`; product CLI is `seiso rl-quant` |
| Dual preset registries | **Keep layered** | Product `rl_quant/presets.py` vs research `easy_config.named_preset` (intentional post RP-05) |
| Compress UI `FALLBACK_STAGES` order | **Fix** (CMP-ORD) | Diverges from backend `STAGE_ORDER` |
| Unreferenced smoke configs | **Adopt or archive** (F6-04) | `smoke_train_gpu.yaml`, `smoke_train_moe_cpu.yaml`, etc. |
| Route helpers `_jobs`/`_pipeline`/`_stream` | **Keep** | Shared by compress/distill/rl-quant |
| Compat `/v1` | **Keep** | Thin adapters; inference key chat-only |
| AST code-exec as “sandbox” | **Deprecate naming** | Localhost-only; remote+code-exec hard-refused |
| Hub-publish / recipes / knowledge job stores | **Implement persistence** (F4-06 / F4-06b) | Not delete |
| `seiso/analysis` | **Keep** | Used by adaptive_quant + `python -m seiso.analysis` |

Suggested cleanup order (after P0/P1 correctness fixes):

1. Fix CMP-ORD FALLBACK + stage sort (correctness + structure).
2. Scrub orphan adaptive-rl-quant naming; document research-only presets.
3. Archive or wire unreferenced smoke configs; expand docs-accuracy.
4. Remove dead `publishToHub` UI client; leave sync route deprecated for scripts.
5. Consolidate provenance helpers (RP-10) and path joins (S1-012) in a dedicated hygiene PR.

---

## Remediation landed (post-report)

| ID | Fix summary |
|----|-------------|
| S1-017 | Bundled cancel awaits shielded future (no 2s timeout) before `release_after_task` |
| S1-018 | Residual norm/decoder patch exception paths clear markers; never strip pre-existing orig |
| S1-019 | Training orch defaults `checkpoints/{user_id}/{job_id}` or refuses missing scope |
| TRN-02/04 | Local AUTO preference probe at TrainConfig; slime `_load_samples` refuses preference rows |
| TRN-05 | TEXT + `train_on_responses_only` without packing refused |
| TRN-03 | Trainer hard-fails packing+responses_only for non-TEXT (no soft-disable) |
| CMP-ORD | `sort_stages` in compress builder; UI FALLBACK + toggle preserve `STAGE_ORDER` |
| F4-06 | `hub_publish_jobs` table + store; route persists redacted config; stream durable |
| F4-06b | Recipes/knowledge documented as intentional ephemeral jobs |
| F4-10 | Durable SSE status emits plain status string |
| F5-05 | `cancelTraining` API client + TrainPage Cancel buttons |
| F5-02 | RL Quant uses `useStagePipelinePage` + `StagePipelineJobsTable` (`error_text`) |
| SLM-02 | `reward_nonzero_std` filters composite reward; `outcome_nonzero_std` for outcomes |
| INF-02 | Explicit router reject in `resolve_local_backend` |
| INF-03 | CLI chat prints device class + resolved inference backend |
| CHAT-01 | Reasoning always stripped; `strip_tool_calls` only gates tool markup |
| NEMO-01 | Path-like `extra_overrides` asserted inside sandbox |
| RP-06 | Product-facing adaptive_quant status/docstrings point at `seiso rl-quant` |
| F6-04 | Smoke configs listed in `docs/CI_LOCAL.md` + docs-accuracy test |
| F6-06 | Local `job_frontend` runs vitest |

---

## Remediation landed (remaining open — second pass)

| ID | Fix summary |
|----|-------------|
| S1-004 | `resolve_training_dataset_path` searches all `USER_SCOPED_DATA_ROOTS` |
| S1-009 | GPU orchestrators drop duplicate `_ACTIVE_RESOURCES`; exclusivity via prepare_for_gpu_task + file lock + GPU_EXECUTOR |
| S1-010 | CSRF Bearer skip requires valid JWT via `decode_token` |
| S1-012 | `safe_join` / `user_dir` for hf_tokens, job_output_root, export/training paths, distill multiseed |
| S1-014 | `Orchestrator.cancel` awaits cancelled task before return |
| S1-016 | Startup refuses `debug` + `allow_remote` |
| RP-09 | Research-only preset docstring on `adaptive_quant.presets` |
| RP-10 | `codellama_compress.replay.content_fingerprint` → `seiso.research.provenance` |
| NEMO-02 | `nemo_rl_ack_recipe_corpus` required for non-dry-run |
| EXP-02-R | Adapter weights without merge ⇒ kind `lora` even with `config.json` |
| HUB-TOCTOU | Hub publish re-prechecks at push (`skip_precheck=False`) |
| F6-05 | Documented opt-in hardware / soft-skip policy in CI_LOCAL |
| F6-07 | Scheduled GHA `live-check` job |
| Dead UI | Removed unused `publishToHub` client |
| Code-exec naming | Registry + policy/sandbox module docs clarify AST limits ≠ OS sandbox |

## Still open (intentionally deferred)

- Optional **OS-level** code-exec sandbox (beyond AST + rlimits); remote+code-exec remains hard-refused.
- Full GPU `@pytest.mark.gpu` suite in PR CI (needs CUDA runners; keep `make test-hardware` / soft-skip).
- Deeper adaptive_quant fingerprint unification (domain `sha256_canonical` kept for replay integrity).

**Keep:** `seiso/slime_single_gpu` thin shim (compat import path; do not delete).

---

*Generated as the 2026-07-23 Full-Codebase Correctness Review; remediation status refreshed after both fix passes.*
