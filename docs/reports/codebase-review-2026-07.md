# Codebase Review — July 2026

Full-tree correctness and engineering audit of SeisoLocalAI (~644 Python files, ~101 TS/TSX, 143 test modules). Refreshes [docs/ANALYSIS.md](../ANALYSIS.md) (2026-06-25).

## Snapshot (Phase 0)

| Metric | Value |
|--------|-------|
| Python files (tracked) | 644 |
| TS/TSX files | 101 |
| Test modules (`tests/*.py`) | 143 |
| Mypy baseline lines | 395 |
| Ruff baseline lines | 0 |
| Largest `seiso/` package | `adaptive_quant` (123 files) |

### Subsystem map

| Layer | Paths | Role |
|-------|-------|------|
| CLI | `seiso_cli/` | Typer entry → core runners |
| Forge API | `forge/api/routes/`, `forge/orchestrators/` | Jobs + SSE |
| Forge services | `forge/services/`, `forge/security/`, `forge/db/` | Auth, paths, persistence |
| UI | `forge-ui/src/` | React + typed API clients |
| Core product | `seiso/{inference,training,slime,export,chat,models,hardware,memory,kernels}` | Shared runners |
| Research wrappers | `seiso/{compress,distill_rl,rl_quant,experiments}` | Product integration |
| Bundled research | `seiso/{adaptive_quant,codellama_compress,analysis}` | Research internals |
| Compat shim | `seiso/slime_single_gpu/` | Re-exports → `seiso.slime` |

### Mypy baseline hotspots (top)

`forge/services/knowledge_context.py`, `forge/db/stores/chat.py`, `adaptive_quant/{torch_policy,policy}.py`, `inference/model_pool/llama_load.py`, `kernels/{hooks,triton_ops}.py`, `slime/trainer.py`.

---

## Severity rubric

| Sev | Meaning |
|-----|---------|
| P0 | Security, data loss, wrong claim/objective, host RCE when feature enabled |
| P1 | Incorrect common-path behavior; cancel/cleanup; product honesty |
| P2 | Redundant path / drift / maintainability |
| P3 | Style, docs, baseline noise |

---

## Findings

### Phase 1 — Safety

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| S1-001 | P1 | SSRF | `forge/security/http_client.py` global `getaddrinfo` pin | Fix: custom transport (larger change) | Open |
| S1-002 | P1 | Jobs | `forge/orchestrators/base.py` cancel ignores PENDING | Fix | Fixed |
| S1-003 | P1 | Paths | Divergent `user_paths` vs `seiso.security` | Consolidate | Partial (recipes root) |
| S1-004 | P2 | Training sandbox | uploads-only vs full user tree | Fix later | Open |
| S1-005 | P2 | Hub publish | No re-assert in orchestrator | Fix | Fixed |
| S1-006 | P2 | Kernels | Per-model restore clears all | Fix later | Open |
| S1-007 | P2 | Inference kernels | No KernelPatchSession | Fix later | Open |
| S1-008 | P2 | Rate limit | Empty IP keys never deleted | Fix | Fixed |
| S1-009 | P2 | GPU serialize | Triple mechanisms | Consolidate later | Open |
| S1-010 | P2 | CSRF | Any Bearer skips CSRF | Keep + tighten later | Open |
| S1-011 | P2 | HTTP | Redirects not explicit false | Fix | Fixed |
| S1-012 | P3 | Paths | Raw `Path/user_id` joins | Consolidate later | Open |
| S1-013 | P3 | Paths | `recipes` missing from scoped roots | Fix | Fixed |
| S1-014 | P3 | Cancel | Does not await GPU finally | Optional | Open |
| S1-015 | P3 | Auth | Local reset-session unauthenticated | Keep (local-only) | Keep |
| S1-016 | P3 | CSP | unsafe-inline when debug | Keep; refuse debug+remote later | Open |

### Phase 2 — Product surfaces

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| INF-01 | P1 | Inference | `available_backends` returns single engine | Fix later | Open |
| SLM-01 | P1 | Slime | vLLM/SGLang old_logprobs from actor | Document/guard later | Open |
| TRN-01 | P1 | Training UI | packing+`auto` guard gap | Fix | Fixed |
| INF-02 | P2 | Inference | ROUTER enum vs local resolver | Document/reject | Open |
| INF-03 | P2 | CLI chat | Prints device class not backend | Fix later | Open |
| TRN-02 | P2 | Preference | auto format gate timing | Open | Open |
| TRN-03 | P2 | Packing | hard-fail vs soft-disable | Unify later | Open |
| SLM-02 | P2 | Slime | `reward_nonzero_std` misnamed | Alias later | Open |
| SLM-03 | P2 | CLI | slime multi-GPU parity | Document | Fixed (docs) |
| EXP-01 | P2 | Export | `BASE` near-dead | Deprecate | Fixed |
| EXP-02 | P2 | Export | LoRA→full copytree | Guard later | Open |
| INF-04/05 | P3 | Inference | Dead empty-set / fallthrough | Fix | Fixed |
| PAR-02 | P3 | Naming | `slime_single_gpu_metrics.jsonl` | Keep + deprecate package | Deprecate note |

**`seiso/slime_single_gpu`:** **Deprecate** (keep thin shim). No production callers; tests + legacy scripts/docs remain.

### Phase 3 — Research pipelines

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| RP-01 | P0 | Claims | llama_cpp without quality → deploy_claimable | Fix | Fixed |
| RP-02 | P0 | Claims | Multiseed/sweep claim boundary empty | Fix | Fixed |
| RP-03 | P1 | Copy | “Deploy …” ignores evidence | Fix | Fixed |
| RP-04 | P1 | Config | `stdlib` vs `python` training_backend | Alias | Fixed |
| RP-05 | P1 | Presets | post_train dual registry | Open (larger) | Open |
| RP-06 | P1 | CLI | orphan `adaptive-rl-quant-*` names | Docs scrub | Partial |
| RP-07 | P1 | Compress | Dual CLI surfaces | Document | Fixed (docs) |
| RP-08 | P2 | Evidence | Nested `evidence.level` misread | Fix | Fixed |
| RP-09–21 | P2–P3 | Provenance/presets | Triplicated helpers, orphan presets | Later | Open |

### Phase 4 — Forge API / DB / tools

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| F4-01 | P0 | Code-exec | AST deny-list ≠ sandbox | Document + fail-closed note | Documented |
| F4-02 | P1 | Compat auth | `compare_digest` length → 500 | Fix | Fixed |
| F4-03–05 | P1 | DB | `recipe_jobs` / `knowledge_bases` / `projects` schema-only | Delete or implement | Open |
| F4-06 | P1 | Hub publish | Memory-only jobs | Persist later | Open |
| F4-07–09 | P2 | Jobs | Status aliases / error_text gaps | Later | Open |

### Phase 5 — UI

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| F5-01 | P1 | Jobs UI | `error_text` missing from types/UI | Fix | Fixed |
| F5-02 | P1 | RL Quant | Not on `useStagePipelinePage` | Later | Open |
| F5-03 | P2 | Export | Dead sync `publishToHub` client | Deprecate comment | Fixed |
| F5-04 | P2 | TrainPage | analyzeDataset ignores configCustomized | Fix | Fixed |

### Phase 6 — CLI / docs / tests

| ID | Sev | Area | Evidence | Action | Status |
|----|-----|------|----------|--------|--------|
| F6-01 | P1 | Docs | `seiso slime` undocumented | Fix | Fixed |
| F6-02 | P1 | Docs | Wrong `seiso compress` snippet | Fix | Fixed |
| F6-03 | P2 | Tests | docs accuracy coverage thin | Expand later | Open |
| F6-04 | P2 | Configs | Unreferenced smoke/examples | Adopt or archive later | Open |
| F6-05 | P2 | Tests | GPU e2e / nvcc brittle | Soft-skip later | Open |

---

## Keep / deprecate / delete

| Candidate | Verdict |
|-----------|---------|
| `seiso/slime_single_gpu` | **Deprecate** (shim retained) |
| `ExportFormat.BASE` | **Deprecate** (alias to FULL behavior documented) |
| Sync `POST /export/publish` + UI `publishToHub` | **Deprecate** (jobs path is canonical) |
| DB `recipe_jobs`, `knowledge_bases`, `projects` | **Delete or implement** (open) |
| Route helpers `_jobs`/`_pipeline`/`_stream` | **Keep** |
| Compat `/v1` stack | **Keep** (thin adapters) |
| AST code-exec as “sandbox” | **Deprecate naming**; harden or gate under remote |
| Orphan adaptive_quant console script names | **Delete from docs**; use `seiso rl-quant` / `python -m` |

---

## Remediation landed in this pass

See git history / working tree for: PENDING cancel, deploy_quality_claimable, claim boundaries for aggregates, packing/`auto` UI guard, compat auth digest, rate-limit prune, explicit no-redirects, recipes scoped root, hub publish re-assert, inference dead fallthrough, docs (slime/compress), UI `error_text`, TrainPage analysis gate, training_backend alias, recommendation copy.

## Follow-ups (not in this pass)

1. Replace global DNS pin with httpx transport (S1-001).
2. Unify path policy into `seiso.security` (S1-003).
3. Multi-backend picker (INF-01); slime engine logprobs (SLM-01).
4. Single RL-quant preset registry (RP-05).
5. Persist or drop dead DB tables (F4-03–05).
6. OS-level code-exec sandbox or hard disable under `allow_remote` (F4-01).

---

*Generated as part of the Full-Codebase Correctness and Engineering Review.*
