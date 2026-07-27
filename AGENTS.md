# Agent and contributor orientation for SeisoLocalAI

Short guide for humans and coding agents. Deep detail lives in the `docs/` tree (start with [docs/README.md](docs/README.md) and [docs/getting-started.md](docs/getting-started.md)).

## Read first

1. **[README.md](README.md)** — What Seiso is, quick start (one-liner `start`), Forge UI at `http://127.0.0.1:8765`, CLI table, paths (`~/Seiso` vs `~/.seiso`), platform notes.
2. **[docs/README.md](docs/README.md)** — Full doc index + learning paths (end-user vs CLI vs dev vs deployment).
3. **[docs/CI_LOCAL.md](docs/CI_LOCAL.md)** + `make ci-fast` — Required quality gate before any PR or significant change. Uses baselines for lint/types.
4. **[docs/ANALYSIS.md](docs/ANALYSIS.md)** — Current architecture overview, feature map, code health, security notes, WIP items, and recommendations (this analysis).

Key commands:
- `start` or `SEISO_START=0 start` (from repo root or on PATH).
- `seiso doctor [--network]`
- `seiso forge` (then open browser)
- `seiso train --config configs/example_lora.yaml`
- `seiso experiment quant-regression -c configs/examples/quant_regression_study.yaml` (research)
- `seiso provenance attest|verify` (Nostr digest attestation; default on, kill with `SEISO_ALLOW_NOSTR=0` — see [docs/provenance-nostr.md](docs/provenance-nostr.md))
- `make ci-fast` (or `python3 scripts/run_ci_local.py --fast`)

## Rules of thumb

- **Always activate the project venv** (`.venv/bin/activate` or let `start` manage) before running Python/CLI commands. The base system python will not have the right extras or pinned deps.
- Prefer the documented entry points (`start`, `seiso`, `scripts/doctor.sh`) over raw `python ...`. They set up paths, HF cache, and runtime config.
- For a CI-equivalent Python env: `python scripts/install_locked_deps.py --editable` (hashed `locks/python.lock`). Refresh with `python scripts/update_dep_locks.py` after pyproject changes.
- **Smoke first**: Use `configs/*_smoke.*` presets for fast iteration. They exist precisely for CI + agent loops.
- **Never delete** `~/.seiso` or its subdirs (user data, caches, checkpoints). Use `SEISO_DATA_DIR` overrides for throwaway experiments.
- Memory-sensitive work: the platform applies guards (`seiso/memory/protection/`, `forge/services/memory_release.py`). Call `prepare_for_gpu_task` patterns when adding new heavy GPU jobs.
- Kernels are **monkey-patched temporarily** — always ensure restore paths run (see `lifecycle.py`, trainer cleanup, memory release). Test both success and exception cases.
- Bundled compression and RL quant code lives under `seiso/` (`seiso/codellama_compress/`, `seiso/adaptive_quant/`, `seiso/analysis/`). Prefer the Seiso wrappers (`config_builder.py`, `runner.py`, `bootstrap.py`, `kernel_integration.py`) when changing integrated workflows.
- Forge jobs stream logs/metrics via orchestrators + SSE. When adding features, update the matching orchestrator + route + UI page together.
- UI: after TS/JS changes run `cd forge-ui && npm run build` (or use `npm run dev` against a running `seiso forge`).
- Before significant changes: `make ci-fast`. Full `make ci` for frontend or big refactors. Respect ruff/mypy baselines unless you intentionally refresh them.

## Security & privacy (critical)

- Default is localhost + encrypted fields + per-user sandboxing under `SEISO_DATA_DIR`.
- All artifact paths must go through `seiso/security/*` helpers or `forge/services/user_paths.py` + `safe_join` / `assert_within`.
- New remote provider or tool code must respect the existing URL policy, rate limiter, and CSRF checks.
- Tokens (HF, etc.) are encrypted at rest in the DB. Prefer the existing `hf_auth` and token storage flows.
- GPU / hardware info is intentionally bounded (see `nvidia_boundary` usage).

Do not relax sandbox or crypto defaults without a very strong documented reason and tests.

## Extending common areas (pointers)

- **New training preset / recs**: `seiso/training/recommendations.py`, `seiso/training/dataset_analysis.py`, `seiso/training/practices.py`, `platform_caps.py`, example YAML in `configs/`. Update TrainPage + `docs/training/quickstart.md` if new knobs appear. Run `pytest tests/test_docs_accuracy.py`.
- **Forge auth**: Nostr npub identity; nsec proves ownership. Default onboarding is keygen → show npub to write down → Continue (`forge/services/nostr_auth.py`, `AuthPage`). No password path.
- **Nostr provenance**: Default path on (`SEISO_ALLOW_NOSTR=1`, public digests-only relays). Kill with `SEISO_ALLOW_NOSTR=0`. Auto-attest still opt-in (`SEISO_NOSTR_ATTEST`). Digests-only events via `seiso/research/nostr/`; keys under `nostr_keys/` (HF-token style). Do not add DMs, agent tools, or always-on social clients without a new design review. See [docs/provenance-nostr.md](docs/provenance-nostr.md).
- **New kernel op**: Add to `seiso/kernels/cuda/` + `cuda_ops.py` + dispatch + hooks + tests. Update low-VRAM profile logic.
- **New pipeline stage (compress/distill/rl)**: Update the stage router / config builder + manifest + the corresponding orchestrator + page.
- **NeMo RL** (`method: nemo_rl`): external launcher in `seiso/nemo_rl/` — requires `SEISO_NEMO_RL_ROOT` pointing at a [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) checkout + `uv`. Do not vendor NeMo RL into this repo.
- **New inference backend**: `seiso/inference/backends.py` + model pool + runner + UI picker.
- **Export format**: `seiso/export/formats.py` + gguf helpers + profiles.
- **API surface**: Add route in `forge/api/routes/`, register in `forge/main.py`, add typed client in `forge-ui/src/lib/api/`, add page or component.
- **Tests**: Mirror file names under `tests/`. Use fixtures from `conftest.py`. Mark slow tests.

## Local dev loop (agent-friendly)

```bash
cd Seiso
source .venv/bin/activate
seiso doctor
# UI dev
cd forge-ui && npm run dev   # in second terminal (against running forge)
# Run one smoke
seiso train --config configs/smoke_train_cpu.yaml || true
# Quality
python3 scripts/run_ci_local.py --job lint --skip-install
python3 scripts/run_ci_local.py --job test --skip-install -k "not slow"
```

See also `docs/troubleshooting.md`, `docs/install.md`, and `docs/forge.md` (for dev mode with hot reload).

## Repository layout (abbrev.)

```
seiso/                 # core (runners, kernels, training, export, compress, rl, ...)
seiso/slime/           # slime RL (HF / multi-GPU DDP / SGLang / vLLM rollouts)
seiso/chat/            # shared chat prompts + output sanitize (CLI + Forge)
seiso/slime_single_gpu/# compat shim → seiso.slime (do not add new code here)
seiso_cli/main.py      # CLI
forge/                 # FastAPI (orchestrators, routes, services, security, db)
forge-ui/              # React sources + built dist/
seiso/codellama_compress/    # bundled LLM compression (research)
seiso/adaptive_quant/        # bundled adaptive RL quant (research)
seiso/analysis/        # bundled RL quant analysis CLI/helpers (research)
seiso/research/        # provenance / determinism helpers (+ optional Nostr attest under seiso/research/nostr/)
configs/               # example + smoke YAML/JSON
scripts/               # install, doctor, run_ci_local, ...
tests/                 # broad pytest coverage
docs/                  # user + dev guides
```

Happy building. Keep it local, keep it safe, keep the memory guards happy.
