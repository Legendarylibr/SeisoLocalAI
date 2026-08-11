# Rust hybrid control plane — migration notes

Companion to [ADR 0001](adr/0001-rust-hybrid-control-plane.md).

## Inventory (Phase 0)

### Surfaces

| Surface | Today | Target |
|---------|-------|--------|
| Forge HTTP | `forge/` FastAPI | `crates/seiso-forge` axum |
| CLI | `seiso_cli/` Typer | `crates/seiso-cli` clap |
| Core ML | `seiso/` | Python worker (`python/seiso_ml_worker` later) |
| UI | `forge-ui/` | Unchanged (contract-stable API) |

### Critical route groups to freeze

Record OpenAPI / golden fixtures before large Python churn:

- Auth / onboarding (`forge/api/routes/auth.py`, `forge/services/nostr_auth.py`)
- System / hardware (`forge/api/routes/system.py`)
- Models / hub (`forge/api/routes/models.py`)
- Inference / chat SSE (`forge/api/routes/inference.py`, `forge/services/inference_chat*.py`)
- Training jobs + SSE (`forge/api/routes/training.py`, orchestrators)
- Settings (encrypted tokens)
- Compat `/v1/*`

Generate snapshot (when forge is runnable):

```bash
# After starting python forge, dump OpenAPI if exposed, or use route inventory:
rg -n '@router\.(get|post|put|patch|delete)' forge/api/routes --glob '*.py'
```

### SQLite / crypto

- Field format: `enc:v1:` + base64(iv12 || ciphertext||tag) — see `seiso/research/nostr/crypto.py`
- Key: 32 bytes, base64 or 64-char hex (`SEISO_DB_ENCRYPTION_KEY`)
- Rust port: `crates/seiso-crypto` (parity tests with fixed vectors)

### Path sandbox

- `safe_join`, `assert_within`, `USER_SCOPED_DATA_ROOTS` — `seiso/security/__init__.py`
- Rust port: `crates/seiso-sandbox`

### Worker protocol

See `crates/seiso-protocol` — versioned JSONL ops: `train.start`, `log`, `metric`, `progress`, `done`, `error`, `cancel`.

## Dual-run

| Env | Meaning |
|-----|---------|
| `SEISO_FORGE_IMPL=python` | Current FastAPI (default until cutover) |
| `SEISO_FORGE_IMPL=rust` | `seiso-forge` binary |

## Building the Rust workspace

```bash
cargo test --workspace
cargo run -p seiso-forge
# health: curl -s http://127.0.0.1:8765/api/health
cargo run -p seiso-cli -- doctor
```

Requires Rust 1.75+ (CI pins stable). On hosts where `gcc` is blocked, use clang:

```bash
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang CC=clang CXX=clang++
cargo test --workspace
```

Phase 1 status:

- [x] Workspace crates: core, crypto, sandbox, protocol, jobs, db, forge, cli
- [x] Python-compatible `enc:v1:` AES-256-GCM vectors
- [x] Path sandbox port + unit tests
- [x] JSONL worker protocol v1 + `python/seiso_ml_worker` smoke
- [x] `seiso-forge` `/api/health`, `/api/system`, `/api/jobs*`
- [x] GitHub Actions `rust.yml`
- [ ] Auth / Nostr onboarding (Phase 2)
- [ ] Real train/export workers (Phase 3)
- [ ] Inference / chat (Phase 4)


## Baselines to capture (Phase 0 checklist)

- [ ] Cold start time of `seiso forge` (Python)
- [ ] Idle RSS of forge process
- [ ] Chat SSE first-token latency (local tiny GGUF)
- [ ] Train job submit → first log event latency (`configs/smoke_train_cpu.yaml`)
