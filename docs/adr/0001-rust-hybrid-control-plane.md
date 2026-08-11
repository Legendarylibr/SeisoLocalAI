# ADR 0001: Hybrid Rust control plane

**Status:** Accepted  
**Date:** 2026-08-11  
**Context:** SeisoLocalAI is ~98k LOC of Python (core + Forge) + React UI, tightly coupled to PyTorch / HF / PEFT / TRL for training and research pipelines.

## Decision

Rewrite the **control plane** (Forge HTTP server, CLI shell, auth, path sandbox, field crypto, job supervision, inference process management) in **Rust**.

Keep **ML workers in Python** (training, slime, distill-rl, compression, CUDA/Triton kernels) behind a **versioned JSONL worker protocol**.

Keep **forge-ui** (React/TypeScript).

Do **not** reimplement the full HF/PyTorch training stack in Candle/Burn for v1.

## Architecture

```
forge-ui  --REST/SSE-->  seiso-forge (Rust/axum)
                              |                \
                              | JSONL worker    \ subprocess
                              v                  v
                        seiso-ml (Python)    llama.cpp / sidecars
```

Crate layout lives under `crates/` (see workspace `Cargo.toml`). Dual-run via `SEISO_FORGE_IMPL=rust|python` (env) until cutover.

## Consequences

**Positive**

- Memory-safe security surface (sandbox, CSRF, crypto).
- Better packaging (cargo binaries; Python only for train extras).
- Incremental delivery with parity gates.

**Negative / costs**

- Dual maintenance until Python Forge is decommissioned.
- Crypto and DB must stay byte-compatible with existing `~/.seiso` data.
- Team must own Rust CI (`cargo fmt/clippy/test`) plus existing Python gates.

## Non-goals

- Pure-Rust training parity with PEFT/TRL/slime.
- Replacing React with a Rust GUI in this ADR.
- Rewriting Lean provenance formalization.

## Phases

0 Inventory / API freeze · 1 Skeleton · 2 Auth/settings · 3 Jobs/workers ·  
4 Inference/chat · 5 CLI/packaging · 6 Harden & decommission Python forge  

Full plan: session plan / project docs (hybrid control plane).

## References

- `seiso/security/__init__.py` — path sandbox (must port with tests)
- `seiso/research/nostr/crypto.py` — `enc:v1:` AES-256-GCM (must be byte-identical)
- `forge/` — FastAPI reference implementation until cutover
