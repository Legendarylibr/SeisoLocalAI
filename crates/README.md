# Seiso Rust control plane

Hybrid rewrite workspace (see `docs/adr/0001-rust-hybrid-control-plane.md`).

| Crate | Role |
|-------|------|
| `seiso-core` | Config, paths, env |
| `seiso-crypto` | `enc:v1:` AES-256-GCM (Python-compatible) |
| `seiso-sandbox` | `safe_join` / path policy |
| `seiso-protocol` | JSONL worker protocol v1 |
| `seiso-jobs` | Job supervisor + Python worker spawn |
| `seiso-db` | SQLite bootstrap |
| `seiso-forge` | axum HTTP server binary |
| `seiso-cli` | `seiso-rs` CLI (`doctor`, `forge`) |

```bash
cargo test --workspace
cargo run -p seiso-forge
# curl http://127.0.0.1:8765/api/health
cargo run -p seiso-cli -- doctor
```

Python worker (smoke): `PYTHONPATH=python python3 -m seiso_ml_worker`
