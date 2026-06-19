# Contributing

Thanks for helping improve Seiso.

## Before you open a PR

1. Install dev dependencies: `start` (or `pip install -e ".[forge,train,dev]"` plus `requirements-dev.txt`).
2. Run the local quality gate: `make ci-fast` (or `make ci` before large UI changes).
3. See [docs/CI_LOCAL.md](docs/CI_LOCAL.md) for job details and the recommended cross-platform matrix.

## Project layout

| Path | Purpose |
|------|---------|
| `forge/` | FastAPI backend for Forge |
| `forge-ui/` | React UI |
| `seiso/` | Core training, inference, and export library |
| `seiso_cli/` | CLI entry points |
| `docs/` | User and developer documentation |
| `tests/` | Python tests |
| `scripts/` | Install, start, and CI helpers |
| `deploy/` | Reverse-proxy and systemd examples |
| `third_party/` | Vendored dependencies (e.g. adaptive RL quant) |

## Documentation

Start at [docs/README.md](docs/README.md) or [docs/getting-started.md](docs/getting-started.md).
