# Troubleshooting

## CUDA kernels fail to compile

**Symptom:** Log shows `CUDA kernel load failed`, training uses PyTorch fallback.

**Fix:**
- Install CUDA toolkit; ensure `nvcc --version` works
- Windows: install Visual Studio Build Tools
- Match PyTorch CUDA version to toolkit

## `bitsandbytes` / QLoRA on macOS

**Symptom:** Training fails loading 4-bit model.

**Fix:** Use `quant: 16bit` in config. bitsandbytes is excluded on Darwin in `pyproject.toml`.

## MLX import error in chat

**Symptom:** `mlx-lm is required for MLX backend`

**Fix:** `pip install -e ".[mlx]"`

## Training loads MLX model on Mac

Fixed: training uses `for_training=True` → always PyTorch. Update Seiso if you see MLX in training logs.

## Fused kernels do nothing on Mac

Expected — no CUDA GPU. Disable fused checkboxes or set `use_triton: false`.

## AMD: no fused speedup

Install Triton manually: `pip install triton` with ROCm PyTorch.

## Forge UI blank

Build frontend from the repo root:

```bash
cd forge-ui && npm install && npm run build && cd ..
seiso forge
```

Or run `./scripts/start.sh` — it builds the UI automatically if `forge-ui/dist` is missing.

For UI development, run `seiso forge` in one terminal and `cd forge-ui && npm run dev` in another — browse http://127.0.0.1:5173. See [forge.md](forge.md).

## Install script fails

**Symptom:** `curl ... | bash` exits with missing Python, Node, or git.

**Fix:**
- Python 3.10+: `python3 --version`
- Node.js 18+: `node --version` and `npm --version` ([nodejs.org](https://nodejs.org/))
- git: `git --version`
- Custom path: `SEISO_INSTALL_DIR=~/code/Seiso curl -fsSL .../scripts/install.sh | bash`
- Re-run from a clone: `./scripts/install.sh`

**Symptom:** `Seiso not found` when running `start.sh`.

**Fix:** Run the installer first, or set `SEISO_INSTALL_DIR` to your clone path.

## Port in use

```bash
SEISO_PORT=8766 seiso forge
```

## HTTPS / reverse proxy

Forge runs HTTP on localhost by default. For HTTPS access, terminate TLS with Caddy or nginx:

```bash
cp deploy/env.https.example .env   # set SEISO_CORS_ORIGINS to your https:// domain
cd forge-ui && npm install && npm run build && cd ..
seiso forge
# then configure deploy/caddy/Caddyfile or deploy/nginx/seiso-forge.conf
```

Full guide: [deployment/reverse-proxy.md](deployment/reverse-proxy.md)

## Check hardware detection

```bash
python -c "from seiso.training.platform_caps import training_capabilities; import json; print(json.dumps(training_capabilities(), indent=2))"
```

## Reset Seiso data

Default data dir: `~/.seiso` (override with `SEISO_DATA_DIR`).
