# Troubleshooting

## Model exceeds available memory

**Symptom:** Chat or Hub blocks a model, or load fails with an out-of-memory error.

**Fix:**
1. Click **Free memory** in Chat or Model Hub (unloads llama.cpp / MLX / PyTorch / Ollama from RAM — does not delete `hf_cache/` files).
2. Close other memory-heavy apps (browsers, other LLM tools).
3. Pick a smaller model or more aggressive quant (Q4_K_M, Q4_0).
4. On Mac ≤24 GB, prefer **llama.cpp + GGUF**; use MLX only when headroom is ample.
5. Optional escape hatch: `SEISO_ALLOW_MEMORY_OVERCOMMIT=1` (not recommended).

See [inference/backends.md](inference/backends.md#memory-management) for RAM-tier guidance.

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

Or run `start` — it builds the UI automatically if `forge-ui/dist` is missing.

**Symptom:** Sidebar shows old pages (for example **Image Compress** instead of **Distill-RL**).

**Fix:** You are running a stale UI build or an old branch. From the repo root:

```bash
git pull
cd forge-ui && npm ci && npm run build && cd ..
seiso forge
```

Hard-refresh the browser (`Cmd+Shift+R` / `Ctrl+Shift+R`) after rebuilding.

For UI development, run `seiso forge` in one terminal and `cd forge-ui && npm run dev` in another — browse http://127.0.0.1:5173. See [forge.md](forge.md).

## Install script fails

**Symptom:** `curl ... | bash` exits with missing Python, Node, or git.

**Fix:**
- Python 3.10+: `python3 --version`
- Node.js 18+: `node --version` and `npm --version` ([nodejs.org](https://nodejs.org/))
- git: `git --version`
- Custom path: `SEISO_INSTALL_DIR="$HOME/code/Seiso" curl …/start | bash`
- Re-run from a clone: `start`

**Symptom:** `Seiso not found` when running `start` or `curl …/start | bash`.

**Fix:** Run the installer first, or set `SEISO_INSTALL_DIR` to your clone path.

**Symptom:** `seiso_run_install_worker: command not found`, `no such file or directory` sourcing `common.sh`, then `Install failed` / `Doctor script not found`.

**Cause:** The install directory exists but the git clone did not finish (network error, non-empty `$HOME/Seiso`, or partial download).

**Fix:**
```bash
rm -rf "$HOME/Seiso"
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Or point at a clean path:
```bash
SEISO_INSTALL_DIR="$HOME/code/Seiso" curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

## Flash Attention / flash-attn wheel build fails

**Symptom:** `pip install` fails building `flash-attn`, or errors about missing `pyproject.toml` / `setup.py` on `C:\` or `/mnt/c/...`.

**Cause:** The repo or pip build temp dir is on a **Windows filesystem** (common in WSL when the clone lives under `/mnt/c/Users/...`). CUDA extension builds need a Linux-native path.

**Fix:**
1. Install on the Linux filesystem: `SEISO_INSTALL_DIR="$HOME/Seiso"` then re-run `start`
2. Skip flash-attn during install: `SEISO_SKIP_FLASH_ATTN=1 start`
3. After a successful main install on `$HOME/Seiso`, optionally run `./scripts/install_flash_attn.sh`
4. Ensure CUDA toolkit (`nvcc --version`) and PyTorch CUDA match your driver ([pytorch.org](https://pytorch.org/get-started/locally/))

Seiso does **not** require flash-attn — training and chat fall back to PyTorch SDPA when it is missing.

## Forge refuses to start (remote / proxy settings)

**Symptom:** `RuntimeError: SEISO_ALLOW_REMOTE=true requires explicit acknowledgement`

**Fix:** Only enable remote if intentional:
```bash
export SEISO_ALLOW_REMOTE=true
export SEISO_REMOTE_ACK=1
```

**Symptom:** `SEISO_TRUST_PROXY=true requires SEISO_TRUSTED_PROXY_IPS`

**Fix:** Set proxy allowlist to your reverse proxy address:
```bash
export SEISO_TRUSTED_PROXY_IPS=127.0.0.1,::1
```

**Symptom:** Remote + tools/code-exec blocked at startup

**Fix:** Do not combine unless you accept the risk, then:
```bash
export SEISO_REMOTE_DANGEROUS_ACK=1
```

## OpenAI `/v1` returns 401

Use the inference-scoped key (not your admin password):

```bash
# Linux / macOS / WSL
cat "$HOME/.seiso/.inference_api_key"
```

```powershell
# Windows
Get-Content "$env:USERPROFILE\.seiso\.inference_api_key"
```

Use header: `Authorization: Bearer seiso_sk_...`

Or log in via Forge and use the session JWT.

## Port in use / Forge already running

Forge binds exclusively to `127.0.0.1:8765` by default. A second `seiso forge` prints:

```text
Error: Forge is already running — cannot bind 127.0.0.1:8765.
```

Stop the existing Forge process, or use a different port **and** data directory:

```bash
SEISO_PORT=8766 SEISO_DATA_DIR=~/.seiso-alt seiso forge
```

If two processes share the same `SEISO_DATA_DIR` on different ports, the data-dir lock (`.forge.lock`) blocks the second instance.

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

Default data dir: `$HOME/.seiso` on Linux/macOS/WSL, `%USERPROFILE%\.seiso` on Windows (override with `SEISO_DATA_DIR`).
