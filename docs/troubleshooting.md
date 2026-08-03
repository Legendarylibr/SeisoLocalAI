# Troubleshooting

## Model exceeds available memory

**Symptom:** Chat or Hub blocks a model, or load fails with an out-of-memory error.

**Fix:**
1. Click **Free memory** in Chat or Model Hub (unloads llama.cpp / MLX / PyTorch from RAM; with llama-swap it also asks the sidecar to unload running model processes — downloaded `hf_cache/` files are kept).
2. Close other memory-heavy apps (browsers, other LLM tools).
3. Pick a smaller model or more aggressive quant (Q4_K_M, Q4_0).
4. On Mac ≤24 GB, prefer **llama.cpp + GGUF**; use MLX only when headroom is ample.
5. Optional escape hatch: `SEISO_ALLOW_MEMORY_OVERCOMMIT=1` (not recommended).

See [inference/backends.md](inference/backends.md#memory-management) for RAM-tier guidance.

## Native Linux GGUF chat says Ollama or llama-swap is required

**Symptom:** Chat fails with setup guidance for Ollama, `llama-swap`, or `SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1`.

**Cause:** On native Linux NVIDIA, Seiso does not silently fall back to in-process llama.cpp because CUDA failures there can kill Forge. The safe default is Ollama-first isolated chat.

**Fix:**
1. Re-run the Linux NVIDIA bootstrap: `curl -fsSL …/scripts/bootstrap/linux-nvidia.sh | bash`
2. Or install/start Ollama manually: `curl -fsSL https://ollama.com/install.sh | sh && ollama serve`
3. Run `seiso doctor` — Ollama API should show OK at `http://127.0.0.1:11434`
4. Optional fallback: install `llama-swap` when Ollama is down
5. Only set `SEISO_LLAMA_ALLOW_INPROCESS_NATIVE_LINUX=1` if you accept that an in-process llama.cpp crash can stop Forge

## CUDA kernels fail to compile

**Symptom:** Log shows `CUDA kernel load failed`, training uses PyTorch fallback.

**Common error (CUDA 13 / PyTorch 2.12):**

```text
ptxas ... fatal : Unsupported .version 9.3; current version is '9.0'
```

The pip `cuda-toolkit[nvcc]==13.0.2` wheel bundles an `nvcc` that emits PTX 9.3 while its `ptxas` only accepts up to 9.0. Upgrade the toolkit inside the Seiso venv, clear the JIT cache, then restart:

```bash
cd ~/Seiso
source .venv/bin/activate
pip install 'cuda-toolkit[nvcc]>=13.1.0'
rm -rf ~/.cache/torch_extensions/*/seiso_cuda_kernels
start
```

Check compatibility:

```bash
.venv/bin/python -c "from seiso.kernels.cuda_env import cuda_toolkit_status; print(cuda_toolkit_status())"
```

`ptxas_compatible` should be `True` and `ptxas_max` should be at least `9.3` when using CUDA 13.

**Other fixes:**
- Install CUDA toolkit; ensure `nvcc --version` works
- Windows: install Visual Studio Build Tools
- Match PyTorch CUDA version to toolkit
- Keep Triton’s bundled `ptxas` off `PATH` ahead of the toolkit (Seiso sanitizes this automatically on kernel JIT)

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

**Symptom:** `Seiso repository incomplete at …/Seiso`, or `seiso_run_install_worker: command not found` / `no such file or directory` sourcing `common.sh`, then `Install failed` / `Doctor script not found`.

**Cause:** The install directory exists but the git clone did not finish (network error, non-empty `$HOME/Seiso`, or partial download). The installer expects `pyproject.toml`, `seiso_cli/`, and `scripts/lib/common.sh`.

**Fix:** If the directory has a `.git` folder, re-run `start` — the installer will try to repair the clone. Otherwise remove the partial directory and install again:

```bash
rm -rf "$HOME/Seiso"
curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

Or point at a clean path:
```bash
SEISO_INSTALL_DIR="$HOME/code/Seiso" curl -fsSL https://raw.githubusercontent.com/Legendarylibr/SeisoLocalAI/main/start | bash
```

**Symptom:** `Seiso CLI missing at …/Seiso/.venv/bin/seiso` or `seiso: command not found` right after install on native Linux.

**Cause:** Usually the pip install did not finish (install TUI reported success too early, or heavy extras like PyTorch / llama.cpp failed). On failure the installer falls back to core `[forge]` first so the CLI exists, then retries optional training/inference extras.

**Fix:**
```bash
SEISO_NO_BANNER=1 seiso-start
# or: SEISO_NO_BANNER=1 start
```

If that still fails, inspect the log and reinstall core extras manually:
```bash
cat "$HOME/Seiso/.seiso-install.log" | tail -50
source "$HOME/Seiso/.venv/bin/activate"
pip install -U pip wheel 'setuptools>=83'
pip install -e "$HOME/Seiso[forge]"
"$HOME/Seiso/.venv/bin/seiso" forge
```

Optional training/inference extras can be added afterward (pick your platform):
```bash
# Linux NVIDIA:
pip install -e "$HOME/Seiso[train,cuda,llamacpp]"
# Linux CPU / ROCm base:
# pip install -e "$HOME/Seiso[train,llamacpp]"
# macOS Apple Silicon:
# pip install -e "$HOME/Seiso[train,llamacpp,mlx]"
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

**Symptom:** Remote + tools blocked at startup

**Fix:** Do not combine unless you accept the risk, then:
```bash
export SEISO_REMOTE_DANGEROUS_ACK=1
```

**Symptom:** Remote + code-exec blocked at startup

**Fix:** Code execution is AST deny-list only (not a full OS sandbox) and **cannot**
be combined with `SEISO_ALLOW_REMOTE`. Disable one of:
```bash
unset SEISO_ALLOW_CODE_EXEC
# or
unset SEISO_ALLOW_REMOTE
```

## NeMo RL checkout or `uv` not found

**Symptom:** `FileNotFoundError: NeMo RL checkout not found` or `uv is required to launch NeMo RL`.

**Cause:** `method: nemo_rl` / `seiso nemo-rl` only launches an **external** [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) tree. Seiso does not ship NeMo RL or its lockfile.

**Fix:**
1. Clone recursively and export the root:
   ```bash
   git clone --recursive https://github.com/NVIDIA-NeMo/RL.git ~/nemo-rl
   export SEISO_NEMO_RL_ROOT=~/nemo-rl
   ```
   Or set `nemo_rl_root: ~/nemo-rl` in the training YAML.
2. Install [`uv`](https://docs.astral.sh/uv/) and ensure it is on `PATH`, or set `SEISO_UV` / `UV` to the executable.
3. Preview Seiso’s projected Hydra command without running NeMo RL:
   ```bash
   seiso train --config configs/smoke_nemo_rl.yaml
   ```
4. For a 10-step install check inside a real checkout, set `nemo_rl_recipe: smoke`.

See [training/quickstart.md § NeMo RL](training/quickstart.md#nemo-rl) (includes upstream citation).

## Compat `/v1` returns 401

Use the inference-scoped key (not your Nostr nsec):

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

**Tool calling on `/v1`:** even with `SEISO_ALLOW_COMPAT_TOOLS=true`, the inference API key stays chat-only. Use a Forge session JWT for Compat tools.

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

## Chat reply stops early or says “continuing (N/M)”

**Symptom:** Songs, essays, or “longer” / “again” replies cut off mid-line, or the UI briefly shows a continue cue.

**Cause:** On native Linux NVIDIA (and other OOM-safe profiles), each generation pass is capped (often ~512–768 tokens). Seiso finishes long replies with **multi-pass auto-continue** (fixed `n_ctx`, linear-decay packing). A short pause with “continuing…” is expected.

**If a reply still ends incomplete:**
1. Ask **`continue`**, **`longer`**, or **`again`** — those are treated as long-form follow-ups.
2. Prefer a stronger instruct/chat model if small models keep emitting EOS mid-sentence.
3. Optional env (restart Forge after setting):
   ```bash
   # Cumulative output ceiling across auto-continue passes (default 32768, max 131072)
   export SEISO_CHAT_AUTO_CONTINUE_TOTAL_TOKENS=65536
   # Extra passes: -1 = auto from budget / per-pass size (0 disables continues)
   export SEISO_CHAT_AUTO_CONTINUE_MAX=-1
   ```
4. Do **not** set `SEISO_LLAMA_UNSAFE_LONG_COMPLETIONS=1` unless you accept higher OOM risk for single-pass size.

## Check hardware detection

```bash
python -c "from seiso.training.platform_caps import training_capabilities; import json; print(json.dumps(training_capabilities(), indent=2))"
```

## Reset Seiso data

Default data dir: `$HOME/.seiso` on Linux/macOS/WSL, `%USERPROFILE%\.seiso` on Windows (override with `SEISO_DATA_DIR`).

## Opt-in pay marketplace / Ark + L402

> **Not functional yet — do not use.** Live Ark and L402 settlement are not wired; faucet/sim only for local smoke tests.

| Symptom | Check |
|---------|--------|
| `seiso pay` refuses to run | `SEISO_ALLOW_PAY=1` must be set |
| Settle / funding fails closed | Set `SEISO_PROTOCOL_TREASURY_ARK` (and operator Ark) **or** use `SEISO_PAY_FAUCET=1` for local tests only |
| `SEISO_ARK_BACKEND=bark\|second` errors | Backend not bundled yet — leave unset / use faucet for smoke tests |
| Expecting live L402 | Live Lightning not wired — set `SEISO_PAY_L402_SIM=1` (or faucet) for sim fund/exchange; use `seiso pay session fund --l402` ([L402 explained](https://lightningfaucet.com/learn/l402-payments-explained/)) |
| Job failed / cancelled but balance missing | Escrow should restore to session (`refunded_sats`); check job receipt `settlement.status=refunded` |
| Buyer can’t reach operator | Hit `SEISO_PAY_URL` (pay sidecar), not Forge; check `GET /.well-known/seiso-pay.json` |
| Accidentally exposed faucet | Turn `SEISO_PAY_FAUCET` **off** on any public market |

Full guide: [pay/marketplace.md](pay/marketplace.md).

## Opt-in Buzz mesh

> Secondary / opt-in Buzz-agent path. Local single-node training stays primary.

| Symptom | Check |
|---------|--------|
| `seiso mesh` refuses | `SEISO_ALLOW_MESH=1`, `BUZZ_PRIVATE_KEY` (nsec), and shared `SEISO_MESH_TOKEN` (≥16 chars; never post to Buzz) |
| Loopback master refused | Real multi-host needs a reachable addr; for single-host smoke only set `SEISO_MESH_ALLOW_LOOPBACK=1` |
| Peers don’t join | Import signed plan (`seiso mesh import-plan`); same token + channel; master addr reachable; ranks claimed with `--rank` |
| Worker only prints overlay | Pass `--base-config` to materialize; `--dry-run` / `--launch` to preview or start train |
| Confused with marketplace | Mesh has **no** protocol fee; paid remote compute is [pay/marketplace.md](pay/marketplace.md) |

Full guide: [training/mesh.md](training/mesh.md).
