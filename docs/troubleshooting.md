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

Build frontend: `cd forge-ui && npm install && npm run build`

## Port in use

```bash
SEISO_PORT=8766 seiso forge
```

## HTTPS / reverse proxy

Forge runs HTTP on localhost by default. For HTTPS access, terminate TLS with Caddy or nginx:

```bash
cp deploy/env.https.example .env   # set SEISO_CORS_ORIGINS to your https:// domain
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
