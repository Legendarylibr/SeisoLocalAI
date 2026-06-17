# Seiso documentation

Choose your platform, install, then run Forge or the CLI.

## Platform guides

| Platform | Guide |
|----------|--------|
| Linux + NVIDIA (recommended) | [linux-nvidia.md](platforms/linux-nvidia.md) |
| Linux + AMD ROCm | [linux-amd-rocm.md](platforms/linux-amd-rocm.md) |
| macOS (Apple Silicon / Intel) | [macos.md](platforms/macos.md) |
| Windows (native) | [windows.md](platforms/windows.md) |
| WSL2 + NVIDIA | [wsl.md](platforms/wsl.md) |

## Tasks

| Task | Guide |
|------|--------|
| Install extras & dependencies | [install.md](install.md) |
| Train (CLI + Forge) | [training/quickstart.md](training/quickstart.md) |
| Fused GPU kernels | [training/kernels.md](training/kernels.md) |
| Multi-GPU | [training/multi-gpu.md](training/multi-gpu.md) |
| Inference backends | [inference/backends.md](inference/backends.md) |
| HTTPS / reverse proxy | [deployment/reverse-proxy.md](deployment/reverse-proxy.md) |
| Problems & fixes | [troubleshooting.md](troubleshooting.md) |

## Quick commands

```bash
# Full stack (Linux NVIDIA)
pip install -e ".[forge,train,cuda,dev]"

# Start Forge (API + web UI)
seiso forge
# → http://127.0.0.1:8765

# Train from config
seiso train --config configs/example_lora.yaml

# Benchmark fused kernels (NVIDIA / ROCm GPU)
seiso-bench-kernels --op all
```
