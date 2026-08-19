"""OS/tier memory defaults applied at Forge startup — setdefault only."""

from __future__ import annotations

import os
import platform
from typing import Any

from seiso.env import env_bool
from seiso.hardware.tiers import (
    HardwareTier,
    classify_tier,
    performance_headroom_mb,
    vram_headroom_mb,
)
from seiso.memory.protection import (
    discrete_gpu_total_mb,
    gpu_batch_tier_caps,
    llama_batch_limits_for_headroom,
)

# Re-exported so existing tests can monkeypatch the symbol; idle apply path
# uses ``_lean_runtime_caps`` and does not call this (avoids torch/MLX/bnb).
from seiso.training.platform_caps import training_capabilities  # noqa: F401


def native_linux_nvidia_llama_batch_caps(
    *,
    tier: HardwareTier,
    headroom_mb: int,
    low: bool,
    gpu_total_mb: int = 0,
) -> tuple[int, int, int]:
    """VRAM-derived llama.cpp batch/ubatch/cache caps for native Linux NVIDIA."""
    total = gpu_total_mb or discrete_gpu_total_mb()
    batch, ubatch = gpu_batch_tier_caps(total, "normal")
    if low:
        low_batch, low_ubatch = gpu_batch_tier_caps(total, "compact")
        batch = min(batch, low_batch)
        ubatch = min(ubatch, low_ubatch, batch)
    cache_cap = min(2048, max(256, batch * 2))
    return batch, ubatch, cache_cap


def _refresh_native_linux_llama_env(*, batch_cap: int, ubatch_cap: int, cache_cap: int) -> None:
    """Pin native Linux llama.cpp batch env to VRAM-derived caps on every Forge start."""
    if env_bool("SEISO_DISABLE_MEMORY_CAPS", False):
        return

    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "false")
    os.environ.setdefault("SEISO_LLAMA_KV_QUANT", "false")
    os.environ.setdefault("SEISO_CHAT_CONTEXT_CHARS", "8000")
    os.environ["SEISO_LLAMA_BATCH"] = str(batch_cap)
    os.environ["SEISO_LLAMA_UBATCH"] = str(min(ubatch_cap, batch_cap))
    os.environ["SEISO_LLAMA_CACHE_MB"] = str(cache_cap)
    if env_bool("SEISO_LLAMA_CONSERVATIVE", False):
        os.environ["SEISO_LLAMA_FLASH_ATTN"] = "false"
        os.environ["SEISO_LLAMA_KV_QUANT"] = "false"


def memory_profile_label(profile: dict[str, Any]) -> str:
    """Derive low vs balanced from hardware budget (not transient free VRAM)."""
    headroom = performance_headroom_mb(profile)
    ram_gb = float(profile.get("ram_gb") or 0)
    if headroom < 4096 or (ram_gb > 0 and ram_gb <= 12):
        return "low"
    if headroom < 12288 or (ram_gb > 0 and ram_gb <= 24):
        return "balanced"
    return "balanced"


def _host_ram_gb() -> float:
    """Cheap RAM probe without importing torch/MLX."""
    try:
        from seiso.hardware.profile import _ram_gb

        return float(_ram_gb())
    except Exception:
        return 0.0


def _seed_skip_mlx_probe_early(*, system: str, ram_gb: float) -> None:
    """Prefer GGUF on tight Apple unified RAM — set before any MLX import path."""
    if system != "Darwin":
        return
    if ram_gb <= 0 or ram_gb > 24:
        return
    os.environ.setdefault("SEISO_SKIP_MLX_PROBE", "1")


def _nvidia_llama_gpu_layers_default() -> str:
    """Return ``0`` only when llama_cpp is already imported and lacks GPU offload."""
    import sys

    if "llama_cpp" not in sys.modules:
        return "-1"
    try:
        from seiso.inference.model_pool import _llama_gpu_offload_ok

        return "-1" if _llama_gpu_offload_ok() else "0"
    except ImportError:
        return "-1"


def _lean_runtime_caps(profile: dict[str, Any]) -> dict[str, Any]:
    """Inference-startup caps without training stacks (bnb/triton/mlx/torch)."""
    raw_gpus = profile.get("gpus")
    gpus: list[Any] = raw_gpus if isinstance(raw_gpus, list) else []
    gpu_count = len(gpus)
    vendor = "none"
    nvidia_hardware = False
    if gpus:
        first = gpus[0] if isinstance(gpus[0], dict) else {}
        name = str(first.get("name") or "").lower()
        vendor_raw = str(first.get("vendor") or "").lower()
        if "nvidia" in vendor_raw or "nvidia" in name or "geforce" in name or "rtx" in name:
            vendor = "nvidia"
            nvidia_hardware = True
        elif "amd" in vendor_raw or "radeon" in name:
            vendor = "amd"
    if not nvidia_hardware:
        try:
            from seiso.security.nvidia_boundary import nvidia_smi_visible

            if nvidia_smi_visible():
                nvidia_hardware = True
                vendor = "nvidia"
                if gpu_count <= 0:
                    gpu_count = 1
        except ImportError:
            pass

    wsl2 = False
    try:
        from seiso.kernels.platform import detect_wsl2

        wsl2 = bool(detect_wsl2())
    except ImportError:
        pass

    train_platform = "cpu"
    if nvidia_hardware:
        train_platform = "cuda"
    elif platform.system() == "Darwin" and str(profile.get("backend") or "") in {
        "metal",
        "mlx",
        "mps",
    }:
        train_platform = "mps"

    return {
        "nvidia_hardware": nvidia_hardware,
        "gpu_count": gpu_count,
        "vendor": vendor,
        "train_platform": train_platform,
        "wsl2": wsl2,
        # Idle path never probes MLX; Train/API can still call training_capabilities().
        "supports_mlx_inference": False,
    }


def apply_platform_memory_profile(*, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Apply lean RAM defaults for this OS/tier.

    Uses os.environ.setdefault — never overrides explicit user configuration.
    """
    system = platform.system()
    # Seed SKIP_MLX before hardware_profile/detect_backend can import mlx.core.
    early_ram = float((profile or {}).get("ram_gb") or 0) or _host_ram_gb()
    _seed_skip_mlx_probe_early(system=system, ram_gb=early_ram)

    if profile is None:
        from seiso.hardware.profile import hardware_profile

        profile = hardware_profile()

    tier = classify_tier(profile)
    free_headroom = vram_headroom_mb(profile)
    headroom = performance_headroom_mb(profile)
    ram_gb = float(profile.get("ram_gb") or early_ram or 0)
    # Re-seed after profile resolve (covers callers that passed profile without RAM).
    _seed_skip_mlx_probe_early(system=system, ram_gb=ram_gb)
    caps = _lean_runtime_caps(profile)
    memory_caps_disabled = env_bool("SEISO_DISABLE_MEMORY_CAPS", False)
    low = not memory_caps_disabled and (
        os.environ.get("SEISO_MEMORY_PROFILE", "").strip().lower() == "low"
        or memory_profile_label(profile) == "low"
    )
    # Apple unified memory only — discrete Linux VRAM must keep VRAM-derived caps.
    tight_apple = not memory_caps_disabled and system == "Darwin" and ram_gb > 0 and ram_gb <= 24

    os.environ.setdefault("SEISO_LLAMA_USE_MMAP", "true")
    os.environ.setdefault("SEISO_LLAMA_USE_MLOCK", "false")
    os.environ.setdefault("SEISO_LLAMA_NO_PERF", "true")

    if tight_apple:
        cache_mb = "0"
    elif tier == HardwareTier.WORKSTATION and ram_gb >= 32:
        cache_mb = "2048"
    else:
        cache_mb = "1024"
    native_linux_nvidia = False
    if system == "Linux":
        try:
            from seiso.platform import is_native_linux_nvidia

            native_linux_nvidia = is_native_linux_nvidia(profile=profile)
        except ImportError:
            pass
    # Prompt cache steals unified RAM on tight Macs; native Linux already disables it.
    os.environ.setdefault(
        "SEISO_LLAMA_PROMPT_CACHE",
        "false" if (native_linux_nvidia or tight_apple) else "true",
    )
    # Native Linux sets cache from batch_caps once below; avoid a second, looser cap.
    if not native_linux_nvidia:
        os.environ.setdefault("SEISO_LLAMA_CACHE_MB", cache_mb)

    if system == "Darwin":
        if tier == HardwareTier.CPU_ONLY:
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")
            os.environ.setdefault(
                "SEISO_LLAMA_THREADS", str(min(max((os.cpu_count() or 4) - 2, 2), 8))
            )
        elif tier == HardwareTier.APPLE_UNIFIED:
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")

    elif system == "Windows":
        if caps.get("nvidia_hardware") or (
            caps.get("gpu_count", 0) > 0 and caps.get("vendor") == "nvidia"
        ):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "-1")
        elif caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            os.environ.setdefault("SEISO_LLAMA_BATCH", "512")

    elif system == "Linux":
        if caps.get("wsl2"):
            data_dir = os.environ.get("SEISO_DATA_DIR", "")
            if data_dir.startswith("/mnt/"):
                import logging

                logging.getLogger(__name__).warning(
                    "SEISO_DATA_DIR on /mnt/ (Windows filesystem) — move to ~/ for better mmap performance"
                )
        if caps.get("nvidia_hardware") or (
            caps.get("gpu_count", 0) > 0 and caps.get("vendor") == "nvidia"
        ):
            # Avoid importing llama_cpp at idle. If already loaded and the wheel
            # cannot offload, seed CPU layers; otherwise prefer full offload.
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", _nvidia_llama_gpu_layers_default())
            if tier in (
                HardwareTier.WORKSTATION,
                HardwareTier.CAPABLE,
                HardwareTier.MODEST,
                HardwareTier.EDGE,
            ):
                os.environ.setdefault(
                    "SEISO_LLAMA_THREADS",
                    str(
                        min(
                            max((os.cpu_count() or 4) - 2, 4),
                            14 if tier == HardwareTier.WORKSTATION else 10,
                        )
                    ),
                )
                if native_linux_nvidia:
                    gpu_total = discrete_gpu_total_mb(profile)
                    batch, ubatch, cache_cap = native_linux_nvidia_llama_batch_caps(
                        tier=tier,
                        headroom_mb=headroom,
                        low=low,
                        gpu_total_mb=gpu_total,
                    )
                    _refresh_native_linux_llama_env(
                        batch_cap=batch,
                        ubatch_cap=ubatch,
                        cache_cap=min(int(cache_mb), cache_cap),
                    )
                else:
                    batch, ubatch = llama_batch_limits_for_headroom(headroom)
                    os.environ.setdefault("SEISO_LLAMA_BATCH", str(batch))
                    os.environ.setdefault("SEISO_LLAMA_UBATCH", str(ubatch))
            elif native_linux_nvidia:
                gpu_total = discrete_gpu_total_mb(profile)
                batch, ubatch, cache_cap = native_linux_nvidia_llama_batch_caps(
                    tier=tier,
                    headroom_mb=headroom,
                    low=low,
                    gpu_total_mb=gpu_total,
                )
                _refresh_native_linux_llama_env(
                    batch_cap=batch,
                    ubatch_cap=ubatch,
                    cache_cap=cache_cap,
                )
            if not low and tier in (HardwareTier.WORKSTATION, HardwareTier.CAPABLE):
                if not native_linux_nvidia:
                    os.environ.setdefault("SEISO_LLAMA_FLASH_ATTN", "true")
                os.environ.setdefault("SEISO_LLAMA_OP_OFFLOAD", "true")
                os.environ.setdefault("SEISO_LLAMA_OFFLOAD_KQV", "true")
                if tier == HardwareTier.WORKSTATION:
                    os.environ.setdefault("SEISO_STREAM_BATCH_CHARS", "4")
        elif caps.get("train_platform") == "cpu" or not caps.get("gpu_count"):
            os.environ.setdefault("SEISO_LLAMA_GPU_LAYERS", "0")
            if native_linux_nvidia:
                os.environ.setdefault("SEISO_LLAMA_CACHE_MB", "256")

    result = {
        "memory_profile": memory_profile_label(profile),
        "tier": tier.value,
        "headroom_mb": headroom,
        "free_headroom_mb": free_headroom,
        "ram_gb": ram_gb,
        "os": system,
    }
    _log_platform_profile_applied(
        profile=profile,
        system=system,
        tier=tier,
        headroom=headroom,
        low=low,
    )
    return result


def _log_platform_profile_applied(
    *,
    profile: dict[str, Any],
    system: str,
    tier: HardwareTier,
    headroom: int,
    low: bool,
) -> None:
    _ = (profile, system, tier, headroom, low)
