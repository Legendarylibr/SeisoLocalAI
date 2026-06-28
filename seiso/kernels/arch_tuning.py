"""Architecture-aware CUDA kernel tuning for Ampere/Ada/Hopper/Blackwell."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from seiso.compat import StrEnum


class GpuArchFamily(StrEnum):
    UNKNOWN = "unknown"
    AMPERE = "ampere"  # sm_80, sm_86
    ADA = "ada"  # sm_89
    HOPPER = "hopper"  # sm_90
    BLACKWELL = "blackwell"  # sm_100+


@dataclass(frozen=True)
class ArchTuningProfile:
    family: GpuArchFamily
    sm: int
    rms_mode: int
    swiglu_vec: int
    lora_tile: int
    use_cuda_graphs: bool
    use_stream_overlap: bool
    use_wmma: bool
    use_persistent_kernels: bool
    prefer_flash_attn: str  # fa3, fa2, sdpa


def _sm_from_capability(major: int, minor: int) -> int:
    return major * 10 + minor


def arch_family_from_capability(major: int, minor: int) -> GpuArchFamily:
    sm = _sm_from_capability(major, minor)
    if sm >= 100:
        return GpuArchFamily.BLACKWELL
    if sm >= 90:
        return GpuArchFamily.HOPPER
    if sm == 89:
        return GpuArchFamily.ADA
    if sm >= 80:
        return GpuArchFamily.AMPERE
    return GpuArchFamily.UNKNOWN


@lru_cache(maxsize=4)
def detect_arch_tuning() -> ArchTuningProfile:
    """Detect GPU architecture and return optimal kernel launch profile."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ArchTuningProfile(
                family=GpuArchFamily.UNKNOWN,
                sm=0,
                rms_mode=0,
                swiglu_vec=8,
                lora_tile=256,
                use_cuda_graphs=False,
                use_stream_overlap=False,
                use_wmma=False,
                use_persistent_kernels=False,
                prefer_flash_attn="sdpa",
            )
        major, minor = torch.cuda.get_device_capability(0)
    except ImportError:
        return ArchTuningProfile(
            family=GpuArchFamily.UNKNOWN,
            sm=0,
            rms_mode=0,
            swiglu_vec=4,
            lora_tile=128,
            use_cuda_graphs=False,
            use_stream_overlap=False,
            use_wmma=False,
            use_persistent_kernels=False,
            prefer_flash_attn="sdpa",
        )

    sm = _sm_from_capability(major, minor)
    family = arch_family_from_capability(major, minor)

    if family == GpuArchFamily.BLACKWELL:
        return ArchTuningProfile(
            family=family,
            sm=sm,
            rms_mode=2,  # parallax + cp.async
            swiglu_vec=8,
            lora_tile=512,
            use_cuda_graphs=True,
            use_stream_overlap=True,
            use_wmma=True,
            use_persistent_kernels=True,
            prefer_flash_attn="fa3",
        )
    if family == GpuArchFamily.HOPPER:
        return ArchTuningProfile(
            family=family,
            sm=sm,
            rms_mode=2,
            swiglu_vec=8,
            lora_tile=384,
            use_cuda_graphs=True,
            use_stream_overlap=True,
            use_wmma=True,
            use_persistent_kernels=True,
            prefer_flash_attn="fa3",
        )
    if family == GpuArchFamily.ADA:
        return ArchTuningProfile(
            family=family,
            sm=sm,
            rms_mode=2,
            swiglu_vec=8,
            lora_tile=256,
            use_cuda_graphs=True,
            use_stream_overlap=True,
            use_wmma=True,
            use_persistent_kernels=True,
            prefer_flash_attn="fa2",
        )
    if family == GpuArchFamily.AMPERE:
        return ArchTuningProfile(
            family=family,
            sm=sm,
            rms_mode=1,  # stripe — lower smem on 80-class
            swiglu_vec=8,
            lora_tile=256,
            use_cuda_graphs=True,
            use_stream_overlap=True,
            use_wmma=True,
            use_persistent_kernels=True,
            prefer_flash_attn="fa2",
        )

    return ArchTuningProfile(
        family=family,
        sm=sm,
        rms_mode=0,
        swiglu_vec=4,
        lora_tile=128,
        use_cuda_graphs=False,
        use_stream_overlap=False,
        use_wmma=False,
        use_persistent_kernels=False,
        prefer_flash_attn="sdpa",
    )


def apply_arch_tuning(*, deterministic: bool = False) -> dict[str, Any]:
    """Apply architecture-specific kernel tuning to the CUDA extension."""
    profile = detect_arch_tuning()
    use_graphs = profile.use_cuda_graphs and not deterministic

    try:
        from seiso.kernels.cuda_ops import set_kernel_tuning

        set_kernel_tuning(
            profile.rms_mode,
            profile.swiglu_vec,
            profile.lora_tile,
            arch_sm=profile.sm,
            use_cuda_graphs=1 if use_graphs else 0,
            use_stream_overlap=1 if profile.use_stream_overlap else 0,
        )
    except (ImportError, AttributeError, RuntimeError):
        pass

    return {
        "arch_family": profile.family.value,
        "arch_sm": profile.sm,
        "rms_mode": profile.rms_mode,
        "swiglu_vec": profile.swiglu_vec,
        "lora_tile": profile.lora_tile,
        "use_cuda_graphs": use_graphs,
        "use_stream_overlap": profile.use_stream_overlap,
        "use_wmma": profile.use_wmma,
        "use_persistent_kernels": profile.use_persistent_kernels,
        "prefer_flash_attn": profile.prefer_flash_attn,
    }
