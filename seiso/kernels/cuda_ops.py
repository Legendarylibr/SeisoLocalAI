"""Native CUDA fused kernels — NVIDIA only; AMD uses Triton via dispatch."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from seiso.kernels.cuda_env import configure_cuda_build_env
from seiso.kernels.fallback_ops import pytorch_rms_norm as _pytorch_rms_norm
from seiso.kernels.platform import GpuVendor, detect_gpu

logger = logging.getLogger(__name__)

# Must run before any torch cpp_extension import (CUDA_HOME is read once).
configure_cuda_build_env()

_CUDA_DIR = Path(__file__).resolve().parent / "cuda"
_EXT: Any | None = None
_EXT_ERROR: str | None = None


def _torch_cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _ensure_cuda_build_env() -> dict[str, str]:
    """Configure CUDA_HOME, PATH, host compiler, and CCCL includes for nvcc."""
    return configure_cuda_build_env()


def _as_cuda_2d(tensor: Any) -> tuple[Any, tuple[int, ...] | None]:
    """Flatten trailing dims to (rows, cols) for native 2D kernels; None if already 2D."""
    if tensor.dim() == 2:
        return tensor, None
    if tensor.dim() < 2:
        raise ValueError(f"expected >=2D tensor, got shape {tuple(tensor.shape)}")
    orig = tuple(tensor.shape)
    flat = tensor.reshape(-1, orig[-1])
    return flat, orig


def _cuda_include_paths() -> list[str]:
    from seiso.kernels.cuda_env import cuda_build_include_paths

    base = [str(_CUDA_DIR / "include"), str(_CUDA_DIR)]
    return base + cuda_build_include_paths()


def _cuda_compile_flags() -> tuple[list[str], list[str]]:
    """Host and device compile flags tuned for native SM arch and WSL2 JIT."""
    platform = detect_gpu()
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-Xptxas",
        "-O3",
    ]

    cc = platform.cuda_compute_capability
    if cc is not None:
        major, minor = cc
        # Direct SM target avoids PTX ISA version mismatches across pip toolkit wheels.
        extra_cuda_cflags.append(f"-arch=sm_{major}{minor}")

    if platform.is_wsl2:
        # WSL2: parallel nvcc threads + lineinfo for profiling on Windows hosts.
        extra_cuda_cflags.extend(["--threads", "0", "-lineinfo"])

    return extra_cflags, extra_cuda_cflags


@lru_cache(maxsize=1)
def is_cuda_available() -> bool:
    """True when native CUDA extension loaded (NVIDIA GPUs only)."""
    if detect_gpu().vendor != GpuVendor.NVIDIA:
        return False
    return _load_extension() is not None


def _load_extension() -> Any | None:
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        return None
    if detect_gpu().vendor != GpuVendor.NVIDIA:
        _EXT_ERROR = "native CUDA kernels require NVIDIA GPU"
        return None
    if not _torch_cuda_available():
        _EXT_ERROR = "torch CUDA unavailable"
        return None

    try:
        from torch.utils.cpp_extension import load

        build_meta = _ensure_cuda_build_env()
        if not build_meta.get("cuda_home"):
            raise OSError(
                "CUDA toolkit (nvcc) not found. Install with: "
                "pip install 'cuda-toolkit[nvcc]==13.0.2' cuda-cccl ninja"
            )
        from seiso.kernels.cuda_env import cuda_link_flags

        cflags, cuda_cflags = _cuda_compile_flags()
        _EXT = load(
            name="seiso_cuda_kernels",
            sources=[
                str(_CUDA_DIR / "extension.cpp"),
                str(_CUDA_DIR / "rms_norm.cu"),
                str(_CUDA_DIR / "fused_swiglu.cu"),
                str(_CUDA_DIR / "fused_lora.cu"),
                str(_CUDA_DIR / "fused_lora_qkv.cu"),
                str(_CUDA_DIR / "fused_mlp.cu"),
                str(_CUDA_DIR / "fused_cross_entropy.cu"),
            ],
            extra_include_paths=_cuda_include_paths(),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=cuda_link_flags(),
            verbose=False,
        )
        plat = detect_gpu()
        target = "WSL2 CUDA" if plat.is_wsl2 else "native CUDA"
        logger.info("Seiso %s fused kernels loaded (SM %s)", target, plat.cuda_compute_capability)
        return _EXT
    except Exception as exc:  # noqa: BLE001
        _EXT_ERROR = str(exc)
        logger.warning("CUDA kernel load failed: %s", exc)
        return None


def cuda_kernel_status() -> dict[str, str | bool | None]:
    """Diagnostic info when native kernels fail to load."""
    from seiso.kernels.cuda_env import cuda_toolkit_status

    status = cuda_toolkit_status()
    status["extension_loaded"] = _EXT is not None
    status["extension_error"] = _EXT_ERROR
    return status


def set_kernel_tuning(
    rms_mode: int,
    swiglu_vec: int,
    lora_tile: int,
    *,
    arch_sm: int = 0,
    use_cuda_graphs: int = 0,
    use_stream_overlap: int = 1,
) -> None:
    """Apply RL-selected CUDA launch configuration to the native extension."""
    ext = _load_extension()
    if ext is None:
        return
    ext.set_kernel_tuning(
        int(rms_mode),
        int(swiglu_vec),
        int(lora_tile),
        int(arch_sm),
        int(use_cuda_graphs),
        int(use_stream_overlap),
    )


def fused_rms_norm(x, weight, eps: float = 1e-6, residual=None):
    """
    Stripe RMSNorm with optional fused residual — NVIDIA native path.

    Falls back to Triton then PyTorch when extension unavailable.
    """

    if not x.is_cuda:
        return _pytorch_rms_norm(x, weight, eps, residual)

    x2, orig_shape = _as_cuda_2d(x)
    res2 = residual
    if residual is not None and residual.dim() != 2:
        res2, _ = _as_cuda_2d(residual)

    ext = _load_extension()
    if ext is not None:
        out = ext.fused_rmsnorm(x2, weight, res2, eps)
        return out.reshape(orig_shape) if orig_shape is not None else out

    return _pytorch_rms_norm(x, weight, eps, residual)


def fused_swiglu(gate, up):
    """``silu(gate) * up`` with vectorized CUDA kernel."""
    import torch

    if not gate.is_cuda:
        return torch.nn.functional.silu(gate) * up

    gate2, orig_shape = _as_cuda_2d(gate)
    up2 = up if up.dim() == 2 else _as_cuda_2d(up)[0]

    ext = _load_extension()
    if ext is not None:
        out = ext.fused_swiglu(gate2, up2)
        return out.reshape(orig_shape) if orig_shape is not None else out
    return torch.nn.functional.silu(gate) * up


def fused_lora_delta(x, lora_A, lora_B, base=None, scale: float = 1.0, *, inplace: bool = False):
    """Fused low-rank delta: ``base + scale * B @ (A @ x)`` for 1D or 2D inputs."""

    if not x.is_cuda:
        hidden = x @ lora_A.t()
        delta = scale * (hidden @ lora_B.t())
        if base is None:
            return delta
        if inplace:
            base.add_(delta.to(base.dtype))
            return base
        return base + delta

    ext = _load_extension()
    rank = lora_A.size(0)
    if ext is not None and rank <= 64 and x.dim() in (1, 2):
        return ext.fused_lora_delta(x, lora_A, lora_B, base, scale, inplace)

    hidden = x @ lora_A.t()
    delta = scale * (hidden @ lora_B.t())
    if base is None:
        return delta
    if inplace:
        base.add_(delta.to(base.dtype))
        return base
    return base + delta


def cross_entropy_forward(logits, labels, ignore_index: int = -100):
    """Returns (row_loss, row_max, row_lse) float tensors."""
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA cross_entropy_forward requires native extension")
    return ext.cross_entropy_forward(logits, labels, ignore_index)


def cross_entropy_backward(
    logits, labels, row_max, row_lse, ignore_index: int = -100, grad_scale: float = 1.0
):
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA cross_entropy_backward requires native extension")
    return ext.cross_entropy_backward(logits, labels, row_max, row_lse, ignore_index, grad_scale)


def fused_lora_qkv_delta(
    x,
    out_q,
    out_k,
    out_v,
    lora_A_q,
    lora_B_q,
    lora_A_k,
    lora_B_k,
    lora_A_v,
    lora_B_v,
    *,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
):
    """In-place fused LoRA deltas for Q/K/V sharing one input read."""
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("fused_lora_qkv_delta requires native CUDA extension")
    return ext.fused_lora_qkv_delta(
        x,
        out_q,
        out_k,
        out_v,
        lora_A_q,
        lora_B_q,
        lora_A_k,
        lora_B_k,
        lora_A_v,
        lora_B_v,
        scale_q,
        scale_k,
        scale_v,
    )


def fused_mlp_swiglu(x, W_gate, W_up):
    """Fused gate/up projection + SwiGLU: silu(x @ W_gate^T) * (x @ W_up^T)."""
    ext = _load_extension()
    if ext is None:
        import torch

        gate = x @ W_gate.t()
        up = x @ W_up.t()
        return torch.nn.functional.silu(gate) * up
    return ext.fused_mlp_swiglu(x, W_gate, W_up)
