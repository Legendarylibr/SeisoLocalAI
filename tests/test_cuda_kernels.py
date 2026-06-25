"""Native CUDA kernel compile + smoke tests."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()",
    reason="CUDA GPU required",
)


def test_cuda_toolkit_discovered():
    from seiso.kernels.cuda_env import cuda_toolkit_status

    status = cuda_toolkit_status()
    assert status["ready"], f"nvcc not found: {status}"


def test_cuda_kernels_compile_and_run():
    torch = pytest.importorskip("torch")
    from seiso.kernels import cuda_ops as co

    co._EXT = None
    co._EXT_ERROR = None
    co.is_cuda_available.cache_clear()

    assert co.is_cuda_available(), co._EXT_ERROR or co.cuda_kernel_status()

    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    y = co.fused_rms_norm(x, w)
    assert y.shape == x.shape

    gate = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    s = co.fused_swiglu(gate, up)
    assert s.shape == gate.shape

    from seiso.kernels.dispatch import active_backend

    assert active_backend() == "cuda"


def test_fused_lora_qkv_matches_reference():
    torch = pytest.importorskip("torch")
    from seiso.kernels import cuda_ops as co
    from seiso.kernels.dispatch import fused_lora_qkv_delta

    co._EXT = None
    co._EXT_ERROR = None
    co.is_cuda_available.cache_clear()
    assert co.is_cuda_available(), co._EXT_ERROR or co.cuda_kernel_status()

    rows, in_dim, out_dim, rank = 4, 64, 48, 8
    dtype = torch.bfloat16
    x = torch.randn(rows, in_dim, device="cuda", dtype=dtype)
    a_q = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_q = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    a_k = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_k = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    a_v = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_v = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02

    out_q = torch.randn(rows, out_dim, device="cuda", dtype=dtype)
    out_k = torch.randn(rows, out_dim, device="cuda", dtype=dtype)
    out_v = torch.randn(rows, out_dim, device="cuda", dtype=dtype)
    ref_q = out_q.clone()
    ref_k = out_k.clone()
    ref_v = out_v.clone()

    fused_lora_qkv_delta(
        x,
        out_q,
        out_k,
        out_v,
        a_q,
        b_q,
        a_k,
        b_k,
        a_v,
        b_v,
        scale_q=0.5,
        scale_k=0.25,
        scale_v=0.125,
    )

    for ref, a, b, scale in (
        (ref_q, a_q, b_q, 0.5),
        (ref_k, a_k, b_k, 0.25),
        (ref_v, a_v, b_v, 0.125),
    ):
        hidden = x @ a.t()
        ref.add_((scale * (hidden @ b.t())).to(ref.dtype))

    assert torch.allclose(out_q, ref_q, atol=5e-2, rtol=5e-2)
    assert torch.allclose(out_k, ref_k, atol=5e-2, rtol=5e-2)
    assert torch.allclose(out_v, ref_v, atol=5e-2, rtol=5e-2)


def test_fused_lora_qkv_cache_invalidates_after_inplace_copy():
    torch = pytest.importorskip("torch")
    from seiso.kernels import cuda_ops as co
    from seiso.kernels.dispatch import fused_lora_qkv_delta

    co._EXT = None
    co._EXT_ERROR = None
    co.is_cuda_available.cache_clear()
    assert co.is_cuda_available(), co._EXT_ERROR or co.cuda_kernel_status()

    rows, in_dim, out_dim, rank = 2, 32, 24, 4
    dtype = torch.bfloat16
    static_x = torch.randn(rows, in_dim, device="cuda", dtype=dtype)
    a_q = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_q = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    a_k = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_k = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    a_v = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_v = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02

    def _run(x):
        out_q = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
        out_k = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
        out_v = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
        fused_lora_qkv_delta(
            x, out_q, out_k, out_v, a_q, b_q, a_k, b_k, a_v, b_v,
        )
        return out_q.clone(), out_k.clone(), out_v.clone()

    first = _run(static_x)
    static_x.copy_(torch.randn_like(static_x))
    second = _run(static_x)
    assert not torch.allclose(first[0], second[0], atol=1e-3)