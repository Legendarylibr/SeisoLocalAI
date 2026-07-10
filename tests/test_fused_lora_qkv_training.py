"""QKV fusion must stay active during training (requires_grad=True)."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not __import__('torch').cuda.is_available()",
        reason="CUDA required",
    ),
]


def test_lora_qkv_uses_cublas_during_training():
    torch = pytest.importorskip("torch")
    from seiso.kernels.dispatch import fused_lora_qkv_delta

    rows, in_dim, out_dim, rank = 128, 1024, 1024, 8
    dtype = torch.bfloat16
    x = torch.randn(rows, in_dim, device="cuda", dtype=dtype, requires_grad=True)
    a_q = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_q = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    a_k = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_k = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    a_v = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b_v = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02

    out_q = torch.randn(rows, out_dim, device="cuda", dtype=dtype)
    out_k = torch.randn(rows, out_dim, device="cuda", dtype=dtype)
    out_v = torch.randn(rows, out_dim, device="cuda", dtype=dtype)
    ref_q, ref_k, ref_v = out_q.clone(), out_k.clone(), out_v.clone()

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
    )

    for ref, a, b in ((ref_q, a_q, b_q), (ref_k, a_k, b_k), (ref_v, a_v, b_v)):
        ref.add_((x @ a.t()) @ b.t())

    torch.testing.assert_close(out_q, ref_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_v, ref_v, rtol=1e-2, atol=1e-2)
    assert x.requires_grad


def test_stacked_a_matmul_matches_separate():
    torch = pytest.importorskip("torch")
    from seiso.kernels.dispatch import _fused_lora_qkv_delta_torch

    rows, in_dim, out_dim, rank = 64, 512, 512, 16
    dtype = torch.float32
    x = torch.randn(rows, in_dim, device="cuda", dtype=dtype)
    mats = [
        torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.01 for _ in range(3)
    ]
    bs = [
        torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.01 for _ in range(3)
    ]
    out_q = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
    out_k = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
    out_v = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
    ref_q, ref_k, ref_v = out_q.clone(), out_k.clone(), out_v.clone()

    _fused_lora_qkv_delta_torch(
        x,
        out_q,
        out_k,
        out_v,
        mats[0],
        bs[0],
        mats[1],
        bs[1],
        mats[2],
        bs[2],
    )

    for ref, a, b in zip((ref_q, ref_k, ref_v), mats, bs, strict=True):
        ref.add_(x @ a.t() @ b.t())

    torch.testing.assert_close(out_q, ref_q, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_v, ref_v, rtol=1e-4, atol=1e-4)


def test_gqa_stacked_kv_a_matmul_matches_separate():
    torch = pytest.importorskip("torch")
    from seiso.kernels.dispatch import _fused_lora_qkv_delta_torch

    rows, in_dim, out_q, out_kv, rank = 64, 512, 512, 128, 16
    dtype = torch.float32
    x = torch.randn(rows, in_dim, device="cuda", dtype=dtype, requires_grad=True)
    a_q = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.01
    b_q = torch.randn(out_q, rank, device="cuda", dtype=dtype) * 0.01
    a_k = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.01
    b_k = torch.randn(out_kv, rank, device="cuda", dtype=dtype) * 0.01
    a_v = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.01
    b_v = torch.randn(out_kv, rank, device="cuda", dtype=dtype) * 0.01
    out_q = torch.zeros(rows, out_q, device="cuda", dtype=dtype)
    out_k = torch.zeros(rows, out_kv, device="cuda", dtype=dtype)
    out_v = torch.zeros(rows, out_kv, device="cuda", dtype=dtype)
    ref_q, ref_k, ref_v = out_q.clone(), out_k.clone(), out_v.clone()

    _fused_lora_qkv_delta_torch(
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
    )

    ref_q.add_(x @ a_q.t() @ b_q.t())
    ref_k.add_(x @ a_k.t() @ b_k.t())
    ref_v.add_(x @ a_v.t() @ b_v.t())

    torch.testing.assert_close(out_q, ref_q, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_v, ref_v, rtol=1e-4, atol=1e-4)
    assert x.requires_grad


def test_gqa_batched_base_qkv_forward():
    torch = pytest.importorskip("torch")
    from seiso.kernels.hooks import _batched_base_qkv_forward

    rows, in_dim, out_q, out_kv = 32, 256, 512, 128
    dtype = torch.float32
    flat_x = torch.randn(rows, in_dim, device="cuda", dtype=dtype)

    class _Linear:
        def __init__(self, out_features: int):
            self.weight = torch.randn(out_features, in_dim, device="cuda", dtype=dtype)
            self.bias = None

        def __call__(self, x):
            return x @ self.weight.t()

    class _Proj:
        def __init__(self, out_features: int):
            self.base_layer = _Linear(out_features)

    q_proj = _Proj(out_q)
    k_proj = _Proj(out_kv)
    v_proj = _Proj(out_kv)

    out_q, out_k, out_v = _batched_base_qkv_forward(flat_x, q_proj, k_proj, v_proj)
    ref_q = q_proj.base_layer(flat_x)
    ref_k = k_proj.base_layer(flat_x)
    ref_v = v_proj.base_layer(flat_x)

    torch.testing.assert_close(out_q, ref_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_v, ref_v, rtol=1e-2, atol=1e-2)


def test_fused_swiglu_preserves_autograd_during_training():
    torch = pytest.importorskip("torch")
    from seiso.kernels.dispatch import fused_swiglu

    gate = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = fused_swiglu(gate, up)
    loss = out.sum()
    loss.backward()
    assert gate.grad is not None
    assert up.grad is not None
