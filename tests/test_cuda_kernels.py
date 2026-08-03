"""Native CUDA kernel compile + smoke tests."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        "not __import__('torch').cuda.is_available()",
        reason="CUDA GPU required",
    ),
]


def test_cuda_toolkit_discovered():
    from seiso.kernels.cuda_env import cuda_toolkit_status

    status = cuda_toolkit_status()
    if not status.get("ready"):
        pytest.skip(f"nvcc/toolkit not discovered on this host: {status}")


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
        return out_q.clone(), out_k.clone(), out_v.clone()

    first = _run(static_x)
    static_x.copy_(torch.randn_like(static_x))
    second = _run(static_x)
    assert not torch.allclose(first[0], second[0], atol=1e-3)


def test_fused_cross_entropy_all_ignored_labels():
    """Batches with only -100 labels must not trip autograd mark_dirty."""
    torch = pytest.importorskip("torch")
    from seiso.kernels.loss import fused_cross_entropy_loss, shift_logits_and_labels

    logits = torch.randn(4, 128, device="cuda", requires_grad=True)
    labels = torch.full((4,), -100, device="cuda", dtype=torch.long)
    loss = fused_cross_entropy_loss(logits, labels)
    loss.backward()
    assert float(loss) == 0.0
    assert logits.grad is not None
    assert torch.all(logits.grad == 0)

    seq_logits = torch.randn(1, 32, 64, device="cuda", requires_grad=True)
    seq_labels = torch.full((1, 32), -100, device="cuda", dtype=torch.long)
    shift_logits, shift_labels = shift_logits_and_labels(seq_logits, seq_labels)
    loss2 = fused_cross_entropy_loss(shift_logits, shift_labels)
    loss2.backward()
    assert float(loss2) == 0.0


def test_fused_rms_norm_matches_pytorch():
    """Stripe RMSNorm must match reference (full-row sum of squares)."""
    torch = pytest.importorskip("torch")
    from seiso.kernels import cuda_ops as co
    from seiso.kernels.fallback_ops import pytorch_rms_norm

    co._EXT = None
    co._EXT_ERROR = None
    co.is_cuda_available.cache_clear()
    assert co.is_cuda_available(), co._EXT_ERROR or co.cuda_kernel_status()

    torch.manual_seed(0)
    x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    y = co.fused_rms_norm(x, w, eps=1e-6)
    ref = pytorch_rms_norm(x, w, 1e-6, None)
    assert torch.allclose(y.float(), ref.float(), atol=2e-2, rtol=2e-2)


def test_fused_cross_entropy_matches_pytorch():
    """Vectorized CE forward/backward must match F.cross_entropy."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from seiso.kernels import cuda_ops as co
    from seiso.kernels.loss import fused_cross_entropy_loss

    co._EXT = None
    co._EXT_ERROR = None
    co.is_cuda_available.cache_clear()
    assert co.is_cuda_available(), co._EXT_ERROR or co.cuda_kernel_status()

    torch.manual_seed(1)
    rows, vocab = 8, 320  # not a multiple of 8 — exercises scalar tail
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, vocab, (rows,), device="cuda")
    labels[0] = -100

    loss = fused_cross_entropy_loss(logits, labels, ignore_index=-100)
    ref = F.cross_entropy(logits, labels, ignore_index=-100)
    assert torch.allclose(loss, ref, atol=1e-5, rtol=1e-5)

    loss.backward()
    grad_fused = logits.grad.detach().clone()
    logits.grad = None
    ref.backward()
    assert torch.allclose(grad_fused, logits.grad, atol=1e-5, rtol=1e-5)


def test_fused_mlp_uses_gemm_epilogue_not_naive_default(monkeypatch):
    """Default MLP path is torch GEMM + fused_swiglu (never scalar CUDA matmul)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from seiso.kernels import cuda_ops as co
    from seiso.kernels.dispatch import fused_mlp_swiglu

    monkeypatch.delenv("SEISO_KERNEL_ALLOW_NAIVE_MLP", raising=False)
    co._EXT = None
    co._EXT_ERROR = None
    co.is_cuda_available.cache_clear()

    torch.manual_seed(2)
    rows, hin, hout = 16, 64, 128
    x = torch.randn(rows, hin, device="cuda", dtype=torch.bfloat16)
    wg = torch.randn(hout, hin, device="cuda", dtype=torch.bfloat16)
    wu = torch.randn(hout, hin, device="cuda", dtype=torch.bfloat16)

    y = fused_mlp_swiglu(x, wg, wu)
    ref = torch.nn.functional.silu(x @ wg.t()) * (x @ wu.t())
    assert y.shape == ref.shape
    assert torch.allclose(y.float(), ref.float(), atol=5e-2, rtol=5e-2)

    # Direct cuda_ops entrypoint must also default to GEMM+epilogue.
    y2 = co.fused_mlp_swiglu(x, wg, wu)
    assert torch.allclose(y2.float(), ref.float(), atol=5e-2, rtol=5e-2)


def test_lora_defaults_to_cublas_path(monkeypatch):
    """Production LoRA must not use the serial A@x CUDA kernel by default."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from seiso.kernels import dispatch as d

    monkeypatch.delenv("SEISO_KERNEL_ALLOW_NAIVE_LORA", raising=False)
    x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    assert d._prefer_cublas_lora(x) is True

    # Opt-in naive only for tiny no-grad microbenches.
    monkeypatch.setenv("SEISO_KERNEL_ALLOW_NAIVE_LORA", "1")
    tiny = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        assert d._prefer_cublas_lora(tiny) is False
        assert d._prefer_cublas_lora(x) is True
    # Grad mode always forces cuBLAS even with the opt-in env.
    assert d._prefer_cublas_lora(tiny) is True


def test_stacked_mlp_matches_two_gemm_reference():
    """Stacked cat(W_gate,W_up) GEMM must match separate gate/up matmuls."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from seiso.kernels.dispatch import fused_mlp_swiglu

    torch.manual_seed(3)
    rows, hin, hout = 32, 128, 256
    # float32: stacked vs dual GEMM can diverge in bf16 accumulation order.
    x = torch.randn(rows, hin, device="cuda", dtype=torch.float32)
    wg = torch.randn(hout, hin, device="cuda", dtype=torch.float32)
    wu = torch.randn(hout, hin, device="cuda", dtype=torch.float32)
    y = fused_mlp_swiglu(x, wg, wu)
    ref = torch.nn.functional.silu(x @ wg.t()) * (x @ wu.t())
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


def test_lora_qkv_batched_b_matches_independent():
    """Equal-rank equal-scale QKV should match independent LoRA pairs."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from seiso.kernels.dispatch import fused_lora_qkv_delta

    torch.manual_seed(4)
    rows, in_dim, out_dim, rank = 16, 64, 48, 8
    dtype = torch.bfloat16
    x = torch.randn(rows, in_dim, device="cuda", dtype=dtype)
    a = torch.randn(rank, in_dim, device="cuda", dtype=dtype) * 0.02
    b = torch.randn(out_dim, rank, device="cuda", dtype=dtype) * 0.02
    # Shared A/B shapes (independent weights with same layout).
    a_q, a_k, a_v = a.clone(), a.clone() + 0.001, a.clone() - 0.001
    b_q, b_k, b_v = b.clone(), b.clone() + 0.001, b.clone() - 0.001

    out_q = torch.zeros(rows, out_dim, device="cuda", dtype=dtype)
    out_k = torch.zeros_like(out_q)
    out_v = torch.zeros_like(out_q)
    fused_lora_qkv_delta(
        x, out_q, out_k, out_v, a_q, b_q, a_k, b_k, a_v, b_v, scale_q=0.5, scale_k=0.5, scale_v=0.5
    )

    def _ref(out, aa, bb, scale):
        h = x @ aa.t()
        out.add_((scale * (h @ bb.t())).to(out.dtype))

    rq = torch.zeros_like(out_q)
    rk = torch.zeros_like(out_k)
    rv = torch.zeros_like(out_v)
    _ref(rq, a_q, b_q, 0.5)
    _ref(rk, a_k, b_k, 0.5)
    _ref(rv, a_v, b_v, 0.5)
    assert torch.allclose(out_q, rq, atol=5e-2, rtol=5e-2)
    assert torch.allclose(out_k, rk, atol=5e-2, rtol=5e-2)
    assert torch.allclose(out_v, rv, atol=5e-2, rtol=5e-2)


def test_attention_resolve_never_empty():
    from seiso.kernels.attention import attention_metadata, resolve_attention_implementation

    impl = resolve_attention_implementation()
    assert impl in {
        "flash_attention_3",
        "flash_attention_2",
        "sdpa",
        "eager",
    }
    meta = attention_metadata()
    assert meta["attn_implementation"] == impl

def test_restore_registry_keeps_modules_on_failure(monkeypatch):
    from seiso.kernels import lifecycle as life

    class Mod:
        pass

    m1, m2 = Mod(), Mod()
    m1._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m2._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m1.forward = lambda x: 1  # type: ignore[attr-defined]
    m2.forward = lambda x: 2  # type: ignore[attr-defined]

    orig_clear = life._clear_patch_markers

    def flaky(module: object) -> None:
        if module is m2:
            raise RuntimeError("restore boom")
        orig_clear(module)

    monkeypatch.setattr(life, "_clear_patch_markers", flaky)
    life._PATCH_REGISTRY.clear()
    life._PATCH_REGISTRY[42] = [m1, m2]
    with pytest.raises(RuntimeError, match="restore boom"):
        life._restore_registry_key(42)
    assert life._PATCH_REGISTRY[42] == [m2]
    life._PATCH_REGISTRY.clear()

def test_patch_session_keeps_modules_on_restore_failure(monkeypatch):
    from seiso.kernels.lifecycle import KernelPatchSession, _clear_patch_markers

    class Mod:
        pass

    m1, m2 = Mod(), Mod()
    m1._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m2._seiso_orig_forward = lambda x: x  # type: ignore[attr-defined]
    m1.forward = lambda x: 1  # type: ignore[attr-defined]
    m2.forward = lambda x: 2  # type: ignore[attr-defined]

    orig_clear = _clear_patch_markers

    def flaky(module: object) -> None:
        if module is m1:
            raise RuntimeError("session restore boom")
        orig_clear(module)

    session = KernelPatchSession()
    session.record(m1)
    session.record(m2)
    monkeypatch.setattr("seiso.kernels.lifecycle._clear_patch_markers", flaky)
    with pytest.raises(RuntimeError, match="session restore boom"):
        session.restore()
    # LIFO: m2 restored first; failure on m1 keeps m1 for retry.
    assert session._modules == [m1]
    monkeypatch.setattr("seiso.kernels.lifecycle._clear_patch_markers", orig_clear)
    assert session.restore() == 1
    assert session._modules == []

