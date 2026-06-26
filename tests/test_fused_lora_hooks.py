"""Unit tests for fused LoRA QKV hooks and residual-norm wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from seiso.kernels.hooks import (
    _DECODER_LAYER_CLASSES,
    _FUSED_RESIDUAL_DECODER_CLASSES,
    apply_fused_residual_norm_kernels,
)


def test_residual_patches_only_fused_decoder_classes(monkeypatch):
    monkeypatch.setattr("seiso.kernels.hooks.active_backend", lambda: "cuda")

    patched_norms: list[str] = []
    patched_decoders: list[str] = []

    def _fake_norm_patch(model, decoder):
        patched_norms.append(type(decoder).__name__)
        return True

    def _fake_decoder_patch(model, decoder):
        patched_decoders.append(type(decoder).__name__)
        return True

    monkeypatch.setattr("seiso.kernels.hooks._patch_post_attention_residual_norm", _fake_norm_patch)
    monkeypatch.setattr("seiso.kernels.hooks._patch_fused_residual_decoder_forward", _fake_decoder_patch)

    class LlamaDecoderLayer:
        input_layernorm = object()
        post_attention_layernorm = object()
        self_attn = object()
        mlp = object()

    class Gemma2DecoderLayer:
        input_layernorm = object()
        post_attention_layernorm = object()
        self_attn = object()
        mlp = object()

    model = SimpleNamespace(
        named_modules=lambda: iter(
            [
                ("layers.0", LlamaDecoderLayer()),
                ("layers.1", Gemma2DecoderLayer()),
            ]
        )
    )

    meta = apply_fused_residual_norm_kernels(model)
    assert patched_norms == ["LlamaDecoderLayer"]
    assert patched_decoders == ["LlamaDecoderLayer"]
    assert meta["fused_residual_norm_patched"] == 1
    assert meta["fused_residual_decoder_patched"] == 1
    assert "Gemma2DecoderLayer" in _DECODER_LAYER_CLASSES
    assert "Gemma2DecoderLayer" not in _FUSED_RESIDUAL_DECODER_CLASSES


@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()",
    reason="CUDA required",
)
def test_residual_fusion_matches_llama_decoder_semantics():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaModel

    from seiso.kernels.hooks import apply_fused_residual_norm_kernels, apply_training_kernels

    config = LlamaConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        vocab_size=100,
    )
    from seiso.kernels.dispatch import active_backend

    model = LlamaModel(config).cuda().eval()
    apply_training_kernels(model, use_cuda=True, use_triton=True, patch_mlp=False)
    if active_backend() != "cuda":
        pytest.skip("native CUDA kernels unavailable")

    meta = apply_fused_residual_norm_kernels(model)
    assert meta["fused_residual_decoder_patched"] == 1
    assert meta["fused_residual_norm_patched"] == 1

    layer = model.layers[0]
    x = torch.randn(2, 8, 64, device="cuda")
    position_ids = torch.arange(8, device="cuda").unsqueeze(0)
    position_embeddings = model.rotary_emb(x, position_ids)

    with torch.no_grad():
        out_patched = layer(
            x,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
        )

    residual = x
    h = layer.input_layernorm(residual)
    h, _ = layer.self_attn(
        h,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
    )
    post_skip = residual + h
    h = layer.post_attention_layernorm._seiso_orig_forward(post_skip)
    h = layer.mlp(h)
    ref = post_skip + h

    assert torch.allclose(out_patched, ref, rtol=0.05, atol=0.2), (
        f"residual fusion mismatch: max diff {(out_patched - ref).abs().max().item()}"
    )