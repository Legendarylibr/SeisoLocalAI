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
    monkeypatch.setattr(
        "seiso.kernels.hooks._patch_fused_residual_decoder_forward", _fake_decoder_patch
    )

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
    # Gemma norm/MLP contracts differ from Llama-style fused kernels; the
    # class must be excluded from both patch sets entirely.
    assert "Gemma2DecoderLayer" not in _DECODER_LAYER_CLASSES
    assert "Gemma2DecoderLayer" not in _FUSED_RESIDUAL_DECODER_CLASSES


@pytest.mark.gpu
@pytest.mark.skipif(
    "not __import__('torch').cuda.is_available()",
    reason="CUDA required",
)
def test_residual_fusion_matches_llama_decoder_semantics():
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaModel

    from seiso.kernels.hooks import (
        apply_fused_residual_norm_kernels,
        apply_training_kernels,
    )

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
    if isinstance(out_patched, tuple):
        # Patched decoder mirrors the HF contract: extras follow hidden_states.
        out_patched = out_patched[0]

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


def test_decoder_forward_preserves_cache_and_attentions():
    """#3: patched decoder forward must not drop use_cache/output_attentions extras."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from seiso.kernels.hooks import _patch_fused_residual_decoder_forward
    from seiso.kernels.lifecycle import restore_kernel_patches

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen: dict = {}

        def forward(self, hidden_states, output_attentions=False, use_cache=False, **kwargs):
            self.seen = {
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                **kwargs,
            }
            out = hidden_states * 2
            extras = []
            if output_attentions:
                extras.append("attn-weights")
            if use_cache:
                extras.append("past-kv")
            return (out, *extras) if extras else out

    class LlamaDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()
            self.self_attn = FakeAttention()
            self.mlp = nn.Identity()

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    model = nn.Module()
    decoder = LlamaDecoderLayer()
    try:
        assert _patch_fused_residual_decoder_forward(model, decoder) is True
        x = torch.randn(2, 3, 4)

        bare = decoder(x)
        assert isinstance(bare, torch.Tensor)

        out = decoder(x, output_attentions=True, use_cache=True, past_key_values="old-kv")
        assert isinstance(out, tuple)
        assert torch.equal(out[0], bare)
        # Cache and attention outputs survive the patched forward.
        assert out[1] == "attn-weights"
        assert out[2] == "past-kv"
        # Flags are still forwarded to the attention module.
        assert decoder.self_attn.seen["past_key_values"] == "old-kv"
        assert decoder.self_attn.seen["output_attentions"] is True
        assert decoder.self_attn.seen["use_cache"] is True
    finally:
        restore_kernel_patches()


def test_norm_patch_rollback_restores_entry_active_forward(monkeypatch):
    """#9: failed re-patch must keep the previously installed patch forward active."""
    from seiso.kernels.hooks import _patch_post_attention_residual_norm
    from seiso.kernels.lifecycle import restore_kernel_patches

    def _original(self, hidden_states):
        return hidden_states

    def _previous_patch(self, hidden_states):
        return hidden_states

    # Simulate an earlier kernel patch: _seiso_orig_forward pre-exists and the
    # entry-active forward is the previously installed patch.
    norm = SimpleNamespace()
    norm.forward = _previous_patch
    norm._seiso_orig_forward = _original
    decoder = SimpleNamespace(post_attention_layernorm=norm)
    model = SimpleNamespace()

    def _boom(_model, _module):
        raise RuntimeError("register failed")

    monkeypatch.setattr("seiso.kernels.hooks.register_patch", _boom)
    try:
        with pytest.raises(RuntimeError, match="register failed"):
            _patch_post_attention_residual_norm(model, decoder)
        assert norm.forward is _previous_patch
        assert norm._seiso_orig_forward is _original
        assert not hasattr(norm, "_seiso_residual_norm_forward")
    finally:
        restore_kernel_patches()


def test_decoder_patch_rollback_restores_entry_active_forward(monkeypatch):
    """#9: failed re-patch must keep the previously installed patch forward active."""
    from seiso.kernels.hooks import _patch_fused_residual_decoder_forward
    from seiso.kernels.lifecycle import restore_kernel_patches

    def _original(self, hidden_states, **kwargs):
        return hidden_states

    def _previous_patch(self, hidden_states, **kwargs):
        return hidden_states

    decoder = SimpleNamespace()
    decoder.forward = _previous_patch
    decoder._seiso_orig_forward = _original
    model = SimpleNamespace()

    def _boom(_model, _module):
        raise RuntimeError("register failed")

    monkeypatch.setattr("seiso.kernels.hooks.register_patch", _boom)
    try:
        with pytest.raises(RuntimeError, match="register failed"):
            _patch_fused_residual_decoder_forward(model, decoder)
        assert decoder.forward is _previous_patch
        assert decoder._seiso_orig_forward is _original
        assert not hasattr(decoder, "_seiso_residual_decoder_forward")
    finally:
        restore_kernel_patches()
