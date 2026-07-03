"""Tests for torch.compile + gradient-checkpointing workarounds."""

from __future__ import annotations

from seiso.training.torch_dynamo import (
    apply_compile_checkpoint_workarounds,
    configure_compile_checkpoint_compat,
    needs_compile_checkpoint_workaround,
    wrap_model_forward_for_dynamic_shapes,
)


def test_needs_compile_checkpoint_workaround():
    assert needs_compile_checkpoint_workaround(
        torch_compile=True, gradient_checkpointing=True
    )
    assert not needs_compile_checkpoint_workaround(
        torch_compile=True, gradient_checkpointing=False
    )
    assert not needs_compile_checkpoint_workaround(
        torch_compile=False, gradient_checkpointing=True
    )


def test_configure_compile_checkpoint_compat_disables_lru_cache(monkeypatch):
    import torch

    import seiso.training.torch_dynamo as mod

    monkeypatch.setattr(mod, "_LRU_CACHE_CONFIGURED", False)
    calls: list[bool] = []
    monkeypatch.setattr(
        torch._C._dynamo.eval_frame,
        "_set_lru_cache",
        lambda value: calls.append(value),
    )
    monkeypatch.setattr(
        mod,
        "env_bool",
        lambda _name, default: default,
    )

    applied = configure_compile_checkpoint_compat(
        torch_compile=True,
        gradient_checkpointing=True,
    )
    assert applied is True
    assert calls == [False]


def test_configure_compile_checkpoint_compat_noop_when_disabled(monkeypatch):
    import torch

    import seiso.training.torch_dynamo as mod

    monkeypatch.setattr(mod, "_LRU_CACHE_CONFIGURED", False)
    calls: list[bool] = []
    monkeypatch.setattr(
        torch._C._dynamo.eval_frame,
        "_set_lru_cache",
        lambda value: calls.append(value),
    )
    monkeypatch.setattr(
        mod,
        "env_bool",
        lambda name, default: False if name == "SEISO_TORCH_COMPILE_CHECKPOINT_FIX" else default,
    )

    applied = configure_compile_checkpoint_compat(
        torch_compile=True,
        gradient_checkpointing=True,
    )
    assert applied is False
    assert calls == []


def test_wrap_model_forward_for_dynamic_shapes_marks_tensors(monkeypatch):
    import torch

    marked: list[tuple[int, int]] = []

    def fake_mark_dynamic(tensor, index, **kwargs):
        marked.append((tuple(tensor.shape), index))

    monkeypatch.setattr("torch._dynamo.mark_dynamic", fake_mark_dynamic)

    class _Model:
        def forward(self, input_ids, attention_mask=None):
            return input_ids

    model = wrap_model_forward_for_dynamic_shapes(_Model())
    model.forward(
        torch.zeros(2, 8, dtype=torch.long),
        attention_mask=torch.ones(2, 8, dtype=torch.long),
    )
    assert (2, 8) in [shape for shape, _ in marked]
    assert 0 in [idx for _, idx in marked]
    assert 1 in [idx for _, idx in marked]


def test_apply_compile_checkpoint_workarounds_idempotent(monkeypatch):
    import seiso.training.torch_dynamo as mod

    monkeypatch.setattr(mod, "_LRU_CACHE_CONFIGURED", False)
    monkeypatch.setattr(
        mod,
        "configure_compile_checkpoint_compat",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "seiso.training.torch_dynamo.env_bool",
        lambda name, default: False if name == "SEISO_TORCH_COMPILE_MARK_DYNAMIC" else default,
    )

    class _Model:
        def forward(self, x):
            return x

    model = _Model()
    out = apply_compile_checkpoint_workarounds(
        model,
        torch_compile=True,
        gradient_checkpointing=True,
    )
    assert out is model
    assert getattr(model, "_seiso_dynamic_shape_forward", False) is False