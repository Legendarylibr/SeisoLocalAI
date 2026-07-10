"""Inference fused-kernel guards for quantized torch models."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_mlp_forward_skips_fused_mlp_for_bitsandbytes_layers(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused MLP inference kernel coverage")
    from torch import nn

    from seiso.kernels.hooks import apply_training_kernels

    class Params4bit:
        quant_state = object()

    class Linear4bit(nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.weight = Params4bit()
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x):
            return torch.nn.functional.linear(
                x,
                torch.ones(self.out_features, self.in_features, device=x.device),
            )

    class Qwen2MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = Linear4bit(8, 16)
            self.up_proj = Linear4bit(8, 16)
            self.down_proj = nn.Linear(16, 8)
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states):
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
            return self.down_proj(self.act_fn(gate) * up)

    calls: list[str] = []

    def _fake_fused_mlp(*_args, **_kwargs):
        calls.append("fused_mlp")
        return torch.zeros(1, 16, device="cuda")

    def _fake_fused_swiglu(gate, up):
        calls.append("fused_swiglu")
        return torch.nn.functional.silu(gate) * up

    monkeypatch.setattr("seiso.kernels.hooks.active_backend", lambda: "cuda")
    monkeypatch.setattr("seiso.kernels.hooks._use_fused_cuda_kernels", lambda _x: True)
    monkeypatch.setattr("seiso.kernels.hooks.fused_mlp_swiglu", _fake_fused_mlp)
    monkeypatch.setattr("seiso.kernels.hooks.fused_swiglu", _fake_fused_swiglu)

    model = nn.Module()
    model.layers = nn.ModuleList([Qwen2MLP().cuda()])

    apply_training_kernels(model, use_cuda=True, use_triton=False, patch_mlp=True)

    x = torch.randn(2, 3, 8, device="cuda")
    with torch.inference_mode():
        model.layers[0](x)

    assert calls == ["fused_swiglu"]
    assert "fused_mlp" not in calls
