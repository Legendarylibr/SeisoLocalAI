from seiso.kernels.platform import GpuVendor, detect_gpu
from seiso.kernels.triton_ops import estimate_vram_savings_pct, is_triton_available


def test_triton_availability():
    assert isinstance(is_triton_available(), bool)


def test_platform_detection():
    platform = detect_gpu()
    assert platform.vendor in (GpuVendor.NVIDIA, GpuVendor.AMD, GpuVendor.CPU)
    assert isinstance(platform.device_count, int)


def test_vram_estimate_triton():
    assert estimate_vram_savings_pct(True, True) >= 60
    assert estimate_vram_savings_pct(False, False) == 0


def test_dispatch_and_patch_restore():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from torch import nn

    from seiso.kernels.dispatch import active_backend, fused_rms_norm, kernel_metadata
    from seiso.kernels.hooks import apply_training_kernels, clear_kernel_patches
    from seiso.kernels.lifecycle import restore_kernel_patches

    meta = kernel_metadata()
    assert meta["kernel_backend"] in ("cuda", "triton", "pytorch")
    assert active_backend() in ("cuda", "triton", "pytorch")

    x = torch.randn(4, 32)
    w = torch.ones(32)
    assert fused_rms_norm(x, w).shape == x.shape

    class FakeRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.variance_epsilon = 1e-6

        def forward(self, hidden):
            return hidden * self.weight

    model = nn.Sequential(FakeRMSNorm(16))
    model[0].__class__.__name__ = "LlamaRMSNorm"

    patch_meta = apply_training_kernels(model)
    assert patch_meta["rmsnorm_patched"] == 1
    assert hasattr(model[0], "_seiso_orig_forward")

    class FakeMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gate_proj = nn.Linear(dim, dim, bias=False)
            self.up_proj = nn.Linear(dim, dim, bias=False)
            self.down_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    mlp_model = nn.Sequential(FakeMLP(16))
    mlp_model[0].__class__.__name__ = "LlamaMLP"
    mlp_meta = apply_training_kernels(mlp_model)
    assert mlp_meta["mlp_patched"] == 1

    clear_kernel_patches(model)
    assert not hasattr(model[0], "_seiso_orig_forward")
    assert restore_kernel_patches(model) == 0
