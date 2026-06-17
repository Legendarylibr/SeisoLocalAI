from seiso.kernels.triton_ops import estimate_vram_savings_pct, is_triton_available


def test_triton_availability():
    # Should not raise regardless of CUDA/triton install
    assert isinstance(is_triton_available(), bool)


def test_vram_estimate():
    assert estimate_vram_savings_pct(True, True) >= 60
    assert estimate_vram_savings_pct(False, False) == 0
