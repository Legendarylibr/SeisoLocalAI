"""Architecture-aware kernel tuning tests."""

from seiso.kernels.arch_tuning import (
    ArchTuningProfile,
    GpuArchFamily,
    arch_family_from_capability,
    detect_arch_tuning,
)
from seiso.kernels.attention import resolve_attention_implementation


def test_arch_family_from_capability():
    assert arch_family_from_capability(8, 9) == GpuArchFamily.ADA
    assert arch_family_from_capability(8, 0) == GpuArchFamily.AMPERE
    assert arch_family_from_capability(9, 0) == GpuArchFamily.HOPPER
    assert arch_family_from_capability(10, 0) == GpuArchFamily.BLACKWELL


def test_detect_arch_tuning_returns_profile():
    profile = detect_arch_tuning()
    assert isinstance(profile, ArchTuningProfile)
    assert profile.swiglu_vec in (4, 8)
    assert profile.lora_tile in (128, 256, 384, 512)


def test_resolve_attention_implementation():
    impl = resolve_attention_implementation()
    assert impl in ("flash_attention_3", "flash_attention_2", "sdpa")
