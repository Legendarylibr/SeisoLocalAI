"""NVIDIA GPU safety invariants — prevent VRAM overcommit that crashes drivers/cards.

These tests use mocked hardware/files and verify sizing math + load ladders stay
within conservative budgets. No real GPU required.
"""

from __future__ import annotations

import os
import platform
from itertools import product

import pytest

from seiso.memory.protection import (
    build_hf_max_memory,
    clamp_llama_load_kwargs,
    llama_batch_limits_for_headroom,
    llama_effective_batch_headroom_mb,
    llama_load_profile_ladder,
    llama_model_is_tight_vram_fit,
    llama_offload_fits_headroom,
)

# Realistic NVIDIA chat scenarios: (label, free_vram_mb, weight_mb, params_b, n_ctx)
_NVIDIA_CHAT_SCENARIOS = [
    ("4090_27b_q4_comfortable", 24576, 17000, 27, 4096),
    ("4090_27b_q4_tight", 19000, 17000, 27, 4096),
    ("4090_27b_q4_low_vram", 12000, 17000, 27, 4096),
    ("4090_13b_q4", 24576, 8500, 13, 4096),
    ("4090_gemma_14b_q4", 24576, 9000, 14, 4096),
    ("3070_7b_q4", 8192, 4500, 7, 4096),
    ("3060_12gb_7b", 12288, 4500, 7, 4096),
    ("a6000_48gb_70b_partial", 49152, 42000, 70, 4096),
    ("5080_16gb_13b", 16384, 8500, 13, 4096),
    ("4090_70b_q4_impossible", 24576, 42000, 70, 4096),
    ("4090_27b_q4_long_ctx", 24576, 17000, 27, 8192),
]

# Swept grid for cross-hardware simulation (no real GPU).
_HEADROOM_GRID_MB = (4096, 8192, 12288, 16384, 24576, 49152)
_WEIGHT_GRID_MB = (2000, 4500, 8500, 17000, 32000, 42000)
_CTX_GRID = (2048, 4096, 8192)


def _assert_layers_fit_vram(
    *,
    headroom_mb: int,
    weight_mb: int,
    n_ctx: int,
    n_gpu_layers: int,
    model_path: str,
    total_layers: int = 64,
) -> None:
    """Full offload claim must leave room for weights + KV within headroom."""
    assert llama_offload_fits_headroom(
        model_path,
        headroom_mb=headroom_mb,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        weight_mb=weight_mb,
        total_layers=total_layers,
    ), (
        f"layers={n_gpu_layers} needs more than {headroom_mb} MB headroom "
        f"(weight={weight_mb}, n_ctx={n_ctx})"
    )


@pytest.fixture
def gguf_path(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"gguf")
    return path


@pytest.mark.parametrize(
    "label,headroom_mb,weight_mb,params_b,n_ctx",
    _NVIDIA_CHAT_SCENARIOS,
    ids=[s[0] for s in _NVIDIA_CHAT_SCENARIOS],
)
def test_fit_llama_gpu_layers_never_claims_full_offload_without_budget(
    monkeypatch, gguf_path, label, headroom_mb, weight_mb, params_b, n_ctx
):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_total_layers", lambda _p: 64
    )

    layers = mp.fit_llama_gpu_layers(
        str(gguf_path), -1, headroom_mb, n_ctx=n_ctx
    )

    if layers == -1:
        _assert_layers_fit_vram(
            headroom_mb=headroom_mb,
            weight_mb=weight_mb,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            model_path=str(gguf_path),
        )
    elif layers > 0:
        _assert_layers_fit_vram(
            headroom_mb=headroom_mb,
            weight_mb=weight_mb,
            n_ctx=n_ctx,
            n_gpu_layers=layers,
            model_path=str(gguf_path),
        )
    else:
        assert layers == 0


@pytest.mark.parametrize(
    "headroom_mb,weight_mb,n_ctx",
    [
        (12288, 8500, 4096),
        (16384, 12000, 4096),
        (24576, 17000, 4096),
    ],
)
def test_native_linux_nvidia_keeps_slack_before_near_capacity_full_offload(
    monkeypatch, gguf_path, headroom_mb, weight_mb, n_ctx
):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_total_layers", lambda _p: 64
    )

    layers = mp.fit_llama_gpu_layers(
        str(gguf_path), -1, headroom_mb, n_ctx=n_ctx
    )

    assert layers != -1


def test_native_linux_nvidia_allows_full_offload_for_comfortable_models(
    monkeypatch, gguf_path
):
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_native_linux_nvidia", lambda: True)
    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_total_layers", lambda _p: 48
    )

    layers = mp.fit_llama_gpu_layers(
        str(gguf_path), -1, 24576, n_ctx=4096
    )

    assert layers == -1

    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count", lambda _p: 48
    )
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf_path),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )
    assert kwargs["n_batch"] == 4096
    assert kwargs["n_ubatch"] == 1024
    assert "flash_attn" not in kwargs


@pytest.mark.parametrize(
    "label,headroom_mb,weight_mb,params_b,n_ctx",
    _NVIDIA_CHAT_SCENARIOS,
    ids=[s[0] for s in _NVIDIA_CHAT_SCENARIOS],
)
def test_clamp_llama_load_kwargs_respects_post_weight_headroom(
    monkeypatch, gguf_path, label, headroom_mb, weight_mb, params_b, n_ctx
):
    monkeypatch.setattr(
        "seiso.memory.protection.headroom_mb", lambda: headroom_mb
    )
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count", lambda _p: 64
    )

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf_path),
            "n_ctx": n_ctx,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )

    assert kwargs["n_batch"] <= 4096
    assert kwargs["n_ubatch"] <= 1024
    assert kwargs["n_ubatch"] <= kwargs["n_batch"]

    if llama_model_is_tight_vram_fit(
        model_path=gguf_path,
        free_mb=headroom_mb,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
    ):
        effective = llama_effective_batch_headroom_mb(
            headroom_mb,
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
        )
        max_batch, max_ubatch = llama_batch_limits_for_headroom(effective)
        assert kwargs["n_batch"] <= max_batch
        assert kwargs["n_ubatch"] <= max_ubatch


@pytest.mark.parametrize("tier", ["normal", "compact", "minimal"])
@pytest.mark.parametrize(
    "headroom_mb,weight_mb,n_ctx",
    [(24576, 17000, 4096), (12000, 17000, 4096), (8192, 4500, 4096)],
)
def test_load_profile_ladder_batches_never_exceed_headroom(
    monkeypatch, gguf_path, tier, headroom_mb, weight_mb, n_ctx
):
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count", lambda _p: 64
    )

    profiles = llama_load_profile_ladder(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        free_mb=headroom_mb,
        base_batch=4096,
        base_ubatch=1024,
        tier=tier,
    )
    effective = llama_effective_batch_headroom_mb(
        headroom_mb,
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
    )
    max_batch, max_ubatch = llama_batch_limits_for_headroom(effective)

    for idx, profile in enumerate(profiles):
        tight = llama_model_is_tight_vram_fit(
            model_path=gguf_path,
            free_mb=headroom_mb,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
        )
        if idx == 0 and tight:
            assert profile["n_batch"] <= max_batch
            assert profile["n_ubatch"] <= max_ubatch
        assert profile["n_ubatch"] <= profile["n_batch"]
        assert profile["n_batch"] <= 4096
        assert profile["n_ubatch"] <= 1024


def test_load_llama_model_attempt_count_is_bounded_on_repeated_oom(
    monkeypatch, tmp_path
):
    """OOM retries must terminate — no infinite GPU allocation loop."""
    import seiso.inference.model_pool as mp

    gguf = tmp_path / "qwen-27b-q4.gguf"
    gguf.write_bytes(b"gguf")
    attempts = 0

    class FakeLlama:
        def __init__(self, *, model_path: str, **kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(mp, "_default_llama_gpu_layers", lambda: -1)
    monkeypatch.setattr(mp, "fit_llama_gpu_layers", lambda _p, _r, _h, **_k: 24)
    monkeypatch.setattr(
        mp,
        "_llama_kv_quant_options",
        lambda _p: [{}, {"type_k": 8, "type_v": 8}],
    )
    monkeypatch.setattr(mp, "_refresh_headroom_stats", lambda *, force=False: None)
    monkeypatch.setattr(
        "seiso.memory.protection.release_cached_memory", lambda sync=False: None
    )
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17000
    )
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 64)
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        type("LlamaCpp", (), {"Llama": FakeLlama}),
    )

    with pytest.raises(RuntimeError, match="Could not load model"):
        mp._load_llama_model(str(gguf), 4096)

    # full_targets empty (fitted=24), partial attempts only — must be finite
    assert 0 < attempts < 200


def test_build_hf_max_memory_reserves_vram(monkeypatch):
    class FakeCudaModule:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(_device):
            return (20 * 1024**3, 24 * 1024**3)

    class FakeTorch:
        cuda = FakeCudaModule

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    max_mem = build_hf_max_memory(reserve_ratio=0.03)
    assert max_mem is not None
    usable_mb = int(max_mem[0].replace("MiB", ""))
    total_mb = 24 * 1024
    assert usable_mb < total_mb * 0.95


def test_platform_profile_native_linux_caps_startup_batches(monkeypatch):
    """Startup batch defaults on native Linux must not exceed safe post-weight caps."""
    from seiso.hardware.tiers import HardwareTier
    from seiso.memory.platform_profile import apply_platform_memory_profile

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)

    profile = {
        "ram_gb": 32,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24576}],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.platform.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 24576
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "gpu_count": 1,
            "train_platform": "cpu",
            "nvidia_hardware": True,
            "vendor": "nvidia",
        },
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    batch = int(os.environ["SEISO_LLAMA_BATCH"])
    ubatch = int(os.environ["SEISO_LLAMA_UBATCH"])
    assert batch == 4096
    assert ubatch == 1024
    assert ubatch <= batch
    assert os.environ.get("SEISO_LLAMA_FLASH_ATTN") == "false"
    assert os.environ.get("SEISO_LLAMA_SPEED_SCALE") == "false"


def test_platform_profile_native_linux_respects_user_flash_attn_env(monkeypatch):
    from seiso.hardware.tiers import HardwareTier
    from seiso.memory.platform_profile import apply_platform_memory_profile

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)

    profile = {
        "ram_gb": 64,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24576}],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setenv("SEISO_LLAMA_FLASH_ATTN", "false")
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 24576
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "gpu_count": 1,
            "train_platform": "cpu",
            "nvidia_hardware": True,
            "vendor": "nvidia",
        },
    )
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    assert os.environ["SEISO_LLAMA_FLASH_ATTN"] == "false"


def test_platform_profile_native_linux_low_ram_caps_startup_batches(monkeypatch):
    """Low host RAM on native Linux must not inherit large VRAM-only batch defaults."""
    from seiso.hardware.tiers import HardwareTier
    from seiso.memory.platform_profile import apply_platform_memory_profile

    for key in list(os.environ):
        if key.startswith("SEISO_LLAMA_"):
            monkeypatch.delenv(key, raising=False)

    profile = {
        "ram_gb": 12,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24576}],
        "backend": "cuda",
        "platform": "Linux",
    }
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr("seiso.platform.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.memory.platform_profile.classify_tier",
        lambda _p: HardwareTier.WORKSTATION,
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.vram_headroom_mb", lambda _p: 24576
    )
    monkeypatch.setattr(
        "seiso.memory.platform_profile.training_capabilities",
        lambda: {
            "gpu_count": 1,
            "train_platform": "cpu",
            "nvidia_hardware": True,
            "vendor": "nvidia",
        },
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )

    apply_platform_memory_profile(profile=profile)

    batch = int(os.environ["SEISO_LLAMA_BATCH"])
    ubatch = int(os.environ["SEISO_LLAMA_UBATCH"])
    assert batch <= 512
    assert ubatch <= 256
    assert ubatch <= batch


def test_deferred_preflight_never_bypasses_when_model_exceeds_gpu_capacity(
    tmp_path, monkeypatch
):
    from seiso.memory.protection import MemoryLoadBlockedError, ensure_load_fits

    gguf = tmp_path / "too-big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    profile = {
        "backend": "cuda",
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vram_total_mb": 24564}],
        "ram_gb": 32,
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )
    monkeypatch.setattr("seiso.platform.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 24564)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 24564)
    monkeypatch.setattr(
        "seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True
    )
    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit",
        lambda _path, mode="chat": {
            "hardware_fit": "unlikely",
            "est_vram_mb": 50000,
            "memory_load_blocked": True,
            "memory_load_blocked_reason": "exceeds GPU",
        },
    )

    with pytest.raises(MemoryLoadBlockedError):
        ensure_load_fits(gguf, mode="chat", backend="llamacpp")


def test_fit_llama_gpu_layers_rejects_full_offload_when_kv_exceeds_headroom(
    monkeypatch, gguf_path
):
    """Old 0.92 weight-only heuristic could claim full offload here — must not."""
    import seiso.inference.model_pool as mp

    headroom_mb = 18500
    weight_mb = 17000
    n_ctx = 2048

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: weight_mb
    )
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 64)

    layers = mp.fit_llama_gpu_layers(
        str(gguf_path), -1, headroom_mb, n_ctx=n_ctx
    )

    assert layers != -1
    if layers > 0:
        _assert_layers_fit_vram(
            headroom_mb=headroom_mb,
            weight_mb=weight_mb,
            n_ctx=n_ctx,
            n_gpu_layers=layers,
            model_path=str(gguf_path),
        )


def test_fit_llama_gpu_layers_grid_never_overclaims_across_hardware(
    monkeypatch, gguf_path
):
    """Simulate layer fitting across VRAM/model/context combinations — no overcommit."""
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 64)

    for headroom_mb, weight_mb, n_ctx in product(
        _HEADROOM_GRID_MB, _WEIGHT_GRID_MB, _CTX_GRID
    ):
        monkeypatch.setattr(
            "seiso.memory.protection.estimate_path_vram_mb",
            lambda _p, w=weight_mb: w,
        )
        layers = mp.fit_llama_gpu_layers(
            str(gguf_path), -1, headroom_mb, n_ctx=n_ctx
        )
        assert layers == -1 or 0 <= layers <= 64
        if layers == -1:
            _assert_layers_fit_vram(
                headroom_mb=headroom_mb,
                weight_mb=weight_mb,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                model_path=str(gguf_path),
            )
        elif layers > 0:
            _assert_layers_fit_vram(
                headroom_mb=headroom_mb,
                weight_mb=weight_mb,
                n_ctx=n_ctx,
                n_gpu_layers=layers,
                model_path=str(gguf_path),
            )


def test_fit_llama_gpu_layers_keeps_full_offload_on_comfortable_hardware(
    monkeypatch, gguf_path
):
    """Generous headroom must still get full GPU offload — no performance regression."""
    import seiso.inference.model_pool as mp

    monkeypatch.setattr(mp, "_llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb", lambda _p: 4500
    )
    monkeypatch.setattr("seiso.inference.backends.gguf_total_layers", lambda _p: 64)

    layers = mp.fit_llama_gpu_layers(str(gguf_path), -1, 49152, n_ctx=4096)
    assert layers == -1
