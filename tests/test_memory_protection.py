"""Tests for cross-cutting memory protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from seiso.memory.protection import (
    MemoryLoadBlockedError,
    apply_rl_memory_guards,
    apply_training_memory_guards,
    assess_path_memory_fit,
    clamp_llama_cache_mb,
    clamp_llama_load_kwargs,
    clamp_llama_n_ctx,
    discrete_gpu_total_mb,
    ensure_load_fits,
    gpu_batch_tier_caps,
    is_oom_error,
    llama_batch_limits_for_headroom,
    llama_effective_batch_headroom_mb,
    llama_host_batch_headroom_mb,
    llama_kv_cache_reserve_mb,
    llama_load_profile_ladder,
    llama_model_is_tight_vram_fit,
    llama_next_recovery_tier,
    llama_offload_fits_headroom,
    llama_prefill_needs_reload,
    sanitize_inference_payload,
    trim_llama_messages_to_context,
)


def _mock_gpu_total(monkeypatch, vram_mb: int) -> None:
    monkeypatch.setattr(
        "seiso.memory.protection.discrete_gpu_total_mb",
        lambda _profile=None: vram_mb,
    )


def _gpu_normal_caps(vram_mb: int) -> tuple[int, int]:
    return gpu_batch_tier_caps(vram_mb, "normal")


def _write_arch_gguf(
    path: Path, architecture: str, *, extra: list[tuple[bytes, int]] | None = None
) -> None:
    import struct

    arch_key = b"general.architecture"
    arch_value = architecture.encode()
    prefix = architecture.split("-", 1)[0]
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
    ]
    for key, value in extra or []:
        payload.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                struct.pack("<I", value),
            ]
        )
    block_key = prefix.encode() + b".block_count"
    payload.extend(
        [
            struct.pack("<Q", len(block_key)),
            block_key,
            struct.pack("<I", 4),
            struct.pack("<I", 32),
        ]
    )
    kv_count = 2 + len(extra or [])
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, kv_count) + b"".join(payload))


def test_llama_offload_fits_headroom_requires_weight_plus_kv(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    assert llama_offload_fits_headroom(
        gguf,
        headroom_mb=24576,
        n_gpu_layers=-1,
        n_ctx=4096,
        weight_mb=17000,
        total_layers=64,
    )
    assert not llama_offload_fits_headroom(
        gguf,
        headroom_mb=18500,
        n_gpu_layers=-1,
        n_ctx=2048,
        weight_mb=17000,
        total_layers=64,
    )


def test_is_oom_error_detects_cuda_message():
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom_error(RuntimeError("failed to allocate Metal buffer"))


def test_is_oom_error_ignores_other_errors():
    assert not is_oom_error(ValueError("bad batch"))
    assert not is_oom_error(RuntimeError("allocation strategy unavailable"))
    assert not is_oom_error(RuntimeError("custom allocator callback"))


def test_sanitize_inference_payload_clamps_max_tokens(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 4096)
    out = sanitize_inference_payload(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999}
    )
    assert 1 <= out["max_tokens"] <= 8192


def test_trim_llama_messages_to_context_drops_old_history_before_prefill():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "old question " * 3000},
        {"role": "assistant", "content": "old answer " * 3000},
        {"role": "user", "content": "current question"},
    ]

    trimmed = trim_llama_messages_to_context(messages, n_ctx=2048, max_tokens=256)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "current question"
    assert len(trimmed) < len(messages)
    assert sum(len(str(m.get("content", ""))) for m in trimmed) < 4000


def test_trim_llama_messages_to_context_leaves_short_prompt_unchanged():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "current question"},
    ]

    assert trim_llama_messages_to_context(messages, n_ctx=4096, max_tokens=256) is messages


def test_clamp_llama_n_ctx_respects_headroom(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 3072)
    n_ctx = clamp_llama_n_ctx(
        8192, messages=[{"role": "user", "content": "x" * 200}], max_tokens=256
    )
    assert 2048 <= n_ctx <= 8192
    assert n_ctx % 512 == 0


def test_clamp_llama_load_kwargs_keeps_requested_batch_on_tight_memory(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 3500)
    kwargs = clamp_llama_load_kwargs(
        {"n_ctx": 4096, "n_batch": 2048, "n_ubatch": 2048, "n_gpu_layers": -1}
    )
    assert kwargs["n_batch"] == 2048
    assert kwargs["n_ubatch"] == 2048


def test_clamp_llama_load_kwargs_does_not_scale_batch_with_large_context(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    kwargs = clamp_llama_load_kwargs(
        {"n_ctx": 8192, "n_batch": 2048, "n_ubatch": 512, "n_gpu_layers": -1}
    )
    assert kwargs["n_batch"] == 2048


def test_clamp_llama_load_kwargs_uses_model_context_ceiling(monkeypatch, tmp_path):
    gguf = tmp_path / "short-context.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    seen: dict[str, object] = {}

    def fake_ceiling(model_path, *, model_format=None, model_name=None):
        seen["model_path"] = model_path
        seen["model_format"] = model_format
        seen["model_name"] = model_name
        return 4096

    monkeypatch.setattr(
        "seiso.inference.context_limits.effective_context_ceiling",
        fake_ceiling,
    )

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 131072,
            "n_batch": 512,
            "n_ubatch": 128,
            "n_gpu_layers": 0,
        }
    )

    assert kwargs["n_ctx"] == 4096
    assert seen["model_path"] == str(gguf)
    assert seen["model_format"] == "gguf"


def test_llama_batch_limits_scale_by_gpu_headroom():
    assert llama_batch_limits_for_headroom(1024) == (256, 128)
    assert llama_batch_limits_for_headroom(4096) == (512, 256)
    assert llama_batch_limits_for_headroom(8192) == (1024, 256)
    assert llama_batch_limits_for_headroom(24576) == (2048, 512)
    assert llama_batch_limits_for_headroom(49152) == (4096, 1024)


def test_llama_next_recovery_tier_sequence():
    assert llama_next_recovery_tier("normal") == "compact"
    assert llama_next_recovery_tier("compact") == "minimal"
    assert llama_next_recovery_tier("minimal") is None


def test_clamp_llama_load_kwargs_scales_batch_for_large_gpu_gguf(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen-27b-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 22000,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 64,
    )

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )

    assert kwargs["n_batch"] == 256
    assert kwargs["n_ubatch"] == 128


def test_clamp_llama_load_kwargs_uses_high_batch_when_large_gpu_has_room(monkeypatch, tmp_path):
    gguf = tmp_path / "small-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 49152)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 8192,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 64,
    )

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )

    assert kwargs["n_batch"] == 4096
    assert kwargs["n_ubatch"] == 1024


def test_llama_batch_headroom_accounts_for_model_weights(monkeypatch, tmp_path):
    from seiso.memory.protection import llama_batch_headroom_mb

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00" * (8 * 1024**2))
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 8192,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 32,
    )
    remaining = llama_batch_headroom_mb(16384, model_path=gguf, n_gpu_layers=-1)
    assert remaining < 8192


def test_llama_batch_headroom_estimates_directory_models(monkeypatch, tmp_path):
    from seiso.memory.protection import llama_batch_headroom_mb

    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"\x00" * (4 * 1024**2))
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 6144,
    )
    remaining = llama_batch_headroom_mb(16384, model_path=model_dir, n_gpu_layers=-1)
    assert remaining < 16384


def test_llama_kv_cache_reserve_scales_with_model_and_context(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen-27b-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 22000,
    )

    small_ctx = llama_kv_cache_reserve_mb(
        gguf,
        n_ctx=2048,
        n_gpu_layers=-1,
        total_layers=64,
        weight_mb=22000,
        free_mb=24576,
    )
    large_ctx = llama_kv_cache_reserve_mb(
        gguf,
        n_ctx=8192,
        n_gpu_layers=-1,
        total_layers=64,
        weight_mb=22000,
        free_mb=24576,
    )

    assert small_ctx > 1024
    assert large_ctx > small_ctx


def test_clamp_llama_cache_mb_keeps_configured_value_on_low_headroom(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 2048)
    assert clamp_llama_cache_mb(1024) == 1024


def test_clamp_llama_cache_mb_caps_on_linux(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 8192)
    assert clamp_llama_cache_mb(1024) == 341


def test_clamp_llama_cache_mb_accounts_for_model_mmap_on_native_linux(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 16384)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 12000,
    )
    assert clamp_llama_cache_mb(1024, model_path=gguf) < 1024


def test_llama_effective_batch_headroom_skips_margin_for_comfortable_models(monkeypatch, tmp_path):
    from seiso.memory.protection import (
        llama_batch_headroom_mb,
        llama_model_is_tight_vram_fit,
    )

    gguf = tmp_path / "gemma-14b-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 9000,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 48,
    )

    assert not llama_model_is_tight_vram_fit(
        model_path=gguf, free_mb=24576, n_gpu_layers=-1, n_ctx=4096
    )
    gpu_only = llama_batch_headroom_mb(24576, model_path=gguf, n_gpu_layers=-1, n_ctx=4096)
    effective = llama_effective_batch_headroom_mb(
        24576, model_path=gguf, n_gpu_layers=-1, n_ctx=4096
    )
    assert effective == gpu_only


def test_llama_effective_batch_headroom_uses_host_ram_on_native_linux(monkeypatch, tmp_path):
    from seiso.memory.protection import llama_batch_headroom_mb

    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 8192)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 6000,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 64,
    )
    gpu_only = llama_batch_headroom_mb(8192, model_path=gguf, n_gpu_layers=-1)
    host_only = llama_host_batch_headroom_mb(model_path=gguf, n_gpu_layers=-1, free_vram_mb=8192)
    effective = llama_effective_batch_headroom_mb(8192, model_path=gguf, n_gpu_layers=-1)
    assert host_only is not None
    raw = min(gpu_only, host_only)
    expected = max(128, int(raw * 0.85) - 256)
    assert effective == expected


def test_llama_load_profile_ladder_upscales_small_model_on_big_gpu(monkeypatch, tmp_path):
    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 1024,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 32,
    )
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)
    monkeypatch.delenv("SEISO_LLAMA_SPEED_SCALE", raising=False)

    profiles = llama_load_profile_ladder(
        model_path=str(gguf),
        n_ctx=4096,
        n_gpu_layers=-1,
        free_mb=24576,
        base_batch=1024,
        base_ubatch=512,
        tier="normal",
    )
    assert profiles[0]["n_batch"] == 2048
    assert profiles[0]["n_ubatch"] == 512


def test_llama_load_profile_ladder_native_linux_keeps_july3_speed_for_roomy_models(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.delenv("SEISO_LLAMA_SPEED_SCALE", raising=False)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    _mock_gpu_total(monkeypatch, 24576)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 1024,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 32,
    )

    profiles = llama_load_profile_ladder(
        model_path=str(gguf),
        n_ctx=4096,
        n_gpu_layers=-1,
        free_mb=24576,
        base_batch=4096,
        base_ubatch=1024,
        tier="normal",
    )

    expected_batch, expected_ubatch = _gpu_normal_caps(24576)
    assert profiles[0]["n_batch"] == expected_batch
    assert profiles[0]["n_ubatch"] == expected_ubatch
    assert profiles[0].get("flash_attn") is False
    assert profiles[-1].get("flash_attn") is False


def test_clamp_llama_load_kwargs_native_linux_roomy_keeps_july3_batches(monkeypatch, tmp_path):
    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 1024)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 32)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    _mock_gpu_total(monkeypatch, 24576)

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
            "flash_attn": True,
        }
    )
    expected_batch, expected_ubatch = _gpu_normal_caps(24576)
    assert kwargs["n_batch"] == expected_batch
    assert kwargs["n_ubatch"] == expected_ubatch
    # Dense/unknown models may keep flash_attn when opted in.
    assert kwargs.get("flash_attn") is True


def test_clamp_llama_load_kwargs_native_linux_borderline_non_tight_caps_batch(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "mid.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 13000)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 48)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_a, **_k: 800,
    )

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )
    assert kwargs["n_batch"] == 1024
    assert kwargs["n_ubatch"] == 256


def test_llama_prefill_guard_keeps_roomy_short_prompt(monkeypatch, tmp_path):
    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    _mock_gpu_total(monkeypatch, 24576)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_effective_batch_headroom_mb",
        lambda *_args, **_kwargs: 24576,
    )

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=1024,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is False
    expected_batch, expected_ubatch = _gpu_normal_caps(24576)
    assert safe_batch == expected_batch
    assert safe_ubatch == expected_ubatch


def test_llama_prefill_guard_keeps_roomy_short_prompt_on_small_headroom_fluctuation(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    _mock_gpu_total(monkeypatch, 24576)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24500)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_effective_batch_headroom_mb",
        lambda *_args, **_kwargs: 24500,
    )

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=1024,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is False
    expected_batch, expected_ubatch = _gpu_normal_caps(24576)
    assert safe_batch == expected_batch
    assert safe_ubatch == expected_ubatch


def test_llama_prefill_guard_reloads_growing_native_linux_prompt(monkeypatch, tmp_path):
    gguf = tmp_path / "mid.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_effective_batch_headroom_mb",
        lambda *_args, **_kwargs: 7600,
    )
    messages = [{"role": "user", "content": "x" * 20000}]

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=messages,
        n_ctx=8192,
        loaded_n_batch=4096,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is True
    assert safe_batch <= 512
    assert safe_ubatch <= 256


def test_llama_prefill_guard_keeps_roomy_12b_after_load(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen2.5-12b-q4.gguf"
    _write_arch_gguf(gguf, "qwen2")
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {"gpus": [{"vram_total_mb": 24576}]})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 15500)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 8000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)
    monkeypatch.setattr(
        "seiso.hardware.tiers.discrete_vram_total_mb",
        lambda _profile: 24576,
    )

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=1024,
        loaded_n_ubatch=256,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=15500,
    )

    assert needs_reload is False
    assert safe_batch >= 512
    assert safe_ubatch >= 128


def test_llama_prefill_guard_tight_gemma_27b_caps_safe_at_loaded_batch(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "gemma3-27b-q4.gguf"
    _write_arch_gguf(gguf, "gemma3", extra=[(b"gemma3.attention.sliding_window", 512)])
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {"gpus": [{"vram_total_mb": 24576}]})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 6500)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 16000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 1024,
    )
    monkeypatch.setattr(
        "seiso.hardware.tiers.discrete_vram_total_mb",
        lambda _profile: 24576,
    )

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=256,
        loaded_n_ubatch=128,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is False
    assert safe_batch <= 256
    assert safe_ubatch <= 128
    assert safe_ubatch <= safe_batch


def test_llama_prefill_guard_reloads_short_prompt_for_borderline_24gb_q4(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen-30b-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 9500)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 15000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=4096,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=9500,
    )

    assert needs_reload is True
    assert safe_batch <= 1024
    assert safe_ubatch <= 256


def test_llama_prefill_guard_reloads_when_loaded_ubatch_exceeds_safe(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen-30b-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 9500)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 15000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=256,
        loaded_n_ubatch=512,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=9500,
    )

    assert needs_reload is True
    assert safe_batch <= 1024
    assert safe_ubatch <= 256


def test_llama_prefill_guard_reloads_when_headroom_shrank_without_15pct_drop(monkeypatch, tmp_path):
    gguf = tmp_path / "mid.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 11000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_effective_batch_headroom_mb",
        lambda *_args, **_kwargs: 7600,
    )
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)

    needs_reload, safe_batch, _safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=1024,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=12000,
    )

    assert needs_reload is True
    assert safe_batch <= 512


def test_llama_prefill_guard_noops_off_native_linux(monkeypatch, tmp_path):
    gguf = tmp_path / "small.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "x" * 20000}],
        n_ctx=8192,
        loaded_n_batch=4096,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
    )

    assert needs_reload is False
    assert safe_batch == 4096
    assert safe_ubatch == 1024


def test_estimate_prompt_tokens_counts_vision_parts_without_base64_chars():
    from seiso.memory.protection import _estimate_prompt_tokens

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    est = _estimate_prompt_tokens(messages)
    assert est >= 1024
    assert est < 2000


def test_llama_prefill_guard_reloads_short_text_with_vision_content(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma-vision.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_effective_batch_headroom_mb",
        lambda *_args, **_kwargs: 7600,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=messages,
        n_ctx=8192,
        loaded_n_batch=4096,
        loaded_n_gpu_layers=-1,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is True
    assert safe_batch <= 512
    assert safe_ubatch <= 256


def test_clamp_llama_load_kwargs_native_linux_vision_mmproj_clamps_batch(monkeypatch, tmp_path):
    gguf = tmp_path / "llava.gguf"
    mmproj = tmp_path / "mmproj-Q8_0.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    mmproj.write_bytes(b"\x00" * 512)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_effective_batch_headroom_mb",
        lambda *_a, **_k: 6500,
    )
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )
    assert kwargs["n_batch"] <= 512
    assert kwargs["n_ubatch"] <= 256


def test_clamp_llama_load_kwargs_native_linux_borderline_roomy(monkeypatch, tmp_path):
    """~60% VRAM roomy dense model may keep explicit flash_attn."""
    gguf = tmp_path / "mid.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 13000)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 48)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.llama_model_is_tight_vram_fit", lambda **_k: False)

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
            "flash_attn": True,
        }
    )
    # Dense/unknown keep flash_attn when not tight; batch may still clamp for headroom.
    assert kwargs.get("flash_attn") is True
    assert kwargs["n_batch"] <= 4096
    assert kwargs["n_ubatch"] <= kwargs["n_batch"]


def test_clamp_llama_load_kwargs_partial_offload_allows_larger_batch_than_full(
    monkeypatch, tmp_path
):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17000)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 64)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)

    full = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
        }
    )
    partial = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": 32,
        }
    )
    assert partial["n_batch"] >= full["n_batch"]


def test_clamp_llama_load_kwargs_native_linux_tight_disables_flash_attn(monkeypatch, tmp_path):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 17000)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 64)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.inference.backends.gguf_is_moe", lambda _p: True)
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_uses_sliding_window_attention",
        lambda _p: False,
    )

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
            "flash_attn": True,
        }
    )
    # Tight-fit native Linux loads avoid first-prefill SWA/flash-attn crash paths.
    assert "flash_attn" not in kwargs
    assert kwargs["n_batch"] <= 512
    assert kwargs["n_ubatch"] <= 128
    assert kwargs.get("op_offload") is False
    assert kwargs.get("offload_kqv") is False


def test_clamp_llama_load_kwargs_borderline_24gb_q4_uses_safe_prefill(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen-30b-q4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 15000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)
    monkeypatch.setattr("seiso.inference.backends.gguf_block_count", lambda _p: 64)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)

    assert llama_model_is_tight_vram_fit(
        model_path=gguf,
        free_mb=24576,
        n_gpu_layers=-1,
        n_ctx=4096,
    )
    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": -1,
            "flash_attn": True,
        }
    )

    assert kwargs["n_batch"] <= 512
    assert kwargs["n_ubatch"] <= 256
    assert "flash_attn" not in kwargs
    assert kwargs.get("offload_kqv") is False


def test_clamp_llama_load_kwargs_cpu_only_gemma_clamps_host_batch(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma3-12b-q4.gguf"
    _write_arch_gguf(gguf, "gemma3", extra=[(b"gemma3.attention.sliding_window", 512)])
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 12000)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)

    kwargs = clamp_llama_load_kwargs(
        {
            "_model_path": str(gguf),
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_gpu_layers": 0,
        }
    )

    assert kwargs["n_batch"] <= 256
    assert kwargs["n_ubatch"] <= 128


def test_llama_prefill_guard_cpu_only_gemma_short_prompt(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma3-12b-q4.gguf"
    _write_arch_gguf(gguf, "gemma3", extra=[(b"gemma3.attention.sliding_window", 512)])
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 12000)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "hi"}],
        n_ctx=4096,
        loaded_n_batch=256,
        loaded_n_ubatch=128,
        loaded_n_gpu_layers=0,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is False
    assert safe_batch <= 256
    assert safe_ubatch <= 128


def test_llama_prefill_guard_cpu_only_gemma_long_prompt(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma3-12b-q4.gguf"
    _write_arch_gguf(gguf, "gemma3", extra=[(b"gemma3.attention.sliding_window", 512)])
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda **_: {})
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 24576)
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 12000)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)

    needs_reload, safe_batch, safe_ubatch = llama_prefill_needs_reload(
        model_path=str(gguf),
        messages=[{"role": "user", "content": "x" * 20000}],
        n_ctx=8192,
        loaded_n_batch=4096,
        loaded_n_ubatch=1024,
        loaded_n_gpu_layers=0,
        load_tier="normal",
        loaded_headroom_mb=24576,
    )

    assert needs_reload is True
    assert safe_batch <= 256
    assert safe_ubatch <= 128


def test_qwen3_14b_roomy_4090_uses_normal_first_profile(monkeypatch, tmp_path):
    gguf = tmp_path / "qwen3-14b-q4.gguf"
    _write_arch_gguf(gguf, "qwen3")
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr("seiso.memory.protection.estimate_path_vram_mb", lambda _p: 9000)
    monkeypatch.setattr(
        "seiso.memory.protection.llama_kv_cache_reserve_mb",
        lambda *_args, **_kwargs: 512,
    )
    monkeypatch.setattr("seiso.memory.protection.available_ram_mb", lambda: 65536)
    monkeypatch.setattr(
        "seiso.hardware.tiers.discrete_vram_total_mb",
        lambda _profile: 24576,
    )

    profiles = llama_load_profile_ladder(
        model_path=str(gguf),
        n_ctx=4096,
        n_gpu_layers=-1,
        free_mb=24576,
        base_batch=1024,
        base_ubatch=256,
        tier="normal",
    )

    assert profiles[0]["n_batch"] >= 512
    assert profiles[0]["n_ubatch"] >= 128


def test_llama_load_profile_ladder_skips_upscale_when_model_fills_gpu(monkeypatch, tmp_path):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 22000,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 64,
    )
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: False)

    profiles = llama_load_profile_ladder(
        model_path=str(gguf),
        n_ctx=4096,
        n_gpu_layers=-1,
        free_mb=24576,
        base_batch=512,
        base_ubatch=128,
        tier="normal",
    )
    assert profiles[0]["n_batch"] <= 512


def test_llama_batch_limits_for_model_uses_post_weight_headroom(monkeypatch, tmp_path):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        "seiso.memory.protection.estimate_path_vram_mb",
        lambda _path: 22000,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_block_count",
        lambda _path: 64,
    )
    headroom = llama_effective_batch_headroom_mb(24576, model_path=gguf, n_gpu_layers=-1)
    batch, ubatch = llama_batch_limits_for_headroom(headroom)
    assert batch == 256
    assert ubatch == 128


def test_apply_training_memory_guards_keeps_user_sizing(monkeypatch):
    from seiso.training.config import TrainConfig

    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 6000, "vram_used_mb": 1000}],
        "ram_gb": 16,
    }
    monkeypatch.setattr("seiso.memory.protection.hardware_profile", lambda: profile)
    monkeypatch.setattr("forge.services.hardware.vram_headroom_mb", lambda _p: 5000)

    cfg = TrainConfig(
        model_id="meta-llama/Llama-3.2-1B",
        dataset="data.jsonl",
        batch_size=8,
        gradient_accumulation_steps=1,
        max_seq_length=8192,
    )
    guarded = apply_training_memory_guards(cfg)
    assert guarded.batch_size == 8
    assert guarded.gradient_accumulation_steps == 1
    assert guarded.max_seq_length == 8192


def test_apply_rl_memory_guards_keeps_user_sizing(monkeypatch):
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 2048)
    flat = {
        "torch_preflight_batch_size": 16384,
        "replay_buffer_on_gpu": True,
        "torch_batch_episodes": 2048,
    }
    out = apply_rl_memory_guards(flat)
    assert out == flat


def test_ensure_load_fits_blocks_oversized_gguf(tmp_path, monkeypatch):
    gguf = tmp_path / "huge.gguf"
    gguf.write_bytes(b"\x00" * (9 * 1024**3))
    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 4096, "vram_used_mb": 0}],
        "ram_gb": 16,
    }
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 2048)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 2048)
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    with pytest.raises(MemoryLoadBlockedError):
        ensure_load_fits(gguf, mode="chat")


def test_ensure_load_fits_forwards_backend_to_pool(tmp_path, monkeypatch):
    from seiso.inference.model_pool import ModelPool

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\x00")
    calls: list[tuple[str, str | None]] = []
    pool = ModelPool()

    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit",
        lambda _path, mode="chat": {"memory_load_blocked": False},
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, target_path, backend=None: calls.append((target_path, backend)) or False,
    )

    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit_for_load",
        lambda path, mode="chat", backend=None: (
            pool.prepare_for_load(str(path), backend) or {"memory_load_blocked": False}
        ),
    )

    ensure_load_fits(gguf, mode="chat", backend="llamacpp")

    assert calls == [(str(gguf), "llamacpp")]


def test_apple_llamacpp_load_gets_best_effort_cpu_offload(tmp_path, monkeypatch):
    gguf = tmp_path / "tight.gguf"
    gguf.write_bytes(b"\x00" * (7 * 1024**3))
    profile = {"backend": "mlx", "gpus": [], "ram_gb": 24, "platform": "darwin"}
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 24 * 1024)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 5200)

    fit = ensure_load_fits(gguf, mode="chat", backend="llamacpp")

    assert fit["memory_load_blocked"] is False
    assert fit["memory_load_blocked_reason"] is None
    assert "Mac CPU offload fallback" in fit["memory_load_warning"]


def test_apple_non_llamacpp_load_still_blocks(tmp_path, monkeypatch):
    gguf = tmp_path / "tight.gguf"
    gguf.write_bytes(b"\x00" * (7 * 1024**3))
    profile = {"backend": "mlx", "gpus": [], "ram_gb": 24, "platform": "darwin"}
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 24 * 1024)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 5200)

    with pytest.raises(MemoryLoadBlockedError):
        ensure_load_fits(gguf, mode="chat", backend="mlx")


def test_apple_llamacpp_preflight_bypass_requires_blocked_fit(monkeypatch):
    import seiso.memory.protection as protection

    profile = {"backend": "mlx", "gpus": [], "ram_gb": 24, "platform": "darwin"}
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )

    assert (
        protection._llamacpp_deferred_preflight_platform(
            {"memory_load_blocked": False},
            backend="llamacpp",
            mode="chat",
        )
        is None
    )


def test_native_linux_llamacpp_load_defers_preflight_when_model_fits_gpu(tmp_path, monkeypatch):
    gguf = tmp_path / "qwen27b.gguf"
    gguf.write_bytes(b"\x00" * (16 * 1024**3))
    profile = {
        "backend": "cuda",
        "gpus": [
            {
                "name": "NVIDIA GeForce RTX 4090",
                "vram_total_mb": 24564,
                "vram_used_mb": 17000,
            }
        ],
        "ram_gb": 32,
        "platform": "Linux",
    }
    monkeypatch.setattr(
        "seiso.memory.protection.hardware_profile", lambda force_refresh=False: profile
    )
    monkeypatch.setattr("seiso.platform.detect_wsl2", lambda: False)
    monkeypatch.setattr("seiso.platform.is_native_linux_nvidia", lambda **_: True)
    monkeypatch.setattr(
        "seiso.hardware.tiers.classify_tier",
        lambda _p: (
            __import__("seiso.hardware.tiers", fromlist=["HardwareTier"]).HardwareTier.WORKSTATION
        ),
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 24564)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 7564)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)

    fit = ensure_load_fits(gguf, mode="chat", backend="llamacpp")

    assert fit["memory_load_blocked"] is False
    assert fit["memory_load_blocked_reason"] is None
    assert "full GPU offload" in fit["memory_load_warning"]


def test_native_linux_llamacpp_preflight_still_blocks_when_exceeds_gpu_capacity(
    tmp_path, monkeypatch
):
    gguf = tmp_path / "huge.gguf"
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
        "seiso.hardware.tiers.classify_tier",
        lambda _p: (
            __import__("seiso.hardware.tiers", fromlist=["HardwareTier"]).HardwareTier.WORKSTATION
        ),
    )
    monkeypatch.setattr(
        "seiso.inference.model_pool.ModelPool.prepare_for_load",
        lambda self, *args, **kwargs: False,
    )
    monkeypatch.setattr("seiso.hardware.fit.fit_headroom_mb", lambda _p: 24564)
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 24564)
    monkeypatch.setattr("seiso.inference.model_pool._llama_gpu_offload_ok", lambda: True)
    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit",
        lambda _path, mode="chat": {
            "hardware_fit": "unlikely",
            "est_vram_mb": 32000,
            "memory_load_blocked": True,
            "memory_load_blocked_reason": "Needs ~31.2 GB at runtime but only ~24.0 GB VRAM is safely available right now.",
        },
    )

    with pytest.raises(MemoryLoadBlockedError):
        ensure_load_fits(gguf, mode="chat", backend="llamacpp")


def test_assess_path_memory_fit_for_small_file(tmp_path, monkeypatch):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"\x00" * (32 * 1024**2))
    monkeypatch.setattr("seiso.memory.protection.headroom_mb", lambda: 16384)
    fit = assess_path_memory_fit(gguf, mode="chat")
    assert fit.get("memory_load_blocked") is False


def test_allow_memory_overcommit_skips_block(tmp_path, monkeypatch):
    gguf = tmp_path / "big.gguf"
    gguf.write_bytes(b"\x00" * (4 * 1024**3))
    monkeypatch.setattr(
        "seiso.memory.protection.assess_path_memory_fit",
        lambda _path, mode="chat": {
            "memory_load_blocked": True,
            "memory_load_blocked_reason": "Model exceeds available memory",
            "hardware_fit": "unlikely",
        },
    )
    monkeypatch.setenv("SEISO_ALLOW_MEMORY_OVERCOMMIT", "1")
    fit = ensure_load_fits(gguf, mode="chat")
    assert fit.get("memory_load_blocked") is True


def test_llama_kv_cache_reserve_uses_sliding_window_cap(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_kv_bytes_per_token",
        lambda _p: 172032,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_uses_sliding_window_attention",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_sliding_window",
        lambda _p: 4096,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_swa_layer_fraction",
        lambda _p: 1.0,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_total_layers",
        lambda _p: 42,
    )

    full_ctx = llama_kv_cache_reserve_mb(
        gguf, n_ctx=8192, n_gpu_layers=-1, weight_mb=5000, free_mb=24576
    )
    swa_ctx = llama_kv_cache_reserve_mb(
        gguf, n_ctx=4096, n_gpu_layers=-1, weight_mb=5000, free_mb=24576
    )

    assert full_ctx == swa_ctx
    assert full_ctx < 900


def test_llama_kv_cache_reserve_blends_mixed_swa_layers(monkeypatch, tmp_path):
    gguf = tmp_path / "gemma4.gguf"
    gguf.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_kv_bytes_per_token",
        lambda _p: 172032,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_uses_sliding_window_attention",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "seiso.inference.backends.gguf_sliding_window",
        lambda _p: 512,
    )

    monkeypatch.setattr(
        "seiso.inference.backends.gguf_swa_layer_fraction",
        lambda _p: 1.0,
    )
    all_local = llama_kv_cache_reserve_mb(
        gguf, n_ctx=8192, n_gpu_layers=-1, weight_mb=5000, free_mb=24576
    )

    monkeypatch.setattr(
        "seiso.inference.backends.gguf_swa_layer_fraction",
        lambda _p: 0.85,
    )
    mixed = llama_kv_cache_reserve_mb(
        gguf, n_ctx=8192, n_gpu_layers=-1, weight_mb=5000, free_mb=24576
    )

    assert mixed > all_local
