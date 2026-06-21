"""Hardware fit heuristics — local-only, no network."""

from forge.services.hardware import (
    assess_catalog_fit,
    classify_tier,
    enrich_catalog_models,
    estimate_chat_vram_gb,
    preferred_inference_backend,
    training_defaults,
)


def test_estimate_chat_vram_scales_with_params():
    small = estimate_chat_vram_gb("3B")
    large = estimate_chat_vram_gb("70B")
    assert small < large
    assert small < 5


def test_moe_uses_active_params():
    full = estimate_chat_vram_gb("30B", tags=())
    moe = estimate_chat_vram_gb("35B", tags=("moe",), repo_id="Qwen/Qwen3.6-35B-A3B")
    assert moe < full


def test_unknown_params_use_conservative_default():
    from seiso.memory.estimates import estimate_gguf_download_bytes

    est = estimate_gguf_download_bytes("?", repo_id="some-org/mystery-gguf")
    assert est > 0
    assert est < 50 * 1024**3
    vram = estimate_chat_vram_gb("?", repo_id="some-org/mystery-gguf")
    assert 0 < vram < 20


def test_classify_tier_cpu_when_no_gpu():
    tier = classify_tier({"backend": "cpu", "gpus": [], "ram_gb": 16})
    assert tier.value == "cpu_only"


def test_detect_gpus_nvidia_smi_fallback(monkeypatch):
    from seiso.hardware.profile import detect_gpus, hardware_profile

    monkeypatch.setattr("seiso.hardware.profile._torch_gpus", lambda: [])
    monkeypatch.setattr("seiso.hardware.profile._mlx_apple_gpu", lambda: [])
    monkeypatch.setattr(
        "seiso.hardware.profile._nvidia_smi_gpus",
        lambda: [
            {
                "index": 0,
                "name": "NVIDIA GeForce RTX 4090",
                "vram_total_mb": 24564,
                "vram_used_mb": None,
                "utilization_pct": None,
                "temperature_c": None,
            }
        ],
    )
    monkeypatch.setattr("seiso.hardware.profile._nvidia_smi_metrics", lambda: {})

    gpus = detect_gpus()
    assert len(gpus) == 1
    assert "4090" in gpus[0]["name"]

    profile = hardware_profile(force_refresh=True)
    assert profile["tier"] == "workstation"
    assert profile["tier_label"] == "Workstation GPU"


def test_detect_backend_linux_nvidia_smi_without_cuda(monkeypatch):
    from seiso.models.loader import Backend, detect_backend

    detect_backend.cache_clear()
    monkeypatch.setattr("seiso.models.loader.platform.system", lambda: "Linux")
    monkeypatch.setattr("seiso.security.nvidia_boundary.nvidia_smi_visible", lambda: True)

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert detect_backend() == Backend.TORCH
    detect_backend.cache_clear()


def test_enrich_catalog_ranks_priority_first():
    profile = {
        "backend": "cpu",
        "gpus": [],
        "ram_gb": 16,
        "platform": "linux",
        "arch": "x86_64",
    }
    models = [
        {
            "repo_id": "big",
            "name": "Big",
            "params": "70B",
            "quant": "Q4_K_M",
            "tags": [],
            "priority": 90,
            "task": "chat",
        },
        {
            "repo_id": "small",
            "name": "Small",
            "params": "1B",
            "quant": "Q4_K_M",
            "tags": [],
            "priority": 50,
            "task": "chat",
        },
    ]
    ranked = enrich_catalog_models(models, profile, fetch_sizes=False)
    assert ranked[0]["params"] == "70B"
    assert ranked[0]["priority"] == 90
    assert ranked[1]["hardware_fit"] in ("ideal", "good")


def test_format_catalog_note_shows_download_and_runtime():
    from forge.services.hardware import (
        HardwareTier,
        estimate_gguf_download_bytes,
        format_catalog_note,
    )

    note = format_catalog_note(
        est_vram_gb=2.9,
        download_bytes=int(19.7 * 1024**3),
        headroom_gb=9.6,
        fit="ideal",
        tier=HardwareTier.APPLE_UNIFIED,
    )
    assert "Download ~19.7 GB" in note
    assert "Runtime ~2.9 GB est." in note

    moe_est = estimate_gguf_download_bytes("35B", tags=("moe",), repo_id="Qwen/Qwen3.6-35B-A3B")
    assert 1.5 * 1024**3 < moe_est < 3.5 * 1024**3


def test_hardware_profile_includes_backend_labels():
    from forge.services.hardware import hardware_profile

    profile = hardware_profile(force_refresh=True)
    labels = profile.get("inference_backend_labels") or {}
    assert labels.get("llamacpp") == "llama.cpp"
    assert labels.get("mlx") == "MLX"


def test_training_defaults_conservative_on_edge():
    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 6000, "vram_used_mb": 1000}],
        "ram_gb": 16,
    }
    defaults = training_defaults(profile)
    assert defaults["batch_size"] >= 1
    assert defaults["gradient_accumulation_steps"] >= 8


def test_preferred_backend_cpu_only_is_llamacpp():
    profile = {"backend": "cpu", "gpus": [], "ram_gb": 16}
    assert preferred_inference_backend(profile) == "llamacpp"


def test_preferred_backend_apple_tight_memory_is_llamacpp(monkeypatch):
    profile = {"backend": "mlx", "gpus": [], "ram_gb": 16}
    monkeypatch.setattr("seiso.hardware.training.vram_headroom_mb", lambda _p: 10240)
    assert classify_tier(profile).value == "apple_unified"
    assert preferred_inference_backend(profile) == "llamacpp"


def test_preferred_backend_apple_plenty_is_mlx(monkeypatch):
    profile = {"backend": "mlx", "gpus": [], "ram_gb": 64}
    monkeypatch.setattr("seiso.hardware.training.vram_headroom_mb", lambda _p: 20480)
    assert preferred_inference_backend(profile) == "mlx"


def test_low_memory_apple_marks_large_models_tight(monkeypatch):
    profile = {"backend": "mlx", "gpus": [], "ram_gb": 16}
    monkeypatch.setattr("seiso.hardware.training.vram_headroom_mb", lambda _p: 10240)
    fit = assess_catalog_fit(
        {"params": "7B", "quant": "Q4_K_M", "tags": [], "repo_id": "x", "task": "chat"},
        profile,
    )
    assert fit["hardware_fit"] == "tight"


def test_assess_hardware_fit_blocks_when_est_exceeds_headroom(monkeypatch):
    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 8192, "vram_used_mb": 0}],
        "ram_gb": 16,
    }
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 4096)
    fit = assess_catalog_fit(
        {"params": "13B", "quant": "Q4_K_M", "tags": [], "repo_id": "x", "task": "chat"},
        profile,
    )
    assert fit["memory_load_blocked"] is True
    assert fit["memory_load_blocked_reason"]
    assert "GB" in fit["memory_load_blocked_reason"]


def test_assess_hardware_fit_allows_when_est_within_headroom(monkeypatch):
    profile = {
        "backend": "cuda",
        "gpus": [{"vram_total_mb": 24576, "vram_used_mb": 0}],
        "ram_gb": 32,
    }
    monkeypatch.setattr("seiso.hardware.fit.vram_headroom_mb", lambda _p: 20480)
    fit = assess_catalog_fit(
        {"params": "7B", "quant": "Q4_K_M", "tags": [], "repo_id": "x", "task": "chat"},
        profile,
    )
    assert fit["memory_load_blocked"] is False
    assert fit["memory_load_blocked_reason"] is None
