"""Tests for Hugging Face Hub connectivity and environment setup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from forge.config import ForgeSettings
from forge.services import hf_connectivity, hf_hub
from forge.services.hf_auth import load_user_hf_token, resolve_hf_token, save_user_hf_token
from forge.services.user_paths import assert_user_path
from seiso.models.hf_env import configure_hf_hub_cache, hf_transfer_stack, resolve_hf_cache_dir
from seiso.security import SecurityError


def test_configure_hf_hub_cache_sets_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_NUM_THREADS", raising=False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.setattr("seiso.models.hf_env._xet_available", lambda: True)
    cache = configure_hf_hub_cache(tmp_path)
    assert cache == tmp_path / "hf_cache"
    assert Path(cache).is_dir()
    assert os.environ["HF_HOME"] == str(tmp_path / "hf_home")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(cache)
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "hf_xet_cache")
    assert (tmp_path / "hf_xet_cache").is_dir()
    assert (tmp_path / "hf_home" / "xet" / "logs").is_dir()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "600"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert os.environ["HF_HUB_NUM_THREADS"] == "16"
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_configure_hf_hub_cache_preserves_user_transfer_settings(monkeypatch, tmp_path):
    custom_cache = tmp_path / "custom-cache"
    custom_xet = tmp_path / "custom-xet"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(custom_cache))
    monkeypatch.setenv("HF_XET_CACHE", str(custom_xet))
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "900")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "45")
    monkeypatch.setenv("HF_HUB_NUM_THREADS", "4")
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
    monkeypatch.setattr("seiso.models.hf_env._xet_available", lambda: True)

    cache = configure_hf_hub_cache(tmp_path)

    assert cache == custom_cache
    assert os.environ["HF_HOME"] == str(tmp_path / "hf_home")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(custom_cache)
    assert os.environ["HF_XET_CACHE"] == str(custom_xet)
    assert custom_xet.is_dir()
    assert (tmp_path / "hf_home" / "xet" / "logs").is_dir()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "900"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "45"
    assert os.environ["HF_HUB_NUM_THREADS"] == "4"
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ


def test_hf_transfer_stack_reports_backend():
    info = hf_transfer_stack()
    assert "backend" in info
    assert "xet_available" in info
    assert "num_threads" in info
    assert "hints" in info
    assert info["backend"] in ("hf_xet", "http")


def test_probe_hf_hub_missing_package(monkeypatch):
    monkeypatch.setattr(hf_connectivity, "HfApi", None, raising=False)

    def _import_error():
        raise ImportError("missing huggingface_hub")

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("missing huggingface_hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = hf_connectivity.probe_hf_hub()
    assert not result.reachable
    assert "huggingface_hub" in (result.error or "").lower()


def test_dep_status_handles_runtime_import_failures(monkeypatch):
    def fail_find_spec(_name):
        raise RuntimeError("No Metal device available")

    monkeypatch.setattr(hf_connectivity, "find_spec", fail_find_spec)
    assert hf_connectivity._dep_status("mlx_lm") is False


def test_resolve_hf_token_ignores_env_placeholder(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    token, source = resolve_hf_token(settings_token="# optional: paste token here")

    assert token is None
    assert source == "none"


def test_user_hf_token_uses_stable_settings_key(tmp_path):
    first = ForgeSettings(data_dir=tmp_path)
    save_user_hf_token(
        first.data_dir,
        "u1",
        "hf_test_token",
        encryption_key=first.hf_token_encryption_key,
    )

    assert (tmp_path / ".hf_token_encryption_key").is_file()

    second = ForgeSettings(data_dir=tmp_path)
    token = load_user_hf_token(
        second.data_dir,
        "u1",
        encryption_key=second.hf_token_encryption_key,
    )

    assert token == "hf_test_token"


def test_probe_hf_hub_anonymous_ok(monkeypatch):
    class FakeApi:
        def model_info(self, repo_id, timeout=None):
            assert repo_id == "gpt2"
            return {"id": repo_id}

    class FakeHfHubHTTPError(Exception):
        pass

    import types

    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.HfApi = FakeApi
    utils_mod = types.ModuleType("huggingface_hub.utils")
    utils_mod.HfHubHTTPError = FakeHfHubHTTPError
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub_mod)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub.utils", utils_mod)

    result = hf_connectivity.probe_hf_hub()
    assert result.reachable
    assert result.anonymous_ok


def test_probe_hf_hub_invalid_token_falls_back_to_anonymous(monkeypatch):
    class FakeApi:
        def whoami(self, token=None):
            raise FakeHfHubHTTPError()

        def model_info(self, repo_id, timeout=None):
            assert repo_id == "gpt2"
            return {"id": repo_id}

    class FakeHfHubHTTPError(Exception):
        response = type("R", (), {"status_code": 401})()

    import types

    hub_mod = types.ModuleType("huggingface_hub")
    hub_mod.HfApi = FakeApi
    utils_mod = types.ModuleType("huggingface_hub.utils")
    utils_mod.HfHubHTTPError = FakeHfHubHTTPError
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub_mod)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub.utils", utils_mod)

    result = hf_connectivity.probe_hf_hub(token="hf_bad")
    assert result.reachable
    assert result.token_invalid
    assert result.anonymous_ok
    assert result.warning


def test_assert_hub_ready_allows_invalid_token_when_anonymous_ok(monkeypatch):
    monkeypatch.setattr(
        hf_connectivity,
        "check_inference_runtime",
        lambda: hf_connectivity.InferenceRuntimeStatus(huggingface_hub=True),
    )
    monkeypatch.setattr(
        hf_connectivity,
        "probe_hf_hub",
        lambda **_: hf_connectivity.HfConnectivityResult(
            reachable=True,
            anonymous_ok=True,
            token_invalid=True,
            warning="bad token",
        ),
    )
    hf_connectivity.assert_hub_ready_for_download()


def test_assert_hub_ready_raises_when_unreachable(monkeypatch):
    monkeypatch.setattr(
        hf_connectivity,
        "check_inference_runtime",
        lambda: hf_connectivity.InferenceRuntimeStatus(huggingface_hub=True),
    )
    monkeypatch.setattr(
        hf_connectivity,
        "probe_hf_hub",
        lambda **_: hf_connectivity.HfConnectivityResult(reachable=False, error="offline"),
    )
    with pytest.raises(ValueError, match="offline"):
        hf_connectivity.assert_hub_ready_for_download()


def test_with_download_retries_transient(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("timed out")
        return "/tmp/model.gguf"

    monkeypatch.setattr(hf_hub.time, "sleep", lambda *_a, **_k: None)
    path = hf_hub._with_download_retries(flaky, repo_id="org/model")
    assert path == "/tmp/model.gguf"
    assert calls["n"] == 2


def test_with_download_retries_auth_fails_fast():
    def gated():
        raise RuntimeError("403 Forbidden")

    with pytest.raises(ValueError, match="Access denied"):
        hf_hub._with_download_retries(gated, repo_id="meta-llama/Llama-3.1-8B")


def test_broken_symlink_rejected(tmp_path):
    user_id = "u1"
    models = tmp_path / "models" / user_id
    models.mkdir(parents=True)
    target = tmp_path / "hf_cache" / "missing.gguf"
    link = models / "broken.gguf"
    link.symlink_to(target)

    with pytest.raises(SecurityError, match="broken"):
        assert_user_path(tmp_path, user_id, link)


def test_resolve_hf_cache_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf_cache"

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert resolve_hf_cache_dir(tmp_path) == tmp_path / "hf" / "hub"
