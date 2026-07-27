"""Install profiles must ship Forge UI locks + Nostr relay deps on every OS path."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_forge_extra_includes_nostr_relay_deps():
    extras = _pyproject()["project"]["optional-dependencies"]
    forge = "\n".join(extras["forge"])
    nostr = "\n".join(extras["nostr"])
    assert "websockets" in forge
    assert "cryptography" in forge
    # Explicit [nostr] alias remains for docs / opt-in installs.
    assert "websockets" in nostr
    assert "cryptography" in nostr


def test_install_profiles_all_include_forge():
    common = (ROOT / "scripts/lib/common.sh").read_text(encoding="utf-8")
    # Every named profile / auto OS path must pull [forge] so Nostr relay deps install.
    for needle in (
        'printf \'%s\\n\' "forge,train,cuda,llamacpp"',
        'printf \'%s\\n\' "forge,train,llamacpp"',
        'printf \'%s\\n\' "forge,train,llamacpp,mlx"',
        'printf \'%s\\n\' "forge,llamacpp,mlx"',
        'printf \'%s\\n\' "forge,llamacpp"',
        'extras="forge,train,llamacpp,mlx"',
        'extras="forge,train,cuda,llamacpp"',
        'extras="forge,train,llamacpp"',
    ):
        assert needle in common, f"missing install extras line: {needle}"
    assert "seiso_ui_install_deps" in common
    assert "bun install --frozen-lockfile" in common
    assert "npm ci" in common


def test_ui_lockfiles_present_for_bun_and_npm_paths():
    """macOS/Linux default Bun; Windows / SEISO_USE_NPM=1 use npm ci."""
    assert (ROOT / "forge-ui/bun.lock").is_file()
    assert (ROOT / "forge-ui/package-lock.json").is_file()
    package = (ROOT / "forge-ui/package.json").read_text(encoding="utf-8")
    for dep in ("@noble/ciphers", "@noble/hashes", "@scure/base"):
        assert dep in package
        assert dep in (ROOT / "forge-ui/bun.lock").read_text(encoding="utf-8")
