"""Install profiles must ship Forge UI locks + Nostr relay deps on every OS path."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# CI runs on Python 3.10; tomllib is stdlib only from 3.11+.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

_UI_CRYPTO_DEPS = ("@noble/ciphers", "@noble/hashes", "@scure/base")
_NAMED_PROFILES = (
    "linux-nvidia",
    "linux-cpu",
    "linux-rocm",
    "wsl-nvidia",
    "macos",
    "chat",
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _common_sh() -> str:
    return (ROOT / "scripts/lib/common.sh").read_text(encoding="utf-8")


def test_forge_extra_includes_nostr_relay_deps():
    extras = _pyproject()["project"]["optional-dependencies"]
    forge = "\n".join(extras["forge"])
    nostr = "\n".join(extras["nostr"])
    assert "websockets" in forge
    assert "cryptography" in forge
    # Explicit [nostr] alias remains for docs / opt-in installs.
    assert "websockets" in nostr
    assert "cryptography" in nostr


def test_nostr_extra_is_subset_of_forge_relay_deps():
    """Installing [forge] alone must satisfy everything [nostr] asks for."""
    extras = _pyproject()["project"]["optional-dependencies"]

    def _names(entries: list[str]) -> set[str]:
        out: set[str] = set()
        for item in entries:
            name = item.split(";", 1)[0].strip()
            name = re.split(r"[<>=!~\[]", name, maxsplit=1)[0].strip()
            out.add(name)
        return out

    assert _names(extras["nostr"]).issubset(_names(extras["forge"]))


def test_install_profiles_all_include_forge():
    common = _common_sh()
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


def test_named_install_profiles_documented_in_common():
    common = _common_sh()
    for profile in _NAMED_PROFILES:
        assert profile in common, f"profile {profile!r} missing from common.sh"
    # Profile help text lists the supported names for curl | bash users.
    assert "linux-nvidia, linux-cpu, linux-rocm, wsl-nvidia, macos, chat" in common
    # macOS ships Bash 3.2 — avoid ${var,,} / ${var!r} in install helpers.
    assert "seiso_tolower" in common
    assert "${1,,}" not in common
    assert "${SEISO_INSTALL_PROFILE,,}" not in common
    assert "${SEISO_INSTALL_PROFILE!r}" not in common


def test_macos_install_profile_works_on_bash32(tmp_path):
    """Documented SEISO_INSTALL_PROFILE=macos must work under macOS /bin/bash 3.2."""
    import os
    import shutil
    import subprocess

    bash = "/bin/bash" if os.path.isfile("/bin/bash") else shutil.which("bash")
    assert bash, "bash required"
    script = tmp_path / "probe.sh"
    script.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                f'source "{ROOT / "scripts/lib/common.sh"}"',
                'test "$(seiso_tolower MacOS)" = "macos"',
                'test "$(SEISO_INSTALL_PROFILE=macos seiso_detect_platform_extras)" = "forge,train,llamacpp,mlx"',
                'test "$(SEISO_INSTALL_PROFILE=MacOS seiso_detect_platform_extras)" = "forge,train,llamacpp,mlx"',
                'chat="$(seiso_install_profile_extras chat)"',
                'case "$(uname -s)" in',
                '  Darwin) test "$chat" = "forge,llamacpp,mlx" ;;',
                '  *) test "$chat" = "forge,llamacpp" ;;',
                "esac",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run([bash, str(script)], check=True, cwd=str(ROOT))


def test_ui_pkg_manager_prefers_bun_unless_npm_forced():
    common = _common_sh()
    assert "SEISO_USE_NPM" in common
    assert "seiso_ui_pkg_manager" in common
    # Bun path prefers frozen lockfile, refreshes if stale; npm uses ci.
    assert re.search(
        r"seiso_ui_install_deps\(\)[\s\S]*?"
        r"bun install --frozen-lockfile[\s\S]*?"
        r"bun install[\s\S]*?"
        r"npm ci",
        common,
    )


def test_ui_lockfiles_present_for_bun_and_npm_paths():
    """macOS/Linux default Bun; Windows / SEISO_USE_NPM=1 use npm ci."""
    assert (ROOT / "forge-ui/bun.lock").is_file()
    assert (ROOT / "forge-ui/package-lock.json").is_file()
    package = (ROOT / "forge-ui/package.json").read_text(encoding="utf-8")
    bun_lock = (ROOT / "forge-ui/bun.lock").read_text(encoding="utf-8")
    npm_lock = (ROOT / "forge-ui/package-lock.json").read_text(encoding="utf-8")
    for dep in _UI_CRYPTO_DEPS:
        assert dep in package
        assert dep in bun_lock
        assert dep in npm_lock


def test_docs_cover_all_os_install_paths_and_nostr_via_forge():
    install = (ROOT / "docs/install.md").read_text(encoding="utf-8")
    provenance = (ROOT / "docs/provenance-nostr.md").read_text(encoding="utf-8")
    for profile in (
        "linux-nvidia",
        "linux-cpu",
        "wsl-nvidia",
        "macos",
    ):
        assert f"SEISO_INSTALL_PROFILE={profile}" in install
    assert "npm ci" in install
    assert "bun.lock" in install or "frozen-lockfile" in install
    assert "websockets" in install
    assert "[forge]" in provenance
    assert "seiso[nostr]" in provenance


def test_doctor_checks_websockets_and_skips_false_hub_fail_offline():
    doctor = (ROOT / "scripts/doctor.sh").read_text(encoding="utf-8")
    assert '"websockets"' in doctor
    assert "Nostr provenance relays" in doctor
    assert "network probe skipped" in doctor
    assert "Hub client libraries ready (network probe skipped)" in doctor
    assert "Local chat backends present" in doctor
    # Network branch still reports the live ready_for_* flags.
    assert "Hub ready for download" in doctor
    assert "ready_for_local_chat" in doctor


def test_relays_import_error_mentions_nostr_extra(monkeypatch):
    import builtins

    from seiso.research.nostr import relays

    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "websockets" or str(name).startswith("websockets."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    with pytest.raises(ImportError, match="seiso\\[nostr\\]"):
        relays._require_websockets()