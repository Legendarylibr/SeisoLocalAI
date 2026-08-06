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
        "printf '%s\\n' \"forge,train,cuda,llamacpp\"",
        "printf '%s\\n' \"forge,train,llamacpp\"",
        "printf '%s\\n' \"forge,train,llamacpp,mlx\"",
        "printf '%s\\n' \"forge,llamacpp,mlx\"",
        "printf '%s\\n' \"forge,llamacpp\"",
        'extras="forge,train,llamacpp,mlx"',
        'extras="forge,train,cuda,llamacpp"',
        'extras="forge,train,llamacpp"',
    ):
        assert needle in common, f"missing install extras line: {needle}"
    assert "seiso_ui_install_deps" in common
    assert "bun install --frozen-lockfile" in common
    assert "npm ci" in common


def test_pip_bootstrap_matches_pyproject_setuptools():
    """Bootstrap must not pin setuptools below the build-system / [dev] floor."""
    common = _common_sh()
    assert "setuptools>=83" in common
    assert "setuptools<82" not in common
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "setuptools>=83" in pyproject


def test_system_deps_check_venv_and_compilers_not_only_cli_names():
    """Minimal images often have python3/git/curl but lack venv or gcc."""
    common = _common_sh()
    assert "seiso_python_venv_ok" in common
    assert "seiso_build_tools_ok" in common
    assert "ensurepip" in common
    assert "python3-venv" in common


def test_start_command_registers_seiso_start():
    common = _common_sh()
    assert "seiso-start" in common
    assert "seiso_link_start_command" in common


def test_sidecar_hard_fail_is_opt_in_only():
    sidecar = (ROOT / "scripts/lib/sidecar_install.sh").read_text(encoding="utf-8")
    # Must not force hard-fail solely from install profile name.
    assert "linux-nvidia|linux-nvidia-native) return 0" not in sidecar
    assert "SEISO_REQUIRE_SIDECAR" in sidecar
    # Seeded .env must not force REQUIRE=1 (breaks offline / no-sudo machines).
    seed_block = sidecar.split("seiso_seed_sidecar_env")[1].split("seiso_sidecar_fallback")[0]
    assert 'SEISO_REQUIRE_SIDECAR" "1"' not in seed_block
    assert 'SEISO_REQUIRE_SIDECAR" "1"' not in seed_block.replace(" ", "")


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


def test_ui_pkg_manager_linux_prefers_npm_macos_prefers_bun():
    common = _common_sh()
    assert "SEISO_USE_NPM" in common
    assert "SEISO_USE_BUN" in common
    assert "seiso_ui_pkg_manager" in common
    assert "seiso_npm_usable" in common
    assert "seiso_ui_npm_ci" in common
    assert "seiso_ensure_npm_available" in common
    # Linux prefers npm when Node 18+ exists; Bun remains default elsewhere / with SEISO_USE_BUN=1.
    assert '$(uname -s)" == "Linux"' in common
    # Bun path: frozen lockfile, unfrozen only on non-timeout failure, npm sticky fallback.
    assert "seiso_run_with_timeout" in common
    assert "SEISO_BUN_INSTALL_TIMEOUT_SEC" in common
    assert "status" in common and "124" in common
    assert "export SEISO_USE_NPM=1" in common
    assert re.search(
        r"seiso_ui_install_deps\(\)[\s\S]*?"
        r"bun install --frozen-lockfile[\s\S]*?"
        r"bun install[\s\S]*?"
        r"seiso_ui_npm_ci",
        common,
    )
    assert re.search(
        r"seiso_ui_install_deps\(\)[\s\S]*?"
        r"status\" -eq 124[\s\S]*?"
        r"skipping unfrozen retry",
        common,
    )


def test_python_modules_probe_uses_cuda_path_for_llama_cpp():
    """Bare llama_cpp import fails without ensure_cuda_library_path on NVIDIA Linux."""
    common = _common_sh()
    assert "seiso_llamacpp_import_ok" in common
    # Module probe must route llama_cpp through the CUDA-aware helper.
    assert re.search(
        r"seiso_python_modules_available\(\)[\s\S]*?"
        r'\[\[ "\$module" == "llama_cpp" \]\][\s\S]*?'
        r"seiso_llamacpp_import_ok",
        common,
    )


def test_build_forge_ui_skips_when_dist_present():
    common = _common_sh()
    assert "seiso_forge_ui_dist_ready" in common
    assert "SEISO_FORCE_UI" in common
    assert re.search(
        r"seiso_build_forge_ui\(\)[\s\S]*?"
        r"seiso_forge_ui_dist_ready[\s\S]*?"
        r"skipping rebuild",
        common,
    )


def test_resolve_repo_walks_up_from_scripts_start(tmp_path):
    """scripts/start.sh must resolve the clone root, not only $HOME/Seiso."""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    assert bash
    script = tmp_path / "probe.sh"
    script.write_text(
        "\n".join(
            [
                "set -euo pipefail",
                f'source "{ROOT / "scripts/lib/common.sh"}"',
                f'root="$(seiso_resolve_repo_for_start "{ROOT / "scripts/start.sh"}")"',
                f'test "$root" = "{ROOT}"',
                f'root2="$(seiso_resolve_repo_for_start "{ROOT / "start"}")"',
                f'test "$root2" = "{ROOT}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run([bash, str(script)], check=True, cwd=str(ROOT))


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
