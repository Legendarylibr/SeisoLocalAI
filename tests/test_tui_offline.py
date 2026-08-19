"""Seiso TUI: local GGUF, live Hub rows, slash commands, autostart path."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from seiso.tui.hub import search_hub
from seiso.tui.offline import (
    discover_local_gguf,
    format_size,
    local_model_label,
    looks_like_digest_name,
    parse_slash,
    pick_default_model,
    resolve_model_choice,
)
from seiso.tui.pages import DASHBOARD_GOALS, NAV_GROUPS, STUDIO_PAGES
from seiso.tui.terminal import draw_frame, nav_page_ids


def _gguf(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)
    return path


def test_discover_skips_mmproj_and_sorts_smallest_first(tmp_path: Path) -> None:
    _gguf(tmp_path / "models" / "big.gguf", 4000)
    _gguf(tmp_path / "hf_cache" / "tiny.gguf", 100)
    _gguf(tmp_path / "models" / "mmproj-BF16.gguf", 50)
    _gguf(tmp_path / "exports" / "mid.gguf", 800)

    found = discover_local_gguf(tmp_path)

    assert [item.label for item in found] == ["tiny.gguf", "mid.gguf", "big.gguf"]
    assert pick_default_model(found) is not None
    assert pick_default_model(found).label == "tiny.gguf"


def test_discover_shows_model_name_not_hf_blob_hash(tmp_path: Path) -> None:
    digest = "7f99c1aeefbcf991f04f67104e3d6f7b899e95170359ce081b0618d4f11878f5"
    repo = tmp_path / "hf_cache" / "models--Qwen--Qwen3-4B-GGUF"
    blob = repo / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"0" * 200)
    snapshot = (
        repo / "snapshots" / "bc640142c66e1fdd12af0bd68f40445458f3869b" / "Qwen3-4B-Q5_0.gguf"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    inventory = tmp_path / "models" / "user" / "Qwen--Qwen3-4B" / "Qwen3-4B-Q5_0.gguf"
    inventory.parent.mkdir(parents=True)
    inventory.symlink_to(snapshot)

    found = discover_local_gguf(tmp_path)

    assert len(found) == 1
    assert found[0].label == "Qwen3-4B-Q5_0.gguf"
    assert digest not in found[0].label
    assert found[0].path.name == "Qwen3-4B-Q5_0.gguf"
    assert looks_like_digest_name(digest)
    assert not looks_like_digest_name("Qwen3-4B-Q5_0.gguf")


def test_discover_snapshot_only_uses_gguf_filename(tmp_path: Path) -> None:
    digest = "c034cdfb0ad5b3edc5cf4ac07bff3b1b214aa541e633b37191c19c3a7b119727"
    repo = tmp_path / "hf_cache" / "models--Qwen--Qwen3-8B-GGUF"
    blob = repo / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"0" * 150)
    snapshot = (
        repo / "snapshots" / "7c41481f57cb95916b40956ab2f0b139b296d974" / "Qwen3-8B-Q5_0.gguf"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    found = discover_local_gguf(tmp_path)

    assert [item.label for item in found] == ["Qwen3-8B-Q5_0.gguf"]
    assert digest not in str(found[0].path)


def test_search_hub_local_title_is_model_name_not_hash(tmp_path: Path) -> None:
    digest = "7f99c1aeefbcf991f04f67104e3d6f7b899e95170359ce081b0618d4f11878f5"
    repo = tmp_path / "hf_cache" / "models--Qwen--Qwen3-4B-GGUF"
    blob = repo / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"0" * 200)
    snapshot = (
        repo / "snapshots" / "bc640142c66e1fdd12af0bd68f40445458f3869b" / "Qwen3-4B-Q5_0.gguf"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    local, _remote, error = search_hub(
        "",
        data_dir=tmp_path,
        catalog_search=lambda *_a, **_k: type("R", (), {"models": []})(),
        gguf_search=lambda **_k: [],
    )
    assert error is None
    assert [row.title for row in local] == ["Qwen3-4B-Q5_0.gguf"]
    assert digest not in local[0].title


def test_local_model_label_from_blob_and_snapshot_paths() -> None:
    digest = "8d0eb319f05cdd6ce88e308cbcdfa9a8c1f7e99416cc7d34b80f158a7f514411"
    blob = Path("/data/hf_cache/models--MaziyarPanahi--Qwen3-4B-Instruct-2507-GGUF/blobs/" + digest)
    snapshot = Path(
        "/data/hf_cache/models--MaziyarPanahi--Qwen3-4B-Instruct-2507-GGUF/"
        "snapshots/aec29f0e8c31130ba811bec2c774c2ef44888f55/"
        "Qwen3-4B-Instruct-2507.Q2_K.gguf"
    )
    assert local_model_label(blob) == "MaziyarPanahi/Qwen3-4B-Instruct-2507-GGUF"
    assert local_model_label(blob, snapshot) == "Qwen3-4B-Instruct-2507.Q2_K.gguf"
    assert looks_like_digest_name(f"sha256-{digest}")


def test_discover_dedups_hardlinks(tmp_path: Path) -> None:
    src = _gguf(tmp_path / "hf_cache" / "same.gguf", 200)
    link = tmp_path / "models" / "same.gguf"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.hardlink_to(src)
    except OSError:
        link.symlink_to(src)

    found = discover_local_gguf(tmp_path)
    assert len(found) == 1


def test_resolve_model_choice_index_and_name(tmp_path: Path) -> None:
    _gguf(tmp_path / "models" / "alpha.gguf", 10)
    _gguf(tmp_path / "models" / "beta-chat.gguf", 20)
    models = discover_local_gguf(tmp_path)

    by_index, err = resolve_model_choice("2", models)
    assert err is None and by_index is not None
    assert by_index.label == "beta-chat.gguf"

    by_name, err = resolve_model_choice("beta", models)
    assert err is None and by_name is not None
    assert by_name.label == "beta-chat.gguf"

    missing, err = resolve_model_choice("nope", models)
    assert missing is None and err is not None


def test_parse_slash_aliases() -> None:
    assert parse_slash("hello") is None
    assert parse_slash("/quit").kind == "quit"
    assert parse_slash("/free").kind == "unload"
    assert parse_slash("/use 2").kind == "use"
    assert parse_slash("/use 2").arg == "2"
    assert parse_slash("/search qwen 7b").kind == "search"
    assert parse_slash("/search qwen 7b").arg == "qwen 7b"
    assert parse_slash("/download 3").kind == "download"
    assert parse_slash("/logout").kind == "logout"
    assert parse_slash("/relays wss://example.com").kind == "relays"
    assert parse_slash("/relays wss://example.com").arg == "wss://example.com"
    assert parse_slash("/harness openclaw").kind == "harness"
    assert parse_slash("/harness openclaw").arg == "openclaw"
    assert parse_slash("/subagents off").kind == "subagents"
    assert parse_slash("/swarm pair").kind == "swarm"
    assert parse_slash("/agent fix tests").kind == "agent"
    assert parse_slash("/agent fix tests").arg == "fix tests"
    assert parse_slash("/wat").kind == "unknown"


def test_format_size() -> None:
    assert format_size(512) == "512 B"
    assert "MB" in format_size(3 * 1024 * 1024)


def test_nav_matches_forge_labels() -> None:
    labels = [item["label"] for group in NAV_GROUPS for item in group["items"]]
    assert labels == [
        "Dashboard",
        "Hub",
        "Chat",
        "Knowledge",
        "Train",
        "Compress",
        "Distill-RL",
        "Export",
        "Recipes",
        "Integrations",
    ]
    assert set(DASHBOARD_GOALS[0]).issuperset({"id", "label", "path", "desc"})
    assert "train" in STUDIO_PAGES
    assert STUDIO_PAGES["train"]["command"].startswith("seiso train")


def test_search_hub_includes_remote_not_just_local(tmp_path: Path) -> None:
    _gguf(tmp_path / "models" / "tiny.gguf", 32)

    def fake_catalog(query: str, limit: int = 16):
        return type(
            "R",
            (),
            {
                "models": [
                    {
                        "repo_id": "Qwen/Qwen3-4B-GGUF",
                        "name": "Qwen3-4B-GGUF",
                        "family": "qwen",
                        "task": "chat",
                        "downloads": 12000,
                        "tags": ["gguf"],
                    }
                ]
            },
        )()

    def fake_gguf(*, query: str, limit: int = 16, trusted_only: bool = False):
        return [{"repo_id": "unsloth/Qwen3-8B-GGUF", "downloads": 8000, "likes": 40}]

    local, remote, error = search_hub(
        "qwen",
        data_dir=tmp_path,
        catalog_search=fake_catalog,
        gguf_search=fake_gguf,
    )
    assert error is None
    assert local and local[0].title == "tiny.gguf"
    repos = {row.repo_id for row in remote}
    assert "Qwen/Qwen3-4B-GGUF" in repos
    assert "unsloth/Qwen3-8B-GGUF" in repos
    assert all(row.source == "hub" for row in remote)


def test_lite_web_ui_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "seiso/tui/server.py").exists()
    assert not (root / "seiso/tui/static").exists()


def test_start_defaults_to_tui() -> None:
    start = (Path(__file__).resolve().parents[1] / "scripts/start.sh").read_text(encoding="utf-8")
    assert "SEISO_UI:-tui" in start or "SEISO_UI:-tui" in start.replace('"', "")
    assert 'exec "$seiso_bin" tui' in start
    assert "/dev/tty" in start
    assert "seiso tui" in start or "tui" in start


def test_install_complete_announces_tui_not_forge_url() -> None:
    root = Path(__file__).resolve().parents[1]
    outro = (root / "scripts/install_tui.py").read_text(encoding="utf-8")
    install = (root / "scripts/install.sh").read_text(encoding="utf-8")
    assert "TUI starting" in outro
    assert "Open Forge:" not in outro
    assert "open {args.url}" not in outro
    assert "TUI starting" in install or "Start the TUI" in install
    assert "seiso_forge_url" not in install.split("install_tui_outro")[1]


def test_tui_command_registered() -> None:
    from seiso_cli.main import app

    names = {cmd.name or getattr(cmd.callback, "__name__", "") for cmd in app.registered_commands}
    assert "tui" in names


def test_terminal_frame_renders(tmp_path: Path) -> None:
    from rich.console import Console

    from seiso.tui.offline import discover_local_gguf

    _gguf(tmp_path / "models" / "tiny.gguf", 32)
    models = discover_local_gguf(tmp_path)
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, color_system=None)
    draw_frame(
        console,
        page="chat",
        models=models,
        messages=[],
        model_label="tiny.gguf",
        data_dir=str(tmp_path),
        status="",
    )
    text = buf.getvalue()
    assert "SEISO" in text
    assert "Chat" in text
    assert "How can I help you today?" in text
    assert "↑↓ scroll" in text
    assert "Enter select" in text
    assert "chat" in nav_page_ids()


def test_jobs_builds_train_command(tmp_path: Path, monkeypatch) -> None:
    from seiso.tui import jobs

    called: list[list[str]] = []

    def fake_popen(cmd, **_kwargs):
        called.append(list(cmd))
        return object()

    monkeypatch.setattr(jobs.subprocess, "Popen", fake_popen)
    msg = jobs.run_cli_job("configs/example_lora.yaml", cwd=tmp_path)
    assert called == [["seiso", "train", "--config", "configs/example_lora.yaml"]]
    assert "Started" in msg
