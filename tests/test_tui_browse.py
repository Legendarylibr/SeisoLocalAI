"""Keyboard browse: parse keys, move ▸, Enter hints, scroll windows."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from seiso.tui.browse import (
    apply_browse_key,
    clamp_index,
    enter_hint,
    index_of_page,
    move_index,
    page_choices,
    resolve_hub_choice,
    sidebar_items,
    visible_window,
)
from seiso.tui.hub import HubRow
from seiso.tui.keys import Key, parse_keys
from seiso.tui.pages import DASHBOARD_GOALS, NAV_GROUPS


def _row(
    *, title: str, source: str = "hub", path: Path | None = None, status: str = "remote"
) -> HubRow:
    return HubRow(
        key=title,
        source=source,
        title=title,
        repo_id=title,
        path=path,
        size_label="1 GB",
        downloads=10,
        likes=None,
        family="qwen",
        task="chat",
        status=status,
        subtitle=title,
    )


def test_parse_keys_arrows_enter_and_utf8() -> None:
    keys = parse_keys(b"\x1b[A\x1b[B\x1b[C\x1b[D\r\x7f")
    assert [k.name for k in keys] == ["up", "down", "right", "left", "enter", "backspace"]
    chars = parse_keys("café".encode())
    assert [k.char for k in chars] == ["c", "a", "f", "é"]
    assert parse_keys(b"\x1b") == [Key("esc")]
    assert parse_keys(b"\t")[0].name == "tab"
    assert parse_keys(b"\x03")[0].name == "ctrl-c"


def test_parse_keys_application_cursor_and_pages() -> None:
    assert parse_keys(b"\x1bOA")[0].name == "up"
    assert parse_keys(b"\x1b[5~")[0].name == "pageup"
    assert parse_keys(b"\x1b[6~")[0].name == "pagedown"
    assert parse_keys(b"\x1b[H")[0].name == "home"
    assert parse_keys(b"\x1b[F")[0].name == "end"


def test_parse_keys_mouse_wheel() -> None:
    assert parse_keys(b"\x1b[<64;8;4M")[0].name == "up"
    assert parse_keys(b"\x1b[<65;8;4M")[0].name == "down"
    assert parse_keys(b"\x1b[<0;8;4m")[0].name == "none"


def test_sidebar_includes_forge_pages_plus_settings() -> None:
    items = sidebar_items()
    labels = [item.label for item in items]
    forge = [raw["label"] for group in NAV_GROUPS for raw in group["items"]]
    assert labels[: len(forge)] == forge
    assert items[-1].id == "settings"
    assert index_of_page("hub") == labels.index("Hub")
    assert index_of_page("settings") == len(items) - 1


def test_move_and_window_clamp() -> None:
    assert move_index(0, -1, 5) == 0
    assert move_index(4, 1, 5) == 4
    assert move_index(2, 3, 5) == 4
    assert clamp_index(9, 0) == 0
    assert visible_window(0, 4, 10) == (0, 4)
    start, end = visible_window(15, 20, 8)
    assert end - start == 8
    assert start <= 15 < end


def test_apply_browse_key_sidebar_and_page() -> None:
    focus, nav, main = apply_browse_key(
        "down", focus="nav", nav_index=0, main_index=0, nav_count=11, main_count=4
    )
    assert (focus, nav, main) == ("nav", 1, 0)
    focus, nav, main = apply_browse_key(
        "right", focus="nav", nav_index=1, main_index=0, nav_count=11, main_count=4
    )
    assert focus == "main"
    focus, nav, main = apply_browse_key(
        "down", focus="main", nav_index=1, main_index=0, nav_count=11, main_count=4
    )
    assert (focus, main) == ("main", 1)
    focus, nav, main = apply_browse_key(
        "tab", focus="main", nav_index=1, main_index=1, nav_count=11, main_count=4
    )
    assert focus == "nav"
    focus, nav, main = apply_browse_key(
        "esc", focus="main", nav_index=1, main_index=1, nav_count=11, main_count=4
    )
    assert focus == "nav"
    # empty page list: up/down return to the sidebar
    focus, nav, main = apply_browse_key(
        "up", focus="main", nav_index=3, main_index=0, nav_count=11, main_count=0
    )
    assert focus == "nav"
    # letters are not motion — they must type, so a message can start with j
    assert apply_browse_key(
        "j", focus="main", nav_index=1, main_index=2, nav_count=11, main_count=4
    ) == ("main", 1, 2)


def test_page_choices_and_enter_hint(tmp_path: Path) -> None:
    local = [_row(title="tiny.gguf", source="local", path=tmp_path / "tiny.gguf", status="ready")]
    remote = [_row(title="Qwen/Qwen3-4B-GGUF")]
    hub = page_choices("hub", local_hub=local, remote_hub=remote)
    assert [c.kind for c in hub] == ["hub", "hub"]
    assert resolve_hub_choice(hub[0], local) == "open"
    assert resolve_hub_choice(hub[1], local) == "download"

    dash = page_choices("dashboard")
    assert len(dash) == len(DASHBOARD_GOALS)
    assert {c.page for c in dash} <= {"chat", "train", "compress"}

    studio = page_choices("train", configs=["configs/example_lora.yaml"])
    assert studio[0].kind == "run"

    settings = page_choices("settings")
    assert settings[0].action == "unload"
    assert {item.action for item in settings} >= {"unload", "logout", "keygen", "reset"}

    assert "opens tiny.gguf" in enter_hint(
        focus="main",
        page="hub",
        nav_index=0,
        main_index=0,
        choices=hub,
        composing=False,
        compose="",
    )
    assert "downloads Qwen/Qwen3-4B-GGUF" in enter_hint(
        focus="main",
        page="hub",
        nav_index=0,
        main_index=1,
        choices=hub,
        composing=False,
        compose="",
    )
    assert "opens Hub" in enter_hint(
        focus="nav",
        page="hub",
        nav_index=index_of_page("hub"),
        main_index=0,
        choices=hub,
        composing=False,
        compose="",
    )
    assert "type a message" in enter_hint(
        focus="main",
        page="chat",
        nav_index=index_of_page("chat"),
        main_index=0,
        choices=[],
        composing=False,
        compose="",
    )
    assert "sends" in enter_hint(
        focus="main",
        page="chat",
        nav_index=0,
        main_index=0,
        choices=[],
        composing=True,
        compose="hi",
    )


def test_frame_shows_scroll_enter_help_and_cursor(tmp_path: Path) -> None:
    from rich.console import Console

    from seiso.tui.offline import discover_local_gguf
    from seiso.tui.terminal import draw_frame

    gguf = tmp_path / "models" / "tiny.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"0" * 32)
    models = discover_local_gguf(tmp_path)
    local = [_row(title="tiny.gguf", source="local", path=gguf, status="ready")]
    remote = [_row(title="Qwen/Qwen3-4B-GGUF")]

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, color_system=None)
    draw_frame(
        console,
        page="hub",
        models=models,
        messages=[],
        model_label="tiny.gguf",
        data_dir=str(tmp_path),
        status="",
        local_hub=local,
        remote_hub=remote,
        hub_selected=2,
        focus="main",
        main_index=1,
        hint="Enter downloads Qwen/Qwen3-4B-GGUF",
    )
    text = buf.getvalue()
    assert "SEISO" in text
    assert "Model Hub" in text
    assert "↑↓ scroll" in text
    assert "Enter select" in text
    assert "Enter downloads Qwen/Qwen3-4B-GGUF" in text
    assert "Qwen/Qwen3-4B-GGUF" in text


def test_dashboard_and_studio_mark_selected_row(tmp_path: Path) -> None:
    from rich.console import Console

    from seiso.tui.offline import discover_local_gguf
    from seiso.tui.terminal import draw_frame

    models = discover_local_gguf(tmp_path)
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, color_system=None)
    draw_frame(
        console,
        page="dashboard",
        models=models,
        messages=[],
        model_label="none",
        data_dir=str(tmp_path),
        status="",
        focus="main",
        main_index=1,
        hint="Enter opens Train/Finetune",
    )
    text = buf.getvalue()
    assert "Train/Finetune" in text
    assert "Enter opens Train/Finetune" in text

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, color_system=None)
    draw_frame(
        console,
        page="train",
        models=models,
        messages=[],
        model_label="none",
        data_dir=str(tmp_path),
        status="",
        configs=["configs/example_lora.yaml", "configs/smoke_train_cpu.yaml"],
        focus="main",
        main_index=1,
        hint="Enter runs configs/smoke_train_cpu.yaml",
    )
    text = buf.getvalue()
    assert "example_lora.yaml" in text
    assert "smoke_train_cpu.yaml" in text
    assert "Enter runs configs/smoke_train_cpu.yaml" in text


def test_auth_and_settings_frames_show_identity(tmp_path: Path) -> None:
    from rich.console import Console

    from seiso.tui.browse import auth_choices, page_choices
    from seiso.tui.offline import discover_local_gguf
    from seiso.tui.terminal import draw_frame

    models = discover_local_gguf(tmp_path)
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, color_system=None)
    draw_frame(
        console,
        page="auth",
        models=models,
        messages=[],
        model_label="none",
        data_dir=str(tmp_path),
        status="",
        auth_phase="reveal",
        auth_nsec="nsec1exampleonlyforrender",
        npub="npub1publicidentity",
        choices=auth_choices("reveal"),
        hint="Enter — I saved my recovery key — continue",
        needs_onboarding=False,
    )
    text = buf.getvalue()
    assert "Save your recovery key" in text
    assert "nsec1exampleonlyforrender" in text
    assert "npub1publicidentity" in text
    assert "I saved my recovery key" in text

    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=40, color_system=None)
    draw_frame(
        console,
        page="settings",
        models=models,
        messages=[],
        model_label="none",
        data_dir=str(tmp_path),
        status="",
        npub="npub1publicidentity",
        storage_mode="persistent",
        choices=page_choices("settings"),
        hint="Enter — Sign out",
        main_index=1,
    )
    text = buf.getvalue()
    assert "Public ID" in text
    assert "npub1publicidentity" in text
    assert "Sign out" in text
    assert "Generate new recovery key" in text


def test_hub_table_windows_long_lists(tmp_path: Path) -> None:
    from rich.console import Console

    from seiso.tui.terminal import render_hub

    local = [
        _row(title=f"local-{i}.gguf", source="local", path=tmp_path / f"{i}.gguf") for i in range(3)
    ]
    remote = [_row(title=f"org/model-{i}") for i in range(20)]
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=120, height=24, color_system=None)
    group = render_hub(
        local,
        remote,
        query="qwen",
        selected=18,
        status="",
        error=None,
        window=6,
        focus="main",
    )
    console.print(group)
    text = buf.getvalue()
    assert "above" in text
    assert "more" in text
    assert "18/23" in text or "18" in text
    assert "org/model-14" in text or "model-14" in text
