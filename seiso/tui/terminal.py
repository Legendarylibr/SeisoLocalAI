"""Forge-shaped terminal chrome — sidebar, pages, hub tables, chat."""

from __future__ import annotations

import os
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from seiso.tui.browse import (
    clamp_index,
    index_of_page,
    sidebar_items,
    visible_window,
)
from seiso.tui.hub import HubRow, format_downloads
from seiso.tui.offline import LocalModel, format_size
from seiso.tui.pages import DASHBOARD_GOALS, NAV_GROUPS, STUDIO_PAGES

ACCENT = "#5ed4c6"
MUTED = "#5c6b82"
SECONDARY = "#96a3b8"
WARM = "#c9a87c"
DANGER = "#e87088"
TEXT = "#eef1f6"
SURFACE = "#12211f"
SELECTED = f"bold {TEXT} on {SURFACE}"
# Fallback when the tty has not reported a size yet.
SIDEBAR_COLS = 18


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    width: int
    height: int
    side: int
    foot: int
    body: int
    inner: int
    rows: int
    compact: bool
    narrow: bool


def nav_page_ids() -> list[str]:
    return [str(item["id"]) for group in NAV_GROUPS for item in group["items"]]


def _tty_size(console: Console) -> tuple[int, int]:
    """Prefer the live tty ioctl so we do not paint 24+ rows into a 23-row pane."""
    stream = getattr(console, "file", None)
    try:
        is_tty = bool(stream is not None and stream.isatty())
    except Exception:
        is_tty = False
    if is_tty:
        for fd in (1, 0, 2):
            try:
                size = os.get_terminal_size(fd)
            except OSError:
                continue
            if size.columns >= 20 and size.lines >= 10:
                return size.columns, size.lines
    try:
        return int(console.size.width), int(console.size.height)
    except Exception:
        return 80, 24


def frame_metrics(console: Console) -> FrameMetrics:
    """Chrome that fits the live tty — never taller than the screen."""
    width, height = _tty_size(console)
    # Stay one cell inside the margin so a full-width row cannot wrap
    # even if the tty still has autowrap on (detached tmux, some SSH).
    width = max(48, width - 1)
    height = max(12, height)
    console.size = (width, height)
    compact = height < 30
    narrow = width < 90
    side = 16 if narrow else 20
    # 1 help + 1 compose. Never a 3-line input panel — that alone overflowed 24-row ttys.
    foot = 2
    body = max(8, height - foot)
    inner = max(4, body - 2)  # panel top/bottom border, no vertical padding
    # List rows that leave room for panel chrome, a 2-line header, and the footer.
    rows = max(3, min(12, height - 12))
    return FrameMetrics(
        width=width,
        height=height,
        side=side,
        foot=foot,
        body=body,
        inner=inner,
        rows=rows,
        compact=compact,
        narrow=narrow,
    )


def hardware_summary(*, max_len: int = 0) -> str:
    try:
        from seiso.hardware.profile import hardware_profile

        profile = hardware_profile()
    except Exception:
        return "hardware probe unavailable"
    gpus = profile.get("gpus") if isinstance(profile, dict) else None
    gpu0 = gpus[0] if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict) else {}
    name = str(gpu0.get("name") or profile.get("cpu") or "CPU")
    vram = gpu0.get("vram_total_mb")
    ram = profile.get("ram_gb")
    bits = [name]
    if vram:
        bits.append(f"{int(vram) / 1024:.0f} GB VRAM")
    if ram:
        bits.append(f"{ram:g} GB RAM")
    backend = profile.get("preferred_inference_backend") or profile.get("backend")
    if backend:
        bits.append(str(backend))
    text = " · ".join(bits)
    if max_len and len(text) > max_len:
        return text[: max(1, max_len - 1)] + "…"
    return text


def render_sidebar(
    page: str,
    *,
    focus: str = "main",
    nav_index: int | None = None,
    auth_gate: bool = False,
    npub_short: str = "",
    height: int = 24,
    compact: bool = False,
) -> Text:
    items = sidebar_items()
    if nav_index is None:
        nav_index = index_of_page(page)
    nav_index = clamp_index(nav_index, len(items))
    out = Text()
    out.append("SEISO", style=f"bold {ACCENT}")
    if not compact:
        out.append("  local ai", style=MUTED)
    out.append("\n")
    if auth_gate:
        out.append("Sign in\n", style=SECONDARY)
        return out

    # Brand (1) + optional npub (1). Window the rest so the pane never overflows.
    budget = max(4, height - 2 - (1 if npub_short and not compact else 0))
    show_groups = (not compact) and budget >= len(items) + 5
    if show_groups:
        cursor = 0
        for group in NAV_GROUPS:
            out.append(f"{str(group['label']).upper()}\n", style=f"bold {MUTED}")
            for item in group["items"]:
                _nav_line(out, item["label"], item["id"] == page, focus, cursor == nav_index)
                cursor += 1
        _nav_line(
            out,
            "Settings",
            page == "settings",
            focus,
            cursor == nav_index,
            muted=True,
        )
    else:
        start, end = visible_window(nav_index, len(items), budget)
        if start > 0:
            out.append(f"  … {start}\n", style=MUTED)
            budget -= 1
            end = min(len(items), start + budget)
        for index in range(start, end):
            item = items[index]
            _nav_line(
                out,
                item.label,
                item.id == page,
                focus,
                index == nav_index,
                muted=item.id == "settings",
            )
        if end < len(items):
            out.append(f"  … {len(items) - end}\n", style=MUTED)
    if npub_short and not compact:
        out.append(npub_short, style=MUTED)
    return out


def _nav_line(
    out: Text,
    label: str,
    current: bool,
    focus: str,
    pointed: bool,
    *,
    muted: bool = False,
) -> None:
    active = pointed or (focus != "nav" and current)
    mark = "▸ " if active else "  "
    if pointed and focus == "nav":
        style = SELECTED
    elif active:
        style = f"bold {ACCENT}"
    else:
        style = MUTED if muted else SECONDARY
    out.append(f"{mark}{label}\n", style=style)


def render_dashboard(
    models: list[LocalModel],
    *,
    data_dir: str,
    hub_count: int,
    status: str,
    focus: str = "main",
    main_index: int = 0,
    compact: bool = False,
    width: int = 80,
) -> Group:
    hw = hardware_summary(max_len=max(20, width - 36))
    lines = Text()
    if not compact:
        lines.append("Everything stays on this machine unless you publish it.\n", style=SECONDARY)
    lines.append(f"{hw}\n", style=ACCENT)
    main_index = clamp_index(main_index, len(DASHBOARD_GOALS))
    for index, goal in enumerate(DASHBOARD_GOALS):
        pointed = focus == "main" and index == main_index
        mark = "▸ " if pointed else "  "
        style = SELECTED if pointed else f"bold {WARM}"
        lines.append(f"{mark}{goal['label']}\n", style=style)
        if pointed or not compact:
            lines.append(f"    {goal['desc']}\n", style=MUTED)
    lines.append(f"{len(models)} GGUF on disk", style="bold")
    lines.append(f"  ·  {hub_count} Hub results loaded\n", style=SECONDARY)
    if models and not compact:
        smallest = models[0]
        lines.append(
            f"Least RAM local: {smallest.label} ({format_size(smallest.size_bytes)})\n",
            style=SECONDARY,
        )
    if not compact:
        lines.append(f"Data dir  {data_dir}\n", style=MUTED)
    if status:
        lines.append(f"{status}\n", style=WARM)
    return Group(lines)


def render_hub(
    local: list[HubRow],
    remote: list[HubRow],
    *,
    query: str,
    selected: int,
    status: str,
    error: str | None,
    window: int = 14,
    focus: str = "main",
    compact: bool = False,
) -> Group:
    header = Text()
    qlabel = query.strip() or "popular · instruct"
    header.append("Model Hub", style="bold")
    header.append(f"  {qlabel}\n", style=ACCENT)
    if not compact:
        header.append(
            "Live Hugging Face search — not just files already on disk.\n", style=SECONDARY
        )

    table = Table(
        box=None,
        show_header=True,
        header_style=f"bold {ACCENT}",
        pad_edge=False,
        expand=True,
        collapse_padding=True,
    )
    table.add_column("#", style=MUTED, width=3)
    table.add_column(" ", width=1)
    table.add_column("Model", overflow="ellipsis", no_wrap=True)
    table.add_column("Src", width=4)
    table.add_column("↓", justify="right", width=5)
    table.add_column("Size", width=7)
    table.add_column("State", width=6)

    combined = _indexed(local, remote)
    window = max(3, window)
    if not combined:
        table.add_row("—", "", "No local files and Hub returned nothing.", "", "", "", "")
        start, end = 0, 0
    else:
        selected = max(1, min(selected, len(combined)))
        start, end = visible_window(selected - 1, len(combined), window)
        if start > 0:
            table.add_row("…", "", f"{start} above", "", "", "", "", style=MUTED)
        for index, row in combined[start:end]:
            pointed = focus == "main" and index == selected
            mark = "▸" if pointed else " "
            src = "disk" if row.source == "local" else "Hub"
            style = SELECTED if pointed else ""
            table.add_row(
                str(index),
                mark,
                row.title,
                src,
                format_downloads(row.downloads),
                row.size_label,
                row.status,
                style=style,
            )
        if end < len(combined):
            table.add_row("…", "", f"{len(combined) - end} more", "", "", "", "", style=MUTED)

    foot = Text()
    if combined and 1 <= selected <= len(combined):
        current = combined[selected - 1][1]
        extra = f"{current.family} · {current.task}" if (current.family or current.task) else ""
        line = current.subtitle
        if extra and extra not in line:
            line = f"{line}  {extra}" if line else extra
        if line:
            foot.append(f"{line}\n", style=MUTED)
        if len(combined) > window:
            foot.append(f"{selected}/{len(combined)}  ", style=MUTED)
    if error:
        foot.append(f"{error}\n", style=DANGER)
    if status:
        foot.append(f"{status}\n", style=WARM)
    action = "open or download"
    if combined and 1 <= selected <= len(combined):
        action = "open" if combined[selected - 1][1].path is not None else "download"
    if not compact:
        foot.append(f"Enter {action}   type to search   /refresh", style=MUTED)
    return Group(header, table, foot)


def _indexed(local: list[HubRow], remote: list[HubRow]) -> list[tuple[int, HubRow]]:
    return list(enumerate([*local, *remote], start=1))


def render_chat(
    messages: list[dict[str, str]],
    *,
    model_label: str,
    backend: str,
    status: str,
    offset: int = 0,
    window: int = 16,
    compact: bool = False,
) -> Group:
    body = Text()
    visible = [item for item in messages if item.get("role") != "system"]
    if not visible:
        body.append("How can I help you today?\n", style="bold")
        body.append("Start a new chat — conversation stays on this machine.\n", style=SECONDARY)
        body.append(f"Model  {model_label}   ", style=ACCENT)
        body.append(f"engine {backend or 'auto'}\n", style=MUTED)
        if not compact:
            body.append("Type below to send. Weights load on the first message.\n", style=MUTED)
    else:
        window = max(3, window)
        offset = max(0, min(offset, max(0, len(visible) - 1)))
        end = len(visible) - offset
        start = max(0, end - window)
        if start:
            body.append(f"↑ {start} earlier\n", style=MUTED)
        gap = "" if compact else "\n"
        for item in visible[start:end]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "user":
                body.append("You  ", style=f"bold {ACCENT}")
            else:
                body.append("Seiso  ", style=f"bold {WARM}")
            body.append(content.rstrip() + gap + "\n", style=TEXT)
        if offset:
            body.append(f"↓ {offset} newer\n", style=MUTED)
    if status:
        body.append(status + "\n", style=WARM)
    return Group(body)


def render_studio(
    page: str,
    *,
    configs: list[str],
    status: str,
    focus: str = "main",
    main_index: int = 0,
    window: int = 12,
    compact: bool = False,
) -> Text:
    spec = STUDIO_PAGES.get(page)
    if spec is None:
        return Text("Unknown studio page", style=DANGER)
    out = Text()
    out.append(f"{spec['title']}\n", style="bold")
    if not compact:
        out.append(f"{spec['subtitle']}\n", style=SECONDARY)
        out.append(f"{spec['note']}\n", style=MUTED)
    out.append(f"{spec['command']}\n", style=ACCENT)
    if configs:
        main_index = clamp_index(main_index, len(configs))
        start, end = visible_window(main_index, len(configs), max(3, window))
        if start > 0:
            out.append(f"  … {start} above\n", style=MUTED)
        for offset, name in enumerate(configs[start:end]):
            index = start + offset
            pointed = focus == "main" and index == main_index
            mark = "▸" if pointed else " "
            style = SELECTED if pointed else SECONDARY
            out.append(f"  {mark} {index + 1:>2}  {name}\n", style=style)
        if end < len(configs):
            out.append(f"  … {len(configs) - end} more\n", style=MUTED)
    else:
        out.append("No repo configs. /run <file>\n", style=MUTED)
    if status:
        out.append(f"{status}\n", style=WARM)
    return out


def render_knowledge(
    data_dir: str,
    status: str,
    *,
    names: list[str] | None = None,
    missing_dir: bool = False,
    focus: str = "main",
    main_index: int = 0,
    window: int = 16,
    compact: bool = False,
) -> Text:
    from pathlib import Path

    out = Text()
    out.append("Knowledge\n", style="bold")
    if not compact:
        out.append("Local RAG corpus — ingest stays on disk.\n", style=SECONDARY)
    if names is None:
        root = Path(data_dir) / "knowledge"
        missing_dir = not root.is_dir()
        names = [] if missing_dir else [p.name for p in sorted(root.iterdir()) if p.name != "."]
    if missing_dir:
        out.append("  No knowledge/ directory yet.\n", style=MUTED)
    elif names:
        main_index = clamp_index(main_index, len(names))
        start, end = visible_window(main_index, len(names), max(3, window))
        if start > 0:
            out.append(f"  … {start} above\n", style=MUTED)
        for offset, name in enumerate(names[start:end]):
            index = start + offset
            pointed = focus == "main" and index == main_index
            mark = "▸ " if pointed else "  "
            out.append(f"{mark}{name}\n", style=SELECTED if pointed else SECONDARY)
        if end < len(names):
            out.append(f"  … {len(names) - end} more\n", style=MUTED)
    else:
        out.append("  (empty)\n", style=MUTED)
    if status:
        out.append(f"{status}\n", style=WARM)
    return out


def _append_choices(
    out: Text,
    choices: list,
    *,
    focus: str,
    main_index: int,
    window: int | None = None,
    compact: bool = False,
) -> None:
    if not choices:
        return
    main_index = clamp_index(main_index, len(choices))
    if window is not None:
        start, end = visible_window(main_index, len(choices), max(2, window))
    else:
        start, end = 0, len(choices)
    if start > 0:
        out.append(f"  … {start} above\n", style=MUTED)
    for index, item in enumerate(choices[start:end], start=start):
        pointed = focus == "main" and index == main_index
        mark = "▸ " if pointed else "  "
        style = SELECTED if pointed else SECONDARY
        out.append(f"{mark}{item.label}\n", style=style)
        detail = getattr(item, "detail", "")
        if detail and (pointed or not compact):
            out.append(f"    {detail}\n", style=MUTED)
    if end < len(choices):
        out.append(f"  … {len(choices) - end} more\n", style=MUTED)


def render_settings(
    data_dir: str,
    model_label: str,
    status: str,
    *,
    focus: str = "main",
    main_index: int = 0,
    npub: str = "",
    storage_mode: str = "",
    nostr: dict | None = None,
    choices: list | None = None,
    compact: bool = False,
    window: int = 8,
) -> Text:
    out = Text()
    out.append("Settings\n", style="bold")
    if not compact:
        out.append("Local account — same owner identity as Forge.\n", style=SECONDARY)
        out.append(f"Data directory   {data_dir}\n", style=SECONDARY)
    out.append(f"Active model     {model_label}\n", style=SECONDARY)
    if not compact:
        out.append(f"Hardware         {hardware_summary(max_len=48)}\n", style=SECONDARY)
    if storage_mode:
        out.append(f"Storage          {storage_mode}\n", style=SECONDARY)
    if npub:
        out.append(f"Public ID        {npub}\n", style=ACCENT)
    if nostr and not compact:
        saved = "yes" if nostr.get("key_saved") else "no"
        match = nostr.get("identity_match")
        match_s = "yes" if match else ("no" if match is False else "—")
        out.append(f"Signing key      {saved}   identity match {match_s}\n", style=SECONDARY)
    if choices:
        _append_choices(
            out,
            choices,
            focus=focus,
            main_index=main_index,
            window=window,
            compact=compact,
        )
    else:
        pointed = focus == "main"
        mark = "▸ " if pointed else "  "
        out.append(f"{mark}Unload RAM/VRAM\n", style=SELECTED if pointed else SECONDARY)
    if status:
        out.append(f"{status}\n", style=WARM)
    return out


def render_integrations(
    *,
    nostr: dict | None,
    status: str,
    focus: str = "main",
    main_index: int = 0,
    choices: list | None = None,
    compact: bool = False,
    window: int = 8,
) -> Text:
    out = Text()
    out.append("Integrations\n", style="bold")
    if not compact:
        out.append("Nostr provenance — digests only, same as Forge.\n", style=SECONDARY)
    if nostr is None:
        out.append("Sign in to manage Nostr identity and relays.\n", style=MUTED)
    else:
        out.append(f"Public ID        {nostr.get('npub') or '—'}\n", style=ACCENT)
        if not compact:
            allowed = "on" if nostr.get("server_allow_nostr") else "off"
            out.append(f"Server allow     {allowed}\n", style=SECONDARY)
            out.append(
                f"Key on disk      {'yes' if nostr.get('key_saved') else 'no'}\n",
                style=SECONDARY,
            )
        out.append(
            f"Auto-attest      {'on' if nostr.get('auto_attest') else 'off'}\n",
            style=ACCENT if nostr.get("auto_attest") else SECONDARY,
        )
        relays = nostr.get("relays") or []
        if relays and not compact:
            out.append("Relays\n", style="bold")
            for relay in relays[:3]:
                out.append(f"  {relay}\n", style=MUTED)
        elif not relays:
            out.append("Relays           (none)  /relays wss://…\n", style=MUTED)
    if choices:
        _append_choices(
            out,
            choices,
            focus=focus,
            main_index=main_index,
            window=window,
            compact=compact,
        )
    if status:
        out.append(f"{status}\n", style=WARM)
    return out


def render_auth(
    *,
    phase: str,
    storage: str,
    nsec: str = "",
    npub: str = "",
    status: str = "",
    focus: str = "main",
    main_index: int = 0,
    choices: list | None = None,
    needs_onboarding: bool = True,
    compact: bool = False,
    window: int = 6,
) -> Text:
    out = Text()
    if phase == "reveal":
        out.append("Save your recovery key\n", style="bold")
        if not compact:
            out.append(
                "This is the only way to sign back in later. Treat it like a password-manager secret.\n",
                style=SECONDARY,
            )
        out.append("Recovery key (private)\n", style=f"bold {WARM}")
        out.append(f"{nsec}\n", style=ACCENT)
        out.append("Public ID (safe to share)\n", style=MUTED)
        out.append(f"{npub}\n", style=SECONDARY)
        out.append("Write this down now. You will not see it again on this screen.\n", style=WARM)
    elif phase == "encrypt_pass":
        out.append("Encrypted backup\n", style="bold")
        out.append("Type a passphrase (at least 8 characters), then Enter.\n", style=SECONDARY)
    elif phase == "encrypt_confirm":
        out.append("Confirm passphrase\n", style="bold")
        out.append("Type the same passphrase again, then Enter.\n", style=SECONDARY)
    elif phase == "import_pass":
        out.append("Backup passphrase\n", style="bold")
        out.append(
            "This looks like an encrypted backup. Type the passphrase, then Enter.\n",
            style=SECONDARY,
        )
    elif phase == "reset_confirm":
        out.append("Start a new session\n", style="bold")
        out.append(
            "Clears the local account, chats, jobs, and registry. Downloaded models stay.\n",
            style=SECONDARY,
        )
        out.append("Type RESET then Enter to confirm.\n", style=WARM)
    elif needs_onboarding:
        out.append("Create your local account\n", style="bold")
        if not compact:
            out.append(
                "One key for this machine — same as Forge. No email or cloud account.\n",
                style=SECONDARY,
            )
        out.append(f"Storage   {storage}\n", style=ACCENT)
    else:
        out.append("Welcome back\n", style="bold")
        out.append("Paste the recovery key you saved for this workspace.\n", style=SECONDARY)
        if npub:
            out.append(f"Public ID  {npub}\n", style=MUTED)
    if choices and phase not in {"encrypt_pass", "encrypt_confirm", "import_pass", "reset_confirm"}:
        _append_choices(
            out,
            choices,
            focus=focus,
            main_index=main_index,
            window=window,
            compact=compact,
        )
    if not compact:
        out.append("Under the hood: nsec / npub. You do not need a Nostr app.\n", style=MUTED)
    if status:
        out.append(f"{status}\n", style=WARM)
    return out


def _help_line(*, focus: str, page: str = "", width: int = 80) -> Text:
    side_style = ACCENT if focus == "nav" else MUTED
    page_style = ACCENT if focus == "main" else MUTED
    bits: list[tuple[str, str]] = [
        ("↑↓ scroll", MUTED),
        ("  ", MUTED),
        ("← sidebar", side_style),
        ("  ", MUTED),
        ("page →", page_style),
        ("  Enter select", MUTED),
    ]
    if width >= 88:
        bits.insert(2, ("  wheel moves", MUTED))
    if page == "hub" and width >= 78:
        bits.append(("  type to search", MUTED))
    elif page == "auth" and width >= 84:
        bits.append(("  paste a recovery key", MUTED))
    elif page == "chat" and width >= 78:
        bits.append(("  type to chat", MUTED))
    elif width >= 96:
        bits.append(("  type to chat or /command", MUTED))
    if width >= 88:
        bits.append(("  Ctrl+C quit", MUTED))
    text = Text.assemble(*bits)
    text.overflow = "ellipsis"
    text.no_wrap = True
    return text


def input_chrome(page: str, *, secret: bool) -> tuple[str, str]:
    if secret or page == "auth":
        return "Key", "paste recovery key"
    if page == "hub":
        return "Find", "type to search   Enter selects ▸"
    if page == "chat":
        return "You", "type a message"
    return "You", "scroll or type"


def _mask_secret(text: str) -> str:
    if not text:
        return text
    return "•" * min(len(text), 64)


def _compose_line(
    *,
    compose: str,
    hint: str,
    secret: bool = False,
    page: str = "",
    cursor: int | None = None,
) -> Text:
    """Single-line composer — a boxed panel costs 2 extra rows and overflowed small ttys."""
    label, fallback = input_chrome(page, secret=secret)
    shown = _mask_secret(compose) if secret and compose else compose
    if not compose:
        text = Text.assemble((f"{label}  ", MUTED), (hint or fallback, MUTED), (" █", ACCENT))
    else:
        if cursor is None:
            cursor = len(shown)
        cursor = max(0, min(cursor, len(shown)))
        text = Text.assemble(
            (f"{label}  ", ACCENT),
            (shown[:cursor], TEXT),
            ("█", ACCENT),
            (shown[cursor:], TEXT),
        )
    text.overflow = "ellipsis"
    text.no_wrap = True
    return text


def list_window(console: Console, reserved: int = 12) -> int:
    metrics = frame_metrics(console)
    return max(3, metrics.height - max(8, reserved))


def draw_frame(
    console: Console,
    *,
    page: str,
    models: list[LocalModel],
    messages: list[dict[str, str]],
    model_label: str,
    data_dir: str,
    status: str,
    local_hub: list[HubRow] | None = None,
    remote_hub: list[HubRow] | None = None,
    hub_query: str = "",
    hub_selected: int = 1,
    hub_error: str | None = None,
    configs: list[str] | None = None,
    backend: str = "",
    focus: str = "main",
    nav_index: int | None = None,
    main_index: int = 0,
    compose: str = "",
    hint: str = "",
    knowledge: list[str] | None = None,
    show_input: bool = True,
    npub: str = "",
    storage_mode: str = "",
    nostr: dict | None = None,
    choices: list | None = None,
    auth_phase: str = "welcome",
    auth_nsec: str = "",
    secret_input: bool = False,
    needs_onboarding: bool = True,
    npub_full: str = "",
    signed_in: bool = False,
    compose_cursor: int | None = None,
    chat_offset: int = 0,
) -> None:
    local_hub = local_hub or []
    remote_hub = remote_hub or []
    configs = configs or []
    metrics = frame_metrics(console)
    window = metrics.rows
    compact = metrics.compact
    main: RenderableType
    if page == "dashboard":
        title, subtitle = "Dashboard", "Overview"
        main = render_dashboard(
            models,
            data_dir=data_dir,
            hub_count=len(remote_hub),
            status=status,
            focus=focus,
            main_index=main_index,
            compact=compact,
            width=metrics.width,
        )
    elif page == "hub":
        title, subtitle = "Hub", "Models"
        main = render_hub(
            local_hub,
            remote_hub,
            query=hub_query,
            selected=hub_selected,
            status=status,
            error=hub_error,
            window=window,
            focus=focus,
            compact=compact,
        )
    elif page == "chat":
        title, subtitle = "Chat", "Models"
        main = render_chat(
            messages,
            model_label=model_label,
            backend=backend,
            status=status,
            offset=chat_offset,
            window=max(3, window - 1),
            compact=compact,
        )
    elif page == "knowledge":
        title, subtitle = "Knowledge", "Models"
        main = render_knowledge(
            data_dir,
            status,
            names=knowledge,
            focus=focus,
            main_index=main_index,
            window=window,
            compact=compact,
        )
    elif page == "settings":
        title, subtitle = "Settings", "Platform"
        main = render_settings(
            data_dir,
            model_label,
            status,
            focus=focus,
            main_index=main_index,
            npub=npub_full or npub,
            storage_mode=storage_mode,
            nostr=nostr,
            choices=choices,
            compact=compact,
            window=max(3, window - 4),
        )
    elif page == "integrations":
        title, subtitle = "Integrations", "Platform"
        main = render_integrations(
            nostr=nostr,
            status=status,
            focus=focus,
            main_index=main_index,
            choices=choices,
            compact=compact,
            window=max(3, window - 4),
        )
    elif page == "auth":
        title, subtitle = "Account", "Sign in"
        main = render_auth(
            phase=auth_phase,
            storage=storage_mode or "persistent",
            nsec=auth_nsec,
            npub=npub_full or npub,
            status=status,
            focus=focus,
            main_index=main_index,
            choices=choices,
            needs_onboarding=needs_onboarding,
            compact=compact,
            window=max(3, window - 4),
        )
    elif page in STUDIO_PAGES:
        spec = STUDIO_PAGES[page]
        title, subtitle = spec["title"], spec["group"]
        main = render_studio(
            page,
            configs=configs,
            status=status,
            focus=focus,
            main_index=main_index,
            window=window,
            compact=compact,
        )
    else:
        title, subtitle, main = page.title(), "Lite", Text(page)

    auth_gate = page == "auth"
    side_border = ACCENT if focus == "nav" or auth_gate else "grey37"
    main_border = ACCENT if focus == "main" else "grey37"
    sidebar = Panel(
        render_sidebar(
            page,
            focus=focus,
            nav_index=nav_index,
            auth_gate=auth_gate,
            npub_short=npub if signed_in else "",
            height=metrics.inner,
            compact=compact,
        ),
        border_style=side_border,
        padding=(0, 1),
        width=metrics.side,
    )
    ident = f"  ·  {npub}" if npub and not auth_gate and not compact else ""
    model_bit = f"  ·  {model_label}" if model_label and model_label != "none" else ""
    header = Text.assemble(
        (title, "bold"),
        (f"  ·  {subtitle}{model_bit}{ident}", MUTED),
        overflow="ellipsis",
        no_wrap=True,
    )
    main_width = max(20, metrics.width - metrics.side)
    main_panel = Panel(
        main,
        title=header,
        border_style=main_border,
        padding=(0, 1),
        width=main_width,
    )
    body = Table.grid(expand=False, padding=0)
    body.add_column(width=metrics.side, no_wrap=True)
    body.add_column(width=main_width, no_wrap=True)
    body.add_row(sidebar, main_panel)
    console.clear()
    console.print(body, overflow="ellipsis", crop=True, soft_wrap=False)
    console.print(
        _help_line(focus=focus, page=page, width=metrics.width),
        overflow="ellipsis",
        crop=True,
        soft_wrap=False,
    )
    if show_input:
        console.print(
            _compose_line(
                compose=compose,
                hint=hint,
                secret=secret_input,
                page=page,
                cursor=compose_cursor,
            ),
            overflow="ellipsis",
            crop=True,
            soft_wrap=False,
        )
