"""Forge-shaped terminal chrome — sidebar, pages, hub tables, chat."""

from __future__ import annotations

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


def nav_page_ids() -> list[str]:
    return [str(item["id"]) for group in NAV_GROUPS for item in group["items"]]


def hardware_summary() -> str:
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
    return " · ".join(bits)


def render_sidebar(
    page: str,
    *,
    focus: str = "main",
    nav_index: int | None = None,
    auth_gate: bool = False,
    npub_short: str = "",
) -> Text:
    if nav_index is None:
        nav_index = index_of_page(page)
    nav_index = clamp_index(nav_index, len(sidebar_items()))
    out = Text()
    out.append("SEISO", style=f"bold {ACCENT}")
    out.append("  local ai\n", style=MUTED)
    if auth_gate:
        out.append("\nSign in to continue.\n\n", style=SECONDARY)
        out.append("One local account.\nSame key as Forge.", style=MUTED)
        return out
    out.append("\n")
    cursor = 0
    for group in NAV_GROUPS:
        out.append(f"{str(group['label']).upper()}\n", style=f"bold {MUTED}")
        for item in group["items"]:
            pointed = (focus == "nav" and cursor == nav_index) or (
                focus != "nav" and item["id"] == page
            )
            mark = "▸ " if pointed else "  "
            style = f"bold {ACCENT}" if pointed else SECONDARY
            out.append(f"{mark}{item['label']}\n", style=style)
            if item.get("desc") and pointed:
                out.append(f"    {item['desc']}\n", style=MUTED)
            cursor += 1
        out.append("\n")
    settings_pointed = (focus == "nav" and cursor == nav_index) or (
        focus != "nav" and page == "settings"
    )
    out.append(
        "▸ Settings\n" if settings_pointed else "  Settings\n",
        style=f"bold {ACCENT}" if settings_pointed else MUTED,
    )
    if npub_short:
        out.append(f"\n{npub_short}\n", style=MUTED)
    out.append("local-first", style=MUTED)
    return out


def render_dashboard(
    models: list[LocalModel],
    *,
    data_dir: str,
    hub_count: int,
    status: str,
    focus: str = "main",
    main_index: int = 0,
) -> Group:
    hw = hardware_summary()
    lines = Text()
    lines.append("Everything stays on this machine unless you publish it.\n\n", style=SECONDARY)
    lines.append(f"{hw}\n\n", style=ACCENT)
    main_index = clamp_index(main_index, len(DASHBOARD_GOALS))
    for index, goal in enumerate(DASHBOARD_GOALS):
        pointed = focus == "main" and index == main_index
        mark = "▸ " if pointed else "  "
        style = f"bold {ACCENT}" if pointed else f"bold {WARM}"
        lines.append(f"{mark}{goal['label']}\n", style=style)
        lines.append(f"    {goal['desc']}\n", style=MUTED)
    lines.append("\n")
    lines.append(f"{len(models)} GGUF on disk", style="bold")
    lines.append(f"  ·  {hub_count} Hub results loaded\n", style=SECONDARY)
    if models:
        smallest = models[0]
        lines.append(
            f"Least RAM local: {smallest.label} ({format_size(smallest.size_bytes)})\n",
            style=SECONDARY,
        )
    lines.append(f"Data dir  {data_dir}\n", style=MUTED)
    if status:
        lines.append(f"\n{status}\n", style=WARM)
    lines.append("\n↑↓ scroll   Enter open   ← sidebar", style=MUTED)
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
) -> Group:
    header = Text()
    qlabel = query.strip() or "popular · instruct"
    header.append("Model Hub\n", style="bold")
    header.append("Live Hugging Face search — not just files already on disk.\n", style=SECONDARY)
    header.append(f"Query  {qlabel}\n\n", style=ACCENT)

    table = Table(
        box=None,
        show_header=True,
        header_style=f"bold {ACCENT}",
        pad_edge=False,
        expand=True,
    )
    table.add_column("#", style=MUTED, width=3)
    table.add_column(" ", width=1)
    table.add_column("Model", overflow="ellipsis")
    table.add_column("Src", width=5)
    table.add_column("↓", justify="right", width=6)
    table.add_column("Size", width=8)
    table.add_column("State", width=8)

    combined = _indexed(local, remote)
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
            style = ACCENT if pointed else ""
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
        foot.append(f"\n{current.subtitle}\n", style=MUTED)
        if current.family or current.task:
            foot.append(f"{current.family} · {current.task}\n", style=MUTED)
        if len(combined) > window:
            foot.append(f"{selected}/{len(combined)}\n", style=MUTED)
    if error:
        foot.append(f"\n{error}\n", style=DANGER)
    if status:
        foot.append(f"{status}\n", style=WARM)
    action = "open or download"
    if combined and 1 <= selected <= len(combined):
        action = "open" if combined[selected - 1][1].path is not None else "download"
    foot.append(f"\n↑↓ scroll   Enter {action}   type to search   /refresh", style=MUTED)
    return Group(header, table, foot)


def _indexed(local: list[HubRow], remote: list[HubRow]) -> list[tuple[int, HubRow]]:
    return list(enumerate([*local, *remote], start=1))


def render_chat(
    messages: list[dict[str, str]],
    *,
    model_label: str,
    backend: str,
    status: str,
) -> Group:
    body = Text()
    visible = [item for item in messages if item.get("role") != "system"]
    hidden = max(0, len(visible) - 16)
    body.append("How can I help you today?\n" if not visible else "", style="bold")
    if not visible:
        body.append("Start a new chat — conversation stays on this machine.\n\n", style=SECONDARY)
        body.append(f"Model    {model_label}\n", style=ACCENT)
        body.append(f"Engine   {backend or 'auto'}\n", style=MUTED)
        body.append("Type below to send. Weights load on the first message.\n", style=MUTED)
    else:
        if hidden:
            body.append(f"↑ {hidden} earlier messages\n\n", style=MUTED)
        for item in visible[-16:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "user":
                body.append("You\n", style=f"bold {ACCENT}")
            else:
                body.append("Seiso\n", style=f"bold {WARM}")
            body.append(content.rstrip() + "\n\n", style="white")
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
) -> Text:
    spec = STUDIO_PAGES.get(page)
    if spec is None:
        return Text("Unknown studio page", style=DANGER)
    out = Text()
    out.append(f"{spec['title']}\n", style="bold")
    out.append(f"{spec['subtitle']}\n\n", style=SECONDARY)
    out.append(f"{spec['note']}\n\n", style=MUTED)
    out.append("Command\n", style=f"bold {ACCENT}")
    out.append(f"  {spec['command']}\n\n", style=ACCENT)
    if configs:
        out.append("Repo configs\n", style="bold")
        main_index = clamp_index(main_index, len(configs))
        start, end = visible_window(main_index, len(configs), window)
        if start > 0:
            out.append(f"  … {start} above\n", style=MUTED)
        for offset, name in enumerate(configs[start:end]):
            index = start + offset
            pointed = focus == "main" and index == main_index
            mark = "▸" if pointed else " "
            style = ACCENT if pointed else SECONDARY
            out.append(f"  {mark} {index + 1:>2}  {name}\n", style=style)
        if end < len(configs):
            out.append(f"  … {len(configs) - end} more\n", style=MUTED)
        out.append("\n↑↓ scroll   Enter run   or /run configs/example_lora.yaml\n", style=MUTED)
    else:
        out.append("\nNo repo configs found. Type /run <file>\n", style=MUTED)
    if status:
        out.append(f"\n{status}\n", style=WARM)
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
) -> Text:
    from pathlib import Path

    out = Text()
    out.append("Knowledge\n", style="bold")
    out.append("Local RAG corpus — ingest stays on disk.\n\n", style=SECONDARY)
    if names is None:
        root = Path(data_dir) / "knowledge"
        missing_dir = not root.is_dir()
        names = [] if missing_dir else [p.name for p in sorted(root.iterdir()) if p.name != "."]
    if missing_dir:
        out.append("  No knowledge/ directory yet.\n", style=MUTED)
    elif names:
        main_index = clamp_index(main_index, len(names))
        start, end = visible_window(main_index, len(names), window)
        if start > 0:
            out.append(f"  … {start} above\n", style=MUTED)
        for offset, name in enumerate(names[start:end]):
            index = start + offset
            pointed = focus == "main" and index == main_index
            mark = "▸ " if pointed else "  "
            out.append(f"{mark}{name}\n", style=ACCENT if pointed else SECONDARY)
        if end < len(names):
            out.append(f"  … {len(names) - end} more\n", style=MUTED)
    else:
        out.append("  (empty)\n", style=MUTED)
    out.append(
        "\nFull ingest/retrieve UI remains in Forge if you need the graph studio.\n", style=MUTED
    )
    if status:
        out.append(f"\n{status}\n", style=WARM)
    return out


def _append_choices(out: Text, choices: list, *, focus: str, main_index: int) -> None:
    if not choices:
        return
    main_index = clamp_index(main_index, len(choices))
    for index, item in enumerate(choices):
        pointed = focus == "main" and index == main_index
        mark = "▸ " if pointed else "  "
        style = f"bold {ACCENT}" if pointed else SECONDARY
        out.append(f"{mark}{item.label}\n", style=style)
        if getattr(item, "detail", ""):
            out.append(f"    {item.detail}\n", style=MUTED)


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
) -> Text:
    out = Text()
    out.append("Settings\n", style="bold")
    out.append("Local account — same owner identity as Forge.\n\n", style=SECONDARY)
    out.append(f"Data directory   {data_dir}\n", style=SECONDARY)
    out.append(f"Active model     {model_label}\n", style=SECONDARY)
    out.append(f"Hardware         {hardware_summary()}\n", style=SECONDARY)
    if storage_mode:
        out.append(f"Storage          {storage_mode}\n", style=SECONDARY)
    if npub:
        out.append("Public ID\n", style=MUTED)
        out.append(f"  {npub}\n", style=ACCENT)
    if nostr:
        saved = "yes" if nostr.get("key_saved") else "no"
        match = nostr.get("identity_match")
        match_s = "yes" if match else ("no" if match is False else "—")
        out.append(f"Signing key      {saved}   identity match {match_s}\n", style=SECONDARY)
    out.append("\n")
    if choices:
        _append_choices(out, choices, focus=focus, main_index=main_index)
    else:
        pointed = focus == "main"
        mark = "▸ " if pointed else "  "
        out.append(f"{mark}Unload RAM/VRAM\n", style=f"bold {ACCENT}" if pointed else SECONDARY)
        out.append("    Keeps downloads on disk. Enter to unload.\n", style=MUTED)
    out.append("\nSEISO_UI=forge start   launch the optional Forge API.\n", style=MUTED)
    if status:
        out.append(f"\n{status}\n", style=WARM)
    return out


def render_integrations(
    *,
    nostr: dict | None,
    status: str,
    focus: str = "main",
    main_index: int = 0,
    choices: list | None = None,
) -> Text:
    out = Text()
    out.append("Integrations\n", style="bold")
    out.append("Nostr provenance — digests only, same as Forge.\n\n", style=SECONDARY)
    if nostr is None:
        out.append("Sign in to manage Nostr identity and relays.\n", style=MUTED)
    else:
        allowed = "on" if nostr.get("server_allow_nostr") else "off"
        out.append(f"Server allow     {allowed}\n", style=SECONDARY)
        out.append(f"Public ID        {nostr.get('npub') or '—'}\n", style=ACCENT)
        out.append(
            f"Key on disk      {'yes' if nostr.get('key_saved') else 'no'}\n", style=SECONDARY
        )
        match = nostr.get("identity_match")
        out.append(
            f"Identity match   {'yes' if match else ('no' if match is False else '—')}\n",
            style=SECONDARY,
        )
        out.append(
            f"Auto-attest      {'on' if nostr.get('auto_attest') else 'off'}\n",
            style=ACCENT if nostr.get("auto_attest") else SECONDARY,
        )
        out.append(
            f"Loopback relays  {'on' if nostr.get('allow_loopback') else 'off'}\n",
            style=SECONDARY,
        )
        relays = nostr.get("relays") or []
        if relays:
            out.append("Relays\n", style="bold")
            for relay in relays:
                out.append(f"  {relay}\n", style=MUTED)
        else:
            out.append("Relays           (none)  /relays wss://…\n", style=MUTED)
    out.append("\n")
    if choices:
        _append_choices(out, choices, focus=focus, main_index=main_index)
    if status:
        out.append(f"\n{status}\n", style=WARM)
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
) -> Text:
    out = Text()
    if phase == "reveal":
        out.append("Save your recovery key\n", style="bold")
        out.append(
            "This is the only way to sign back in later. Treat it like a password-manager secret.\n\n",
            style=SECONDARY,
        )
        out.append("Recovery key (private)\n", style=f"bold {WARM}")
        out.append(f"{nsec}\n\n", style=ACCENT)
        out.append("Public ID (safe to share)\n", style=MUTED)
        out.append(f"{npub}\n\n", style=SECONDARY)
        out.append(
            "Write this down now. You will not see it again on this screen.\n",
            style=WARM,
        )
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
            "This clears the local account, chats, jobs, and registry. Downloaded models stay.\n\n",
            style=SECONDARY,
        )
        out.append("Type RESET then Enter to confirm.\n", style=WARM)
    elif needs_onboarding:
        out.append("Create your local account\n", style="bold")
        out.append(
            "One key for this machine — same as Forge. No email or cloud account.\n\n",
            style=SECONDARY,
        )
        out.append(f"Storage   {storage}\n\n", style=ACCENT)
    else:
        out.append("Welcome back\n", style="bold")
        out.append("Paste the recovery key you saved for this workspace.\n", style=SECONDARY)
        if npub:
            out.append(f"\nPublic ID  {npub}\n", style=MUTED)
        out.append("\n")
    if choices and phase not in {"encrypt_pass", "encrypt_confirm", "import_pass", "reset_confirm"}:
        out.append("\n")
        _append_choices(out, choices, focus=focus, main_index=main_index)
    out.append(
        "\nUnder the hood: nsec / npub. You do not need a Nostr app.\n",
        style=MUTED,
    )
    if status:
        out.append(f"\n{status}\n", style=WARM)
    return out


def _help_line(*, focus: str, page: str = "") -> Text:
    side_style = ACCENT if focus == "nav" else MUTED
    page_style = ACCENT if focus == "main" else MUTED
    extra = "   type to chat or /command   Ctrl+C quit"
    if page == "hub":
        extra = "   type to search   Ctrl+C quit"
    elif page == "auth":
        extra = "   paste a recovery key   Ctrl+C quit"
    elif page == "chat":
        extra = "   type to chat   Ctrl+C quit"
    return Text.assemble(
        ("↑↓ scroll   ", MUTED),
        ("← sidebar", side_style),
        ("   ", MUTED),
        ("page →", page_style),
        ("   Enter select", MUTED),
        (extra, MUTED),
    )


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


def _compose_line(*, compose: str, hint: str, secret: bool = False, page: str = "") -> Text:
    label, fallback = input_chrome(page, secret=secret)
    label_style = ACCENT if compose else MUTED
    shown = _mask_secret(compose) if secret and compose else compose
    if compose:
        return Text.assemble((f"{label}  ", label_style), (shown, "white"), ("█", ACCENT))
    return Text.assemble((f"{label}  ", MUTED), (hint or fallback, MUTED))


def list_window(console: Console, reserved: int = 18) -> int:
    try:
        height = int(console.size.height)
    except Exception:
        height = 40
    return max(6, height - reserved)


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
) -> None:
    local_hub = local_hub or []
    remote_hub = remote_hub or []
    configs = configs or []
    window = list_window(console)
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
        )
    elif page == "chat":
        title, subtitle = "Chat", "Models"
        main = render_chat(messages, model_label=model_label, backend=backend, status=status)
    elif page == "knowledge":
        title, subtitle = "Knowledge", "Models"
        main = render_knowledge(
            data_dir,
            status,
            names=knowledge,
            focus=focus,
            main_index=main_index,
            window=window,
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
        )
    elif page == "integrations":
        title, subtitle = "Integrations", "Platform"
        main = render_integrations(
            nostr=nostr,
            status=status,
            focus=focus,
            main_index=main_index,
            choices=choices,
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
        ),
        border_style=side_border,
        padding=(0, 1),
        width=24,
    )
    ident = f"  ·  {npub}" if npub and not auth_gate else ""
    model_bit = f"  ·  {model_label}" if model_label and model_label != "none" else ""
    header = Text.assemble(
        (title, "bold"),
        (f"  ·  {subtitle}{model_bit}{ident}", MUTED),
    )
    main_panel = Panel(main, title=header, border_style=main_border, padding=(1, 2))
    layout = Table.grid(expand=True, padding=0)
    layout.add_column(width=26)
    layout.add_column(ratio=1)
    layout.add_row(sidebar, main_panel)
    console.clear()
    console.print(layout)
    console.print(_help_line(focus=focus, page=page))
    if show_input:
        console.print(_compose_line(compose=compose, hint=hint, secret=secret_input, page=page))
