"""Forge-shaped terminal chrome — sidebar, pages, hub tables, chat."""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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


def render_sidebar(page: str) -> Text:
    out = Text()
    out.append("SEISO", style=f"bold {ACCENT}")
    out.append("  local ai\n", style=MUTED)
    out.append("no browser\n\n", style=f"italic {MUTED}")
    for group in NAV_GROUPS:
        out.append(f"{str(group['label']).upper()}\n", style=f"bold {MUTED}")
        for item in group["items"]:
            active = item["id"] == page
            mark = "▸ " if active else "  "
            style = f"bold {ACCENT}" if active else SECONDARY
            out.append(f"{mark}{item['label']}\n", style=style)
            if item.get("desc") and active:
                out.append(f"    {item['desc']}\n", style=MUTED)
        out.append("\n")
    out.append("Settings  /settings\n", style=MUTED)
    out.append("Offline · local-first", style=MUTED)
    return out


def render_dashboard(
    models: list[LocalModel],
    *,
    data_dir: str,
    hub_count: int,
    status: str,
) -> Group:
    hw = hardware_summary()
    lines = Text()
    lines.append("Dashboard\n", style="bold")
    lines.append("Everything stays on this machine unless you publish it.\n\n", style=SECONDARY)
    lines.append(f"{hw}\n\n", style=ACCENT)
    for goal in DASHBOARD_GOALS:
        lines.append(f"  {goal['label']}\n", style=f"bold {WARM}")
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
    lines.append("\n/hub to browse Hugging Face   /chat to talk   /train for CLI studio", style=MUTED)
    return Group(lines)


def render_hub(
    local: list[HubRow],
    remote: list[HubRow],
    *,
    query: str,
    selected: int,
    status: str,
    error: str | None,
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
    for index, row in combined:
        mark = "▸" if index == selected else " "
        src = "disk" if row.source == "local" else "Hub"
        state = row.status
        style = ACCENT if index == selected else ""
        table.add_row(
            str(index),
            mark,
            row.title,
            src,
            format_downloads(row.downloads),
            row.size_label,
            state,
            style=style,
        )

    foot = Text()
    if combined and 1 <= selected <= len(combined):
        current = combined[selected - 1][1]
        foot.append(f"\n{current.subtitle}\n", style=MUTED)
        if current.family or current.task:
            foot.append(f"{current.family} · {current.task}\n", style=MUTED)
    if error:
        foot.append(f"\n{error}\n", style=DANGER)
    if status:
        foot.append(f"{status}\n", style=WARM)
    foot.append(
        "\n/search qwen   /download N   /open N   /chat   /refresh   j/k via numbers",
        style=MUTED,
    )
    return Group(header, table, foot)


def _indexed(local: list[HubRow], remote: list[HubRow]) -> list[tuple[int, HubRow]]:
    return list(enumerate([*local, *remote], start=1))


def combined_rows(local: list[HubRow], remote: list[HubRow]) -> list[HubRow]:
    return [row for _, row in _indexed(local, remote)]


def render_chat(
    messages: list[dict[str, str]],
    *,
    model_label: str,
    backend: str,
    status: str,
) -> Group:
    body = Text()
    body.append("How can I help you today?\n" if not messages else "", style="bold")
    if not messages:
        body.append("Start a new chat — conversation stays on this machine.\n\n", style=SECONDARY)
        body.append(f"Loaded target  {model_label}\n", style=ACCENT)
        body.append(f"Engine         {backend or 'auto'}\n", style=MUTED)
        body.append("Weights load on the first message.\n", style=MUTED)
    else:
        for item in messages[-16:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "user":
                body.append("You\n", style=f"bold {ACCENT}")
            elif role == "system":
                continue
            else:
                body.append("Seiso\n", style=f"bold {WARM}")
            body.append(content.rstrip() + "\n\n", style="white")
    if status:
        body.append(status + "\n", style=WARM)
    return Group(body)


def render_studio(page: str, *, configs: list[str], status: str) -> Text:
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
        for index, name in enumerate(configs[:12], start=1):
            out.append(f"  {index:>2}  {name}\n", style=SECONDARY)
        out.append("\n/run <config>   e.g. /run configs/example_lora.yaml\n", style=MUTED)
    if status:
        out.append(f"\n{status}\n", style=WARM)
    return out


def render_knowledge(data_dir: str, status: str) -> Text:
    from pathlib import Path

    out = Text()
    out.append("Knowledge\n", style="bold")
    out.append("Local RAG corpus — ingest stays on disk.\n\n", style=SECONDARY)
    root = Path(data_dir) / "knowledge"
    if root.is_dir():
        kids = [p for p in root.iterdir() if p.name != "."]
        if kids:
            for path in kids[:20]:
                out.append(f"  {path.name}\n", style=SECONDARY)
        else:
            out.append("  (empty)\n", style=MUTED)
    else:
        out.append("  No knowledge/ directory yet.\n", style=MUTED)
    out.append("\nFull ingest/retrieve UI remains in Forge if you need the graph studio.\n", style=MUTED)
    if status:
        out.append(f"\n{status}\n", style=WARM)
    return out


def render_settings(data_dir: str, model_label: str, status: str) -> Text:
    out = Text()
    out.append("Settings\n", style="bold")
    out.append("Lite TUI — no account, no remote providers.\n\n", style=SECONDARY)
    out.append(f"Data directory   {data_dir}\n", style=SECONDARY)
    out.append(f"Active model     {model_label}\n", style=SECONDARY)
    out.append(f"Hardware         {hardware_summary()}\n\n", style=SECONDARY)
    out.append("/unload frees RAM/VRAM without deleting downloads.\n", style=MUTED)
    out.append("SEISO_UI=forge start   launch the optional Forge API.\n", style=MUTED)
    if status:
        out.append(f"\n{status}\n", style=WARM)
    return out


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
) -> None:
    local_hub = local_hub or []
    remote_hub = remote_hub or []
    configs = configs or []
    main: RenderableType
    if page == "dashboard":
        title, subtitle = "Dashboard", "Overview"
        main = render_dashboard(
            models, data_dir=data_dir, hub_count=len(remote_hub), status=status
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
        )
    elif page == "chat":
        title, subtitle = "Chat", "Models"
        main = render_chat(messages, model_label=model_label, backend=backend, status=status)
    elif page == "knowledge":
        title, subtitle = "Knowledge", "Models"
        main = render_knowledge(data_dir, status)
    elif page == "settings":
        title, subtitle = "Settings", "Platform"
        main = render_settings(data_dir, model_label, status)
    elif page in STUDIO_PAGES:
        spec = STUDIO_PAGES[page]
        title, subtitle = spec["title"], spec["group"]
        main = render_studio(page, configs=configs, status=status)
    else:
        title, subtitle, main = page.title(), "Lite", Text(page)

    sidebar = Panel(render_sidebar(page), border_style=ACCENT, padding=(0, 1), width=24)
    header = Text.assemble(
        (title, "bold"),
        (f"  ·  {subtitle}  ·  {model_label}  ·  {hardware_summary()}", MUTED),
    )
    main_panel = Panel(main, title=header, border_style="grey37", padding=(1, 2))
    layout = Table.grid(expand=True, padding=0)
    layout.add_column(width=26)
    layout.add_column(ratio=1)
    layout.add_row(sidebar, main_panel)
    console.clear()
    console.print(layout)
    console.print(
        Text(
            "/hub /search <q> /download N /chat /train /unload /quit    ·    terminal UI · no web",
            style=MUTED,
        )
    )
