"""Interactive Seiso TUI — Forge pages, live Hub, local chat."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from seiso.tui.hub import HubRow, combined_rows, download_hub_repo, search_hub
from seiso.tui.offline import (
    complete_offline_chat,
    discover_local_gguf,
    parse_slash,
    release_offline_weights,
    resolve_model_choice,
)
from seiso.tui.pages import STUDIO_PAGES
from seiso.tui.terminal import draw_frame, nav_page_ids


def _repo_configs(root: Path) -> list[str]:
    configs = root / "configs"
    if not configs.is_dir():
        return []
    names: list[str] = []
    for path in sorted(configs.glob("*.yaml")) + sorted(configs.glob("*.json")):
        names.append(str(path.relative_to(root)))
    return names


def _backend_for(model_path: str) -> str:
    try:
        from seiso.inference.backends import recommend_backend, resolve_local_backend

        try:
            return resolve_local_backend(model_path=model_path, model_format=None, requested=None)
        except Exception:
            return recommend_backend(model_path=model_path, model_format=None)
    except Exception:
        return "auto"


def run_tui(
    *,
    data_dir: Path,
    initial_model: str = "",
    console: Console | None = None,
    repo_root: Path | None = None,
) -> None:
    console = console or Console()
    root = repo_root or Path(__file__).resolve().parents[2]
    models = discover_local_gguf(data_dir)
    current = initial_model
    current_label = "none"
    if current:
        match = next((item for item in models if str(item.path) == current), None)
        current_label = match.label if match else Path(current).name
    elif models:
        current = str(models[0].path)
        current_label = models[0].label

    messages: list[dict[str, str]] = []
    page = "hub" if not current else "chat"
    status = "Searching Hugging Face Hub…"
    hub_query = ""
    hub_selected = 1
    local_hub: list[HubRow] = []
    remote_hub: list[HubRow] = []
    hub_error: str | None = None
    configs = _repo_configs(root)
    pages = set(nav_page_ids()) | {"settings"}
    backend = _backend_for(current) if current else "auto"

    def refresh_hub() -> None:
        nonlocal local_hub, remote_hub, hub_error, models, status
        local_hub, remote_hub, hub_error = search_hub(hub_query, data_dir=data_dir)
        models = discover_local_gguf(data_dir)
        n_remote = len(remote_hub)
        n_local = len(local_hub)
        status = f"{n_local} on disk · {n_remote} from Hugging Face"

    def paint() -> None:
        draw_frame(
            console,
            page=page,
            models=models,
            messages=messages,
            model_label=current_label,
            data_dir=str(data_dir),
            status=status,
            local_hub=local_hub,
            remote_hub=remote_hub,
            hub_query=hub_query,
            hub_selected=hub_selected,
            hub_error=hub_error,
            configs=configs,
            backend=backend,
        )

    def row_at(index: int) -> HubRow | None:
        rows = combined_rows(local_hub, remote_hub)
        if 1 <= index <= len(rows):
            return rows[index - 1]
        return None

    def adopt_local(path: Path, label: str) -> None:
        nonlocal current, current_label, backend, page, status, messages
        current = str(path)
        current_label = label
        backend = _backend_for(current)
        messages = []
        page = "chat"
        status = f"Using {label}. Weights load on the next message."

    refresh_hub()
    if current:
        page = "chat"
        status = "Ready. Type to chat, or /hub to browse Hugging Face."
    try:
        while True:
            paint()
            try:
                line = typer.prompt("You")
            except (EOFError, KeyboardInterrupt):
                break

            stripped = line.strip()
            if page == "hub" and stripped.isdigit():
                hub_selected = int(stripped)
                picked = row_at(hub_selected)
                if picked is None:
                    status = f"No row #{hub_selected}"
                elif picked.path is not None:
                    adopt_local(picked.path, picked.title)
                else:
                    status = f"Selected {picked.repo_id} — /download {hub_selected} or /open {hub_selected}"
                continue

            cmd = parse_slash(line)
            if cmd is None:
                if not stripped:
                    continue
                if page != "chat":
                    page = "chat"
                if not current:
                    status = "Pick a local model from /hub first (or /download N)."
                    continue
                messages.append({"role": "user", "content": line})
                status = "Generating…"
                paint()
                try:
                    reply = complete_offline_chat(current, messages)
                except Exception as exc:
                    status = f"Chat failed: {exc}"
                    messages.pop()
                    continue
                messages.append({"role": "assistant", "content": reply})
                status = ""
                continue

            if cmd.kind == "quit":
                break
            if cmd.kind == "help":
                status = (
                    "/hub /search q /download N /open N /chat /train /models "
                    "/use N /unload /clear /refresh /quit"
                )
            elif cmd.kind == "clear":
                messages.clear()
                status = "Chat cleared."
            elif cmd.kind == "models":
                page = "hub"
                refresh_hub()
            elif cmd.kind == "refresh":
                page = "hub"
                status = "Refreshing Hub…"
                paint()
                refresh_hub()
            elif cmd.kind == "search":
                hub_query = cmd.arg
                page = "hub"
                status = f"Searching Hugging Face for {hub_query or 'popular models'}…"
                paint()
                refresh_hub()
                hub_selected = 1
            elif cmd.kind == "download":
                page = "hub"
                target = row_at(int(cmd.arg)) if cmd.arg.isdigit() else None
                repo = target.repo_id if target else cmd.arg
                if not repo:
                    status = "Usage: /download N   or   /download org/model"
                else:
                    status = f"Downloading {repo} from Hugging Face…"
                    paint()
                    try:
                        got = download_hub_repo(repo, data_dir=data_dir)
                    except Exception as exc:
                        status = f"Download failed: {exc}"
                    else:
                        refresh_hub()
                        if got.path is not None:
                            adopt_local(got.path, got.title)
                        else:
                            status = f"Downloaded {repo}. /hub to open it."
            elif cmd.kind == "open":
                page = "hub"
                target = row_at(int(cmd.arg)) if cmd.arg.isdigit() else None
                if target is None:
                    status = "Usage: /open N"
                elif target.path is not None:
                    adopt_local(target.path, target.title)
                else:
                    status = f"{target.repo_id} is not local yet — /download {cmd.arg}"
            elif cmd.kind == "use":
                picked, err = resolve_model_choice(cmd.arg, models)
                if err or picked is None:
                    status = err or "No model"
                else:
                    adopt_local(picked.path, picked.label)
                    release_offline_weights()
            elif cmd.kind == "unload":
                release_offline_weights()
                status = "Memory freed. Downloads stay on disk."
            elif cmd.kind == "run":
                from seiso.tui.jobs import run_cli_job

                page = page if page in STUDIO_PAGES else "train"
                status = run_cli_job(cmd.arg, cwd=root)
            elif cmd.kind == "unknown" and cmd.arg in pages | set(STUDIO_PAGES):
                page = cmd.arg
                if page == "hub":
                    refresh_hub()
                else:
                    status = ""
            else:
                status = "Unknown command. /help"
    finally:
        release_offline_weights()
