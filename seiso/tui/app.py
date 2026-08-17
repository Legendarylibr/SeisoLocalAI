"""Interactive Seiso TUI — Forge pages, live Hub, local chat."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import typer
from rich.console import Console

from seiso.tui.auth import (
    BACKUP_FILENAME,
    AuthError,
    AuthUser,
    TuiAuth,
    looks_like_ncryptsec,
    looks_like_nsec,
    resolve_secret,
    write_encrypted_backup,
)
from seiso.tui.browse import (
    Choice,
    apply_browse_key,
    clamp_index,
    enter_hint,
    index_of_page,
    knowledge_names,
    page_choices,
    resolve_hub_choice,
    sidebar_items,
)
from seiso.tui.hub import HubRow, combined_rows, download_hub_repo, search_hub
from seiso.tui.keys import Key, KeyReader, stdin_is_interactive
from seiso.tui.offline import (
    complete_offline_chat,
    discover_local_gguf,
    parse_slash,
    release_offline_weights,
    resolve_model_choice,
)
from seiso.tui.pages import STUDIO_PAGES
from seiso.tui.terminal import draw_frame, list_window, nav_page_ids


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
    interactive: bool | None = None,
    keys: Iterable[Key] | None = None,
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
    auth = TuiAuth(data_dir)
    auth_snap = auth.status()
    account: AuthUser | None = auth_snap.user
    auth_phase = (
        "ready" if account is not None else ("welcome" if auth_snap.needs_onboarding else "login")
    )
    storage_choice = auth_snap.storage_mode if auth_snap.storage_mode_configured else "persistent"
    pending_nsec = ""
    pending_npub = auth_snap.owner_npub or ""
    pass1 = ""
    import_blob = ""
    nostr_info: dict | None = None
    page = "auth" if account is None else ("hub" if not current else "chat")
    status = "" if account is None else "Searching Hugging Face Hub…"
    hub_query = ""
    hub_selected = 1
    local_hub: list[HubRow] = []
    remote_hub: list[HubRow] = []
    hub_error: str | None = None
    configs = _repo_configs(root)
    pages = set(nav_page_ids()) | {"settings"}
    backend = _backend_for(current) if current else "auto"
    focus = "main"
    nav_index = index_of_page(page)
    main_index = 0
    compose = ""
    knowledge = knowledge_names(data_dir)
    scripted = keys is not None
    key_iter = iter(keys or ())
    live = True if scripted else (stdin_is_interactive() if interactive is None else interactive)

    def gated() -> bool:
        return account is None or auth_phase != "ready"

    def refresh_nostr() -> None:
        nonlocal nostr_info
        if account is None:
            nostr_info = None
            return
        try:
            nostr_info = auth.nostr_status(account.id, account.nostr_pubkey)
        except Exception:
            nostr_info = None

    def choices() -> list[Choice]:
        return page_choices(
            page,
            local_hub=local_hub,
            remote_hub=remote_hub,
            configs=configs,
            knowledge=knowledge,
            auth_phase=auth_phase if page == "auth" else "welcome",
            storage_mode=storage_choice,
        )

    def clamp_cursors() -> None:
        nonlocal nav_index, main_index, hub_selected
        nav_index = clamp_index(nav_index, len(sidebar_items()))
        items = choices()
        main_index = clamp_index(main_index, len(items))
        if page == "hub":
            hub_selected = main_index + 1 if items else 1

    def hint() -> str:
        return enter_hint(
            focus=focus,
            page=page,
            nav_index=nav_index,
            main_index=main_index,
            choices=choices(),
            composing=bool(compose),
            compose=compose,
        )

    def refresh_hub() -> None:
        nonlocal local_hub, remote_hub, hub_error, models, status
        local_hub, remote_hub, hub_error = search_hub(hub_query, data_dir=data_dir)
        models = discover_local_gguf(data_dir)
        n_remote = len(remote_hub)
        n_local = len(local_hub)
        status = f"{n_local} on disk · {n_remote} from Hugging Face"
        clamp_cursors()

    def paint() -> None:
        clamp_cursors()
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
            focus=focus,
            nav_index=nav_index,
            main_index=main_index,
            compose=compose,
            hint=hint(),
            knowledge=knowledge,
            show_input=live,
            npub=(account.npub if account and auth_phase == "ready" else pending_npub),
            storage_mode=storage_choice,
            nostr=nostr_info,
            choices=choices() if page in {"auth", "settings", "integrations"} else None,
            auth_phase=auth_phase,
            auth_nsec=pending_nsec,
            secret_input=page == "auth"
            or auth_phase in {"encrypt_pass", "encrypt_confirm", "import_pass", "reset_confirm"}
            or looks_like_nsec(compose)
            or looks_like_ncryptsec(compose),
            needs_onboarding=auth_phase == "welcome",
        )

    def row_at(index: int) -> HubRow | None:
        rows: list[HubRow] = combined_rows(local_hub, remote_hub)
        if 1 <= index <= len(rows):
            chosen: HubRow = rows[index - 1]
            return chosen
        return None

    def adopt_local(path: Path, label: str) -> None:
        nonlocal current, current_label, backend, page, status, messages, focus
        nonlocal nav_index, main_index
        current = str(path)
        current_label = label
        backend = _backend_for(current)
        messages = []
        page = "chat"
        focus = "main"
        nav_index = index_of_page(page)
        main_index = 0
        status = f"Using {label}. Weights load on the next message."

    def goto_page(name: str) -> None:
        nonlocal page, status, focus, nav_index, main_index, knowledge
        if gated() and name != "auth":
            status = "Sign in first — ↑↓ then Enter."
            page = "auth"
            return
        if name not in pages | set(STUDIO_PAGES):
            return
        page = name
        focus = "main"
        nav_index = index_of_page(page)
        main_index = 0
        if page == "hub":
            refresh_hub()
        elif page == "knowledge":
            knowledge = knowledge_names(data_dir)
            status = ""
        elif page in {"settings", "integrations"}:
            refresh_nostr()
            status = ""
        else:
            status = ""
        clamp_cursors()

    def enter_workspace() -> None:
        nonlocal page, status, auth_phase, focus, nav_index, main_index, pending_nsec
        auth_phase = "ready"
        pending_nsec = ""
        focus = "main"
        if current:
            page = "chat"
            nav_index = index_of_page(page)
            status = "Ready. Type to chat, or ↑↓ / Enter to browse."
        else:
            page = "hub"
            nav_index = index_of_page(page)
            status = "Searching Hugging Face Hub…"
            paint()
            refresh_hub()
        main_index = 0
        refresh_nostr()

    def begin_reveal(user: AuthUser, nsec: str) -> None:
        nonlocal account, auth_phase, pending_nsec, pending_npub, page, focus, main_index, status
        account = user
        pending_nsec = nsec
        pending_npub = user.npub
        auth_phase = "reveal"
        page = "auth"
        focus = "main"
        main_index = 0
        status = "Write down the recovery key, then continue."

    def apply_recovery_key(raw: str, passphrase: str | None = None) -> None:
        nonlocal status, import_blob, auth_phase, compose
        text = raw.strip()
        if looks_like_ncryptsec(text) and not passphrase:
            import_blob = text
            auth_phase = "import_pass"
            status = "Type the backup passphrase, then Enter."
            return
        try:
            nsec = resolve_secret(text, passphrase)
        except AuthError as exc:
            status = str(exc)
            return
        try:
            if account is not None and page in {"settings", "integrations"}:
                npub = auth.import_key(account.id, nsec)
                refresh_account()
                status = f"Imported {npub}"
            elif account is None and auth.status().needs_onboarding:
                user, shown = auth.register(generate=False, nsec=nsec, storage_mode=storage_choice)
                account_login(user)
                if shown:
                    begin_reveal(user, shown)
                else:
                    enter_workspace()
                    status = f"Signed in as {user.short_npub}"
            else:
                user = auth.login(nsec)
                account_login(user)
                enter_workspace()
                status = f"Signed in as {user.short_npub}"
        except AuthError as exc:
            status = str(exc)

    def account_login(user: AuthUser) -> None:
        nonlocal account
        account = user

    def refresh_account() -> None:
        nonlocal account
        snap = auth.status()
        account = snap.user or account
        refresh_nostr()

    def save_backup(passphrase: str) -> None:
        nonlocal status, auth_phase, pass1
        if not pending_nsec or not pending_npub:
            status = "No recovery key in memory. Generate or sign in first."
            return
        dest = data_dir / BACKUP_FILENAME
        try:
            write_encrypted_backup(pending_nsec, pending_npub, passphrase, dest)
        except AuthError as exc:
            status = str(exc)
            return
        auth_phase = "reveal"
        pass1 = ""
        status = f"Wrote {dest} (mode 0600). Continue when you have saved the key."

    def run_action(action: str) -> None:
        nonlocal storage_choice, auth_phase, status, pass1, import_blob, pending_nsec, pending_npub
        nonlocal account, page, compose
        if action == "storage_persistent":
            storage_choice = "persistent"
            status = "Workspace will persist on this machine."
        elif action == "storage_ephemeral":
            storage_choice = "ephemeral"
            status = "Temporary session — wiped when you quit."
        elif action == "create":
            try:
                user, nsec = auth.register(generate=True, storage_mode=storage_choice)
            except AuthError as exc:
                status = str(exc)
                return
            if nsec:
                begin_reveal(user, nsec)
            else:
                account_login(user)
                enter_workspace()
        elif action == "restore":
            if compose.strip():
                apply_recovery_key(compose)
                compose = ""
            else:
                status = "Paste your recovery key (nsec or backup), then Enter."
        elif action == "login":
            if compose.strip():
                apply_recovery_key(compose)
                compose = ""
            else:
                status = "Paste your recovery key, then Enter."
        elif action == "confirm_backup":
            enter_workspace()
        elif action == "encrypt_backup":
            auth_phase = "encrypt_pass"
            status = "Type a passphrase (8+ characters), then Enter."
        elif action == "submit_secret":
            if compose.strip():
                handle_auth_secret(compose)
                compose = ""
        elif action == "logout":
            auth.logout()
            account = None
            auth_phase = "login"
            page = "auth"
            pending_nsec = ""
            nostr_info_clear()
            status = "Signed out. Paste your recovery key to continue."
        elif action == "keygen":
            if account is None:
                status = "Sign in first."
                return
            try:
                nsec, npub = auth.rotate_key(account.id)
            except AuthError as exc:
                status = str(exc)
                return
            refresh_account()
            begin_reveal(
                account or AuthUser(id="", npub=npub, nostr_pubkey="", display_name=""), nsec
            )
            pending_npub = npub
            status = "New recovery key — write it down."
        elif action == "import_key":
            if compose.strip():
                apply_recovery_key(compose)
                compose = ""
            else:
                status = "Paste nsec or an encrypted backup, then Enter."
        elif action == "reset":
            auth_phase = "reset_confirm"
            page = "auth"
            status = "Type RESET then Enter to wipe the local account."
        elif action == "unload":
            release_offline_weights()
            status = "Memory freed. Downloads stay on disk."
        elif action == "attest_toggle":
            toggle_pref("auto_attest")
        elif action == "loopback_toggle":
            toggle_pref("allow_loopback")

    def nostr_info_clear() -> None:
        nonlocal nostr_info
        nostr_info = None

    def toggle_pref(field: str) -> None:
        nonlocal status
        if account is None:
            status = "Sign in first."
            return
        refresh_nostr()
        current_prefs = nostr_info or {}
        auto = bool(current_prefs.get("auto_attest"))
        loop = bool(current_prefs.get("allow_loopback"))
        relays = list(current_prefs.get("relays") or [])
        if field == "auto_attest":
            auto = not auto
        else:
            loop = not loop
        try:
            auth.save_prefs(account.id, auto_attest=auto, relays=relays, allow_loopback=loop)
        except AuthError as exc:
            status = str(exc)
            return
        refresh_nostr()
        status = f"{field.replace('_', ' ')} {'on' if (auto if field == 'auto_attest' else loop) else 'off'}"

    def handle_auth_secret(raw: str) -> None:
        nonlocal auth_phase, pass1, status, import_blob
        text = raw.strip()
        if auth_phase == "encrypt_pass":
            if len(text) < 8:
                status = "Passphrase must be at least 8 characters."
                return
            pass1 = text
            auth_phase = "encrypt_confirm"
            status = "Type the same passphrase again."
            return
        if auth_phase == "encrypt_confirm":
            if text != pass1:
                status = "Passphrases do not match. Try again."
                auth_phase = "encrypt_pass"
                pass1 = ""
                return
            save_backup(text)
            return
        if auth_phase == "import_pass":
            apply_recovery_key(import_blob, text)
            import_blob = ""
            return
        if auth_phase == "reset_confirm":
            try:
                auth.reset_session(text)
            except AuthError as exc:
                status = str(exc)
                return
            reset_to_onboarding()
            return
        apply_recovery_key(text)

    def reset_to_onboarding() -> None:
        nonlocal account, auth_phase, page, pending_nsec, pending_npub, storage_choice, status
        account = None
        auth_phase = "welcome"
        page = "auth"
        pending_nsec = ""
        pending_npub = ""
        storage_choice = "persistent"
        nostr_info_clear()
        status = "Local account cleared. Create a new one or restore a key."

    def activate() -> None:
        nonlocal status, focus, page
        if focus == "nav":
            if gated():
                status = "Sign in first."
                return
            items = sidebar_items()
            if not items:
                return
            goto_page(items[clamp_index(nav_index, len(items))].id)
            return
        items = choices()
        if not items:
            if gated():
                return
            if page != "chat" or not current:
                status = "Pick a model on Hub — ↑↓ then Enter."
                goto_page("hub")
            return
        choice = items[clamp_index(main_index, len(items))]
        if choice.kind == "goto":
            goto_page(choice.page)
        elif choice.kind == "run":
            from seiso.tui.jobs import run_cli_job

            status = run_cli_job(choice.config, cwd=root)
        elif choice.kind == "action":
            run_action(choice.action)
        elif choice.kind == "info":
            status = choice.label
        elif choice.kind == "hub":
            activate_hub(choice.hub_index)

    def activate_hub(index: int) -> None:
        nonlocal status
        selected_row = row_at(index)
        if selected_row is None:
            status = f"No row #{index}"
            return
        items = choices()
        choice = next((item for item in items if item.hub_index == index), None)
        action = (
            resolve_hub_choice(choice, local_hub)
            if choice is not None
            else ("open" if selected_row.path is not None else "download")
        )
        if action == "open":
            path = selected_row.path
            label = selected_row.title
            if path is None and choice is not None:
                for row in local_hub:
                    if row.path is None:
                        continue
                    if (
                        row.title.lower() == choice.label.lower()
                        or row.repo_id.lower() == choice.repo_id.lower()
                    ):
                        path, label = row.path, row.title
                        break
            if path is not None:
                adopt_local(path, label)
                return
        download_repo(selected_row.repo_id)

    def download_repo(repo: str) -> None:
        nonlocal status, page
        if not repo:
            status = "Usage: /download N   or   /download org/model"
            return
        page = "hub"
        status = f"Downloading {repo} from Hugging Face…"
        paint()
        try:
            got = download_hub_repo(repo, data_dir=data_dir)
        except KeyboardInterrupt:
            status = "Download cancelled."
        except Exception as exc:
            status = f"Download failed: {exc}"
        else:
            refresh_hub()
            if got.path is not None:
                adopt_local(got.path, got.title)
            else:
                status = f"Downloaded {repo}. Enter to open it."

    def handle_line(line: str) -> bool:
        """Return False to quit."""
        nonlocal page, status, hub_query, hub_selected, main_index, focus
        stripped = line.strip()
        if gated() and not stripped.startswith("/"):
            if not stripped:
                activate()
                return True
            handle_auth_secret(stripped)
            return True
        if account is not None and (looks_like_nsec(stripped) or looks_like_ncryptsec(stripped)):
            apply_recovery_key(stripped)
            return True
        if page == "hub" and stripped.isdigit():
            hub_selected = int(stripped)
            main_index = max(0, hub_selected - 1)
            activate_hub(hub_selected)
            return True

        cmd = parse_slash(line)
        if cmd is None:
            if not stripped:
                return True
            if page != "chat":
                page = "chat"
                focus = "main"
                nav_index_sync()
            if not current:
                status = "Pick a local model from Hub first (↑↓ then Enter)."
                return True
            messages.append({"role": "user", "content": line})
            status = "Generating…"
            paint()
            try:
                reply = complete_offline_chat(current, messages)
            except KeyboardInterrupt:
                status = "Cancelled."
                messages.pop()
                return True
            except Exception as exc:
                status = f"Chat failed: {exc}"
                messages.pop()
                return True
            messages.append({"role": "assistant", "content": reply})
            status = ""
            return True

        if cmd.kind == "quit":
            return False
        if cmd.kind == "help":
            status = (
                "↑↓ scroll  ←→ sidebar/page  Enter select  "
                "/hub /search q /download N /chat /train /unload /logout /quit"
            )
        elif cmd.kind == "clear":
            messages.clear()
            status = "Chat cleared."
        elif cmd.kind == "models":
            goto_page("hub")
        elif cmd.kind == "refresh":
            page = "hub"
            focus = "main"
            nav_index_sync()
            status = "Refreshing Hub…"
            paint()
            refresh_hub()
        elif cmd.kind == "search":
            hub_query = cmd.arg
            page = "hub"
            focus = "main"
            nav_index_sync()
            main_index = 0
            hub_selected = 1
            status = f"Searching Hugging Face for {hub_query or 'popular models'}…"
            paint()
            refresh_hub()
        elif cmd.kind == "download":
            page = "hub"
            target = row_at(int(cmd.arg)) if cmd.arg.isdigit() else None
            repo = target.repo_id if target else cmd.arg
            download_repo(repo)
        elif cmd.kind == "open":
            page = "hub"
            if cmd.arg.isdigit():
                activate_hub(int(cmd.arg))
            else:
                status = "Usage: /open N   or scroll and press Enter"
        elif cmd.kind == "use":
            local_pick, err = resolve_model_choice(cmd.arg, models)
            if err or local_pick is None:
                status = err or "No model"
            else:
                adopt_local(local_pick.path, local_pick.label)
                release_offline_weights()
        elif cmd.kind == "unload":
            release_offline_weights()
            status = "Memory freed. Downloads stay on disk."
        elif cmd.kind == "logout":
            run_action("logout")
        elif cmd.kind == "relays":
            if account is None:
                status = "Sign in first."
            else:
                relays = [
                    part.strip() for part in cmd.arg.replace(",", " ").split() if part.strip()
                ]
                refresh_nostr()
                current_prefs = nostr_info or {}
                try:
                    auth.save_prefs(
                        account.id,
                        auto_attest=bool(current_prefs.get("auto_attest")),
                        relays=relays,
                        allow_loopback=bool(current_prefs.get("allow_loopback")),
                    )
                except AuthError as exc:
                    status = str(exc)
                else:
                    refresh_nostr()
                    status = f"Relays: {', '.join(relays) or '(none)'}"
                    page = "integrations"
        elif cmd.kind == "run":
            from seiso.tui.jobs import run_cli_job

            if page not in STUDIO_PAGES:
                goto_page("train")
            status = run_cli_job(cmd.arg, cwd=root)
        elif cmd.kind == "unknown" and cmd.arg in pages | set(STUDIO_PAGES):
            goto_page(cmd.arg)
        else:
            status = "Unknown command. /help   or ↑↓ then Enter"
        return True

    def nav_index_sync() -> None:
        nonlocal nav_index
        nav_index = index_of_page(page)

    def edit_compose(key: Key) -> None:
        nonlocal compose, auth_phase, page
        if key.name == "backspace":
            compose = compose[:-1]
        elif key.name == "ctrl-u":
            compose = ""
        elif key.name == "ctrl-w":
            compose = compose.rstrip()
            compose = compose[: compose.rfind(" ") + 1] if " " in compose else ""
        elif key.name == "esc":
            compose = ""
            if auth_phase in {"encrypt_pass", "encrypt_confirm"}:
                auth_phase = "reveal"
            elif auth_phase == "import_pass":
                auth_phase = "welcome" if account is None else "ready"
                if auth_phase == "ready":
                    page = "settings"
            elif auth_phase == "reset_confirm" and account is not None:
                auth_phase = "ready"
                page = "settings"
        elif key.name == "char":
            compose += key.char

    def handle_key(key: Key) -> bool:
        nonlocal compose, focus, nav_index, main_index
        if key.name in {"ctrl-c", "ctrl-d", "eof"}:
            return False
        if key.name == "none":
            return True
        if compose:
            if key.name == "enter":
                line = compose
                compose = ""
                if not line.strip():
                    activate()
                    return True
                return handle_line(line)
            if key.name in {"backspace", "ctrl-u", "ctrl-w", "esc", "char"}:
                edit_compose(key)
                return True
            return True
        if key.name == "enter":
            activate()
            return True
        if key.name in {
            "up",
            "down",
            "left",
            "right",
            "tab",
            "pageup",
            "pagedown",
            "home",
            "end",
            "esc",
        }:
            focus, nav_index, main_index = apply_browse_key(
                key.name,
                focus=focus,
                nav_index=nav_index,
                main_index=main_index,
                nav_count=len(sidebar_items()),
                main_count=len(choices()),
                page_step=list_window(console),
            )
            clamp_cursors()
            return True
        if key.name == "char":
            compose = key.char
            return True
        return True

    if account is not None:
        refresh_nostr()
        refresh_hub()
        if current:
            page = "chat"
            focus = "main"
            nav_index = index_of_page(page)
            status = "Ready. Type to chat, or ↑↓ / Enter to browse."
    else:
        focus = "main"
        nav_index = index_of_page("dashboard")
        status = (
            "Create an account or paste a recovery key."
            if auth_phase == "welcome"
            else "Paste your recovery key to unlock this workspace."
        )
    reader = KeyReader()
    try:
        if live and not scripted:
            reader.enable()
        while True:
            paint()
            if live:
                try:
                    key = next(key_iter, Key("eof")) if scripted else reader.read()
                except (EOFError, KeyboardInterrupt, OSError):
                    break
                if not handle_key(key):
                    break
                continue
            try:
                line = typer.prompt("You")
            except (EOFError, KeyboardInterrupt):
                break
            if not handle_line(line):
                break
    finally:
        reader.disable()
        release_offline_weights()
