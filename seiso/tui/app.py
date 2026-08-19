"""Interactive Seiso TUI — Forge pages, live Hub, local chat."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import typer
from rich.console import Console

from forge.config import StorageMode
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
    default_main_index,
    enter_hint,
    index_of_page,
    knowledge_names,
    page_choices,
    resolve_hub_choice,
    sidebar_items,
)
from seiso.tui.harnesses import (
    cycle_harness,
    cycle_preset,
    cycle_role_model,
    cycle_route,
    cycle_source,
    detect_all,
    load_settings,
    save_settings,
    set_role_prompt,
    summary_line,
    toggle_role,
    toggle_role_llm,
    toggle_subagents,
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
    storage_choice: StorageMode = (
        auth_snap.storage_mode if auth_snap.storage_mode_configured else "persistent"
    )
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
    main_index = default_main_index(page, auth_phase)
    compose = ""
    compose_cursor = 0
    chat_offset = 0
    knowledge = knowledge_names(data_dir)
    agent_settings = load_settings(data_dir)
    harness_detect = list(detect_all())
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
            agent_settings=agent_settings,
            harness_detect=harness_detect,
            model_labels=[item.label for item in models],
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
            compose_cursor=compose_cursor,
            chat_offset=chat_offset,
            hint=hint(),
            knowledge=knowledge,
            show_input=live,
            npub=(
                account.short_npub
                if account and auth_phase == "ready"
                else (account.npub if account else pending_npub)
            ),
            npub_full=(account.npub if account else pending_npub),
            signed_in=bool(account) and auth_phase == "ready",
            harness_summary=summary_line(agent_settings),
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
        nonlocal nav_index, main_index, chat_offset
        current = str(path)
        current_label = label
        backend = _backend_for(current)
        messages = []
        chat_offset = 0
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
        nonlocal account, page, compose, main_index, focus
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
            main_index = default_main_index(page, auth_phase)
            focus = "main"
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
        elif action == "cycle_harness":
            persist_agent(cycle_harness(agent_settings), "Harness")
        elif action == "cycle_source":
            persist_agent(cycle_source(agent_settings), "Model source")
        elif action == "cycle_preset":
            persist_agent(cycle_preset(agent_settings), "Swarm preset")
        elif action == "cycle_route":
            persist_agent(cycle_route(agent_settings), "Route class")
        elif action == "toggle_subagents":
            persist_agent(toggle_subagents(agent_settings), "Seiso subagents")
        elif action == "refresh_harnesses":
            refresh_harnesses()
        elif action == "swarm_dry_run":
            run_swarm_from_tui(dry_run=True)
        elif action == "swarm_run":
            run_swarm_from_tui(dry_run=False)
        elif action.startswith("toggle_role_"):
            persist_agent(toggle_role(agent_settings, action[len("toggle_role_") :]), "Subagent")
        elif action.startswith("toggle_llm_"):
            persist_agent(
                toggle_role_llm(agent_settings, action[len("toggle_llm_") :]), "Subagent LLM"
            )
        elif action.startswith("cycle_model_"):
            persist_agent(
                cycle_role_model(
                    agent_settings,
                    action[len("cycle_model_") :],
                    [item.label for item in models],
                ),
                "Subagent model",
            )
        elif action.startswith("prompt_"):
            role = action[len("prompt_") :]
            if compose.strip():
                persist_agent(
                    set_role_prompt(agent_settings, role, compose.strip()), f"{role} prompt"
                )
                compose = ""
            else:
                status = f"Type {role} instructions, then Enter."

    def nostr_info_clear() -> None:
        nonlocal nostr_info
        nostr_info = None

    def persist_agent(updated, label: str) -> None:
        nonlocal agent_settings, status
        agent_settings = updated
        save_settings(data_dir, agent_settings)
        status = f"{label}: {summary_line(agent_settings)}"

    def refresh_harnesses() -> None:
        nonlocal harness_detect, status
        harness_detect = list(detect_all())
        found = sum(1 for item in harness_detect if item.installed)
        status = f"{found} harness CLI(s) on PATH"

    def run_swarm_from_tui(*, dry_run: bool, goal: str = "") -> None:
        nonlocal status, messages
        from seiso.agent.adapters import get_adapter
        from seiso.agent.harness import HarnessContext
        from seiso.agent.swarm.run import run_swarm
        from seiso.routing.types import Candidate
        from seiso.tui.harnesses import prepare_endpoint

        text = (goal or compose).strip() or "dry-run"
        endpoint = prepare_endpoint(data_dir, agent_settings, probe=False)
        if not dry_run and not endpoint.url:
            status = "No local endpoint. Load a GGUF, start Ollama, or enable the Smart Router."
            return
        inventory = (
            Candidate(
                model_id=endpoint.model_id or "local-default",
                backend="llamacpp",
                role="code",
                context_tokens=8192,
                vram_mb=4096,
                downloaded=True,
                params_b=7.0,
            ),
        )
        adapter = get_adapter(agent_settings.harness)
        result = run_swarm(
            text,
            agent_settings,
            HarnessContext(
                local_healthy=True,
                inventory=inventory,
                dry_run=dry_run,
                confirm=not dry_run,
            ),
            worker=None if dry_run else adapter.launch,
            workdir=root,
            isolated_dir=data_dir / "agent" / "harnesses" / agent_settings.harness,
            endpoint_url=endpoint.url,
            model_id=endpoint.model_id,
            api_key=endpoint.api_key,
        )
        status = f"Swarm {result.status}" + (
            f" ({result.blocked_reason})" if result.blocked_reason else ""
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"[{agent_settings.harness}] {status} — {text[:120]}",
            }
        )

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
        nonlocal main_index, focus
        account = None
        auth_phase = "welcome"
        page = "auth"
        pending_nsec = ""
        pending_npub = ""
        storage_choice = "persistent"
        main_index = default_main_index(page, auth_phase)
        focus = "main"
        nostr_info_clear()
        status = "Local account cleared. Create a new one or restore a key."

    def activate() -> None:
        nonlocal status, focus, page
        if focus == "nav":
            if gated():
                status = "Sign in first."
                return
            nav_items = sidebar_items()
            if not nav_items:
                return
            goto_page(nav_items[clamp_index(nav_index, len(nav_items))].id)
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
        nonlocal page, status, hub_query, hub_selected, main_index, focus, chat_offset
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
        if stripped.isdigit() and page != "chat":
            picked = int(stripped)
            items = choices()
            if 1 <= picked <= len(items):
                main_index = picked - 1
                if page == "hub":
                    hub_selected = picked
                    activate_hub(picked)
                else:
                    activate()
                return True

        cmd = parse_slash(line)
        if cmd is None:
            if not stripped:
                return True
            if page == "hub":
                hub_query = stripped
                focus = "main"
                nav_index_sync()
                main_index = 0
                hub_selected = 1
                status = f"Searching Hugging Face for {hub_query}…"
                paint()
                refresh_hub()
                return True
            if page != "chat":
                page = "chat"
                focus = "main"
                nav_index_sync()
            if not current:
                status = "Pick a local model from Hub first (↑↓ then Enter)."
                return True
            messages.append({"role": "user", "content": line})
            chat_offset = 0
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
                "/hub /search q /download N /chat /harness /subagents /agent /unload /quit"
            )
        elif cmd.kind == "clear":
            messages.clear()
            chat_offset = 0
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
        elif cmd.kind == "harness":
            from seiso.agent.adapters.types import parse_harness_id

            try:
                agent_settings.harness = parse_harness_id(cmd.arg)
            except ValueError as exc:
                status = str(exc)
            else:
                persist_agent(agent_settings, "Harness")
        elif cmd.kind == "subagents":
            flag = cmd.arg.strip().lower()
            if flag in {"on", "1", "true", "yes"}:
                agent_settings.activate_subagents()
            elif flag in {"off", "0", "false", "no"}:
                agent_settings.deactivate_subagents()
            else:
                toggle_subagents(agent_settings)
            persist_agent(agent_settings, "Seiso subagents")
        elif cmd.kind == "swarm":
            from seiso.agent.swarm.types import parse_preset

            try:
                agent_settings.preset = parse_preset(cmd.arg)
            except ValueError as exc:
                status = str(exc)
            else:
                persist_agent(agent_settings, "Swarm preset")
        elif cmd.kind == "agent":
            if not cmd.arg.strip():
                status = "Usage: /agent <goal>"
            else:
                run_swarm_from_tui(dry_run=False, goal=cmd.arg)
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
        nonlocal compose, compose_cursor, auth_phase, page
        if key.name == "backspace":
            if compose_cursor > 0:
                compose = compose[: compose_cursor - 1] + compose[compose_cursor:]
                compose_cursor -= 1
        elif key.name == "delete":
            compose = compose[:compose_cursor] + compose[compose_cursor + 1 :]
        elif key.name == "ctrl-u":
            compose = compose[compose_cursor:]
            compose_cursor = 0
        elif key.name == "ctrl-w":
            head = compose[:compose_cursor].rstrip()
            cut = head.rfind(" ") + 1 if " " in head else 0
            compose = compose[:cut] + compose[compose_cursor:]
            compose_cursor = cut
        elif key.name == "left":
            compose_cursor = max(0, compose_cursor - 1)
        elif key.name == "right":
            compose_cursor = min(len(compose), compose_cursor + 1)
        elif key.name == "home":
            compose_cursor = 0
        elif key.name == "end":
            compose_cursor = len(compose)
        elif key.name == "esc":
            compose = ""
            compose_cursor = 0
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
            compose = compose[:compose_cursor] + key.char + compose[compose_cursor:]
            compose_cursor += len(key.char)

    def _scroll_chat(delta: int) -> bool:
        nonlocal chat_offset
        visible = [item for item in messages if item.get("role") != "system"]
        if not visible:
            return False
        chat_offset = max(0, min(chat_offset + delta, max(0, len(visible) - 1)))
        return True

    def apply_click(key: Key) -> None:
        nonlocal focus
        # Left ~26 cols is the sidebar; the rest is the page. Wheel already
        # arrives as up/down — this is only a button press.
        if key.x and key.x <= 26:
            focus = "nav"
        elif key.x:
            focus = "main"

    def browse(name: str) -> None:
        nonlocal focus, nav_index, main_index
        if page == "chat" and name in {"up", "down", "pageup", "pagedown"}:
            step = 1 if name in {"up", "down"} else max(4, list_window(console) // 2)
            if name in {"up", "pageup"}:
                if _scroll_chat(step):
                    return
            elif _scroll_chat(-step):
                return
        focus, nav_index, main_index = apply_browse_key(
            name,
            focus=focus,
            nav_index=nav_index,
            main_index=main_index,
            nav_count=len(sidebar_items()),
            main_count=len(choices()),
            page_step=list_window(console),
        )
        clamp_cursors()

    def handle_key(key: Key) -> bool:
        nonlocal compose, compose_cursor, focus, nav_index, main_index, chat_offset
        if key.name in {"ctrl-c", "ctrl-d", "eof"}:
            return False
        if key.name in {"none", "resize", "ctrl-l", "paste-start", "paste-end"}:
            return True
        if key.name == "click":
            apply_click(key)
            return True
        if compose:
            if key.name == "enter":
                line = compose
                compose = ""
                compose_cursor = 0
                if not line.strip():
                    activate()
                    return True
                return handle_line(line)
            if key.name in {
                "backspace",
                "delete",
                "ctrl-u",
                "ctrl-w",
                "esc",
                "char",
                "left",
                "right",
                "home",
                "end",
            }:
                edit_compose(key)
                return True
            if key.name in {"up", "down", "tab", "pageup", "pagedown"}:
                browse(key.name)
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
            browse(key.name)
            return True
        if key.name == "char" and key.char == "?":
            handle_line("/help")
            return True
        if key.name == "char":
            compose = key.char
            compose_cursor = len(compose)
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
