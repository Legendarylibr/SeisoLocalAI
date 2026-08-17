"""Selection model: sidebar + page lists, scroll windows, Enter actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seiso.tui.hub import HubRow, combined_rows
from seiso.tui.pages import DASHBOARD_GOALS, NAV_GROUPS, STUDIO_PAGES


@dataclass(frozen=True, slots=True)
class NavItem:
    id: str
    label: str
    desc: str
    group: str


@dataclass(frozen=True, slots=True)
class Choice:
    kind: str
    label: str
    page: str = ""
    hub_index: int = 0
    config: str = ""
    path: Path | None = None
    repo_id: str = ""
    status: str = ""
    source: str = ""
    detail: str = ""
    action: str = ""


def sidebar_items() -> list[NavItem]:
    items: list[NavItem] = []
    for group in NAV_GROUPS:
        for raw in group["items"]:
            items.append(
                NavItem(
                    id=str(raw["id"]),
                    label=str(raw["label"]),
                    desc=str(raw.get("desc") or ""),
                    group=str(group["label"]),
                )
            )
    items.append(
        NavItem(id="settings", label="Settings", desc="Lite TUI settings", group="Platform")
    )
    return items


def default_main_index(page: str, auth_phase: str = "welcome") -> int:
    """First highlight: Create account on onboarding, otherwise the top row."""
    if page == "auth" and auth_phase == "welcome":
        return 2
    return 0


def index_of_page(page: str) -> int:
    for index, item in enumerate(sidebar_items()):
        if item.id == page:
            return index
    return 0


def clamp_index(index: int, count: int) -> int:
    if count <= 0:
        return 0
    return max(0, min(index, count - 1))


def move_index(index: int, delta: int, count: int) -> int:
    return clamp_index(index + delta, count)


def visible_window(index: int, count: int, height: int) -> tuple[int, int]:
    """Return [start, end) so *index* stays in view."""
    if count <= 0:
        return 0, 0
    height = max(1, height)
    if count <= height:
        return 0, count
    start = min(max(0, index - height // 3), count - height)
    return start, start + height


def goal_page(goal: dict[str, str]) -> str:
    path = str(goal.get("path") or goal.get("id") or "dashboard")
    ident = path.strip("/").split("/", 1)[0]
    if ident == "inference":
        return "chat"
    return ident or "dashboard"


def page_choices(
    page: str,
    *,
    local_hub: list[HubRow] | None = None,
    remote_hub: list[HubRow] | None = None,
    configs: list[str] | None = None,
    knowledge: list[str] | None = None,
    auth_phase: str = "welcome",
    storage_mode: str = "persistent",
) -> list[Choice]:
    local_hub = local_hub or []
    remote_hub = remote_hub or []
    configs = configs or []
    knowledge = knowledge or []
    if page == "dashboard":
        return [
            Choice(
                kind="goto",
                label=str(goal["label"]),
                page=goal_page(goal),
                detail=str(goal.get("desc") or ""),
            )
            for goal in DASHBOARD_GOALS
        ]
    if page == "hub":
        rows = combined_rows(local_hub, remote_hub)
        out: list[Choice] = []
        for index, row in enumerate(rows, start=1):
            out.append(
                Choice(
                    kind="hub",
                    label=row.title,
                    hub_index=index,
                    path=row.path,
                    repo_id=row.repo_id,
                    status=row.status,
                    source=row.source,
                    detail=row.subtitle,
                )
            )
        return out
    if page == "knowledge":
        return [Choice(kind="info", label=name, detail="Local file") for name in knowledge]
    if page == "integrations":
        return [
            Choice(
                kind="action",
                label="Toggle auto-attest",
                action="attest_toggle",
                detail="Publish run digests to allowlisted relays",
            ),
            Choice(
                kind="action",
                label="Toggle loopback relays",
                action="loopback_toggle",
                detail="Allow ws://127.0.0.1 relays",
            ),
            Choice(
                kind="action",
                label="Generate new recovery key",
                action="keygen",
                detail="Same rotate as Settings",
            ),
            Choice(
                kind="action",
                label="Import recovery key",
                action="import_key",
                detail="Paste nsec or ncryptsec + passphrase",
            ),
        ]
    if page in STUDIO_PAGES:
        return [Choice(kind="run", label=name, config=name, page=page) for name in configs]
    if page == "settings":
        return [
            Choice(
                kind="action",
                label="Unload RAM/VRAM",
                action="unload",
                detail="Keeps downloads on disk",
            ),
            Choice(
                kind="action",
                label="Sign out",
                action="logout",
                detail="Keep the account; require the recovery key next time",
            ),
            Choice(
                kind="action",
                label="Generate new recovery key",
                action="keygen",
                detail="Rotates public ID + attest key. Shown once.",
            ),
            Choice(
                kind="action",
                label="Import recovery key",
                action="import_key",
                detail="Paste nsec or an encrypted backup, then Enter",
            ),
            Choice(
                kind="action",
                label="Start a new session",
                action="reset",
                detail="Clears the local account. Type RESET to confirm.",
            ),
        ]
    if page == "auth":
        return auth_choices(auth_phase, storage_mode)
    return []


def resolve_hub_choice(choice: Choice, local_hub: list[HubRow]) -> str:
    """Return 'open' or 'download' for a hub row."""
    if choice.path is not None:
        return "open"
    needle_title = choice.label.lower()
    needle_repo = choice.repo_id.lower()
    for row in local_hub:
        if row.path is None:
            continue
        if row.title.lower() == needle_title or row.repo_id.lower() == needle_repo:
            return "open"
    if choice.status == "ready":
        return "open"
    return "download"


def enter_hint(
    *,
    focus: str,
    page: str,
    nav_index: int,
    main_index: int,
    choices: list[Choice],
    composing: bool,
    compose: str,
) -> str:
    if composing:
        text = compose.strip()
        if not text:
            return "Enter sends   Esc cancels"
        if text.startswith("/"):
            return "Enter runs command   Esc cancels"
        if page == "chat":
            return "Enter sends   Esc cancels"
        if text.isdigit() and page == "hub":
            return f"Enter selects #{text}   Esc cancels"
        return "Enter submits   Esc cancels"
    if focus == "nav":
        items = sidebar_items()
        if not items:
            return "↑↓ scroll   Enter select"
        item = items[clamp_index(nav_index, len(items))]
        return f"Enter opens {item.label}"
    if page == "chat":
        return "type a message   ← Hub to switch models"
    if not choices:
        if page in STUDIO_PAGES:
            return "type /run configs/example_lora.yaml"
        return "↑↓ scroll   ← sidebar"
    choice = choices[clamp_index(main_index, len(choices))]
    if choice.kind == "hub":
        action = "open" if choice.path is not None or choice.status == "ready" else "download"
        return f"Enter {action}s {choice.label}"
    if choice.kind == "goto":
        return f"Enter opens {choice.label}"
    if choice.kind == "run":
        return f"Enter runs {choice.label}"
    if choice.kind == "unload":
        return "Enter unloads weights from RAM/VRAM"
    if choice.kind == "action":
        return f"Enter — {choice.label}"
    if choice.kind == "info":
        return choice.label
    return "Enter select"


def auth_choices(phase: str, storage: str = "persistent") -> list[Choice]:
    if phase == "reveal":
        return [
            Choice(
                kind="action",
                label="I saved my recovery key — continue",
                action="confirm_backup",
                detail="You will not see the key again on this screen",
            ),
            Choice(
                kind="action",
                label="Save encrypted backup",
                action="encrypt_backup",
                detail="Passphrase-locked .txt (no raw recovery key)",
            ),
        ]
    if phase in {"encrypt_pass", "encrypt_confirm", "import_pass", "reset_confirm"}:
        return [
            Choice(
                kind="action",
                label="Submit",
                action="submit_secret",
                detail="Enter after typing",
            )
        ]
    if phase == "login":
        return [
            Choice(
                kind="action",
                label="Sign in",
                action="login",
                detail="Paste your recovery key, then Enter",
            ),
            Choice(
                kind="action",
                label="Lost your recovery key? Start a new session",
                action="reset",
                detail="Clears the local account. Downloaded models stay.",
            ),
        ]
    persist = storage == "persistent"
    return [
        Choice(
            kind="action",
            label="Keep my workspace" + ("  (selected)" if persist else ""),
            action="storage_persistent",
            detail="Chats, models, jobs, and settings survive restarts",
        ),
        Choice(
            kind="action",
            label="Temporary session" + ("" if persist else "  (selected)"),
            action="storage_ephemeral",
            detail="Wiped when you quit. Nothing written for identity.",
        ),
        Choice(
            kind="action",
            label="Create account and continue",
            action="create",
            detail="Generates a private recovery key on this machine",
        ),
        Choice(
            kind="action",
            label="Restore from recovery key",
            action="restore",
            detail="Paste nsec or an encrypted backup, then Enter",
        ),
    ]


def apply_browse_key(
    name: str,
    *,
    focus: str,
    nav_index: int,
    main_index: int,
    nav_count: int,
    main_count: int,
    page_step: int = 8,
) -> tuple[str, int, int]:
    """Move focus/cursors. *name* is a Key.name. Compose is handled by the app."""
    nav_index = clamp_index(nav_index, nav_count)
    main_index = clamp_index(main_index, main_count)
    if name == "left":
        return "nav", nav_index, main_index
    if name == "tab":
        return ("main" if focus == "nav" else "nav"), nav_index, main_index
    if name == "right":
        return "main", nav_index, main_index
    if name == "up":
        if focus == "nav":
            return focus, move_index(nav_index, -1, nav_count), main_index
        if main_count <= 0:
            return "nav", nav_index, main_index
        return focus, nav_index, move_index(main_index, -1, main_count)
    if name == "down":
        if focus == "nav":
            return focus, move_index(nav_index, 1, nav_count), main_index
        if main_count <= 0:
            return "nav", nav_index, main_index
        return focus, nav_index, move_index(main_index, 1, main_count)
    if name == "pageup":
        step = -max(1, page_step)
        if focus == "nav":
            return focus, move_index(nav_index, step, nav_count), main_index
        return focus, nav_index, move_index(main_index, step, main_count)
    if name == "pagedown":
        step = max(1, page_step)
        if focus == "nav":
            return focus, move_index(nav_index, step, nav_count), main_index
        return focus, nav_index, move_index(main_index, step, main_count)
    if name == "home":
        if focus == "nav":
            return focus, 0, main_index
        return focus, nav_index, 0
    if name == "end":
        if focus == "nav":
            return focus, max(0, nav_count - 1), main_index
        return focus, nav_index, max(0, main_count - 1)
    if name == "esc" and focus == "main":
        return "nav", nav_index, main_index
    return focus, nav_index, main_index


def knowledge_names(data_dir: str | Path) -> list[str]:
    root = Path(data_dir) / "knowledge"
    if not root.is_dir():
        return []
    return [p.name for p in sorted(root.iterdir()) if p.name != "."]
