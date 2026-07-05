"""GitHub CLI tools for the Seiso Code agent — read-only by default, PR create opt-in."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from forge.security.code_policy import scrub_secrets
from forge.services.terminal_policy import scrubbed_subprocess_env
from seiso.security import SecurityError

_GH_TIMEOUT = 45

_READ_SUBCOMMANDS = frozenset({"repo", "pr", "issue", "api"})
_WRITE_SUBCOMMANDS = frozenset({"pr"})


def gh_available() -> bool:
    return shutil.which("gh") is not None


def _run_gh(
    root: Path,
    args: list[str],
    *,
    write: bool = False,
) -> str:
    if not gh_available():
        raise RuntimeError(
            "GitHub CLI (gh) is not installed. Install from https://cli.github.com/ "
            "and run `gh auth login` in this workspace."
        )
    if not args or args[0] != "gh":
        raise ValueError("Internal error: gh argv must start with gh")

    sub = args[1].lower() if len(args) > 1 else ""
    if write:
        if sub not in _WRITE_SUBCOMMANDS:
            raise SecurityError(f"GitHub write subcommand not allowed: {sub}")
    elif sub not in _READ_SUBCOMMANDS:
        raise SecurityError(f"GitHub read subcommand not allowed: {sub}")

    completed = subprocess.run(
        args,
        cwd=root,
        shell=False,
        capture_output=True,
        text=True,
        timeout=_GH_TIMEOUT,
        env=scrubbed_subprocess_env(),
    )
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        detail = stderr or stdout or f"gh exited {completed.returncode}"
        raise RuntimeError(scrub_secrets(detail))
    return scrub_secrets(stdout)


def github_repo_info(root: Path) -> dict[str, Any]:
    raw = _run_gh(
        root,
        [
            "gh",
            "repo",
            "view",
            "--json",
            "nameWithOwner,url,description,defaultBranchRef,isPrivate,viewerPermission",
        ],
    )
    data = json.loads(raw) if raw else {}
    branch = None
    ref = data.get("defaultBranchRef")
    if isinstance(ref, dict):
        branch = ref.get("name")
    return {
        "name_with_owner": data.get("nameWithOwner"),
        "url": data.get("url"),
        "description": data.get("description"),
        "default_branch": branch,
        "is_private": data.get("isPrivate"),
        "viewer_permission": data.get("viewerPermission"),
    }


def github_list_prs(root: Path, *, state: str = "open", limit: int = 10) -> dict[str, Any]:
    state = state if state in {"open", "closed", "merged", "all"} else "open"
    limit = max(1, min(int(limit), 30))
    raw = _run_gh(
        root,
        [
            "gh",
            "pr",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,state,url,headRefName,author",
        ],
    )
    rows = json.loads(raw) if raw else []
    return {"state": state, "count": len(rows), "pull_requests": rows}


def github_list_issues(root: Path, *, state: str = "open", limit: int = 10) -> dict[str, Any]:
    state = state if state in {"open", "closed", "all"} else "open"
    limit = max(1, min(int(limit), 30))
    raw = _run_gh(
        root,
        [
            "gh",
            "issue",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,state,url,author",
        ],
    )
    rows = json.loads(raw) if raw else []
    return {"state": state, "count": len(rows), "issues": rows}


def github_create_pr(
    root: Path,
    *,
    title: str,
    body: str = "",
    draft: bool = True,
    base: str | None = None,
    head: str | None = None,
) -> dict[str, Any]:
    title = (title or "").strip()
    if not title:
        raise ValueError("PR title is required")

    argv = ["gh", "pr", "create", "--title", title]
    if body.strip():
        argv.extend(["--body", body.strip()])
    if draft:
        argv.append("--draft")
    if base:
        argv.extend(["--base", base.strip()])
    if head:
        argv.extend(["--head", head.strip()])

    output = _run_gh(root, argv, write=True)
    return {"title": title, "draft": draft, "url": output.strip(), "output": output.strip()}
