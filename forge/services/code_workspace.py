"""Local code workspace — file tree, git status, search, terminal."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from forge.security.code_policy import (
    assert_delete_allowed,
    assert_symlink_free,
    assert_write_allowed,
    is_sensitive_path,
    scrub_secrets,
)
from forge.services.terminal_policy import (
    audit_terminal_command,
    scrubbed_subprocess_env,
    validate_terminal_command,
)
from seiso.security import SecurityError, safe_join

logger = logging.getLogger(__name__)

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".next",
        ".turbo",
    }
)
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_TERMINAL_OUTPUT = 256 * 1024
_TERMINAL_TIMEOUT = 120
_SEARCH_PREVIEW = 240
_MAX_TERMINAL_LOG = 12

_terminal_log: dict[str, list[dict[str, Any]]] = {}


def resolve_code_workspace(settings) -> Path:
    raw = os.environ.get("SEISO_CODE_WORKSPACE", "").strip()
    if raw:
        root = Path(raw).expanduser().resolve()
    elif getattr(settings, "code_workspace", None) and str(settings.code_workspace).strip():
        configured = Path(str(settings.code_workspace)).expanduser()
        root = (
            configured.resolve()
            if configured.is_absolute()
            else (Path.cwd() / configured).resolve()
        )
    else:
        root = Path.cwd().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Code workspace not found: {root}")
    return root


def workspace_overlaps_data_dir(root: Path, data_dir: Path) -> bool:
    """True when workspace and Seiso data dir share or nest paths."""
    root = root.resolve()
    data_dir = data_dir.resolve()
    try:
        root.relative_to(data_dir)
        return True
    except ValueError:
        pass
    try:
        data_dir.relative_to(root)
        return True
    except ValueError:
        return root == data_dir


def code_workspace_explicitly_configured(settings) -> bool:
    raw = os.environ.get("SEISO_CODE_WORKSPACE", "").strip()
    if raw:
        return True
    configured = getattr(settings, "code_workspace", None)
    if configured is None:
        return False
    text = str(configured).strip()
    return bool(text and text not in {".", ""})


def warn_workspace_data_overlap(root: Path, data_dir: Path) -> None:
    """Log when the code workspace overlaps the Seiso data directory."""
    if workspace_overlaps_data_dir(root, data_dir):
        logger.warning(
            "Code workspace %s overlaps SEISO data directory %s — "
            "agents may read local secrets or credentials",
            root.resolve(),
            data_dir.resolve(),
        )


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve_path(root: Path, rel_path: str) -> Path:
    rel = rel_path.strip().replace("\\", "/").lstrip("/")
    if not rel:
        raise SecurityError("Path is required")
    parts = [p for p in rel.split("/") if p]
    assert_symlink_free(root, rel)
    return safe_join(root, *parts)


def _language_for(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mapping = {
        "ts": "typescript",
        "tsx": "typescript",
        "js": "javascript",
        "jsx": "javascript",
        "py": "python",
        "rs": "rust",
        "go": "go",
        "css": "css",
        "html": "html",
        "json": "json",
        "md": "markdown",
        "sh": "shell",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
    }
    return mapping.get(ext, "plaintext")


def _recent_store(settings) -> Path:
    path = settings.data_dir / "code_recent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_recent(settings) -> list[str]:
    store = _recent_store(settings)
    if not store.exists():
        return []
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data[:40]]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_recent(settings, paths: list[str]) -> None:
    store = _recent_store(settings)
    store.write_text(json.dumps(paths[:40], indent=0), encoding="utf-8")


def touch_recent(settings, rel_path: str) -> None:
    current = _load_recent(settings)
    ordered = [rel_path] + [p for p in current if p != rel_path]
    _save_recent(settings, ordered)


def workspace_snapshot(root: Path, settings) -> dict[str, Any]:
    branch = None
    head = None
    dirty = False
    changes: list[dict[str, Any]] = []
    remotes: list[dict[str, Any]] = []

    git_dir = root / ".git"
    if git_dir.exists():
        try:
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                or None
            )
            head = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                or None
            )
            status = subprocess.check_output(
                ["git", "status", "--porcelain", "-b"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in status.splitlines():
                if line.startswith("##"):
                    continue
                if len(line) < 4:
                    continue
                index_status = line[0]
                worktree_status = line[1]
                path_part = line[3:].strip()
                if " -> " in path_part:
                    path_part = path_part.split(" -> ", 1)[1]
                status_code = f"{index_status}{worktree_status}".strip()
                changes.append(
                    {
                        "path": path_part.replace("\\", "/"),
                        "status": status_code,
                        "staged": index_status != " " and index_status != "?",
                    }
                )
            dirty = bool(changes)
            remote_lines = subprocess.check_output(
                ["git", "remote", "-v"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()
            seen: set[str] = set()
            for line in remote_lines:
                parts = line.split()
                if len(parts) < 2:
                    continue
                name, url = parts[0], parts[1]
                if name in seen:
                    continue
                seen.add(name)
                provider: Literal["github", "gitlab", "bitbucket", "unknown"] = "unknown"
                lower = url.lower()
                if "github" in lower:
                    provider = "github"
                elif "gitlab" in lower:
                    provider = "gitlab"
                elif "bitbucket" in lower:
                    provider = "bitbucket"
                remotes.append({"name": name, "url": url, "provider": provider})
        except (subprocess.CalledProcessError, OSError):
            pass

    from forge.services.github_agent import gh_available

    github_remotes = [r for r in remotes if r["provider"] == "github"]
    gh_ready = bool(github_remotes) and gh_available()
    tools = [
        {"id": "repo.read", "label": "Repo read", "safe": True, "description": "Browse workspace files"},
        {"id": "code.search", "label": "Code search", "safe": True, "description": "Search the codebase"},
        {"id": "git.diff", "label": "Diff preview", "safe": True, "description": "View git diffs"},
        {"id": "terminal.read", "label": "Terminal read", "safe": True, "description": "Read recent terminal output"},
        {"id": "terminal.run", "label": "Run commands", "safe": False, "description": "Execute shell commands"},
        {"id": "tests.run", "label": "Run tests", "safe": False, "description": "Execute test suites"},
        {"id": "files.write", "label": "Patch files", "safe": False, "description": "Create and edit files"},
    ]
    if gh_ready:
        tools.extend(
            [
                {"id": "github.read", "label": "GitHub read", "safe": True, "description": "Read GitHub metadata"},
                {"id": "github.pr", "label": "Draft PRs", "safe": False, "description": "Draft pull requests"},
            ]
        )
    if settings.web_search_enabled:
        tools.append(
            {
                "id": "web.search",
                "label": "Web search",
                "safe": False,
                "description": "Search public HTTPS sources for docs and context",
            }
        )
    return {
        "repo_root": str(root),
        "branch": branch,
        "head": head,
        "dirty": dirty,
        "remotes": remotes,
        "changes": changes,
        "recent_files": _load_recent(settings),
        "github": {
            "connected": bool(github_remotes),
            "cli_available": gh_available(),
            "ready": gh_ready,
            "provider": "github" if github_remotes else None,
            "remotes": github_remotes,
        },
        "tools": tools,
    }


def list_tree(root: Path, rel_path: str = "") -> list[dict[str, Any]]:
    base = root if not rel_path else _resolve_path(root, rel_path)
    if not base.is_dir():
        raise FileNotFoundError(rel_path or str(base))

    entries: list[dict[str, Any]] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise PermissionError(str(exc)) from exc

    for child in children:
        if child.name in _SKIP_DIRS:
            continue
        rel = _rel(root, child)
        if child.is_dir():
            entries.append({"name": child.name, "path": rel, "type": "dir"})
        elif child.is_file():
            entries.append({"name": child.name, "path": rel, "type": "file"})
    return entries


def read_file(root: Path, settings, rel_path: str) -> dict[str, Any]:
    path = _resolve_path(root, rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    size = path.stat().st_size
    truncated = size > _MAX_FILE_BYTES
    content = path.read_text(encoding="utf-8", errors="replace")
    if truncated:
        content = content[:_MAX_FILE_BYTES]
    touch_recent(settings, rel_path)
    return {
        "path": rel_path.replace("\\", "/"),
        "language": _language_for(path),
        "content": content,
        "size": size,
        "truncated": truncated,
        "sensitive": is_sensitive_path(rel_path),
    }


def write_file(
    root: Path, settings, rel_path: str, content: str, *, destructive_ack: bool = False
) -> dict[str, Any]:
    assert_write_allowed(rel_path, destructive_ack=destructive_ack)
    path = _resolve_path(root, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError("File exceeds maximum size")
    path.write_text(content, encoding="utf-8")
    touch_recent(settings, rel_path)
    return read_file(root, settings, rel_path)


def create_entry(
    root: Path,
    settings,
    rel_path: str,
    kind: Literal["file", "dir"],
    content: str = "",
    *,
    destructive_ack: bool = False,
) -> dict[str, Any]:
    assert_write_allowed(rel_path, destructive_ack=destructive_ack)
    path = _resolve_path(root, rel_path)
    if path.exists():
        raise FileExistsError(rel_path)
    if kind == "dir":
        path.mkdir(parents=True, exist_ok=False)
        return {"path": rel_path.replace("\\", "/"), "type": "dir", "created": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    touch_recent(settings, rel_path)
    return {"path": rel_path.replace("\\", "/"), "type": "file", "created": True}


def delete_entry(root: Path, rel_path: str, *, destructive_ack: bool = False) -> dict[str, Any]:
    assert_delete_allowed(rel_path, destructive_ack=destructive_ack)
    path = _resolve_path(root, rel_path)
    if not path.exists():
        raise FileNotFoundError(rel_path)
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    return {"path": rel_path.replace("\\", "/"), "deleted": True}


def file_diff(root: Path, rel_path: str | None = None) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"path": rel_path, "diff": "", "truncated": False, "line_count": 0}
    cmd = ["git", "diff", "--no-color"]
    if rel_path:
        cmd.append(rel_path)
    try:
        diff = subprocess.check_output(cmd, cwd=root, text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        diff = ""
    truncated = len(diff) > _MAX_FILE_BYTES
    if truncated:
        diff = diff[:_MAX_FILE_BYTES]
    return {
        "path": rel_path,
        "diff": diff,
        "truncated": truncated,
        "line_count": diff.count("\n") + (1 if diff else 0),
    }


def search_code(root: Path, query: str, limit: int = 40) -> dict[str, Any]:
    query = query.strip()
    if not query:
        return {"query": query, "results": [], "count": 0, "engine": "python"}

    results: list[dict[str, Any]] = []
    engine: Literal["rg", "python"] = "python"

    try:
        proc = subprocess.run(
            [
                "rg",
                "--line-number",
                "--column",
                "--no-heading",
                "--color=never",
                "--max-count",
                str(max(limit, 1)),
                query,
                ".",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode in (0, 1):
            engine = "rg"
            for line in proc.stdout.splitlines():
                if len(results) >= limit:
                    break
                match = re.match(r"^(.+?):(\d+):(\d+):(.*)$", line)
                if not match:
                    continue
                rel, ln, col, preview = match.groups()
                rel = rel.replace("\\", "/").lstrip("./")
                if any(part in _SKIP_DIRS for part in rel.split("/")):
                    continue
                results.append(
                    {
                        "path": rel,
                        "line": int(ln),
                        "column": int(col),
                        "preview": preview.strip()[:_SEARCH_PREVIEW],
                    }
                )
            return {"query": query, "results": results, "count": len(results), "engine": engine}
    except (OSError, subprocess.TimeoutExpired):
        pass

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if len(results) >= limit:
                break
            full = Path(dirpath) / name
            rel = _rel(root, full)
            if any(part in _SKIP_DIRS for part in rel.split("/")):
                continue
            try:
                if full.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = full.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    col = pattern.search(line).start() + 1  # type: ignore[union-attr]
                    results.append(
                        {
                            "path": rel,
                            "line": idx,
                            "column": col,
                            "preview": line.strip()[:_SEARCH_PREVIEW],
                        }
                    )
                    if len(results) >= limit:
                        break
    return {"query": query, "results": results, "count": len(results), "engine": engine}


def _record_terminal(root: Path, result: dict[str, Any]) -> None:
    key = str(root.resolve())
    entries = _terminal_log.setdefault(key, [])
    entries.append(
        {
            "command": result.get("command"),
            "cwd": result.get("cwd"),
            "exit_code": result.get("exit_code"),
            "output": result.get("output"),
            "truncated": result.get("truncated"),
        }
    )
    _terminal_log[key] = entries[-_MAX_TERMINAL_LOG:]


def recent_terminal_output(root: Path, *, limit: int = 5) -> dict[str, Any]:
    key = str(root.resolve())
    rows = list(_terminal_log.get(key, []))
    limit = max(1, min(int(limit), _MAX_TERMINAL_LOG))
    return {"count": len(rows), "entries": rows[-limit:]}


def run_terminal(
    root: Path,
    command: str,
    cwd: str | None = None,
    *,
    user_id: str | None = None,
    destructive_ack: bool = False,
) -> dict[str, Any]:
    argv = validate_terminal_command(command, destructive_ack=destructive_ack)
    workdir = _resolve_path(root, cwd) if cwd else root
    if not workdir.is_dir():
        raise NotADirectoryError(cwd or str(root))

    audit_terminal_command(command, user_id=user_id)
    completed = subprocess.run(
        argv,
        cwd=workdir,
        shell=False,
        capture_output=True,
        text=True,
        timeout=_TERMINAL_TIMEOUT,
        env=scrubbed_subprocess_env(),
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = stdout + (f"\n{stderr}" if stderr else "")
    truncated = len(combined) > _MAX_TERMINAL_OUTPUT
    if truncated:
        combined = combined[:_MAX_TERMINAL_OUTPUT] + "\n… (output truncated)"
    combined = scrub_secrets(combined)
    result = {
        "command": command,
        "cwd": _rel(root, workdir) if workdir != root else "",
        "exit_code": completed.returncode,
        "output": combined,
        "truncated": truncated,
    }
    _record_terminal(root, result)
    return result
