"""Lean offline chat helpers: local GGUF discovery and slash commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from seiso.io.files import iter_matching_files

_SCAN_ROOTS = ("models", "exports", "hf_cache")
_SKIP_NAME_MARKERS = ("mmproj", ".incomplete")
# Hugging Face cache blobs are named by digest (sha256 hex, optional prefix).
_DIGEST_NAME = re.compile(
    r"^(?:sha256[-_])?[0-9a-f]{64}(?:\.gguf)?$|^[0-9a-f]{40}(?:\.gguf)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LocalModel:
    path: Path
    size_bytes: int
    label: str


@dataclass(frozen=True, slots=True)
class SlashCommand:
    kind: str
    arg: str = ""


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    value = float(size_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1024.0
        if value < 1024.0:
            if unit in {"GB", "TB"}:
                return f"{value:.2f} {unit}"
            return f"{value:.1f} {unit}"
    return f"{value:.2f} PB"


def _should_skip(path: Path) -> bool:
    name = path.name.lower()
    if any(marker in name for marker in _SKIP_NAME_MARKERS):
        return True
    # Content-addressed HF blobs have no filename — use snapshot/inventory links.
    return any(part == "blobs" for part in path.parts)


def looks_like_digest_name(name: str) -> bool:
    """True when *name* is a SHA-256 / git digest, not a model filename."""
    return bool(_DIGEST_NAME.fullmatch(Path(name).name))


def _repo_id_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("models--"):
            return part.removeprefix("models--").replace("--", "/") or None
        if (
            "--" in part
            and part not in {"hf_cache", "hf_home"}
            and not looks_like_digest_name(part)
        ):
            # Inventory dirs look like ``Qwen--Qwen3-4B``.
            owner, _, name = part.partition("--")
            if owner and name and "/" not in part:
                return f"{owner}/{name.replace('--', '/')}"
    return None


def _snapshot_gguf_name(path: Path) -> str | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "snapshots" or index + 2 >= len(parts):
            continue
        name = parts[-1]
        if name.lower().endswith(".gguf") and not looks_like_digest_name(name):
            return name
    return None


def local_model_label(*paths: Path) -> str:
    """Human-readable GGUF label. Never a raw Hugging Face blob SHA-256."""
    gguf_names: list[str] = []
    repos: list[str] = []
    fallback = "model"
    for path in paths:
        fallback = path.name or fallback
        name = path.name
        if name.lower().endswith(".gguf") and not looks_like_digest_name(name):
            gguf_names.append(name)
        snap = _snapshot_gguf_name(path)
        if snap:
            gguf_names.append(snap)
        repo = _repo_id_from_path(path)
        if repo:
            repos.append(repo)
    if gguf_names:
        return gguf_names[0]
    if repos:
        return repos[0]
    return fallback


def _prefer_local_model(existing: LocalModel | None, candidate: LocalModel) -> LocalModel:
    if existing is None:
        return candidate
    existing_hash = looks_like_digest_name(existing.label)
    candidate_hash = looks_like_digest_name(candidate.label)
    if existing_hash and not candidate_hash:
        return candidate
    if candidate_hash and not existing_hash:
        return existing
    existing_named = existing.path.suffix.lower() == ".gguf" and not looks_like_digest_name(
        existing.path.name
    )
    candidate_named = candidate.path.suffix.lower() == ".gguf" and not looks_like_digest_name(
        candidate.path.name
    )
    if candidate_named and not existing_named:
        return candidate
    return existing


def _file_key(path: Path) -> tuple[int, int] | str:
    try:
        st = path.stat()
        return (st.st_dev, st.st_ino)
    except OSError:
        return str(path)


def discover_local_gguf(data_dir: Path) -> list[LocalModel]:
    """Return unique local GGUF files, smallest first (least RAM)."""
    found: dict[tuple[int, int] | str, LocalModel] = {}
    for name in _SCAN_ROOTS:
        root = data_dir / name
        if not root.exists():
            continue
        for path in iter_matching_files(root, suffixes={".gguf"}):
            if _should_skip(path):
                continue
            try:
                resolved = path.resolve()
                if resolved.is_dir():
                    continue
                size = resolved.stat().st_size
            except OSError:
                continue
            if size <= 0:
                continue
            named = path if not looks_like_digest_name(path.name) else resolved
            candidate = LocalModel(
                path=named,
                size_bytes=size,
                label=local_model_label(path, resolved),
            )
            key = _file_key(resolved)
            found[key] = _prefer_local_model(found.get(key), candidate)
    return sorted(found.values(), key=lambda item: (item.size_bytes, item.label.lower()))


def pick_default_model(models: list[LocalModel]) -> LocalModel | None:
    return models[0] if models else None


def resolve_model_choice(
    choice: str,
    models: list[LocalModel],
    *,
    cwd: Path | None = None,
) -> tuple[LocalModel | None, str | None]:
    """Resolve a path, 1-based index, or unique name substring."""
    raw = choice.strip()
    if not raw:
        model = pick_default_model(models)
        if model is None:
            return None, "No local GGUF models found under models/, exports/, or hf_cache/."
        return model, None

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() and cwd is not None:
        candidate = cwd / candidate
    if candidate.is_file() and candidate.suffix.lower() == ".gguf":
        try:
            resolved = candidate.resolve()
            size = resolved.stat().st_size
        except OSError as exc:
            return None, f"Cannot read {candidate}: {exc}"
        return (
            LocalModel(
                path=candidate if not looks_like_digest_name(candidate.name) else resolved,
                size_bytes=size,
                label=local_model_label(candidate, resolved),
            ),
            None,
        )

    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(models):
            return models[index - 1], None
        return None, f"No model #{index}. Use /models for the list."

    needle = raw.lower()
    hits = [
        model
        for model in models
        if needle in model.label.lower() or needle in str(model.path).lower()
    ]
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, f"No local GGUF matches {raw!r}."
    labels = ", ".join(hit.label for hit in hits[:6])
    extra = "…" if len(hits) > 6 else ""
    return None, f"Ambiguous model {raw!r}: {labels}{extra}. Use an index from /models."


def parse_slash(line: str) -> SlashCommand | None:
    text = line.strip()
    if not text.startswith("/"):
        return None
    parts = text[1:].split(None, 1)
    if not parts:
        return SlashCommand(kind="unknown")
    verb = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    aliases = {
        "help": "help",
        "h": "help",
        "?": "help",
        "quit": "quit",
        "exit": "quit",
        "q": "quit",
        "clear": "clear",
        "reset": "clear",
        "unload": "unload",
        "free": "unload",
        "models": "models",
        "ls": "models",
        "use": "use",
        "model": "use",
        "search": "search",
        "s": "search",
        "find": "search",
        "download": "download",
        "dl": "download",
        "get": "download",
        "refresh": "refresh",
        "r": "refresh",
        "run": "run",
        "open": "open",
        "logout": "logout",
        "signout": "logout",
        "relays": "relays",
    }
    kind = aliases.get(verb)
    if kind is None:
        return SlashCommand(kind="unknown", arg=verb)
    return SlashCommand(kind=kind, arg=arg)


def complete_offline_chat(model: str, messages: list[dict[str, str]]) -> str:
    """Run one local reply. Weights load on first call, not at UI start."""
    import asyncio

    from seiso.chat.prompts import chat_system_prompt, resolve_model_key
    from seiso.chat.sanitize import sanitize_llm_output
    from seiso.inference.runner import run_chat

    model_key = resolve_model_key(model_path=model)
    system = chat_system_prompt(model_key, tools_enabled=False)
    payload_messages: list[dict[str, str]] = list(messages)
    if system and not any(item.get("role") == "system" for item in payload_messages):
        payload_messages = [{"role": "system", "content": system}, *payload_messages]
    raw = asyncio.run(run_chat({"model_path": model, "messages": payload_messages}))
    return sanitize_llm_output(raw, strip_tool_calls=True)


def release_offline_weights() -> None:
    """Drop loaded inference weights without touching files on disk."""
    from seiso.inference.model_pool import get_model_pool
    from seiso.memory.protection import release_cached_memory

    get_model_pool().unload_all()
    release_cached_memory(sync=False)
