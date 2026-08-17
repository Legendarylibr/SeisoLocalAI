"""Lean offline chat helpers: local GGUF discovery and slash commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seiso.io.files import iter_matching_files

_SCAN_ROOTS = ("models", "exports", "hf_cache")
_SKIP_NAME_MARKERS = ("mmproj", ".incomplete")


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
    return any(marker in name for marker in _SKIP_NAME_MARKERS)


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
                size = resolved.stat().st_size
            except OSError:
                continue
            if size <= 0:
                continue
            found[_file_key(resolved)] = LocalModel(
                path=resolved,
                size_bytes=size,
                label=resolved.name,
            )
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
        return LocalModel(path=resolved, size_bytes=size, label=resolved.name), None

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
