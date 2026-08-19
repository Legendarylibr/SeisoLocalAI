"""Seiso terminal UI — Forge-shaped, Hugging Face Hub, no browser."""

from seiso.tui.offline import (
    LocalModel,
    SlashCommand,
    complete_offline_chat,
    discover_local_gguf,
    format_size,
    parse_slash,
    pick_default_model,
    release_offline_weights,
    resolve_model_choice,
)
from seiso.tui.pages import DASHBOARD_GOALS, NAV_GROUPS, STUDIO_PAGES

__all__ = [
    "DASHBOARD_GOALS",
    "LocalModel",
    "NAV_GROUPS",
    "STUDIO_PAGES",
    "SlashCommand",
    "complete_offline_chat",
    "discover_local_gguf",
    "format_size",
    "parse_slash",
    "pick_default_model",
    "release_offline_weights",
    "resolve_model_choice",
]
