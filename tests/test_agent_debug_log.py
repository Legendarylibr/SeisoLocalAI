"""Agent debug logging defaults."""

from __future__ import annotations

import seiso.agent_debug_log as debug


def test_agent_debug_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SEISO_AGENT_DEBUG", raising=False)
    assert debug.agent_debug_enabled() is False


def test_agent_debug_enabled_when_env_set(monkeypatch):
    monkeypatch.setenv("SEISO_AGENT_DEBUG", "1")
    assert debug.agent_debug_enabled() is True
