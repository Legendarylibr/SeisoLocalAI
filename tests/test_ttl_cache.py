"""Tests for seiso.ttl_cache."""

from __future__ import annotations

import time

from seiso.ttl_cache import TtlCache


def test_ttl_cache_hit_and_miss():
    cache: TtlCache[str, int] = TtlCache(ttl_s=60.0, max_entries=4)
    cache.set("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None


def test_ttl_cache_expires_entries():
    cache: TtlCache[str, int] = TtlCache(ttl_s=0.05, max_entries=4)
    cache.set("a", 1)
    time.sleep(0.06)
    assert cache.get("a") is None


def test_ttl_cache_evicts_oldest_at_capacity():
    cache: TtlCache[str, int] = TtlCache(ttl_s=60.0, max_entries=2)
    cache.set("first", 1)
    time.sleep(0.01)
    cache.set("second", 2)
    cache.set("third", 3)
    assert cache.get("first") is None
    assert cache.get("second") == 2
    assert cache.get("third") == 3


def test_ttl_cache_clear():
    cache: TtlCache[str, int] = TtlCache(ttl_s=60.0, max_entries=4)
    cache.set("a", 1)
    cache.clear()
    assert cache.get("a") is None
