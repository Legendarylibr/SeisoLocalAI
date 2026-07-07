"""Shared minimal GGUF file builders for inference tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from seiso.inference.backends import clear_gguf_caches


@pytest.fixture(autouse=True)
def reset_gguf_cache():
    clear_gguf_caches()
    yield
    clear_gguf_caches()


def write_arch_gguf(
    path: Path,
    architecture: str,
    *,
    extra: list[tuple[bytes, int]] | None = None,
    block_count: int = 32,
) -> None:
    """Write a tiny GGUF stub with ``general.architecture`` and optional KV pairs."""
    arch_key = b"general.architecture"
    arch_value = architecture.encode()
    prefix = architecture.split("-", 1)[0]
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
    ]
    for key, value in extra or []:
        payload.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                struct.pack("<I", value),
            ]
        )
    block_key = prefix.encode() + b".block_count"
    payload.extend(
        [
            struct.pack("<Q", len(block_key)),
            block_key,
            struct.pack("<I", 4),
            struct.pack("<I", block_count),
        ]
    )
    kv_count = 2 + len(extra or [])
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, kv_count) + b"".join(payload))


def write_gguf_u32_metadata(path: Path, pairs: list[tuple[bytes, int]]) -> None:
    """Write a tiny GGUF stub with ``llama`` architecture and u32 metadata pairs."""
    arch_key = b"general.architecture"
    arch_value = b"llama"
    payload = [
        struct.pack("<Q", len(arch_key)),
        arch_key,
        struct.pack("<I", 8),
        struct.pack("<Q", len(arch_value)),
        arch_value,
    ]
    for key, value in pairs:
        payload.extend(
            [
                struct.pack("<Q", len(key)),
                key,
                struct.pack("<I", 4),
                struct.pack("<I", value),
            ]
        )
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1 + len(pairs)) + b"".join(payload)
    )
