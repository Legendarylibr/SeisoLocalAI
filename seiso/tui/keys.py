"""Single-keystroke reader for the Seiso TUI (no extra deps)."""

from __future__ import annotations

import select
import sys
from dataclasses import dataclass
from typing import IO, TextIO

_ESC_TIMEOUT_S = 0.05

# Windows second-byte codes after 0x00 / 0xE0.
_WIN_SPECIAL = {
    72: "up",
    80: "down",
    75: "left",
    77: "right",
    71: "home",
    79: "end",
    73: "pageup",
    81: "pagedown",
    83: "delete",
}


@dataclass(frozen=True, slots=True)
class Key:
    name: str
    char: str = ""


def parse_keys(data: bytes) -> list[Key]:
    """Turn a raw stdin chunk (escape sequences, UTF-8, mouse) into keys."""
    keys: list[Key] = []
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0x1B:
            key, consumed = _parse_escape(data, i)
            keys.append(key)
            i += consumed
            continue
        if b in {0x0D, 0x0A}:
            keys.append(Key("enter"))
            i += 1
            continue
        if b in {0x7F, 0x08}:
            keys.append(Key("backspace"))
            i += 1
            continue
        if b == 0x09:
            keys.append(Key("tab"))
            i += 1
            continue
        if b == 0x03:
            keys.append(Key("ctrl-c"))
            i += 1
            continue
        if b == 0x04:
            keys.append(Key("ctrl-d"))
            i += 1
            continue
        if b == 0x15:
            keys.append(Key("ctrl-u"))
            i += 1
            continue
        if b == 0x17:
            keys.append(Key("ctrl-w"))
            i += 1
            continue
        if b < 0x20:
            i += 1
            continue
        char, consumed = _decode_utf8(data, i)
        if char:
            keys.append(Key("char", char))
        i += consumed
    return keys


def _decode_utf8(data: bytes, start: int) -> tuple[str, int]:
    lead = data[start]
    if lead < 0x80:
        return chr(lead), 1
    if lead & 0xE0 == 0xC0:
        need = 2
    elif lead & 0xF0 == 0xE0:
        need = 3
    elif lead & 0xF8 == 0xF0:
        need = 4
    else:
        return "\ufffd", 1
    chunk = data[start : start + need]
    if len(chunk) < need:
        return "", len(chunk) or 1
    try:
        return chunk.decode("utf-8"), need
    except UnicodeDecodeError:
        return "\ufffd", 1


def _parse_escape(data: bytes, start: int) -> tuple[Key, int]:
    rest = data[start + 1 :]
    if not rest:
        return Key("esc"), 1
    # SGR mouse: ESC [ < btn ; x ; y M/m
    if rest.startswith(b"[<"):
        end = 2
        while end < len(rest) and rest[end] not in {0x4D, 0x6D}:  # M / m
            end += 1
        if end < len(rest):
            payload = rest[2:end].decode("ascii", errors="ignore")
            suffix = rest[end]
            key = _mouse_key(payload, released=suffix == 0x6D)
            return key, 1 + end + 1
        return Key("esc"), 1
    # CSI arrows / home / end / delete / pages
    if rest[0] == 0x5B:  # [
        if len(rest) >= 2 and rest[1] in {0x41, 0x42, 0x43, 0x44, 0x48, 0x46}:
            names = {
                0x41: "up",
                0x42: "down",
                0x43: "right",
                0x44: "left",
                0x48: "home",
                0x46: "end",
            }
            return Key(names[rest[1]]), 3
        if len(rest) >= 3 and rest[2] == 0x7E:
            extra = {0x31: "home", 0x33: "delete", 0x34: "end", 0x35: "pageup", 0x36: "pagedown"}
            name = extra.get(rest[1])
            if name:
                return Key(name), 4
        # swallow unknown CSI
        consumed = 1
        while consumed < len(rest) and not (0x40 <= rest[consumed] <= 0x7E):
            consumed += 1
        return Key("none"), 1 + min(consumed + 1, len(rest))
    # SS3 (application cursor keys): ESC O A
    if rest[0] == 0x4F and len(rest) >= 2 and rest[1] in {0x41, 0x42, 0x43, 0x44}:
        names = {0x41: "up", 0x42: "down", 0x43: "right", 0x44: "left"}
        return Key(names[rest[1]]), 3
    return Key("esc"), 1


def _mouse_key(payload: str, *, released: bool) -> Key:
    if released:
        return Key("none")
    parts = payload.split(";")
    if not parts:
        return Key("none")
    try:
        btn = int(parts[0])
    except ValueError:
        return Key("none")
    # Wheel: 64 up, 65 down (sometimes 68/69 with modifiers)
    if btn in {64, 68}:
        return Key("up")
    if btn in {65, 69}:
        return Key("down")
    return Key("none")


def stdin_is_interactive(stream: TextIO | None = None) -> bool:
    src = stream if stream is not None else sys.stdin
    out = sys.stdout
    try:
        return bool(src.isatty() and out.isatty())
    except Exception:
        return False


class KeyReader:
    """Read one logical key. Enables wheel tracking on POSIX TTYs."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self._mouse = False

    def enable(self) -> None:
        if not stdin_is_interactive(self.stream):
            return
        try:
            sys.stdout.write("\x1b[?1000h\x1b[?1006h")
            sys.stdout.flush()
            self._mouse = True
        except Exception:
            self._mouse = False

    def disable(self) -> None:
        if not self._mouse:
            return
        try:
            sys.stdout.write("\x1b[?1006l\x1b[?1000l")
            sys.stdout.flush()
        except Exception:
            pass
        self._mouse = False

    def read(self) -> Key:
        if sys.platform == "win32":
            return self._read_windows()
        return self._read_posix()

    def _read_windows(self) -> Key:
        try:
            import msvcrt
        except ImportError:
            return Key("eof")
        getwch = getattr(msvcrt, "getwch", None)
        if getwch is None:
            return Key("eof")
        ch = getwch()
        if ch in {"\x00", "\xe0"}:
            code = ord(getwch())
            return Key(_WIN_SPECIAL.get(code, "none"))
        if ch in {"\r", "\n"}:
            return Key("enter")
        if ch in {"\x08", "\x7f"}:
            return Key("backspace")
        if ch == "\t":
            return Key("tab")
        if ch == "\x1b":
            return Key("esc")
        if ch == "\x03":
            return Key("ctrl-c")
        if ch == "\x04":
            return Key("ctrl-d")
        if ch == "\x1a":
            return Key("eof")
        if ch:
            return Key("char", ch)
        return Key("eof")

    def _read_posix(self) -> Key:
        fd = self.stream.fileno()
        import termios
        import tty

        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            data = self._read_posix_chunk(fd)
        except OSError:
            return Key("eof")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if not data:
            return Key("eof")
        keys = parse_keys(data)
        return keys[0] if keys else Key("none")

    def _read_posix_chunk(self, fd: int) -> bytes:
        buf: IO[bytes] | None = getattr(self.stream, "buffer", None)
        first = buf.read(1) if buf is not None else self.stream.read(1).encode("utf-8")
        if not first:
            return b""
        # If this is ESC, wait briefly for the rest of a sequence / wheel event.
        if first == b"\x1b":
            extra = self._drain(fd, buf, timeout=_ESC_TIMEOUT_S)
            return first + extra
        extra = self._drain(fd, buf, timeout=0.0)
        return first + extra

    def _drain(self, fd: int, buf: IO[bytes] | None, timeout: float) -> bytes:
        try:
            ready, _, _ = select.select([fd], [], [], timeout)
        except (OSError, ValueError):
            return b""
        if not ready:
            return b""
        chunks = bytearray()
        while True:
            try:
                more_ready, _, _ = select.select([fd], [], [], 0.0)
            except (OSError, ValueError):
                break
            if not more_ready:
                break
            piece = buf.read(1) if buf is not None else self.stream.read(1).encode("utf-8")
            if not piece:
                break
            chunks.extend(piece)
            if len(chunks) > 64:
                break
        return bytes(chunks)
