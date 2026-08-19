"""Single-keystroke reader for the Seiso TUI (no extra deps)."""

from __future__ import annotations

import os
import select
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import IO, Any, TextIO

_ESC_TIMEOUT_S = 0.03
_READ_CHUNK = 4096

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

# CSI / SS3 final bytes for arrows and simple keys.
_ARROW = {0x41: "up", 0x42: "down", 0x43: "right", 0x44: "left", 0x48: "home", 0x46: "end"}
_TILDE = {1: "home", 2: "insert", 3: "delete", 4: "end", 5: "pageup", 6: "pagedown"}

# Mouse / alt-screen / paste / cursor — enable and reverse in matching order.
_TERM_ENABLE = (
    "\x1b[?1049h"  # alternate screen (leave the user's scrollback alone)
    "\x1b[?25l"  # hide hardware cursor; we draw our own block
    "\x1b[?7l"  # no autowrap — a full-width row must not push the frame down
    "\x1b[?1000h"  # mouse button + wheel
    "\x1b[?1006h"  # SGR mouse (btn;x;yM) — without this, wheels type digits
    "\x1b[?1007h"  # alternate-scroll: wheel → arrows if SGR is unavailable
    "\x1b[?2004h"  # bracketed paste
)
_TERM_DISABLE = "\x1b[?2004l\x1b[?1007l\x1b[?1006l\x1b[?1000l\x1b[?7h\x1b[?25h\x1b[?1049l"


@dataclass(frozen=True, slots=True)
class Key:
    name: str
    char: str = ""
    x: int = 0
    y: int = 0


def parse_keys(data: bytes) -> list[Key]:
    """Turn a raw stdin chunk (escape sequences, UTF-8, mouse) into keys.

    A trailing lone ESC is emitted as ``esc`` so a real Escape key still works.
    An incomplete CSI / mouse / UTF-8 tail is dropped — never turned into
    digits — because wheel events often arrive split across reads.
    """
    keys, rest = parse_keys_incremental(data)
    if rest == b"\x1b":
        keys.append(Key("esc"))
    return keys


def parse_keys_incremental(data: bytes) -> tuple[list[Key], bytes]:
    """Parse *data*. Incomplete tail is returned unconsumed (never as chars)."""
    keys: list[Key] = []
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0x1B:
            key, consumed, complete = _parse_escape(data, i)
            if not complete:
                return keys, data[i:]
            if key.name != "none":
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
        if b == 0x0C:
            keys.append(Key("ctrl-l"))
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
        char, consumed, complete = _decode_utf8(data, i)
        if not complete:
            return keys, data[i:]
        if char:
            keys.append(Key("char", char))
        i += consumed
    return keys, b""


def _decode_utf8(data: bytes, start: int) -> tuple[str, int, bool]:
    lead = data[start]
    if lead < 0x80:
        return chr(lead), 1, True
    if lead & 0xE0 == 0xC0:
        need = 2
    elif lead & 0xF0 == 0xE0:
        need = 3
    elif lead & 0xF8 == 0xF0:
        need = 4
    else:
        return "\ufffd", 1, True
    chunk = data[start : start + need]
    if len(chunk) < need:
        return "", 0, False
    try:
        return chunk.decode("utf-8"), need, True
    except UnicodeDecodeError:
        return "\ufffd", 1, True


def _parse_escape(data: bytes, start: int) -> tuple[Key, int, bool]:
    """Return (key, bytes_consumed, complete). Incomplete → hold the tail."""
    rest = data[start + 1 :]
    if not rest:
        return Key("esc"), 1, False
    # SGR mouse: ESC [ < btn ; x ; y M/m
    if rest.startswith(b"[<"):
        return _parse_sgr_mouse(rest)
    # X10 mouse: ESC [ M Cb Cx Cy  (what you get if 1006 is not honoured)
    if rest.startswith(b"[M"):
        if len(rest) < 5:
            return Key("none"), 0, False
        key = _x10_mouse(rest[2], rest[3], rest[4])
        return key, 6, True
    # CSI
    if rest[0] == 0x5B:  # [
        return _parse_csi(rest)
    # SS3 (application cursor keys): ESC O A
    if rest[0] == 0x4F:
        if len(rest) < 2:
            return Key("none"), 0, False
        name = _ARROW.get(rest[1])
        if name:
            return Key(name), 3, True
        return Key("esc"), 1, True
    # ESC + leftover printable is a real Escape (or Alt+key we ignore)
    return Key("esc"), 1, True


def _parse_sgr_mouse(rest: bytes) -> tuple[Key, int, bool]:
    """*rest* starts with ``[<``. Consumed count is relative to the ESC."""
    end = 2
    while end < len(rest) and rest[end] not in {0x4D, 0x6D}:  # M / m
        byte = rest[end]
        if not (0x30 <= byte <= 0x39 or byte == 0x3B):  # 0-9 or ;
            # Garbage after `[<` — swallow so digits never type.
            return Key("none"), 1 + end, True
        end += 1
    if end >= len(rest):
        return Key("none"), 0, False
    payload = rest[2:end].decode("ascii", errors="ignore")
    suffix = rest[end]
    key = _sgr_mouse_key(payload, released=suffix == 0x6D)
    return key, 1 + end + 1, True


def _parse_csi(rest: bytes) -> tuple[Key, int, bool]:
    """*rest* starts with ``[``. Consumed count is relative to the ESC."""
    if len(rest) >= 2 and rest[1] in _ARROW:
        return Key(_ARROW[rest[1]]), 3, True
    # Find CSI final byte (0x40–0x7E). Hold if it has not arrived yet.
    end = 1
    while end < len(rest) and not (0x40 <= rest[end] <= 0x7E):
        end += 1
    if end >= len(rest):
        return Key("none"), 0, False
    final = rest[end]
    params = rest[1:end]
    consumed = 1 + end + 1  # ESC + [..final]
    if final == 0x7E:  # ~
        code = _csi_first_int(params)
        if code in _TILDE:
            return Key(_TILDE[code]), consumed, True
        if code == 200:
            return Key("paste-start"), consumed, True
        if code == 201:
            return Key("paste-end"), consumed, True
        return Key("none"), consumed, True
    # urxvt mouse: ESC [ btn ; x ; y M   (no '<')
    if final in {0x4D, 0x6D} and b";" in params:
        payload = params.decode("ascii", errors="ignore")
        return _sgr_mouse_key(payload, released=final == 0x6D), consumed, True
    name = _ARROW.get(final)
    if name and not params:
        return Key(name), consumed, True
    return Key("none"), consumed, True


def _csi_first_int(params: bytes) -> int | None:
    if not params:
        return None
    token = params.split(b";", 1)[0]
    if not token.isdigit():
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _sgr_mouse_key(payload: str, *, released: bool) -> Key:
    parts = payload.split(";")
    if not parts:
        return Key("none")
    try:
        btn = int(parts[0])
        x = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        y = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    except ValueError:
        return Key("none")
    return _mouse_from_button(btn, x=x, y=y, released=released)


def _x10_mouse(cb: int, cx: int, cy: int) -> Key:
    # X10 encodes button as 32+btn and cells as 32+coord (1-based).
    btn = (cb - 32) & 0xFF
    x = max(0, cx - 32)
    y = max(0, cy - 32)
    return _mouse_from_button(btn, x=x, y=y, released=False)


def _mouse_from_button(btn: int, *, x: int, y: int, released: bool) -> Key:
    if released or btn & 32:
        return Key("none", x=x, y=y)
    # Strip shift (4) / alt (8) / ctrl (16); keep wheel / button.
    base = btn & ~0x1C
    if base == 64:
        return Key("up", x=x, y=y)
    if base == 65:
        return Key("down", x=x, y=y)
    if base == 66:
        return Key("left", x=x, y=y)
    if base == 67:
        return Key("right", x=x, y=y)
    if base == 0:
        return Key("click", x=x, y=y)
    if base == 2:
        return Key("click-right", x=x, y=y)
    return Key("none", x=x, y=y)


def stdin_is_interactive(stream: TextIO | None = None) -> bool:
    src = stream if stream is not None else sys.stdin
    out = sys.stdout
    try:
        return bool(src.isatty() and out.isatty())
    except Exception:
        return False


class KeyReader:
    """Read logical keys. Stays in raw mode and tracks the wheel for the session."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdin
        self._armed = False
        self._old_termios: list | None = None
        self._buf = b""
        self._queue: deque[Key] = deque()
        self._resized = False
        self._prev_winch: Callable[[int, FrameType | None], Any] | int | None = None

    def enable(self) -> None:
        if self._armed or not stdin_is_interactive(self.stream):
            return
        if sys.platform != "win32":
            self._enter_raw()
        try:
            sys.stdout.write(_TERM_ENABLE)
            sys.stdout.flush()
        except Exception:
            pass
        self._install_winch()
        self._armed = True

    def disable(self) -> None:
        if not self._armed:
            return
        try:
            sys.stdout.write(_TERM_DISABLE)
            sys.stdout.flush()
        except Exception:
            pass
        self._restore_winch()
        self._leave_raw()
        self._armed = False
        self._buf = b""
        self._queue.clear()

    def read(self) -> Key:
        if self._queue:
            return self._queue.popleft()
        if sys.platform == "win32":
            return self._read_windows()
        return self._read_posix()

    def _enter_raw(self) -> None:
        try:
            import termios
            import tty
        except ImportError:
            return
        fd = self.stream.fileno()
        try:
            self._old_termios = termios.tcgetattr(fd)
            # cbreak, not raw: keep OPOST so Rich newlines still do CR+LF.
            # Full setraw left the cursor on the last column and hid the UI.
            tty.setcbreak(fd)
            mode = termios.tcgetattr(fd)
            mode[3] &= ~termios.ISIG  # Ctrl+C is a key, not a signal
            termios.tcsetattr(fd, termios.TCSADRAIN, mode)
        except Exception:
            self._old_termios = None

    def _leave_raw(self) -> None:
        if self._old_termios is None:
            return
        try:
            import termios

            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        self._old_termios = None

    def _install_winch(self) -> None:
        if sys.platform == "win32":
            return
        try:
            import signal

            self._prev_winch = signal.getsignal(signal.SIGWINCH)

            def _on_winch(_signum: int, _frame: object) -> None:
                self._resized = True

            signal.signal(signal.SIGWINCH, _on_winch)
        except Exception:
            self._prev_winch = None

    def _restore_winch(self) -> None:
        if self._prev_winch is None:
            return
        try:
            import signal

            signal.signal(signal.SIGWINCH, self._prev_winch)
        except Exception:
            pass
        self._prev_winch = None

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
        if ch == "\x0c":
            return Key("ctrl-l")
        if ch == "\x1a":
            return Key("eof")
        if ch:
            return Key("char", ch)
        return Key("eof")

    def _read_posix(self) -> Key:
        fd = self.stream.fileno()
        if not self._armed:
            return self._read_posix_legacy(fd)
        while True:
            if self._resized:
                self._resized = False
                return Key("resize")
            chunk = self._read_available(fd, timeout=None if not self._buf else _ESC_TIMEOUT_S)
            if chunk is None:
                return Key("eof")
            if chunk:
                self._buf += chunk
                self._flush_parsed()
                if self._queue:
                    return self._queue.popleft()
                # Incomplete sequence — wait for the rest, never type it.
                continue
            if self._buf == b"\x1b":
                self._buf = b""
                return Key("esc")
            if self._buf:
                # Timed out on a junk tail (should be rare). Drop it.
                self._buf = b""
            if self._resized:
                self._resized = False
                return Key("resize")

    def _flush_parsed(self) -> None:
        keys, rest = parse_keys_incremental(self._buf)
        self._buf = rest
        self._queue.extend(keys)

    def _read_available(self, fd: int, timeout: float | None) -> bytes | None:
        try:
            ready, _, _ = select.select([fd], [], [], timeout)
        except InterruptedError:
            return b""
        except (OSError, ValueError):
            return None
        if not ready:
            return b""
        try:
            piece = os.read(fd, _READ_CHUNK)
        except InterruptedError:
            return b""
        except OSError:
            return None
        return piece if piece else None

    def _read_posix_legacy(self, fd: int) -> Key:
        """Per-keystroke raw mode — only used if enable() was not called."""
        import termios
        import tty

        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            data = self._legacy_chunk(fd)
        except OSError:
            return Key("eof")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if not data:
            return Key("eof")
        keys, rest = parse_keys_incremental(data)
        if rest == b"\x1b" and not keys:
            return Key("esc")
        return keys[0] if keys else Key("none")

    def _legacy_chunk(self, fd: int) -> bytes:
        buf: IO[bytes] | None = getattr(self.stream, "buffer", None)
        first = buf.read(1) if buf is not None else self.stream.read(1).encode("utf-8")
        if not first:
            return b""
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
            if len(chunks) > 512:
                break
        return bytes(chunks)
