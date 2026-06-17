#!/usr/bin/env python3
"""Glitch ASCII rain → sunlight install TUI for scripts/install.sh."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable

# Rain palette (cold, glitchy)
C_SKY = "\033[38;5;24m"
C_RAIN = "\033[38;5;39m"
C_GLITCH = "\033[38;5;45m"
C_MUTE = "\033[38;5;238m"
C_PERSON = "\033[38;5;252m"
C_WARN = "\033[38;5;208m"

# Sun palette (warm, hopeful)
C_SUN = "\033[38;5;220m"
C_GLOW = "\033[38;5;228m"
C_WARM = "\033[38;5;214m"
C_GRASS = "\033[38;5;82m"
C_SKY_CLEAR = "\033[38;5;117m"

RESET = "\033[0m"
HIDE = "\033[?25l"
SHOW = "\033[?25h"
CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
INV = "\033[7m"

GLITCH_CHARS = "▓▒░█╔╗╚╝║═@#$%&*~^/\\|<>"

RAIN_SCENES = [
    r"""
      ·   :   ·   :   ·   :   ·   :   ·
    :   ·   :   ·   :   ·   :   ·   :
  {sky}▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔{r}
  {mute}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░{r}
  {rain}╲  │ ╱ │╲  │ ╱ │╲  │ ╱ │╲  │ ╱ │╲  │ ╱{r}
  {mute}▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒{r}
  {rain}≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈{r}
  {mute}░░░░░░░░░░░░░░░░░░░░░░░░░░░░  {person}╭─╮{r}
  {rain}│ ╱ ╲ │ ╱ ╲ │ ╱ ╲ │ ╱ ╲ │ ╱ ╲ │  {person}│o│{r}
  {mute}░░░░░░░░░░░░░░░░░░░░░░░░░░░░  {person}╰┬╯{r}
  {rain}≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈  {person}╱ ╲{r}
""",
    r"""
    ·  :  ·  :  ·  :  ·  :  ·  :  ·  :
  {sky}▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔{r}
  {glitch}█▓▒░ GL1TCH ST0RM ░▒▓█ ░▒▓█ ░▒▓█ ░▒▓█{r}
  {rain}╱│╲│╱│╲│╱│╲│╱│╲│╱│╲│╱│╲│╱│╲│╱│╲│╱│╲│{r}
  {mute}▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒{r}
  {rain}≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈{r}
  {mute}░░░░░░░░░░░░░░░░░░░░░░░░░░░░  {person}┌─┐{r}
  {rain}╲ │ ╱ ╲ │ ╱ ╲ │ ╱ ╲ │ ╱ ╲ │   {person}│●│{r}
  {mute}░░░░░░░░░░░░░░░░░░░░░░░░░░░░  {person}├─┤{r}
  {rain}≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈  {person}╱ ╲{r}
""",
]

SUN_SCENE = r"""
  {warm}        \   |   /{r}
  {sun}      ---- ☼ ----{r}
  {warm}        /   |   \{r}
  {sky}▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁{r}
  {glow}✦ ✧  sunlight breaks through  ✧ ✦{r}
  {grass}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  {person}╭──╮{r}
  {glow}░ optimism ░ hope ░ clarity ░   {person}│ ^ │{r}
  {grass}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  {person}╰┬─┬╯{r}
  {warm}    the storm is over — forge ahead   {person} ╱   ╲{r}
"""


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("SEISO_NO_BANNER", "0") != "1"


def _size() -> tuple[int, int]:
    try:
        s = shutil.get_terminal_size(fallback=(80, 24))
        return s.columns, s.lines
    except OSError:
        return 80, 24


def _fmt(template: str) -> str:
    return template.format(
        sky=C_SKY,
        rain=C_RAIN,
        glitch=C_GLITCH,
        mute=C_MUTE,
        person=C_PERSON,
        sun=C_SUN,
        glow=C_GLOW,
        warm=C_WARM,
        grass=C_GRASS,
        r=RESET,
    )


def _glitch_text(text: str, intensity: float) -> str:
    if intensity <= 0:
        return text
    out: list[str] = []
    for ch in text:
        if ch in " \n" or random.random() > intensity:
            out.append(ch)
        else:
            out.append(random.choice(GLITCH_CHARS))
    line = "".join(out)
    if random.random() < intensity * 0.35:
        shift = random.randint(-3, 3)
        if shift > 0:
            line = " " * shift + line
        elif shift < 0:
            line = line[-shift:]
    return line


def _draw(
    body: str,
    *,
    title: str,
    subtitle: str,
    footer: str,
    glitch: float,
    invert_flash: bool = False,
) -> None:
    cols, _ = _size()
    bar = "═" * min(cols - 2, 58)
    head = f"{C_GLITCH}{BOLD}▛▀▀ {title} ▀▀▜{RESET}"
    sub = f"{DIM}{subtitle}{RESET}"
    foot = f"{C_WARN if glitch > 0.2 else C_GLOW}{footer}{RESET}"

    lines = [_glitch_text(_fmt(body), glitch) for body in [body]]
    art = lines[0].strip("\n").splitlines()

    buf = [CLEAR, head, sub, ""]
    buf.extend(art)
    buf.extend(["", foot, f"{C_MUTE}{bar}{RESET}"])
    if invert_flash:
        buf.append(f"{INV}{C_GLITCH} SIGNAL CORRUPT ░▒▓ RESET... {RESET}")

    sys.stdout.write("\n".join(buf) + "\n")
    sys.stdout.flush()


def _tail_log(path: str, max_lines: int = 2) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return "working..."
    if not lines:
        return "starting..."
    last = lines[-1].strip()
    if len(last) > 64:
        last = last[:61] + "..."
    return last or "working..."


def cmd_intro(_: argparse.Namespace) -> int:
    if not _tty():
        return 0
    sys.stdout.write(HIDE)
    try:
        for i in range(6):
            _draw(
                RAIN_SCENES[i % len(RAIN_SCENES)],
                title="SEISO INSTALL",
                subtitle="signal locked · rain on the wire",
                footer="initializing glitch buffer...",
                glitch=0.15 + (i % 3) * 0.08,
                invert_flash=i == 2,
            )
            time.sleep(0.12)
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()
    return 0


def cmd_during(args: argparse.Namespace) -> int:
    if not _tty():
        return subprocess.call(["wait", str(args.wait_pid)]) if args.wait_pid else 0

    sys.stdout.write(HIDE)
    proc = None
    if args.wait_pid:
        try:
            os.kill(args.wait_pid, 0)
        except OSError:
            return 0
        proc = args.wait_pid

    log_path = args.log or ""
    label = args.label or "installing"
    frame = 0
    try:
        while True:
            if proc is not None:
                try:
                    os.kill(proc, 0)
                except OSError:
                    break
            glitch = 0.12 + (frame % 5) * 0.05 + random.random() * 0.08
            detail = _tail_log(log_path) if log_path else label
            _draw(
                RAIN_SCENES[frame % len(RAIN_SCENES)],
                title="SEISO INSTALL",
                subtitle=f"{label} · someone waits in the rain",
                footer=f"▸ {detail}",
                glitch=glitch,
                invert_flash=frame % 17 == 0,
            )
            frame += 1
            time.sleep(0.09)
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()

    if proc is not None:
        _, status = os.waitpid(proc, 0)
        return os.waitstatus_to_exitcode(status)
    return 0


def cmd_outro(_: argparse.Namespace) -> int:
    if not _tty():
        return 0
    sys.stdout.write(HIDE)
    try:
        # Rain fading — less glitch each frame
        for i in range(8):
            _draw(
                RAIN_SCENES[(7 - i) % len(RAIN_SCENES)],
                title="SEISO INSTALL",
                subtitle="clouds breaking · static clearing",
                footer="almost there...",
                glitch=max(0.02, 0.35 - i * 0.04),
                invert_flash=False,
            )
            time.sleep(0.08)

        # Sun burst
        for i in range(10):
            body = SUN_SCENE
            if i < 3:
                body = RAIN_SCENES[0] + "\n" + SUN_SCENE
            _draw(
                body,
                title="SEISO READY",
                subtitle="optimism · sunlight · clear signal",
                footer="the storm stopped — you're good to go",
                glitch=max(0.0, 0.12 - i * 0.012),
                invert_flash=i == 1,
            )
            time.sleep(0.11)

        sys.stdout.write(
            f"\n{C_SUN}{BOLD}✦ Seiso installed — open http://127.0.0.1:8765 ✦{RESET}\n\n"
        )
        sys.stdout.flush()
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seiso glitch install TUI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_intro = sub.add_parser("intro")
    p_intro.set_defaults(func=cmd_intro)

    p_during = sub.add_parser("during")
    p_during.add_argument("--wait-pid", type=int, required=True)
    p_during.add_argument("--log", default="")
    p_during.add_argument("--label", default="installing")
    p_during.set_defaults(func=cmd_during)

    p_outro = sub.add_parser("outro")
    p_outro.set_defaults(func=cmd_outro)

    args = parser.parse_args(list(argv) if argv is not None else None)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
