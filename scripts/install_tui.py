#!/usr/bin/env python3
"""Glitch ASCII rain → sunlight install TUI for scripts/install.sh."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import signal
import sys
import time
from collections.abc import Iterable

BRAND = "SeisoLocalAI"

# Rain palette (cold, glitchy)
C_SKY = "\033[38;5;24m"
C_RAIN = "\033[38;5;39m"
C_GLITCH = "\033[38;5;45m"
C_MUTE = "\033[38;5;238m"
C_PERSON = "\033[38;5;252m"

# Sun palette (warm, hopeful)
C_SUN = "\033[38;5;220m"
C_GLOW = "\033[38;5;228m"
C_WARM = "\033[38;5;214m"
C_GRASS = "\033[38;5;82m"

RESET = "\033[0m"
HIDE = "\033[?25l"
SHOW = "\033[?25h"
CLEAR = "\033[2J\033[H"
BOLD = "\033[1m"
DIM = "\033[2m"
INV = "\033[7m"

GLITCH_CHARS = "▓▒░█@#$%&*~^/\\|<>"

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
        shift = random.randint(-2, 2)
        if shift > 0:
            line = " " * shift + line
        elif shift < 0:
            line = line[-shift:]
    return line


def _flash_brand(frame: int, glitch: float) -> str:
    colors = (C_RAIN, C_GLITCH, C_GLOW, C_SUN, C_WARM)
    text = BRAND
    if frame % 4 == 0:
        text = text.upper()
    if frame % 7 != 0:
        text = _glitch_text(text, glitch * 0.45)
    color = colors[frame % len(colors)]
    cols, _ = _size()
    pad = max(0, (cols - len(BRAND)) // 2)
    flash = " " * pad + text
    if frame % 6 < 3:
        return f"{INV}{color}{BOLD}{flash}{RESET}"
    return f"{color}{BOLD}{flash}{RESET}"


def _draw(
    body: str,
    *,
    subtitle: str,
    brand_line: str,
    glitch: float,
    invert_flash: bool = False,
) -> None:
    cols, _ = _size()
    bar = "═" * min(cols - 2, 58)
    sub = f"{DIM}{subtitle}{RESET}"

    art = _glitch_text(_fmt(body), glitch).strip("\n").splitlines()

    buf = [CLEAR, brand_line, "", sub, ""]
    buf.extend(art)
    buf.extend(["", brand_line, f"{C_MUTE}{bar}{RESET}"])
    if invert_flash:
        buf.append(f"{INV}{C_GLITCH} ░▒▓ SIGNAL ░▒▓ {RESET}")

    sys.stdout.write("\n".join(buf) + "\n")
    sys.stdout.flush()


def cmd_intro(_: argparse.Namespace) -> int:
    if not _tty():
        return 0
    sys.stdout.write(HIDE)
    try:
        for i in range(6):
            glitch = 0.15 + (i % 3) * 0.08
            _draw(
                RAIN_SCENES[i % len(RAIN_SCENES)],
                subtitle="rain on the wire",
                brand_line=_flash_brand(i, glitch),
                glitch=glitch,
                invert_flash=i == 2,
            )
            time.sleep(0.12)
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()
    return 0


def _poll_exitcode(pid: int) -> int | None:
    done_pid, status = os.waitpid(pid, os.WNOHANG)
    if done_pid == 0:
        return None
    return os.waitstatus_to_exitcode(status)


def cmd_during(args: argparse.Namespace) -> int:
    if not _tty():
        if args.wait_pid:
            _, status = os.waitpid(args.wait_pid, 0)
            return os.waitstatus_to_exitcode(status)
        return 0

    sys.stdout.write(HIDE)
    proc = args.wait_pid
    exit_code: int | None = None
    if proc:
        try:
            exit_code = _poll_exitcode(proc)
        except ChildProcessError:
            return 0

    frame = 0
    try:
        while exit_code is None:
            if proc is not None:
                try:
                    exit_code = _poll_exitcode(proc)
                except ChildProcessError:
                    exit_code = 0
                if exit_code is not None:
                    break
            glitch = 0.12 + (frame % 5) * 0.05 + random.random() * 0.08
            _draw(
                RAIN_SCENES[frame % len(RAIN_SCENES)],
                subtitle="installing · someone waits in the rain",
                brand_line=_flash_brand(frame, glitch),
                glitch=glitch,
                invert_flash=frame % 17 == 0,
            )
            frame += 1
            time.sleep(0.09)
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()

    return exit_code or 0


def cmd_outro(args: argparse.Namespace) -> int:
    if not _tty():
        return 0
    sys.stdout.write(HIDE)
    try:
        _draw(
            SUN_SCENE,
            subtitle=f"install complete · open {args.url}",
            brand_line=f"{C_SUN}{BOLD}{BRAND}{RESET}",
            glitch=0.0,
        )
        sys.stdout.write(
            f"\n{C_SUN}{BOLD}Install complete.{RESET}\n"
            f"Open Forge: {BOLD}{args.url}{RESET}\n\n"
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
    p_during.set_defaults(func=cmd_during)

    p_outro = sub.add_parser("outro")
    p_outro.add_argument("--url", default="http://127.0.0.1:8765")
    p_outro.set_defaults(func=cmd_outro)

    args = parser.parse_args(list(argv) if argv is not None else None)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
