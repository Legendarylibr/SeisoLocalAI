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

INTRO_FRAMES = 2
INTRO_SLEEP_S = 0.05
DURING_SLEEP_S = 0.06

_FMT = {
    "sky": C_SKY,
    "rain": C_RAIN,
    "glitch": C_GLITCH,
    "mute": C_MUTE,
    "person": C_PERSON,
    "sun": C_SUN,
    "glow": C_GLOW,
    "warm": C_WARM,
    "grass": C_GRASS,
    "r": RESET,
}

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

_FORMATTED_RAIN_SCENES = tuple(scene.format(**_FMT) for scene in RAIN_SCENES)
_FORMATTED_SUN_SCENE = SUN_SCENE.format(**_FMT)


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("SEISO_NO_BANNER", "0") != "1"


def _size() -> tuple[int, int]:
    try:
        s = shutil.get_terminal_size(fallback=(80, 24))
        return s.columns, s.lines
    except OSError:
        return 80, 24


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


def _flash_brand(frame: int, glitch: float, cols: int) -> str:
    colors = (C_RAIN, C_GLITCH, C_GLOW, C_SUN, C_WARM)
    text = BRAND
    if frame % 4 == 0:
        text = text.upper()
    if frame % 7 != 0:
        text = _glitch_text(text, glitch * 0.45)
    color = colors[frame % len(colors)]
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
    art = _glitch_text(body, glitch).strip("\n").splitlines()
    parts = [CLEAR, brand_line, "", sub, "", *art, "", brand_line, f"{C_MUTE}{bar}{RESET}"]
    if invert_flash:
        parts.append(f"{INV}{C_GLITCH} ░▒▓ SIGNAL ░▒▓ {RESET}")
    sys.stdout.write("\n".join(parts) + "\n")
    sys.stdout.flush()


def _play_intro_frames(cols: int) -> None:
    for i in range(INTRO_FRAMES):
        glitch = 0.15 + (i % 3) * 0.08
        _draw(
            _FORMATTED_RAIN_SCENES[i % len(_FORMATTED_RAIN_SCENES)],
            subtitle="rain on the wire",
            brand_line=_flash_brand(i, glitch, cols),
            glitch=glitch,
            invert_flash=i == 1,
        )
        time.sleep(INTRO_SLEEP_S)


def cmd_intro(_: argparse.Namespace) -> int:
    if not _tty():
        return 0
    sys.stdout.write(HIDE)
    try:
        cols, _ = _size()
        _play_intro_frames(cols)
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()
    return 0


def cmd_during(_: argparse.Namespace) -> int:
    if not _tty():
        return 0

    stop = False

    def _handle_stop(_signum: int, _frame: object | None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    sys.stdout.write(HIDE)
    frame = 0
    cols, _ = _size()
    try:
        _play_intro_frames(cols)
        while not stop:
            glitch = 0.12 + (frame % 5) * 0.05 + random.random() * 0.08
            _draw(
                _FORMATTED_RAIN_SCENES[frame % len(_FORMATTED_RAIN_SCENES)],
                subtitle="installing · someone waits in the rain",
                brand_line=_flash_brand(frame, glitch, cols),
                glitch=glitch,
                invert_flash=frame % 17 == 0,
            )
            frame += 1
            time.sleep(DURING_SLEEP_S)
    finally:
        sys.stdout.write(SHOW)
        sys.stdout.flush()

    return 0


def cmd_outro(args: argparse.Namespace) -> int:
    if not _tty():
        return 0
    sys.stdout.write(HIDE)
    try:
        _draw(
            _FORMATTED_SUN_SCENE,
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
    p_during.set_defaults(func=cmd_during)

    p_outro = sub.add_parser("outro")
    p_outro.add_argument("--url", default="http://127.0.0.1:8765")
    p_outro.set_defaults(func=cmd_outro)

    args = parser.parse_args(list(argv) if argv is not None else None)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
