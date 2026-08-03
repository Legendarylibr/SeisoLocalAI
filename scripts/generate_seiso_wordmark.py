#!/usr/bin/env python3
"""Generate a detailed SEISO wordmark matching the Seiso mascot image style.

Warm peach/gold/lavender palette, dithered shading, vertical rain noise,
and soft block glitches — letters stay clear. Subline: LOCAL · AI.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


W, H = 1600, 900
TEXT = "SEISO"
SEED = 42

# Palette pulled from forge-ui/src/assets/seiso-mascot.png
PEACH = (245, 210, 195)
SKIN = (232, 188, 168)
GOLD = (214, 168, 110)
BROWN = (92, 58, 42)
DEEP = (48, 30, 28)
LAVENDER = (168, 140, 210)
PINK = (232, 150, 168)
CYAN_GLITCH = (120, 230, 235)
MAGENTA_GLITCH = (230, 90, 160)
CREAM = (255, 244, 232)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
        "/Library/Fonts/Arial Black.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _center_xy(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return (W - tw) // 2 - bbox[0], (H - th) // 2 - bbox[1] - 28


def _load_mascot() -> Image.Image | None:
    candidates = [
        _repo_root() / "forge-ui" / "src" / "assets" / "seiso-mascot.png",
        _repo_root() / "assets" / "seiso-mascot.png",
    ]
    for path in candidates:
        if path.exists():
            return Image.open(path).convert("RGBA")
    return None


def _warm_bg(rng: random.Random, mascot: Image.Image | None) -> Image.Image:
    img = Image.new("RGB", (W, H), DEEP)
    px = img.load()
    for y in range(H):
        t = y / H
        r = int(DEEP[0] + (GOLD[0] - DEEP[0]) * (0.35 + 0.45 * t))
        g = int(DEEP[1] + (LAVENDER[1] - DEEP[1]) * (0.25 + 0.4 * (1 - abs(t - 0.4))))
        b = int(DEEP[2] + (LAVENDER[2] - DEEP[2]) * (0.35 + 0.35 * (1 - t)))
        for x in range(W):
            n = rng.randint(-10, 10)
            px[x, y] = (
                max(0, min(255, r + n)),
                max(0, min(255, g + n // 2)),
                max(0, min(255, b + n)),
            )

    base = img.convert("RGBA")
    # soft warm orbs (mascot glow)
    for cx, cy, rad, color in (
        (W // 2, H // 2 - 40, 420, (*GOLD, 55)),
        (260, 200, 300, (*PINK, 40)),
        (1340, 680, 340, (*LAVENDER, 48)),
        (900, 120, 240, (*PEACH, 35)),
    ):
        orb = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(orb)
        od.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=color)
        orb = orb.filter(ImageFilter.GaussianBlur(95))
        base = Image.alpha_composite(base, orb)

    if mascot is not None:
        # Soft, large mascot wash behind the word — style anchor, not a logo crop.
        m = mascot.copy()
        m_w = int(W * 0.72)
        ratio = m_w / m.width
        m = m.resize((m_w, int(m.height * ratio)), Image.Resampling.LANCZOS)
        # desaturate slightly + warm tint so type stays readable
        m_rgb = ImageEnhance.Color(m.convert("RGB")).enhance(0.85)
        m_rgb = ImageEnhance.Brightness(m_rgb).enhance(0.55)
        m = Image.blend(m_rgb, Image.new("RGB", m_rgb.size, GOLD), 0.18).convert("RGBA")
        alpha = m.split()[3].point(lambda a: int(a * 0.28))
        m.putalpha(alpha)
        m = m.filter(ImageFilter.GaussianBlur(6))
        paste = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        paste.paste(m, ((W - m.width) // 2, (H - m.height) // 2 - 30), m)
        base = Image.alpha_composite(base, paste)

        # smaller clearer mascot vignette on the right for character presence
        side = mascot.copy()
        sw = 420
        side = side.resize((sw, int(side.height * sw / side.width)), Image.Resampling.LANCZOS)
        side_a = side.split()[3].point(lambda a: int(a * 0.55))
        side.putalpha(side_a)
        # feather edges
        mask = Image.new("L", side.size, 0)
        md = ImageDraw.Draw(mask)
        md.ellipse((8, 8, side.width - 8, side.height - 8), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(28))
        side.putalpha(ImageChops.multiply(side.split()[3], mask))
        paste2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        paste2.paste(side, (W - side.width - 40, (H - side.height) // 2 + 20), side)
        base = Image.alpha_composite(base, paste2)

    return base.convert("RGB")


def _text_layer(
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None = None,
    outline_width: int = 0,
    dy: int = 0,
    dx: int = 0,
) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = _center_xy(draw, text, font)
    kwargs: dict = {"font": font, "fill": fill}
    if outline and outline_width:
        kwargs["stroke_width"] = outline_width
        kwargs["stroke_fill"] = outline
    draw.text((x + dx, y + dy), text, **kwargs)
    return layer


def _dither_fill(font: ImageFont.ImageFont) -> Image.Image:
    """Ordered-dither peach→gold letter face like the mascot shading."""
    mask = _text_layer(TEXT, font, (255, 255, 255, 255))
    # vertical gradient peach → gold → cream
    grad = Image.new("RGB", (W, H))
    gp = grad.load()
    cy0 = H // 2 - 120
    cy1 = H // 2 + 120
    for y in range(H):
        if y < cy0:
            t = 0.0
        elif y > cy1:
            t = 1.0
        else:
            t = (y - cy0) / max(1, cy1 - cy0)
        # two-stop: cream highlight → peach → gold shadow
        if t < 0.45:
            u = t / 0.45
            r = int(CREAM[0] * (1 - u) + PEACH[0] * u)
            g = int(CREAM[1] * (1 - u) + PEACH[1] * u)
            b = int(CREAM[2] * (1 - u) + PEACH[2] * u)
        else:
            u = (t - 0.45) / 0.55
            r = int(PEACH[0] * (1 - u) + GOLD[0] * u)
            g = int(PEACH[1] * (1 - u) + GOLD[1] * u)
            b = int(PEACH[2] * (1 - u) + GOLD[2] * u)
        for x in range(W):
            gp[x, y] = (r, g, b)

    # 2x2 Bayer dither against lavender/brown for mascot-like pixel shading
    dithered = Image.new("RGB", (W, H))
    dp = dithered.load()
    bayer = ((0, 8, 2, 10), (12, 4, 14, 6), (3, 11, 1, 9), (15, 7, 13, 5))
    for y in range(H):
        for x in range(W):
            r, g, b = gp[x, y]
            thr = bayer[y & 3][x & 3] / 16.0
            # mix a lavender shade into darker regions via threshold
            lum = (r + g + b) / (3 * 255)
            if lum + 0.08 < thr:
                dp[x, y] = (
                    int(r * 0.75 + LAVENDER[0] * 0.25),
                    int(g * 0.7 + LAVENDER[1] * 0.3),
                    int(b * 0.65 + LAVENDER[2] * 0.35),
                )
            elif lum > thr + 0.55:
                dp[x, y] = CREAM
            else:
                dp[x, y] = (r, g, b)

    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(dithered, (0, 0), mask.split()[3])
    return out


def _word_stack(font: ImageFont.ImageFont) -> Image.Image:
    stack = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = _text_layer(TEXT, font, (0, 0, 0, 150), dy=10, dx=5)
    shadow = shadow.filter(ImageFilter.GaussianBlur(7))
    stack = Image.alpha_composite(stack, shadow)

    # soft chromatic ghosts (mascot cyan/magenta pops)
    stack = Image.alpha_composite(
        stack, _text_layer(TEXT, font, (*MAGENTA_GLITCH, 70), dx=-5, dy=1)
    )
    stack = Image.alpha_composite(
        stack, _text_layer(TEXT, font, (*CYAN_GLITCH, 70), dx=5, dy=-1)
    )

    # dark aliased outline like mascot linework
    stack = Image.alpha_composite(
        stack,
        _text_layer(
            TEXT,
            font,
            (*DEEP, 255),
            outline=(*BROWN, 255),
            outline_width=14,
        ),
    )
    # warm gold rim
    stack = Image.alpha_composite(
        stack,
        _text_layer(TEXT, font, (0, 0, 0, 0), outline=(*GOLD, 200), outline_width=6),
    )
    # lavender outer kiss
    stack = Image.alpha_composite(
        stack,
        _text_layer(TEXT, font, (0, 0, 0, 0), outline=(*LAVENDER, 110), outline_width=3),
    )

    face = _dither_fill(font)
    stack = Image.alpha_composite(stack, face)

    # crisp readable top fill (keeps letters clear over dither)
    stack = Image.alpha_composite(
        stack, _text_layer(TEXT, font, (*CREAM, 210))
    )
    hi = _text_layer(TEXT, font, (255, 255, 255, 50), dy=-5, dx=-2)
    hi = hi.filter(ImageFilter.GaussianBlur(1.2))
    stack = Image.alpha_composite(stack, hi)
    return stack


def _vertical_rain(img: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(0, W, 3):
        draw.line([(x, 0), (x, H)], fill=(255, 230, 210, 18), width=1)
    for _ in range(220):
        x = rng.randint(0, W - 1)
        y0 = rng.randint(0, H - 1)
        length = rng.randint(12, 90)
        color = rng.choice(
            [
                (*CREAM, rng.randint(25, 55)),
                (*CYAN_GLITCH, rng.randint(20, 50)),
                (*PINK, rng.randint(20, 45)),
                (*GOLD, rng.randint(20, 45)),
            ]
        )
        draw.line([(x, y0), (x, min(H - 1, y0 + length))], fill=color, width=1)
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def _block_glitches(img: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(55):
        x = rng.randint(0, W - 8)
        y = rng.randint(0, H - 8)
        # keep dense blocks out of letter core
        if abs(y - H // 2) < 70 and 280 < x < W - 280:
            continue
        s = rng.choice([2, 3, 4, 5, 7])
        color = rng.choice(
            [
                (*CYAN_GLITCH, rng.randint(120, 220)),
                (*MAGENTA_GLITCH, rng.randint(100, 200)),
                (*CREAM, rng.randint(140, 230)),
                (*LAVENDER, rng.randint(100, 180)),
                (*GOLD, rng.randint(100, 180)),
            ]
        )
        draw.rectangle((x, y, x + s, y + s), fill=color)
    # a few horizontal micro-tears outside the word band
    for _ in range(8):
        y = rng.choice(
            [rng.randint(40, H // 2 - 150), rng.randint(H // 2 + 150, H - 40)]
        )
        x0 = rng.randint(40, W - 200)
        draw.rectangle(
            (x0, y, x0 + rng.randint(40, 160), y + rng.randint(1, 3)),
            fill=(*CYAN_GLITCH, rng.randint(60, 120)),
        )
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def _subtle_slice(img: Image.Image, rng: random.Random) -> Image.Image:
    out = img.copy()
    for _ in range(5):
        y = rng.choice(
            [rng.randint(60, H // 2 - 150), rng.randint(H // 2 + 150, H - 60)]
        )
        h = rng.randint(2, 6)
        shift = rng.randint(-14, 14)
        if shift == 0:
            continue
        region = out.crop((0, y, W, min(H, y + h)))
        out.paste(region, (shift, y))
    # one hairline across letters only
    y = rng.randint(H // 2 - 20, H // 2 + 20)
    region = out.crop((0, y, W, y + 2))
    out.paste(region, (rng.choice([-5, 5]), y))
    return out


def _subtitle(img: Image.Image, font_small: ImageFont.ImageFont) -> Image.Image:
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    label = "LOCAL  ·  AI"
    bbox = draw.textbbox((0, 0), label, font=font_small)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    y = H // 2 + 150
    draw.text((x + 1, y + 1), label, font=font_small, fill=(*DEEP, 180))
    draw.text((x, y), label, font=font_small, fill=(*PEACH, 230))
    draw.rectangle((x - 72, y + 9, x - 14, y + 12), fill=(*PINK, 200))
    draw.rectangle((x + tw + 14, y + 9, x + tw + 72, y + 12), fill=(*GOLD, 200))
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def generate(out: Path, seed: int = SEED) -> Path:
    rng = random.Random(seed)
    font = _font(238)
    font_small = _font(26)
    mascot = _load_mascot()

    bg = _warm_bg(rng, mascot)
    bg = _vertical_rain(bg, rng)
    bg = _block_glitches(bg, rng)
    bg = _subtle_slice(bg.convert("RGB"), rng)

    word = _word_stack(font)
    word = word.filter(ImageFilter.UnsharpMask(radius=1.1, percent=150, threshold=2))

    composed = Image.alpha_composite(bg.convert("RGBA"), word)
    composed = _subtitle(composed, font_small)
    # light film grain
    grain = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gp = grain.load()
    for _ in range(6000):
        x, y = rng.randint(0, W - 1), rng.randint(0, H - 1)
        a = rng.randint(10, 35)
        gp[x, y] = (255, 240, 220, a)
    composed = Image.alpha_composite(composed, grain)

    rgb = composed.convert("RGB")
    rgb = ImageEnhance.Color(rgb).enhance(1.08)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.06)
    # final clarity pass on SEISO
    crisp = _text_layer(TEXT, font, (*CREAM, 230))
    rgb = Image.alpha_composite(rgb.convert("RGBA"), crisp)
    rim = _text_layer(
        TEXT, font, (255, 255, 255, 0), outline=(*BROWN, 140), outline_width=2
    )
    rgb = Image.alpha_composite(rgb, rim).convert("RGB")

    out.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out, "PNG", optimize=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("assets/seiso_wordmark.png"),
        help="Output PNG path",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    path = generate(args.output, seed=args.seed)
    print(path.resolve())


if __name__ == "__main__":
    main()
