"""Build stylized team portraits for the deck.

These are placeholders made with an image filter (smooth duotone in the deck
palette), not AI-generated images. To use AI portraits instead, drop
square images at team/ai/<slug>.png (or .jpg); main.tex picks them up
automatically. See team/ai/PROMPTS.md.

Usage: python3 tools/make_portraits.py <dir-with-headshots>
Headshots are expected as <slug>.jpg (e.g. from texasucf.org/members).
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

TEAM = [
    "abitpal-gyawali",
    "andrew-li",
    "ishan-guthikonda",
    "kaylee-zheng",
    "karol-tang",
    "vishal-sundaram",
]

# Deck palette, dark to light (smooth gradient map, not posterized).
STOPS = [
    (0.00, (0x1B, 0x1F, 0x24)),  # ink
    (0.30, (0x5A, 0x2A, 0x10)),  # dark umber
    (0.60, (0xBF, 0x57, 0x00)),  # burnt orange
    (0.85, (0xF2, 0xC9, 0xA8)),  # peach
    (1.00, (0xFB, 0xF7, 0xF2)),  # cream
]
SIZE = 600


def ramp(v):
    v /= 255
    for (a, ca), (b, cb) in zip(STOPS, STOPS[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return tuple(round(x + (y - x) * t) for x, y in zip(ca, cb))
    return STOPS[-1][1]


LUT = [ramp(i) for i in range(256)]


def stylize(src: Path, dst: Path):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    # Zoom toward the face: keep the central 72%, biased upward.
    c = int(min(w, h) * 0.72)
    left, top = (w - c) // 2, int((h - c) * 0.35)
    im = im.crop((left, top, left + c, top + c)).resize((SIZE, SIZE), Image.LANCZOS)
    g = ImageOps.grayscale(im)
    g = ImageOps.autocontrast(g, cutoff=1)
    # Lift shadows a little so faces stay readable in the duotone.
    g = g.point(lambda v: round(255 * (v / 255) ** 0.85))
    rgb = Image.merge(
        "RGB", [g.point([c[i] for c in LUT]) for i in range(3)]
    )
    # Circular crop.
    mask = Image.new("L", (SIZE * 4, SIZE * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, SIZE * 4 - 1, SIZE * 4 - 1), fill=255)
    mask = mask.resize((SIZE, SIZE), Image.LANCZOS)
    out = rgb.convert("RGBA")
    out.putalpha(mask)
    out.save(dst)


def main():
    src_dir = Path(sys.argv[1])
    out_dir = Path(__file__).resolve().parent.parent / "team"
    for slug in TEAM:
        stylize(src_dir / f"{slug}.jpg", out_dir / f"{slug}.png")
        print("wrote", out_dir / f"{slug}.png")


if __name__ == "__main__":
    main()
