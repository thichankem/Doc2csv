"""Generate assets/icon.ico for the desktop shortcut (run once; output committed).

Draws a simple document → CSV grid mark on a blue rounded square. Uses Pillow.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
OUT.parent.mkdir(parents=True, exist_ok=True)

SIZE = 256
BG = (37, 99, 235)        # blue
BG2 = (29, 78, 216)
PAPER = (255, 255, 255)
GRID = (37, 99, 235)
ACCENT = (16, 185, 129)   # green


def rounded(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def render(size: int) -> Image.Image:
    s = 256
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # background tile
    rounded(d, (8, 8, s - 8, s - 8), 48, BG)
    rounded(d, (8, 8, s - 8, s - 8 - 6), 48, BG2)

    # document sheet
    doc = (54, 40, 150, 216)
    rounded(d, doc, 12, PAPER)
    # text lines on the doc
    for i, y in enumerate(range(66, 150, 20)):
        w = 116 if i % 2 == 0 else 100
        d.rounded_rectangle((70, y, 70 + (w - 56), y + 8), radius=4, fill=(150, 170, 210))

    # CSV grid (table) overlapping bottom-right
    gx0, gy0, gx1, gy1 = 120, 132, 214, 212
    rounded(d, (gx0, gy0, gx1, gy1), 10, PAPER)
    d.rounded_rectangle((gx0, gy0, gx1, gy0 + 22), radius=10, fill=ACCENT)
    # grid lines
    for x in (gx0 + 31, gx0 + 62):
        d.line((x, gy0 + 22, x, gy1), fill=GRID, width=3)
    for y in (gy0 + 22, gy0 + 44, gy0 + 66):
        d.line((gx0, y, gx1, y), fill=GRID, width=3)

    if size != s:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    sizes = [16, 24, 32, 48, 64, 128, 256]
    base = render(256)
    imgs = [base.resize((n, n), Image.LANCZOS) for n in sizes]
    base.save(OUT, format="ICO", sizes=[(n, n) for n in sizes])
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
