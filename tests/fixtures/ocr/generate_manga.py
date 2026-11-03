"""Generate English comic typography samples using locally installed Windows fonts.

These synthetic samples exercise lettering styles, not scans of published manga.
Font binaries are not distributed. Run with the OCR helper virtualenv.
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TEXT = "WAIT! WE HAVE TO GET OUT OF HERE!"
CASES = [
    ("manga_comic.png", "comic.ttf", 22, 1.0, "black", "white", 0),
    ("manga_italic.png", "comicz.ttf", 22, 1.0, "black", "white", 0),
    ("manga_condensed.png", "comicbd.ttf", 24, 0.32, "black", "white", 0),
    ("manga_heavy.png", "impact.ttf", 26, 1.0, "black", "white", 0),
    ("manga_outline.png", "comicbd.ttf", 22, 1.0, "white", (80, 100, 120), 2),
    ("manga_low_contrast.png", "comic.ttf", 18, 1.0, (202, 202, 202), (205, 205, 205), 0),
    ("manga_handwritten.png", "segoepr.ttf", 16, 1.0, "black", "white", 0),
]


def main():
    root = Path(__file__).resolve().parent
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name, family, size, squeeze, foreground, background, outline in CASES:
        font = ImageFont.truetype(str(fonts / family), size)
        width = int(font.getlength(TEXT)) + 24
        lettering = Image.new("RGB", (width, 54), background)
        ImageDraw.Draw(lettering).text((12, 8), TEXT, font=font, fill=foreground,
                                      stroke_width=outline, stroke_fill="black")
        lettering = lettering.resize((round(width * squeeze), 54), Image.Resampling.LANCZOS)
        image = Image.new("RGB", (700, 130), background)
        image.paste(lettering, (35, 35))
        image.save(root / name)


if __name__ == "__main__":
    main()
