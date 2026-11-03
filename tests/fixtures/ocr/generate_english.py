"""Reproduce English typography fixtures with Pillow and Windows Segoe UI.

Run with the OCR helper virtualenv from any directory. Fonts are read locally;
font binaries are not bundled. The committed PNGs are the actual test inputs.
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TEXT = "Don't quit! iPhone saves state-of-the-art screenshots."
CASES = [
    ("english_thin_10.png", "segoeui.ttf", 10, (20, 70), "white", (25, 25, 25)),
    ("english_dark_14.png", "segoeui.ttf", 14, (20, 70), "black", (235, 235, 235)),
    ("english_italic_14.png", "segoeuii.ttf", 14, (20, 70), "white", (25, 25, 25)),
    ("english_edge_14.png", "segoeui.ttf", 14, (1, 1), "white", (25, 25, 25)),
]


def main():
    destination = Path(__file__).resolve().parent
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for filename, font, pixels, position, foreground, background in CASES:
        image = Image.new("RGB", (900, 160), background)
        ImageDraw.Draw(image).text(position, TEXT, font=ImageFont.truetype(str(fonts / font), pixels),
                                   fill=foreground)
        image.save(destination / filename)


if __name__ == "__main__":
    main()
