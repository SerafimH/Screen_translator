"""Reproduce a short dialogue change on a 1080p page using a local Windows font."""

from pathlib import Path
import os

from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parent
    fonts = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
    font = ImageFont.truetype(str(fonts / 'comic.ttf'), 32)
    for word in ('YES', 'NO'):
        image = Image.new('RGB', (1920, 1080), 'white')
        ImageDraw.Draw(image).text((800, 650), word, font=font, fill='black')
        image.save(root / f'dialogue_{word.lower()}.png')


if __name__ == '__main__':
    main()
