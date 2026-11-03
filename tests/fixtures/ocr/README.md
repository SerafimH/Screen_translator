# OCR quality corpus

`corpus.json` contains manually verified text for the four existing screenshots,
plus two deterministic synthetic English subtitle images. The synthetic text is
drawn with OpenCV Hershey Simplex (scale 0.55, antialiasing, a black outline) on a
solid background and on a sinusoidal color pattern. They exercise roughly 14px
letters; they are not representative of all game fonts.

Four additional English fixtures cover native Segoe UI 10px text, 14px dark text,
14px italics, and text close to the image boundary. The sentence includes
`Don't`, `iPhone`, and `state-of-the-art`. These cases require zero word errors.
Reproduce them with `generate_english.py` using the OCR helper virtualenv on
Windows; it reads installed Segoe UI fonts without bundling font binaries.

Run from the repository root:

```powershell
.\ocr_helper\.venv\Scripts\python.exe -m ocr_helper.benchmark --output target/ocr-quality.json
```

The local RapidOCR report includes character error rate (CER), word error rate (WER), missing lines,
recognized text, and processing time per image. Model initialization time is
reported separately. Case and repeated whitespace are normalized; punctuation
errors still count; equivalent apostrophe and hyphen glyphs are normalized.
A line is considered missing if no recognized line is within
20% edit distance. A nonzero exit status means at least one case exceeds its CER
limit, its optional WER limit, or loses a line. No translation API is used.

These are regression checks, not a claim of accuracy on arbitrary screenshots.
Add real failing captures and their exact text before tuning thresholds.
