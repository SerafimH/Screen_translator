# OCR quality corpus

`corpus.json` contains manually verified text for the four existing screenshots,
plus two deterministic synthetic English subtitle images. The synthetic text is
drawn with OpenCV Hershey Simplex (scale 0.55, antialiasing, a black outline) on a
solid background and on a sinusoidal color pattern. They exercise roughly 14px
letters; they are not representative of all game fonts.

Run from the repository root:

```powershell
.\ocr_helper\.venv\Scripts\python.exe -m ocr_helper.benchmark --output target/ocr-quality.json
```

The local RapidOCR report includes character error rate (CER), missing lines,
recognized text, and processing time per image. Model initialization time is
reported separately. Case and repeated whitespace are normalized; punctuation
errors still count. A line is considered missing if no recognized line is within
20% edit distance. A nonzero exit status means at least one case exceeds its CER
limit or loses a line. No translation API is used.

These are regression checks, not a claim of accuracy on arbitrary screenshots.
Add real failing captures and their exact text before tuning thresholds.
