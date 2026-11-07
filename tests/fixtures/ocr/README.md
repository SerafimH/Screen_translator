# OCR quality corpus

`corpus.json` contains manually verified text for the four existing screenshots,
plus two deterministic synthetic English subtitle images. The synthetic text is
drawn with OpenCV Hershey Simplex (scale 0.55, antialiasing, a black outline) on a
solid background and on a sinusoidal color pattern. They exercise roughly 14px
letters and require zero character/word errors. They reproduce a high-confidence
`mission.n.` ending: correction requires two agreeing normalized word crops,
with the detected word geometry preserved. They are not representative of all game fonts.

Four additional English fixtures cover native Segoe UI 10px text, 14px dark text,
14px italics, and text close to the image boundary. The sentence includes
`Don't`, `iPhone`, and `state-of-the-art`. These cases require zero word errors.
Reproduce them with `generate_english.py` using the OCR helper virtualenv on
Windows; it reads installed Segoe UI fonts without bundling font binaries.

Seven synthetic English comic lettering samples cover Comic Sans, bold italics,
compressed bold lettering, Impact, outlined white text, handwriting, and very
faint dark text. All require zero word errors. Reproduce with `generate_manga.py`
on Windows using installed fonts. These are typography checks, not manga scans
and not proof of accuracy for arbitrary artistic lettering. The condensed and
faint cases reproduce missing spacing and an entirely missed line respectively.

`manga_handwritten_black.png` is the user-provided screenshot containing
`D-Did he / find out?!`. Besides the OCR text check, Rust integration tests
exercise Auto mode, complete sentence grouping, and cursor confirmation of all
four words. The rotated line boxes overlap vertically; WinRT can return `finA`
for `find`, so this case also verifies conflict resolution using two strong,
distinct RapidOCR crop readings.

Run from the repository root:

```powershell
.\ocr_helper\.venv\Scripts\python.exe -m ocr_helper.benchmark --output target/ocr-quality.json
```

`manhwa_01.png` through `manhwa_17.png` are the 17 supplied screenshots,
preserved as received. Every line requires zero normalized character and word
errors. Their `blocks` fields also specify separate speech bubbles. The Rust
`supplied_manhwa_auto_roi_and_shift_preserve_complete_dialogue` integration test
runs real Auto OCR, a cropped ROI for each bubble, and Shift word confirmation
with complete paragraph context, including the numeric `10'S` label and negative
desktop coordinates. It also checks that the Shift capture window contains the
complete paragraph from both its first and last lines. These checks do not call
a translation service.

The explicit network test `live_google_supplied_manhwa_translation_report`
translates the supplied dialogue with its preceding context and writes
`target/manhwa-google-translations.json`. It is ignored by the default test run.
Successful requests prove service integration and paragraph separation, not
literary translation quality; the report must be reviewed for semantic mistakes.

The local RapidOCR report includes character error rate (CER), word error rate (WER), missing lines,
recognized text, and processing time per image. Model initialization time is
reported separately. Case and repeated whitespace are normalized; punctuation
errors still count; equivalent apostrophe and hyphen glyphs are normalized.
A line is considered missing if no recognized line is within
20% edit distance. A nonzero exit status means at least one case exceeds its CER
limit, its optional WER limit, or loses a line. No translation API is used.

These are regression checks, not a claim of accuracy on arbitrary screenshots.
Add real failing captures and their exact text before tuning thresholds.
