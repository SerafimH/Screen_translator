# Dialogue change detection

`dialogue_yes.png` and `dialogue_no.png` are synthetic 1920x1080 pages with a
short 32px Comic Sans word at the same position. The old 32x16 average-luminance
signature misses this change. The capture regression compares fine edge changes
inside the previously recognized text rectangle, while the existing coarse
signature continues detecting scene changes.

Regenerate on Windows with the OCR helper virtualenv:

```powershell
.\ocr_helper\.venv\Scripts\python.exe tests/fixtures/capture/generate_dialogue.py
```

The font is read from Windows and is not bundled. Separate Rust checks verify
that a moving 10x10 particle outside the text rectangle does not block the 250ms
stability check, while changes inside the rectangle do. Desktop offsets on a
secondary monitor are included in that check.
