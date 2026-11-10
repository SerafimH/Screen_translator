#!/usr/bin/env python3
"""
Screen Translator - RapidOCR Python Helper Server
Communicates with the Rust application via stdin/stdout binary protocol.
Stdout is strictly reserved for single-line JSON messages ending with '\n'.
All logs, diagnostic messages, and warnings are directed to stderr.
"""

from collections import OrderedDict
from contextlib import redirect_stdout
import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

# -----------------------------------------------------------------------------
# Настройки отладки и фильтрации OCR (константы с порогами)
# -----------------------------------------------------------------------------
SCREEN_TRANSLATOR_OCR_DEBUG_ENV = "SCREEN_TRANSLATOR_OCR_DEBUG"
SCREEN_TRANSLATOR_OCR_VARIANTS_ENV = "SCREEN_TRANSLATOR_OCR_VARIANTS"
SCREEN_TRANSLATOR_OCR_LANG_ENV = "SCREEN_TRANSLATOR_OCR_LANG"

# Пороги масштабирования малых кадров
SMALL_FRAME_PIXELS = 1_500_000
UPSCALE_SMALL_FRAMES = 2
MAX_RETRY_CALLS = 8
MAX_RETRY_SECONDS = 1.5

# Пороги фильтрации мусора
MIN_SCORE = 0.50              # Default matches helper_min_score in the Rust client.
MIN_SINGLE_CHAR_SCORE = 0.85  # Повышенная уверенность для односимвольных строк (меньше -> "too_short")
MAX_SYMBOL_RATIO = 0.25       # Максимальная доля мусорных спецсимволов (больше -> "symbols")
MIN_BOX_WIDTH = 6             # Минимальная ширина рамки в px (меньше -> "tiny_box")
MIN_BOX_HEIGHT = 6            # Минимальная высота рамки в px (меньше -> "tiny_box")
MAX_DEBUG_RUNS = 20           # Хранить не более 20 последних дампов отладки

# Допустимые стандартные знаки препинания (не считаются мусорными символами)
STANDARD_PUNCTUATION = set(".,!?:;'\"-—…()«»“”‘’/&%")

# Configure logging to sys.stderr exclusively
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ocr_helper] %(message)s",
)
logger = logging.getLogger("ocr_helper")


def read_exact(stream, num_bytes: int) -> Optional[bytes]:
    """Reads exactly num_bytes from stream, returning None on EOF."""
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = stream.read(num_bytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


RAPIDOCR_LANG_MAP = {
    "ja": "japan",
    "ko": "korean",
    "ru": "cyrillic",
    "zh": "ch",
}


def create_ocr_engine(models_dir: Optional[str] = None, lang: str = "en"):
    """Initializes RapidOCR engine using local models if present."""
    rec_lang = RAPIDOCR_LANG_MAP.get(lang.lower(), lang.lower())
    try:
        from rapidocr import ModelType, OCRVersion, RapidOCR

        params = {"Rec.lang_type": rec_lang}
        if rec_lang in {"cyrillic", "korean"}:
            # PP-OCRv6 small does not include these recognition languages.
            params.update({"Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE})
        if rec_lang == "en":
            # Thin subtitle/cursor crops must not be stretched until their short
            # side reaches 736px. Small frames already get a separate 2x pass;
            # keep the detector within the same 2000px ceiling as RapidOCR input.
            params.update({"Det.limit_type": "max", "Det.limit_side_len": 2000})
        if models_dir and os.path.exists(models_dir):
            params["Global.model_root_dir"] = models_dir
        logger.info(
            f"Initializing RapidOCR engine ({rec_lang}) with models_dir={models_dir}..."
        )
        return RapidOCR(params=params if params else None)
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR

            logger.info(
                f"Initializing RapidOCR ({rec_lang}) via rapidocr_onnxruntime fallback..."
            )
            return RapidOCR()
        except ImportError as e:
            logger.error(f"RapidOCR is not installed: {e}")
            raise


class EngineManager:
    """Manages language-specific RapidOCR engine instances with lazy loading."""

    def __init__(self, models_dir: Optional[str] = None, default_engine=None):
        self.models_dir = models_dir
        self.engines: OrderedDict[str, Any] = OrderedDict()
        if default_engine is not None:
            self.engines["default"] = default_engine

    def get_engine(self, lang: str):
        base_lang = lang.split("-")[0].split("_")[0].lower() if lang else "en"
        if "default" in self.engines:
            return self.engines["default"]
        if base_lang in self.engines:
            self.engines.move_to_end(base_lang)
            return self.engines[base_lang]
        target_dir = (
            self.models_dir
            if (self.models_dir and os.path.exists(self.models_dir))
            else None
        )
        logger.info(f"Loading RapidOCR engine for language: {base_lang}")
        engine = create_ocr_engine(target_dir, lang=base_lang)
        while len(self.engines) >= 2:
            self.engines.popitem(last=False)
        self.engines[base_lang] = engine
        return engine


def extract_lines_from_result(result) -> List[Dict[str, Any]]:
    """
    Extracts [{"text": ..., "box": [[x,y],[x,y],[x,y],[x,y]], "score": ...}]
    from RapidOCR output, supporting both RapidOCR 3.x and 1.x formats.
    """
    if result is None:
        return []

    lines = []
    # RapidOCR 3.x RapidOCROutput
    if (
        hasattr(result, "boxes")
        and hasattr(result, "txts")
        and hasattr(result, "scores")
    ):
        boxes = result.boxes
        txts = result.txts
        scores = result.scores
        word_results = getattr(result, "word_results", None)
        if boxes is not None and txts is not None and scores is not None:
            for idx, (box, text, score) in enumerate(zip(boxes, txts, scores)):
                box_list = box.tolist() if hasattr(box, "tolist") else list(box)
                line_data = {
                    "text": str(text),
                    "box": box_list,
                    "score": float(score),
                }
                if word_results and idx < len(word_results) and word_results[idx]:
                    words = []
                    for w in word_results[idx]:
                        w_box = w[2].tolist() if hasattr(w[2], "tolist") else list(w[2])
                        words.append({
                            "text": str(w[0]),
                            "score": float(w[1]),
                            "box": w_box,
                        })
                    line_data["words"] = words
                lines.append(line_data)
        return lines

    # RapidOCR 1.x or tuple/list format: (result_list, elapse) or result_list
    res_list = result
    if isinstance(result, tuple) and len(result) >= 1:
        res_list = result[0]

    if isinstance(res_list, (list, tuple)):
        for item in res_list:
            if isinstance(item, dict):
                lines.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                box = item[0]
                text = item[1]
                score = item[2]
                box_list = box.tolist() if hasattr(box, "tolist") else list(box)
                line_data = {
                    "text": str(text),
                    "box": box_list,
                    "score": float(score),
                }
                if len(item) >= 4 and isinstance(item[3], (list, tuple)):
                    words = []
                    for w in item[3]:
                        if isinstance(w, dict):
                            words.append(w)
                        elif isinstance(w, (list, tuple)) and len(w) >= 3:
                            w_box = w[2].tolist() if hasattr(w[2], "tolist") else list(w[2])
                            words.append({"text": str(w[0]), "score": float(w[1]), "box": w_box})
                    if words:
                        line_data["words"] = words
                lines.append(line_data)
    return lines


def align_detected_words(line):
    """Join detected fragments to the confirmed tokens without inventing boxes."""
    words = line.get("words") or []
    tokens = line.get("text", "").split()
    if not words or "".join(tokens) != "".join(w["text"] for w in words):
        return line
    if len(tokens) == len(words) and all(token == word["text"] for token, word in zip(tokens, words)):
        return line
    aligned, index = [], 0
    for token in tokens:
        pieces, text = [], ""
        while index < len(words) and len(text) < len(token):
            pieces.append(words[index])
            text += words[index]["text"]
            index += 1
        if text != token:
            return line
        xs = [p[0] for word in pieces for p in word["box"]]
        ys = [p[1] for word in pieces for p in word["box"]]
        aligned.append({"text": token, "score": min(w.get("score", 0) for w in pieces),
                        "box": [[min(xs), min(ys)], [max(xs), min(ys)],
                                [max(xs), max(ys)], [min(xs), max(ys)]]})
    return {**line, "words": aligned}


def filter_ocr_lines(lines, min_score=MIN_SCORE):
    """
    Чистая функция фильтрации OCR-строк по правилам:
    1. score < min_score -> отбросить ("low_score")
    2. Пустой текст или нет ни одного alnum символа -> отбросить ("no_alnum")
    3. Доля мусорных спецсимволов среди непробельных > MAX_SYMBOL_RATIO (0.25) -> отбросить ("symbols")
    4. Односимвольные строки при score < MIN_SINGLE_CHAR_SCORE (0.85) -> отбросить ("too_short")
    5. Рамка с высотой < 6 px или шириной < 6 px -> отбросить ("tiny_box")

    Возвращает (kept_lines, rejected_lines), где rejected_lines содержит кортежи (line, reason).
    """
    kept = []
    rejected = []

    for item in lines:
        if isinstance(item, dict):
            box = item.get("box")
            text = item.get("text", "")
            score = item.get("score", 0.0)
        else:
            box, text, score = item[0], item[1], item[2]

        score = float(score) if score is not None else 0.0
        text = str(text) if text is not None else ""

        # 1. Порог уверенности
        if not math.isfinite(score) or score < min_score:
            rejected.append((item, "low_score"))
            continue

        # 2. Непустой текст с хотя бы одним алфавитно-цифровым символом
        text_clean = text.strip()
        if not text_clean or not any(ch.isalnum() for ch in text_clean):
            rejected.append((item, "no_alnum"))
            continue

        # 5. Минимальный размер рамки (высота и ширина не меньше 6 px)
        if box and len(box) >= 4:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
                rejected.append((item, "tiny_box"))
                continue

        # 4. Односимвольные строки требуют повышенной уверенности (>= 0.85)
        if len(text_clean) == 1 and score < MIN_SINGLE_CHAR_SCORE:
            rejected.append((item, "too_short"))
            continue

        # 3. Доля мусорных спецсимволов среди непробельных
        non_spaces = [ch for ch in text_clean if not ch.isspace()]
        if non_spaces:
            garbage_count = sum(
                1
                for ch in non_spaces
                if not ch.isalnum() and ch not in STANDARD_PUNCTUATION
                and not (not ch.isascii() and unicodedata.category(ch).startswith("P"))
            )
            if (garbage_count / len(non_spaces)) > MAX_SYMBOL_RATIO:
                rejected.append((item, "symbols"))
                continue

        kept.append(item)

    return kept, rejected


def _compute_box_iou(box_a, box_b) -> float:
    """Вычисляет IoU (пересечение/объединение) для прямоугольников, описывающих две 4-точечные рамки."""
    xs_a = [p[0] for p in box_a]
    ys_a = [p[1] for p in box_a]
    xa1, ya1, xa2, ya2 = min(xs_a), min(ys_a), max(xs_a), max(ys_a)

    xs_b = [p[0] for p in box_b]
    ys_b = [p[1] for p in box_b]
    xb1, yb1, xb2, yb2 = min(xs_b), min(ys_b), max(xs_b), max(ys_b)

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def merge_ocr_lines(variant_groups, min_score=MIN_SCORE, *, filter_before_merge=True):
    """
    Чистая функция объединения строк от нескольких вариантов кадра:
    1. К каждому варианту отдельно применяется filter_ocr_lines, если filter_before_merge=True.
       Перед уточняющим распознаванием фильтрация откладывается, чтобы сохранить проблемные рамки.
    2. Все прошедшие фильтр строки объединяются.
    3. Полная строка сохраняется вместо охваченного ею обрезанного фрагмента.
    4. Уверенность сравнивается для дубликатов и конкурирующих чтений той же области.
    """
    if not variant_groups:
        return []

    # Если передан плоский список строк вместо списка групп, оборачиваем в группу
    if isinstance(variant_groups[0], (tuple, dict)):
        variant_groups = [variant_groups]

    all_kept = []
    for group in variant_groups:
        if not group:
            continue
        kept = filter_ocr_lines(group, min_score)[0] if filter_before_merge else group
        all_kept.extend(kept)

    if not all_kept:
        return []

    def _get_score(item):
        if isinstance(item, dict):
            return float(item.get("score", 0.0))
        return float(item[2])

    def _get_box(item):
        if isinstance(item, dict):
            return item.get("box")
        return item[0]

    def _text(item):
        return str(item.get("text", "") if isinstance(item, dict) else item[1]).casefold()

    def _tokens(item):
        return tuple(token for word in _text(item).split()
                     if (token := word.strip('.,!?:;"“”«»()[]{}')))

    def _is_clipped(full, fragment):
        full_tokens, part_tokens = _tokens(full), _tokens(fragment)
        if not part_tokens or len(full_tokens) <= len(part_tokens):
            return False
        if not any(full_tokens[i:i + len(part_tokens)] == part_tokens
                   for i in range(len(full_tokens) - len(part_tokens) + 1)):
            return False
        outer, inner = _get_box(full), _get_box(fragment)
        fx1, fx2 = min(p[0] for p in outer), max(p[0] for p in outer)
        fy1, fy2 = min(p[1] for p in outer), max(p[1] for p in outer)
        sx1, sx2 = min(p[0] for p in inner), max(p[0] for p in inner)
        sy1, sy2 = min(p[1] for p in inner), max(p[1] for p in inner)
        # Extra words alone can be hallucinations. Require additional image
        # coverage as evidence of a complete line, allowing box rounding.
        return (sx1 >= fx1 - 2 and sx2 <= fx2 + 2
                and sy1 >= fy1 - 2 and sy2 <= fy2 + 2
                and (fx2 - fx1) * (fy2 - fy1) > (sx2 - sx1) * (sy2 - sy1) * 1.03)

    def _unique_fragment_coverage(left, right):
        a, b = _tokens(left), _tokens(right)
        if not any(a[-length:] == b[:length] for length in range(1, min(len(a), len(b)))):
            return False
        box_a, box_b = _get_box(left), _get_box(right)
        return (min(p[0] for p in box_a) < min(p[0] for p in box_b) - 2
                and max(p[0] for p in box_a) < max(p[0] for p in box_b) - 2)

    # Сортируем кандидатов по убыванию оценки уверенности
    candidates = sorted(all_kept, key=_get_score, reverse=True)
    merged = []

    for cand in candidates:
        cand_box = _get_box(cand)
        if not cand_box or len(cand_box) < 4:
            continue

        overlaps = False
        replaced = []
        for index, selected in enumerate(merged):
            sel_box = _get_box(selected)
            if sel_box and len(sel_box) >= 4:
                iou = _compute_box_iou(cand_box, sel_box)
                if iou <= 0.5:
                    continue
                spacing_only = "".join(_text(cand).split()) == "".join(_text(selected).split())
                if spacing_only and not filter_before_merge and isinstance(cand, dict) and isinstance(selected, dict):
                    def disputed(item):
                        spaced = " ".join(w["text"] for w in item.get("words", []))
                        return (item.get("_spacing_check", False) or
                                (spaced != item["text"] and "".join(spaced.split()) == "".join(item["text"].split())))
                    if disputed(cand) or disputed(selected) or _text(cand) != _text(selected):
                        cand["_spacing_check"] = selected["_spacing_check"] = True
                if (not spacing_only and not filter_before_merge and iou >= 0.85
                        and isinstance(cand, dict) and isinstance(selected, dict)
                        and (cand.get("_word_check") or selected.get("_word_check")
                             or any(0.9 <= _get_score(item) < 0.97 for item in (cand, selected)))):
                    cand["_word_check"] = selected["_word_check"] = True
                detected_spacing = (isinstance(cand, dict) and isinstance(selected, dict)
                                    and len(cand.get("words") or []) > len(selected.get("words") or []))
                if spacing_only and detected_spacing and _get_score(cand) >= max(0.95, _get_score(selected) - 0.01):
                    replaced.append(index)
                elif _is_clipped(cand, selected):
                    replaced.append(index)
                elif _is_clipped(selected, cand):
                    overlaps = True
                    break
                elif (_unique_fragment_coverage(cand, selected)
                      or _unique_fragment_coverage(selected, cand)):
                    continue
                elif (_tokens(cand) == _tokens(selected)
                      or "".join(_text(cand).split()) == "".join(_text(selected).split())
                      or iou >= 0.85):
                    # Confidence-sorted candidates compete only when they read
                    # the same text or almost exactly the same image region.
                    overlaps = True
                    break

        if not overlaps:
            for index in reversed(replaced):
                del merged[index]
            merged.append(cand)

    return merged


def should_upscale_frame(width: int, height: int, cv2_available: bool) -> bool:
    """Определяет, нужно ли увеличивать кадр для улучшения распознавания мелкого текста."""
    return cv2_available and (width * height <= SMALL_FRAME_PIXELS)


def rescale_boxes_down(
    lines: List[Dict[str, Any]], scale_factor: float, scale_y: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Пересчитывает координаты 4-точечных рамок найденных строк обратно в систему исходного кадра."""
    scale_y = scale_factor if scale_y is None else scale_y
    if (scale_factor == 1.0 and scale_y == 1.0) or not lines:
        return lines
    rescaled = []
    for line in lines:
        new_line = dict(line)
        box = line.get("box")
        if box:
            new_line["box"] = [
                [float(p[0]) / scale_factor, float(p[1]) / scale_y] for p in box
            ]
        words = line.get("words")
        if words:
            new_words = []
            for w in words:
                nw = dict(w)
                w_box = w.get("box")
                if w_box:
                    nw["box"] = [
                        [float(p[0]) / scale_factor, float(p[1]) / scale_y] for p in w_box
                    ]
                new_words.append(nw)
            new_line["words"] = new_words
        rescaled.append(new_line)
    return rescaled


def call_engine(ocr_exec, img, min_score=MIN_SCORE):
    """Executes OCR with RapidOCR options if supported, falling back gracefully for mock engines."""
    try:
        return ocr_exec(
            img,
            return_word_box=True,
            use_cls=False,
            box_thresh=0.5,
            unclip_ratio=1.6,
            text_score=min_score,
        )
    except TypeError:
        return ocr_exec(img)


def _shift_box_coords(item: Dict[str, Any], dx: float, dy: float) -> None:
    box = item.get("box")
    if box:
        item["box"] = [[float(p[0]) + dx, float(p[1]) + dy] for p in box]
    words = item.get("words")
    if words:
        for w in words:
            w_box = w.get("box")
            if w_box:
                w["box"] = [[float(p[0]) + dx, float(p[1]) + dy] for p in w_box]


def is_problematic_line(line: Dict[str, Any]) -> bool:
    score = float(line.get("score", 1.0) or 0.0)
    text = str(line.get("text", "") or "").strip()
    if score < 0.85:
        return True
    non_spaces = [c for c in text if not c.isspace()]
    if not non_spaces:
        return True
    alnum_count = sum(1 for c in non_spaces if c.isalnum())
    if (alnum_count / len(non_spaces)) < 0.60:
        return True
    for c in text:
        if c in "•@#$%^&*~`|\\_":
            return True
    return False


def score_ocr_candidate(text: str, score: float) -> float:
    trimmed = text.strip()
    if not trimmed:
        return 0.0
    non_spaces = [c for c in trimmed if not c.isspace()]
    if not non_spaces:
        return 0.0
    if not math.isfinite(score):
        return 0.0
    garbage_count = sum(
        not c.isalnum() and c not in STANDARD_PUNCTUATION
        and not (not c.isascii() and unicodedata.category(c).startswith("P"))
        for c in non_spaces
    )
    # Character count is not evidence that a token is an English word. The old
    # per-word bonus let several low-confidence nonsense tokens outrank a clear
    # reading. Preserve confidence and penalize actual corruption, not normal
    # apostrophes/hyphens, names, or the number of words in a crop.
    return min(1.0, max(0.0, score)) - garbage_count / len(non_spaces) * 0.6


def is_condensed_lettering(line: Dict[str, Any]) -> bool:
    """Detect unusually narrow Latin lettering, without guessing a font family."""
    text = str(line.get("text", ""))
    letters = sum(c.isascii() and c.isalpha() for c in text)
    box = line.get("box")
    if letters < 6 or not box or len(box) < 4:
        return False
    width = max(p[0] for p in box) - min(p[0] for p in box)
    height = max(p[1] for p in box) - min(p[1] for p in box)
    return height > 0 and width / letters / height < 0.32


def lettering_variants(img, condensed=False):
    """At most two contrast/stroke-normalized views for English comic lettering.

    Return actual x/y scale factors so both line and word boxes can be mapped
    back exactly. Polarity and stroke weight come from pixels, not OCR text.
    Original pixels are always tried separately by the caller.
    """
    if cv2 is None or np is None or not isinstance(img, np.ndarray):
        return
    height, width = img.shape[:2]
    if min(height, width) < 8 or width * height > SMALL_FRAME_PIXELS:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if int(gray.max()) == int(gray.min()):
        return
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Speech bubbles are usually light, but narration may use a dark panel.
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    if np.median(border) < 128:
        binary = 255 - binary
    ink = (binary == 0).astype(np.uint8)
    fraction = float(ink.mean())
    if not 0.001 <= fraction <= 0.45:
        return

    # Measure glyph height and stroke radius, ignoring tiny antialiasing specks
    # and panel-sized objects such as speech bubble outlines.
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    glyphs = [stat for stat in stats[1:count]
              if stat[cv2.CC_STAT_HEIGHT] >= 4 and stat[cv2.CC_STAT_AREA] >= 4
              and stat[cv2.CC_STAT_HEIGHT] <= height * 0.85]
    if not glyphs:
        return
    glyph_height = float(np.median([s[cv2.CC_STAT_HEIGHT] for s in glyphs]))
    radius = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke_ratio = float(np.percentile(radius[ink != 0], 90)) / glyph_height

    stretch = 1.8 if condensed else 1.0
    previous_size = None
    for scale in (1.0, 2.0):
        fit = min(scale, 2000 / max(width * stretch, height))
        out_w, out_h = max(1, round(width * stretch * fit)), max(1, round(height * fit))
        if (out_w, out_h) == previous_size:
            break  # the size ceiling must not turn one view into two votes
        previous_size = (out_w, out_h)
        prepared = cv2.resize(binary, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if scale == 2.0 and fit >= 1.5:
            kernel = np.ones((2, 2), dtype=np.uint8)
            if stroke_ratio < 0.09:
                prepared = cv2.erode(prepared, kernel)  # strengthen thin ink
            elif stroke_ratio > 0.18:
                prepared = cv2.dilate(prepared, kernel)  # open heavy glyph counters
        yield cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR), out_w / width, out_h / height


class _RetryBudget:
    """Bound optional OCR work across all crops and normalization variants."""

    def __init__(self, calls=MAX_RETRY_CALLS, seconds=MAX_RETRY_SECONDS):
        self.remaining = calls
        self.deadline = time.perf_counter() + seconds

    def available(self):
        return self.remaining > 0 and time.perf_counter() < self.deadline

    def take(self):
        if not self.available():
            return False
        self.remaining -= 1
        return True


def _retry_lines(ocr_exec, image, min_score, budget):
    if not budget.take():
        return []
    return extract_lines_from_result(call_engine(ocr_exec, image, min_score))


def _refinement_loses_text(original, candidate, min_score):
    """A more confident fragment must not erase an already credible reading."""
    if not filter_ocr_lines([original], min(min_score, MIN_SCORE))[0]:
        return False
    tokens = lambda text: tuple(word.strip('.,!?:;"“”«»()[]{}').casefold()
                                for word in str(text).split())
    full, part = tokens(original.get("text", "")), tokens(candidate.get("text", ""))
    if not part or len(part) >= len(full):
        return False
    if any(full[i:i + len(part)] == part for i in range(len(full) - len(part) + 1)):
        return True
    outer, inner = original.get("box"), candidate.get("box")
    if not outer or not inner:
        return False
    return (min(p[0] for p in inner) >= min(p[0] for p in outer) - 2
            and max(p[0] for p in inner) <= max(p[0] for p in outer) + 2
            and min(p[1] for p in inner) >= min(p[1] for p in outer) - 2
            and max(p[1] for p in inner) <= max(p[1] for p in outer) + 2
            and _compute_box_iou(outer, inner) < 0.97)


def refine_confirmed_words(img, line, ocr_exec, min_score, budget):
    """Resolve credible line errors and digit/letter ambiguity by image agreement."""
    words = line.get("words") or []
    if not words or not budget.available() or cv2 is None or np is None:
        return line
    ambiguous = [i for i, word in enumerate(words) if re.search(r"\d[OIlo]|[OIlo]\d", word["text"])]
    weak_line = line.get("_word_check", False) or 0.9 <= float(line.get("score", 0)) < 0.97
    spaced = " ".join(word["text"] for word in words)
    spacing_conflict = (line.get("_spacing_check", False) or
                        (spaced != line["text"] and "".join(spaced.split()) == "".join(line["text"].split())))
    if not ambiguous and not weak_line and not spacing_conflict:
        return line
    # A whole-line consensus may correct one word while retaining another word
    # whose readings conflict. Numeric ambiguities get a tight independent crop.
    targets = ambiguous if ambiguous else [None]
    replacements = {}
    for target in targets:
        box = words[target]["box"] if target is not None else line["box"]
        pad = 4 if target is not None else 12
        x1 = max(0, int(min(p[0] for p in box)) - pad)
        y1 = max(0, int(min(p[1] for p in box)) - pad)
        x2 = min(img.shape[1], int(max(p[0] for p in box)) + pad)
        y2 = min(img.shape[0], int(max(p[1] for p in box)) + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        views, line_views = [], []
        for prepared, sx, sy in lettering_variants(img[y1:y2, x1:x2], target is not None and target not in ambiguous):
            if not budget.available():
                break
            candidates = rescale_boxes_down(_retry_lines(ocr_exec, prepared, min_score, budget), sx, sy)
            candidates, _ = filter_ocr_lines(candidates, max(0.85 if target is not None else 0.9, min_score))
            detected = []
            matching_lines = []
            for candidate in candidates:
                _shift_box_coords(candidate, x1, y1)
                if _compute_box_iou(box, candidate["box"]) > 0.5:
                    matching_lines.append(candidate)
                detected.extend(candidate.get("words") or [candidate])
            views.append(detected)
            line_views.append(matching_lines)
        if len(views) != 2:
            continue
        if (target is None and spacing_conflict and all(len(view) == 1 for view in line_views)
                and line_views[0][0]["text"] == line_views[1][0]["text"]
                and "".join(line_views[0][0]["text"].split()) == "".join(line["text"].split())):
            return align_detected_words(line_views[0][0])
        if spacing_conflict:
            continue
        for index in ([target] if target is not None else range(len(words))):
            original = words[index]
            votes = []
            for view in views:
                matches = [word for word in view if float(word.get("score", 0)) >= max(0.95, min_score)
                           and _compute_box_iou(original["box"], word["box"]) > 0.5]
                if len(matches) != 1:
                    break
                votes.append(matches[0])
            if len(votes) == 2 and votes[0]["text"].casefold() != votes[1]["text"].casefold():
                if target is None and index not in targets and len(targets) < 3:
                    targets.append(index)
                continue
            if len(votes) != 2:
                continue
            confirmed = min(votes, key=lambda word: word["score"])
            if confirmed["text"].strip(".,!?:;…").casefold() == original["text"].strip(".,!?:;…").casefold():
                continue
            text = confirmed["text"].upper() if original["text"].isupper() else confirmed["text"]
            replacements[index] = {**confirmed, "text": text}
    if not replacements:
        return line
    updated = [replacements.get(i, word) for i, word in enumerate(words)]
    return {**line, "text": " ".join(word["text"] for word in updated), "words": updated}


def refine_suspicious_ending(img_bgr, line, ocr_exec, min_score, budget):
    """Confirm a damaged terminal token with two normalized word crops."""
    words = line.get("words") or []
    if not words or not budget.available() or cv2 is None or np is None:
        return line
    word = words[-1]
    text = str(word.get("text", ""))
    # This only selects a crop to re-read; punctuation is never deleted by a
    # spelling rule. Acronyms, domains and ordinary sentence endings stay alone.
    suspicious = re.fullmatch(r"([A-Za-z][A-Za-z'’\-]{2,})[.!?][A-Za-z]{1,2}[.!?]", text)
    box = word.get("box")
    if not suspicious or not box or len(box) < 4 or not line.get("text", "").rstrip().endswith(text):
        return line
    height, width = img_bgr.shape[:2]
    x1 = max(0, int(min(p[0] for p in box)) - 8)
    y1 = max(0, int(min(p[1] for p in box)) - 8)
    x2 = min(width, int(max(p[0] for p in box)) + 8)
    y2 = min(height, int(max(p[1] for p in box)) + 8)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return line
    readings = []
    for prepared, sx, sy in lettering_variants(img_bgr[y1:y2, x1:x2]):
        if not budget.available():
            return line
        candidates = rescale_boxes_down(_retry_lines(ocr_exec, prepared, min_score, budget), sx, sy)
        candidates, _ = filter_ocr_lines(candidates, max(0.95, min_score))
        matching = []
        for candidate in candidates:
            _shift_box_coords(candidate, x1, y1)
            if (candidate["text"].rstrip(".!?").casefold() == suspicious.group(1).casefold()
                    and _compute_box_iou(box, candidate["box"]) > 0.5):
                matching.append(candidate)
        if len(matching) != 1:
            return line
        readings.append(matching[0])
    if (len(readings) != 2 or readings[0]["text"] != readings[1]["text"]
            or _compute_box_iou(readings[0]["box"], readings[1]["box"]) <= 0.5):
        return line
    confirmed = min(readings, key=lambda candidate: candidate["score"])
    # Keep the surrounding sentence and its geometry; update the word geometry
    # too, so cursor lookups cannot retain the rejected terminal fragment.
    return {**line, "text": line["text"].rstrip()[:-len(text)] + confirmed["text"],
            "words": [*words[:-1], confirmed],
            "score": min(float(line.get("score", 0)), confirmed["score"])}


def refine_problematic_box(img_bgr, line: Dict[str, Any], ocr_exec,
                           min_score=MIN_SCORE, english_lettering=False, budget=None) -> Dict[str, Any]:
    budget = _RetryBudget() if budget is None else budget
    box = line.get("box")
    if (not budget.available() or not box or len(box) < 4 or cv2 is None
            or np is None or not isinstance(img_bgr, np.ndarray)):
        return line
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    h_img, w_img = img_bgr.shape[:2]
    pad = 20
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w_img, int(max(xs)) + pad)
    y2 = min(h_img, int(max(ys)) + pad)
    if x2 <= x1 + 5 or y2 <= y1 + 5:
        return line

    crop = img_bgr[y1:y2, x1:x2]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return line

    best_cand = line
    best_eval = score_ocr_candidate(line.get("text", ""), float(line.get("score", 0.0) or 0.0))

    def consider(candidates, sx=1.0, sy=1.0, spacing_margin=False):
        nonlocal best_cand, best_eval
        candidates = rescale_boxes_down(candidates, sx, sy)
        candidates, _ = filter_ocr_lines(candidates, min_score)
        for candidate in candidates:
            _shift_box_coords(candidate, x1, y1)
            if _compute_box_iou(box, candidate["box"]) <= 0.5:
                continue
            if (_refinement_loses_text(line, candidate, min_score)
                    or _refinement_loses_text(best_cand, candidate, min_score)):
                continue
            value = score_ocr_candidate(candidate["text"], float(candidate["score"]))
            same_letters = ("".join(str(line.get("text", "")).split())
                            == "".join(candidate["text"].split()))
            margin = 0.005 if spacing_margin and same_letters else 0.05
            if value > best_eval + margin:
                best_eval, best_cand = value, candidate

    min_ch = np.min(crop, axis=2)

    # Variant (b): min(R,G,B) -> negative
    neg_min = (255 - min_ch).astype(np.uint8)
    crop_neg = np.stack([neg_min, neg_min, neg_min], axis=2)
    consider(_retry_lines(ocr_exec, crop_neg, min_score, budget))
    if not budget.available():
        return best_cand

    # Variant (c): bright mask min(R,G,B) >= 220 -> black text on white
    mask_ch = np.where(min_ch >= 220, 0, 255).astype(np.uint8)
    crop_mask = np.stack([mask_ch, mask_ch, mask_ch], axis=2)
    consider(_retry_lines(ocr_exec, crop_mask, min_score, budget))

    if english_lettering:
        for prepared, sx, sy in lettering_variants(crop, is_condensed_lettering(line)):
            if not budget.available():
                break
            consider(_retry_lines(ocr_exec, prepared, min_score, budget), sx, sy, spacing_margin=True)
    return best_cand


def _run_ocr_with_variants(img, ocr_exec, min_score=MIN_SCORE):
    """Прогоняет OCR на изображении и двух его вариантах (инверсия и белая маска)."""
    res_orig = call_engine(ocr_exec, img, min_score)
    lines_orig = extract_lines_from_result(res_orig)

    img_inv = 255 - img
    res_inv = call_engine(ocr_exec, img_inv, min_score)
    lines_inv = extract_lines_from_result(res_inv)

    is_white = np.min(img, axis=2) >= 200
    mask_ch = np.where(is_white, 0, 255).astype(np.uint8)
    img_mask = np.stack([mask_ch, mask_ch, mask_ch], axis=2)
    res_mask = call_engine(ocr_exec, img_mask, min_score)
    lines_mask = extract_lines_from_result(res_mask)
    return lines_orig, lines_inv, lines_mask


def save_ocr_debug_dump(
    img_bgr, lines_with_status: List[Dict[str, Any]], req_id: Any
) -> None:
    """
    Сохраняет отладочный дамп кадра и распознанных строк при SCREEN_TRANSLATOR_OCR_DEBUG=1.
    Ошибки логируются и не прерывают работу.
    """
    try:
        import shutil

        server_dir = os.path.dirname(os.path.abspath(__file__))
        debug_dir = os.path.join(server_dir, "ocr_debug")
        os.makedirs(debug_dir, exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"frame_{timestamp_str}_{req_id}"
        frame_dir = os.path.join(debug_dir, run_name)
        os.makedirs(frame_dir, exist_ok=True)

        # Сохранение JSON со всеми сырыми строками и полями kept/reason
        json_path = os.path.join(frame_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(lines_with_status, f, ensure_ascii=False, indent=2)

        # Если доступен cv2 и img_bgr является numpy-массивом — сохраняем кадры
        if cv2 is not None and np is not None and isinstance(img_bgr, np.ndarray):
            input_path = os.path.join(frame_dir, "input.png")
            cv2.imwrite(input_path, img_bgr)

            annotated = img_bgr.copy()
            for item in lines_with_status:
                box = item.get("box")
                idx = item.get("index", 0)
                kept = item.get("kept", True)
                if box and len(box) >= 4:
                    pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                    color = (0, 255, 0) if kept else (0, 0, 255)
                    cv2.polylines(
                        annotated, [pts], isClosed=True, color=color, thickness=2
                    )
                    pt = (int(pts[0][0][0]), int(pts[0][0][1]))
                    cv2.putText(
                        annotated,
                        str(idx),
                        pt,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            annotated_path = os.path.join(frame_dir, "annotated.png")
            cv2.imwrite(annotated_path, annotated)

        # Хранить не более MAX_DEBUG_RUNS последних наборов файлов
        existing = [
            os.path.join(debug_dir, d)
            for d in os.listdir(debug_dir)
            if os.path.isdir(os.path.join(debug_dir, d))
        ]
        if len(existing) > MAX_DEBUG_RUNS:
            existing.sort(key=os.path.getctime)
            for old_dir in existing[: len(existing) - MAX_DEBUG_RUNS]:
                shutil.rmtree(old_dir, ignore_errors=True)
    except Exception as e:
        logger.debug(f"Failed to write OCR debug dump: {e}")


def process_request(
    header: Dict[str, Any], raw_rgb: Optional[bytes], engine
) -> Dict[str, Any]:
    """Processes a single parsed request and produces a response dictionary."""
    cmd = header.get("cmd")
    if cmd == "ping":
        return {"ok": True, "cmd": "ping"}
    if cmd == "shutdown":
        return {"ok": True, "cmd": "shutdown"}

    req_id = header.get("id")
    if req_id is None:
        return {"ok": False, "error": "Missing 'id' or 'cmd' in request"}

    min_score = header.get("min_score", MIN_SCORE)
    if (isinstance(min_score, bool) or not isinstance(min_score, (int, float))
            or not math.isfinite(min_score) or not 0 <= min_score <= 1):
        return {"id": req_id, "ok": False, "error": "min_score must be between 0 and 1"}

    lang = str(header.get("lang", "")).strip().lower()
    base_lang = lang.split("-")[0].split("_")[0]
    supported_langs = {"en", "ch", "zh", "ja", "ko", "ru"}
    if base_lang not in supported_langs:
        return {
            "id": req_id,
            "ok": False,
            "error": f"Language '{lang}' is not supported yet (supported: {', '.join(sorted(supported_langs))})",
        }

    width = header.get("width")
    height = header.get("height")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return {
            "id": req_id,
            "ok": False,
            "error": f"Invalid frame dimensions: width={width}, height={height}",
        }

    expected_len = width * height * 3
    if raw_rgb is None or len(raw_rgb) != expected_len:
        actual_len = len(raw_rgb) if raw_rgb else 0
        return {
            "id": req_id,
            "ok": False,
            "error": f"Invalid RGB buffer size: expected {expected_len}, got {actual_len}",
        }

    t0 = time.perf_counter()
    try:
        if np is not None:
            # Convert raw BGR directly to numpy array (H, W, 3) for RapidOCR/OpenCV
            img_bgr = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(
                (height, width, 3)
            )
        else:
            # In test environments without numpy installed, pass raw buffer directly to mock engine
            img_bgr = raw_rgb

        if hasattr(engine, "get_engine"):
            ocr_exec = engine.get_engine(base_lang)
        elif isinstance(engine, dict):
            ocr_exec = engine.get(base_lang) or next(iter(engine.values()))
        else:
            ocr_exec = engine

        use_upscale = should_upscale_frame(
            width,
            height,
            cv2 is not None and np is not None and isinstance(img_bgr, np.ndarray),
        )

        use_variants = (
            not use_upscale
            and os.environ.get(SCREEN_TRANSLATOR_OCR_VARIANTS_ENV) == "1"
            and np is not None
            and isinstance(img_bgr, np.ndarray)
        )
        if use_upscale:
            original = extract_lines_from_result(call_engine(ocr_exec, img_bgr, min_score))
            up_w = width * UPSCALE_SMALL_FRAMES
            up_h = height * UPSCALE_SMALL_FRAMES
            img_up = cv2.resize(
                img_bgr, (up_w, up_h), interpolation=cv2.INTER_CUBIC
            )
            result = call_engine(ocr_exec, img_up, min_score)
            raw_lines = extract_lines_from_result(result)
            raw_lines = rescale_boxes_down(raw_lines, UPSCALE_SMALL_FRAMES)
            raw_lines = original + raw_lines
        elif use_variants:
            lines_orig, lines_inv, lines_mask = _run_ocr_with_variants(
                img_bgr, ocr_exec, min_score
            )
            raw_lines = lines_orig + lines_inv + lines_mask
        else:
            result = call_engine(ocr_exec, img_bgr, min_score)
            raw_lines = extract_lines_from_result(result)

        # Step 3: Targeted retry pass on problematic boxes. English lettering
        # passes are bounded and keep every word box in original coordinates.
        if np is not None and isinstance(img_bgr, np.ndarray):
            # Deduplicate before spending the shared retry budget, retaining
            # damaged readings for rescue. Apply text/score filters afterwards.
            raw_lines = merge_ocr_lines([raw_lines], filter_before_merge=False)
            # Resolve spacing evidence before spending remaining retries on
            # lower-confidence letter readings elsewhere in a dense paragraph.
            raw_lines.sort(key=lambda item: ("".join(item["text"].split()) !=
                "".join(w["text"] for w in item.get("words", [])) or
                " ".join(w["text"] for w in item.get("words", [])) == item["text"]))
            refined = []
            lettering_budget = 4
            retry_budget = _RetryBudget()
            for item in raw_lines:
                if base_lang == "en":
                    item = refine_confirmed_words(img_bgr, item, ocr_exec, min_score, retry_budget)
                    item = refine_suspicious_ending(img_bgr, item, ocr_exec, min_score, retry_budget)
                font_retry = (base_lang == "en" and lettering_budget > 0
                              and (is_problematic_line(item) or is_condensed_lettering(item)))
                if retry_budget.available() and (is_problematic_line(item) or font_retry):
                    item = refine_problematic_box(img_bgr, item, ocr_exec, min_score,
                                                  english_lettering=font_retry, budget=retry_budget)
                    lettering_budget -= int(font_retry)
                refined.append(item)
            raw_lines = refined

            # If faint lettering was missed entirely, recover it only when the
            # two normalized views agree. Never run this on full-size pages.
            if base_lang == "en" and not filter_ocr_lines(raw_lines, min_score)[0]:
                recovered = []
                for prepared, sx, sy in lettering_variants(img_bgr):
                    if not retry_budget.available():
                        break
                    candidates = rescale_boxes_down(
                        _retry_lines(ocr_exec, prepared, min_score, retry_budget), sx, sy)
                    recovered.append(filter_ocr_lines(candidates, max(0.85, min_score))[0])
                if len(recovered) == 2:
                    raw_lines.extend(a for a in recovered[0] if any(
                        " ".join(a["text"].split()) == " ".join(b["text"].split())
                        and _compute_box_iou(a["box"], b["box"]) > 0.5
                        for b in recovered[1]))

        kept_lines, rejected_lines = filter_ocr_lines(raw_lines, min_score)
        if use_variants or use_upscale:
            kept_lines = merge_ocr_lines([kept_lines], min_score)
        kept_ids = {id(line) for line in kept_lines}
        kept_lines = [align_detected_words(line) for line in kept_lines]
        for line in kept_lines:
            line.pop("_spacing_check", None)
            line.pop("_word_check", None)

        if os.environ.get(SCREEN_TRANSLATOR_OCR_DEBUG_ENV) == "1":
            rejection_map = {id(item): reason for item, reason in rejected_lines}
            debug_items = []
            for i, item in enumerate(raw_lines):
                box = item.get("box") if isinstance(item, dict) else item[0]
                text = item.get("text") if isinstance(item, dict) else item[1]
                score = item.get("score") if isinstance(item, dict) else item[2]
                is_kept = id(item) in kept_ids
                reason = rejection_map.get(id(item)) if not is_kept else None
                debug_items.append(
                    {
                        "index": i,
                        "box": box,
                        "text": text,
                        "score": score,
                        "kept": is_kept,
                        "reason": reason,
                    }
                )
            save_ocr_debug_dump(img_bgr, debug_items, req_id)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "id": req_id,
            "ok": True,
            "lines": kept_lines,
            "ms": elapsed_ms,
        }
    except Exception as e:
        logger.exception("Error during OCR recognition")
        return {"id": req_id, "ok": False, "error": str(e)}


def resolve_models_dir() -> Optional[str]:
    """
    Locates the models directory across all execution modes:
    1. PyInstaller bundled temporary directory (sys._MEIPASS / models)
    2. Next to the standalone executable (sys.executable dir / models)
    3. Next to the server.py script file
    4. Working directory fallback (./models or ./ocr_helper/models)
    """
    candidates = []

    # 1. PyInstaller --onefile temp directory
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, "models"))

    # 2. Next to standalone binary (when models/ is placed alongside ocr_helper.exe)
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        candidates.append(os.path.join(exe_dir, "models"))
        candidates.append(os.path.join(exe_dir, "ocr_helper", "models"))

    # 3. Next to this source file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "models"))

    # 4. Current working directory
    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, "models"))
    candidates.append(os.path.join(cwd, "ocr_helper", "models"))

    for path in candidates:
        if os.path.isdir(path) and os.path.exists(path):
            return path
    return None


def run_server(stdin=None, stdout=None, engine=None):
    if stdin is None:
        stdin = sys.stdin.buffer
    if stdout is None:
        stdout = sys.stdout.buffer

    if engine is None:
        models_dir = resolve_models_dir()
        engine = EngineManager(models_dir)
        # The client waits up to 30 seconds for ready, whereas individual OCR
        # requests use a shorter budget. Load the configured model here so a
        # cold model download cannot trigger an endless request/restart loop.
        # Third-party initialization may print; stdout remains protocol-only.
        try:
            with redirect_stdout(sys.stderr):
                ocr_engine = engine.get_engine(os.environ.get(SCREEN_TRANSLATOR_OCR_LANG_ENV) or "en")
                # RapidOCR 3.10 constructs its ONNX sessions lazily. Initialize
                # each enabled model explicitly: an empty-image inference may
                # never reach the recognizer. Legacy engines load eagerly.
                for stage in ("det", "cls", "rec"):
                    load_model = getattr(ocr_engine, f"_load_{stage}_model", None)
                    if getattr(ocr_engine, f"use_{stage}", True) and callable(load_model):
                        load_model()
        except Exception:
            logger.exception("Failed to initialize OCR models before ready")
            raise

    # Send ready notification to client
    ready_msg = json.dumps({"event": "ready"}).encode("utf-8") + b"\n"
    stdout.write(ready_msg)
    stdout.flush()
    logger.info("OCR helper server is ready.")

    while True:
        line = stdin.readline()
        if not line:
            logger.info("stdin EOF received, exiting server.")
            break

        line_str = line.decode("utf-8", errors="replace").strip()
        if not line_str:
            continue

        try:
            header = json.loads(line_str)
        except Exception as e:
            resp = (
                json.dumps({"ok": False, "error": f"Invalid JSON: {e}"}).encode(
                    "utf-8"
                )
                + b"\n"
            )
            stdout.write(resp)
            stdout.flush()
            continue

        raw_rgb = None
        cmd = header.get("cmd")
        if cmd == "shutdown":
            resp = (
                json.dumps({"ok": True, "cmd": "shutdown"}).encode("utf-8")
                + b"\n"
            )
            stdout.write(resp)
            stdout.flush()
            logger.info("Shutdown command processed. Exiting.")
            break

        if not cmd:
            width = header.get("width", 0)
            height = header.get("height", 0)
            if (
                isinstance(width, int)
                and isinstance(height, int)
                and width > 0
                and height > 0
            ):
                expected_bytes = width * height * 3
                raw_rgb = read_exact(stdin, expected_bytes)
                if raw_rgb is None:
                    logger.warning(
                        "EOF encountered while reading RGB image bytes."
                    )
                    break

        resp_dict = process_request(header, raw_rgb, engine)
        resp_bytes = (
            json.dumps(resp_dict, ensure_ascii=False).encode("utf-8") + b"\n"
        )
        stdout.write(resp_bytes)
        stdout.flush()


if __name__ == "__main__":
    run_server()
