#!/usr/bin/env python3
"""
Screen Translator - RapidOCR Python Helper Server
Communicates with the Rust application via stdin/stdout binary protocol.
Stdout is strictly reserved for single-line JSON messages ending with '\n'.
All logs, diagnostic messages, and warnings are directed to stderr.
"""

from collections import OrderedDict
import json
import logging
import os
import sys
import time
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

# Пороги масштабирования малых кадров
SMALL_FRAME_PIXELS = 1_500_000
UPSCALE_SMALL_FRAMES = 2

# Пороги фильтрации мусора
MIN_SCORE = 0.60              # Минимальная уверенность RapidOCR (меньше -> "low_score")
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
        from rapidocr import RapidOCR

        params = {"Rec.lang_type": rec_lang}
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
        if boxes is not None and txts is not None and scores is not None:
            for box, text, score in zip(boxes, txts, scores):
                box_list = box.tolist() if hasattr(box, "tolist") else list(box)
                lines.append(
                    {
                        "text": str(text),
                        "box": box_list,
                        "score": float(score),
                    }
                )
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
                lines.append(
                    {
                        "text": str(text),
                        "box": box_list,
                        "score": float(score),
                    }
                )
    return lines


def filter_ocr_lines(lines):
    """
    Чистая функция фильтрации OCR-строк по правилам:
    1. score < MIN_SCORE (0.60) -> отбросить ("low_score")
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
        if score < MIN_SCORE:
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


def merge_ocr_lines(variant_groups):
    """
    Чистая функция объединения строк от нескольких вариантов кадра:
    1. К каждому варианту отдельно применяется filter_ocr_lines.
    2. Все прошедшие фильтр строки объединяются.
    3. При сильном перекрытии рамок (IoU прямоугольников > 0.5) оставляется строка с большей оценкой.
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
        kept, _ = filter_ocr_lines(group)
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

    # Сортируем кандидатов по убыванию оценки уверенности
    candidates = sorted(all_kept, key=_get_score, reverse=True)
    merged = []

    for cand in candidates:
        cand_box = _get_box(cand)
        if not cand_box or len(cand_box) < 4:
            continue

        overlaps = False
        for selected in merged:
            sel_box = _get_box(selected)
            if sel_box and len(sel_box) >= 4:
                if _compute_box_iou(cand_box, sel_box) > 0.5:
                    overlaps = True
                    break

        if not overlaps:
            merged.append(cand)

    return merged


def should_upscale_frame(width: int, height: int, cv2_available: bool) -> bool:
    """Определяет, нужно ли увеличивать кадр для улучшения распознавания мелкого текста."""
    return cv2_available and (width * height <= SMALL_FRAME_PIXELS)


def rescale_boxes_down(
    lines: List[Dict[str, Any]], scale_factor: float
) -> List[Dict[str, Any]]:
    """Пересчитывает координаты 4-точечных рамок найденных строк обратно в систему исходного кадра."""
    if scale_factor <= 1.0 or not lines:
        return lines
    rescaled = []
    for line in lines:
        new_line = dict(line)
        box = line.get("box")
        if box:
            new_line["box"] = [
                [float(p[0]) / scale_factor, float(p[1]) / scale_factor] for p in box
            ]
        rescaled.append(new_line)
    return rescaled


def _run_ocr_with_variants(img, ocr_exec):
    """Прогоняет OCR на изображении и двух его вариантах (инверсия и белая маска)."""
    res_orig = ocr_exec(img)
    lines_orig = extract_lines_from_result(res_orig)

    img_inv = 255 - img
    res_inv = ocr_exec(img_inv)
    lines_inv = extract_lines_from_result(res_inv)

    is_white = np.min(img, axis=2) >= 200
    mask_ch = np.where(is_white, 0, 255).astype(np.uint8)
    img_mask = np.stack([mask_ch, mask_ch, mask_ch], axis=2)
    res_mask = ocr_exec(img_mask)
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

        if use_upscale:
            up_w = width * UPSCALE_SMALL_FRAMES
            up_h = height * UPSCALE_SMALL_FRAMES
            img_up = cv2.resize(
                img_bgr, (up_w, up_h), interpolation=cv2.INTER_CUBIC
            )
            lines_orig, lines_inv, lines_mask = _run_ocr_with_variants(
                img_up, ocr_exec
            )

            # Пересчитываем координаты рамок обратно в систему исходного кадра ДО фильтрации и объединения
            lines_orig = rescale_boxes_down(lines_orig, UPSCALE_SMALL_FRAMES)
            lines_inv = rescale_boxes_down(lines_inv, UPSCALE_SMALL_FRAMES)
            lines_mask = rescale_boxes_down(lines_mask, UPSCALE_SMALL_FRAMES)

            kept_lines = merge_ocr_lines([lines_orig, lines_inv, lines_mask])
            raw_lines = lines_orig + lines_inv + lines_mask
            rejected_lines = []
        elif (
            os.environ.get(SCREEN_TRANSLATOR_OCR_VARIANTS_ENV) == "1"
            and np is not None
            and isinstance(img_bgr, np.ndarray)
        ):
            lines_orig, lines_inv, lines_mask = _run_ocr_with_variants(
                img_bgr, ocr_exec
            )
            kept_lines = merge_ocr_lines([lines_orig, lines_inv, lines_mask])
            raw_lines = lines_orig + lines_inv + lines_mask
            rejected_lines = []
        else:
            result = ocr_exec(img_bgr)
            raw_lines = extract_lines_from_result(result)
            kept_lines, rejected_lines = filter_ocr_lines(raw_lines)

        if os.environ.get(SCREEN_TRANSLATOR_OCR_DEBUG_ENV) == "1":
            rejection_map = {id(item): reason for item, reason in rejected_lines}
            kept_ids = {id(item) for item in kept_lines}
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
