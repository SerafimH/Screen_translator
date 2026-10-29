#!/usr/bin/env python3
"""
Screen Translator - RapidOCR Python Helper Server
Communicates with the Rust application via stdin/stdout binary protocol.
Stdout is strictly reserved for single-line JSON messages ending with '\n'.
All logs, diagnostic messages, and warnings are directed to stderr.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

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


def create_ocr_engine(models_dir: Optional[str] = None):
    """Initializes RapidOCR engine using local models if present."""
    try:
        from rapidocr import RapidOCR

        params = {}
        if models_dir and os.path.exists(models_dir):
            params["Global.model_root_dir"] = models_dir
            params["Det.lang_type"] = "en"
            params["Rec.lang_type"] = "en"
            params["Cls.lang_type"] = "ch"
        logger.info(f"Initializing RapidOCR engine with models_dir={models_dir}...")
        return RapidOCR(params=params if params else None)
    except ImportError:
        try:
            from rapidocr_onnxruntime import RapidOCR

            logger.info("Initializing RapidOCR via rapidocr_onnxruntime fallback...")
            return RapidOCR()
        except ImportError as e:
            logger.error(f"RapidOCR is not installed: {e}")
            raise


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
    if lang != "en":
        return {
            "id": req_id,
            "ok": False,
            "error": f"Language '{lang}' is not supported yet (only 'en' is supported)",
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
        try:
            # Convert raw RGB to numpy array (H, W, 3) and invert to BGR for RapidOCR/OpenCV
            import numpy as np

            img_rgb = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(
                (height, width, 3)
            )
            img_bgr = img_rgb[:, :, ::-1]
        except ImportError:
            # In test environments without numpy installed, pass raw buffer directly to mock engine
            img_bgr = raw_rgb

        result = engine(img_bgr)
        lines = extract_lines_from_result(result)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "id": req_id,
            "ok": True,
            "lines": lines,
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
        try:
            engine = create_ocr_engine(models_dir)
        except Exception as e:
            logger.error(f"Failed to create OCR engine: {e}")
            sys.exit(1)

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
