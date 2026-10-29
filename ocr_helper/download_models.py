#!/usr/bin/env python3
"""
Downloads PP-OCR models for RapidOCR ahead of time into ocr_helper/models/
Run this script once after running setup.ps1:
    .venv/Scripts/python.exe download_models.py
"""

import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading RapidOCR PP-OCR models into: {models_dir}")

    try:
        from rapidocr.inference_engine.base import FileInfo
        from rapidocr.utils.download_models import download_task
        from rapidocr.utils.typings import EngineType, ModelType, OCRVersion, TaskType

        # 1. Text detection model (PP-OCRv6 English / multilingual)
        print("1/3 Downloading Text Detection model (PP-OCRv6)...")
        download_task(
            models_dir,
            FileInfo(
                engine_type=EngineType.ONNXRUNTIME,
                ocr_version=OCRVersion.PPOCRV6,
                task_type=TaskType.DET,
                lang_type="en",
                model_type=ModelType.SMALL,
            ),
        )

        # 2. Text direction classifier model (PP-OCRv4 mobile, default)
        print("2/3 Downloading Text Classifier model (PP-OCRv4)...")
        download_task(
            models_dir,
            FileInfo(
                engine_type=EngineType.ONNXRUNTIME,
                ocr_version=OCRVersion.PPOCRV4,
                task_type=TaskType.CLS,
                lang_type="ch",
                model_type=ModelType.MOBILE,
            ),
        )

        # 3. Text recognition model (PP-OCRv6 English / multilingual)
        print("3/3 Downloading Text Recognition model (PP-OCRv6)...")
        download_task(
            models_dir,
            FileInfo(
                engine_type=EngineType.ONNXRUNTIME,
                ocr_version=OCRVersion.PPOCRV6,
                task_type=TaskType.REC,
                lang_type="en",
                model_type=ModelType.SMALL,
            ),
        )

        print(f"\nAll models downloaded successfully into {models_dir}!")
    except ImportError as e:
        print(f"ERROR: rapidocr is not installed: {e}", file=sys.stderr)
        print(
            "Please run setup.ps1 first to install dependencies in .venv.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download models: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
