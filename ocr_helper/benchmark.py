"""Local OCR quality report: python -m ocr_helper.benchmark [--output path]."""

import argparse
import json
from pathlib import Path
import time

from ocr_helper.server import EngineManager, cv2, process_request, resolve_models_dir


def normalize(text):
    return " ".join(text.upper().split())


def edit_distance(expected, actual):
    previous = list(range(len(actual) + 1))
    for i, a in enumerate(expected, 1):
        current = [i]
        for j, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def quality(expected_lines, actual_lines):
    expected = normalize(" ".join(expected_lines))
    actual = normalize(" ".join(actual_lines))
    # A line is missing when no recognized line matches it within 20% character errors.
    missing = sum(
        not any(edit_distance(normalize(line), normalize(candidate))
                / max(1, len(normalize(line))) <= 0.2 for candidate in actual_lines)
        for line in expected_lines
    )
    return {"cer": edit_distance(expected, actual) / max(1, len(expected)),
            "missing_lines": missing, "text": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if cv2 is None:
        parser.error("OpenCV is required; install ocr_helper/requirements.txt")
    root = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ocr"
    cases = json.loads((root / "corpus.json").read_text(encoding="utf-8"))
    manager = EngineManager(resolve_models_dir())
    start = time.perf_counter()
    manager.get_engine("en")
    load_ms = (time.perf_counter() - start) * 1000
    results = []
    for i, case in enumerate(cases):
        img = cv2.imread(str(root / case["file"]))
        if img is None:
            raise RuntimeError(f"Cannot read {case['file']}")
        response = process_request({"id": i, "lang": "en", "width": img.shape[1],
                                    "height": img.shape[0], "min_score": 0.5},
                                   img.tobytes(), manager)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        lines = sorted(response["lines"], key=lambda line: min(p[1] for p in line["box"]))
        metrics = quality(case["lines"], [line["text"] for line in lines])
        results.append({"file": case["file"], **metrics, "ms": response["ms"],
                        "pass": metrics["cer"] <= case["max_cer"] and metrics["missing_lines"] == 0})
    report = {"engine": "RapidOCR", "load_ms": round(load_ms, 2), "results": results}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0 if all(item["pass"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
