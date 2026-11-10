"""Compare Google request variants on the audited English dialogue (network required)."""
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def translate(text, extra=""):
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ru&dt=t&dj=1" + extra
    request = urllib.request.Request(url, data=urllib.parse.urlencode({"q": text}).encode(),
                                     headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        body = json.load(response)
    return "".join(part.get("trans", "") for part in body.get("sentences", []))


def main():
    cases = json.loads((ROOT / "tests/fixtures/ocr/corpus.json").read_text(encoding="utf-8"))
    destination = ROOT / "target/google-request-comparison.json"
    previous = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else []
    cached = {(item["file"], item["variant"], item["input"]): item for item in previous}
    report, history = [], []
    for case in cases:
        if not case["file"].startswith("manhwa_"):
            continue
        texts = case["blocks"]
        if case["file"] in ("manhwa_08.png", "manhwa_11.png", "manhwa_17.png"):
            for text in texts[:1]:
                context = "\n[---]\n".join(history[-5:] + [text])
                variants = {
                    "upper_context": (context, ""),
                    "upper_alone": (text, ""),
                    "sentence_alone": (text.capitalize(), ""),
                    "upper_context_nmt": (context, "&model=nmt"),
                    "upper_alone_nmt": (text, "&model=nmt"),
                    "sentence_context": ("\n[---]\n".join(x.capitalize() for x in history[-5:] + [text]), ""),
                    "sentence_period": (text.capitalize().rstrip(".") + ".", ""),
                    "sentence_unicode_ellipsis": (text.capitalize().replace("...", "…"), ""),
                    "sentence_pronoun": (text.capitalize().replace(", i ", ", I "), ""),
                    "sentence_pronoun_context": ("\n[---]\n".join(x.capitalize().replace(", i ", ", I ") for x in history[-5:] + [text]), ""),
                    "english_idiom_paraphrase": (text.capitalize().replace(", i ", ", I ").replace(
                        "a hell of a", "an extremely intense").replace("Come to think of it", "Now that I think about it"), ""),
                    "tough_fight_paraphrase": (text.capitalize().replace(", i ", ", I ").replace(
                        "a hell of a fight", "a very tough fight").replace("Come to think of it", "Now that I think about it"), ""),
                    "conservative_case": (" ".join(word.lower() if word.strip(".,!?:;") in {
                        "JUST", "LIKE", "YOU", "SAID,", "HAD", "A", "OF", "HIM", "FROM", "WHERE", "THEY", "WERE",
                        "HOLDING", "AND", "SENT", "TO", "THINK", "IT...", "COME"} else word for word in text.split()), ""),
                }
                for variant, (source, extra) in variants.items():
                    key = case["file"], variant, source
                    result = cached.get(key) or {"file": case["file"], "variant": variant, "input": source,
                                                "translation": translate(source, extra)}
                    report.append(result)
                    if key not in cached:
                        print(json.dumps(result, ensure_ascii=False), flush=True)
        history.extend(texts)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
