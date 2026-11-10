"""Verify audit fixes and measure WinRT without modifying Rust application sources.

Run `cargo test` first, then `python audit/probes.py`. Uses its compiled dependencies.
The probes now assert the corrected behavior. Original evidence is in results/.
Rust source is passed to rustc on stdin; no .rs files are created or modified.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
include!(r"__MAIN__");

#[cfg(test)]
mod audit_probes {
    use super::*;
    use types::{RecognizedLine, RecognizedWord, ScreenRect, AppConfig, TranslationProvider};

    fn line(text: &str, y: i32) -> RecognizedLine {
        let rect = ScreenRect::new(100, y, 240, 20);
        RecognizedLine { text: text.into(), rect, confidence: Some(0.99),
            words: vec![RecognizedWord { text: text.split_whitespace().next().unwrap().into(),
                rect: ScreenRect::new(100, y, 50, 20), confidence: Some(0.99), is_estimated: false }] }
    }

    #[test]
    fn paragraph_context_keeps_negation_at_an_accepted_line_gap() {
        let blocks = ocr::cluster_lines_into_blocks(vec![line("DO NOT", 100), line("OPEN THE DOOR", 140)], None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "DO NOT OPEN THE DOOR");
        let cursor = types::find_text_under_cursor(&blocks, 110, 150).unwrap();
        assert_eq!(cursor.sentence_text, "DO NOT OPEN THE DOOR");
        println!("VERIFIED gap: paragraph={:?}, cursor={:?}", blocks[0].text, cursor.sentence_text);
    }

    #[test]
    fn paragraph_context_keeps_wrapped_words() {
        for (first, second, expected) in [
            ("well-", "known", "well-known"),
            ("D-", "Did he find out?!", "D-Did he find out?!"),
        ] {
            let blocks = ocr::cluster_lines_into_blocks(vec![line(first, 100), line(second, 124)], None);
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].text, expected);
            let cursor = types::find_text_under_cursor(&blocks, 110, 134).unwrap();
            assert_eq!(cursor.sentence_text, expected);
            println!("VERIFIED hyphen: paragraph={:?}, cursor={:?}", blocks[0].text, cursor.sentence_text);
        }
    }

    #[test]
    fn vision_parser_rejects_reversed_and_out_of_image_boxes() {
        let roi = ScreenRect::new(100, 200, 800, 600);
        for coords in ["[-100,-100,2000,2000]", "[900,900,100,100]"] {
            let input = format!(r#"[{{"original":"Hello","translation":"Привет","box_2d":{coords}}}]"#);
            assert!(translator::parse_vision_blocks_json_with_box_2d(&input, &roi).is_err());
            println!("VERIFIED vision geometry rejected: {coords}");
        }
    }

    #[test]
    fn vision_parser_rejects_huge_model_coordinates_without_panicking() {
        let input = r#"[{"original":"Hello","translation":"Привет","box_2d":[0,1e100,100,1e100]}]"#;
        let result = std::panic::catch_unwind(|| translator::parse_vision_blocks_json_with_box_2d(
            input, &ScreenRect::new(100, 200, 800, 600)));
        assert!(result.unwrap().is_err());
        println!("VERIFIED vision overflow: invalid coordinates return an error without panic");
    }

    async fn mock_vision() -> (String, tokio::task::JoinHandle<serde_json::Value>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let body_start;
            loop {
                let mut buffer = [0; 4096];
                let read = stream.read(&mut buffer).await.unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
                if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    let header = String::from_utf8_lossy(&request[..end]);
                    let length: usize = header.lines().filter_map(|line| line.split_once(':'))
                        .find(|(name, _)| name.eq_ignore_ascii_case("content-length")).unwrap().1.trim().parse().unwrap();
                    if request.len() >= end + 4 + length { body_start = end + 4; break; }
                }
            }
            let body = serde_json::json!({"choices":[{"message":{"content":
                r#"[{"original":"Hello","translation":"Привет","box_2d":[0,0,1000,1000]}]"#}}]}).to_string();
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            stream.write_all(response.as_bytes()).await.unwrap();
            serde_json::from_slice(&request[body_start..]).unwrap()
        });
        (endpoint, task)
    }

    #[tokio::test]
    async fn vision_uses_dialogue_history_and_glossary() {
        let (endpoint, request) = mock_vision().await;
        let config = AppConfig { provider: TranslationProvider::OpenAICompatible,
            openai_base_url: endpoint, openai_api_key: Some("audit-dummy-key".into()),
            openai_vision_model: Some("mock-vision".into()), ..AppConfig::default() };
        let service = translator::TranslatorService::new();
        service.context().record_translation("HISTORY_SENTINEL".into(), "История".into());
        service.context().add_glossary_term("GLOSSARY_SENTINEL", "Термин", None);
        let frame = capture::CapturedFrame { bgra_bytes: vec![0,0,0,255], width: 1, height: 1, offset_x: 0, offset_y: 0 };
        service.translate_image_direct_vision(&frame, &config).await.unwrap();
        let request = request.await.unwrap().to_string();
        assert!(request.contains("HISTORY_SENTINEL"));
        assert!(request.contains("GLOSSARY_SENTINEL"));
        assert_eq!(service.context().get_recent_history().len(), 2);
        println!("VERIFIED vision context: history/glossary sent; result recorded in history");
    }

    #[tokio::test]
    async fn winrt_corpus_report() {
        let engine = ocr::WinrtOcrEngine::new("en").expect("English WinRT OCR must be available for this audit");
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            root.join("tests/fixtures/ocr/corpus.json")).unwrap()).unwrap();
        let mut results = Vec::new();
        for case in cases.as_array().unwrap() {
            let file = case["file"].as_str().unwrap();
            let image = image::open(root.join("tests/fixtures/ocr").join(file)).unwrap().to_rgba8();
            let (width, height) = image.dimensions();
            let mut bytes = image.into_raw();
            for pixel in bytes.chunks_exact_mut(4) { pixel.swap(0, 2); }
            let frame = capture::CapturedFrame { bgra_bytes: bytes, width, height, offset_x: 0, offset_y: 0 };
            let start = std::time::Instant::now();
            match engine.recognize(&frame).await {
                Ok(blocks) => results.push(serde_json::json!({"file":file,
                    "lines":blocks.iter().flat_map(|block| block.lines.iter().map(|line| &line.text)).collect::<Vec<_>>(),
                    "blocks":blocks.iter().map(|block| &block.text).collect::<Vec<_>>(),
                    "ms":start.elapsed().as_secs_f64()*1000.0})),
                Err(error) => results.push(serde_json::json!({"file":file,"error":error,"lines":[],"blocks":[],
                    "ms":start.elapsed().as_secs_f64()*1000.0})),
            }
        }
        std::fs::write(root.join("target/audit-winrt-quality.json"),
            serde_json::to_string_pretty(&results).unwrap()).unwrap();
        println!("WinRT corpus: {} images measured; see target/audit-winrt-quality.json", results.len());
    }
}
'''


def main():
    deps = ROOT / "target/debug/deps"
    output = ROOT / "target/audit-probes.exe"
    args = ["rustc", "--edition=2024", "--test", "-", "--crate-name", "screen_translator_audit",
            "-L", f"dependency={deps}", "-o", str(output)]
    # Select the exact dependency artifacts by their Cargo fingerprint values.
    fingerprints = [path for path in (ROOT / "target/debug/.fingerprint").glob("screen_translator-*/test-bin-screen_translator.json")
                    if (deps / f"{path.parent.name}.exe").exists()]
    if not fingerprints:
        raise RuntimeError("No compiled test executable found. Run cargo test first.")
    fingerprint = json.loads(max(fingerprints, key=lambda path: path.stat().st_mtime).read_text())
    for _, name, _, expected in fingerprint["deps"]:
        matches = []
        for artifact in deps.glob(f"lib{name}-*.rlib"):
            suffix = artifact.stem.rsplit("-", 1)[1]
            for directory in (ROOT / "target/debug/.fingerprint").glob(f"*-{suffix}"):
                stamp = directory / f"lib-{name}"
                if stamp.exists() and int.from_bytes(bytes.fromhex(stamp.read_text().strip()), "little") == expected:
                    matches.append(artifact)
        if len(matches) != 1:
            raise RuntimeError(f"Cannot uniquely resolve compiled dependency {name}: {matches}. Run cargo test first.")
        args += ["--extern", f"{name}={matches[0]}"]
    for directory in (ROOT / "target/debug/build").glob("*/out"):
        args += ["-L", f"native={directory}"]
    result = subprocess.run(args, input=SOURCE.replace("__MAIN__", (ROOT / "src/main.rs").as_posix()),
                            text=True, encoding="utf-8", cwd=ROOT, capture_output=True,
                            env={**os.environ, "CARGO_MANIFEST_DIR": str(ROOT)})
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    result = subprocess.run([str(output), "audit_probes::", "--nocapture", "--test-threads=1"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    log = result.stdout + result.stderr
    (ROOT / "target/audit-probes.log").write_text(log, encoding="utf-8")
    print(log)
    if result.returncode == 0:
        sys.path.insert(0, str(ROOT))
        from ocr_helper.benchmark import quality
        cases = {case["file"]: case for case in json.loads((ROOT / "tests/fixtures/ocr/corpus.json").read_text(encoding="utf-8"))}
        results = json.loads((ROOT / "target/audit-winrt-quality.json").read_text(encoding="utf-8"))
        for item in results:
            case = cases[item["file"]]
            item.update(quality(case["lines"], item["lines"]))
            item["pass"] = (not item.get("error") and item["cer"] <= case["max_cer"]
                            and item["wer"] <= case.get("max_wer", float("inf")) and item["missing_lines"] == 0)
        (ROOT / "target/audit-winrt-quality.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WinRT corpus quality gates: {sum(item['pass'] for item in results)}/{len(results)} passed")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
