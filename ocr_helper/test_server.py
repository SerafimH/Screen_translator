"""
Tests for OCR Helper protocol server using mock engines (no models needed).
Can be run via pytest (`python -m pytest ocr_helper`) or unittest.
"""

import io
import json
import os
import unittest
from unittest.mock import patch

from ocr_helper.server import (
    EngineManager,
    RAPIDOCR_LANG_MAP,
    SCREEN_TRANSLATOR_OCR_DEBUG_ENV,
    SCREEN_TRANSLATOR_OCR_VARIANTS_ENV,
    create_ocr_engine,
    extract_lines_from_result,
    filter_ocr_lines,
    merge_ocr_lines,
    process_request,
    read_exact,
    resolve_models_dir,
    run_server,
    save_ocr_debug_dump,
)


class MockEngine:
    def __init__(self, return_val=None):
        self.return_val = return_val

    def __call__(self, img_bgr):
        return self.return_val


class TestOcrHelperProtocol(unittest.TestCase):
    def test_read_exact(self):
        data = b"1234567890"
        stream = io.BytesIO(data)
        self.assertEqual(read_exact(stream, 5), b"12345")
        self.assertEqual(read_exact(stream, 5), b"67890")
        self.assertIsNone(read_exact(stream, 1))

    def test_ping_command(self):
        engine = MockEngine()
        resp = process_request({"cmd": "ping"}, None, engine)
        self.assertTrue(resp.get("ok"))
        self.assertEqual(resp.get("cmd"), "ping")

    def test_shutdown_command(self):
        engine = MockEngine()
        resp = process_request({"cmd": "shutdown"}, None, engine)
        self.assertTrue(resp.get("ok"))
        self.assertEqual(resp.get("cmd"), "shutdown")

    def test_unsupported_language(self):
        engine = MockEngine()
        resp = process_request(
            {"id": 1, "width": 10, "height": 10, "lang": "xyz"}, None, engine
        )
        self.assertFalse(resp.get("ok"))
        self.assertEqual(resp.get("id"), 1)
        self.assertIn("not supported yet", resp.get("error", ""))

    def test_en_variants_supported(self):
        engine = MockEngine([])
        raw_bgr = bytes([0, 0, 0] * 4)
        for lang_code in ("en", "en-US", "en_US", "en-GB"):
            resp = process_request(
                {"id": 10, "width": 2, "height": 2, "lang": lang_code}, raw_bgr, engine
            )
            self.assertTrue(resp.get("ok"), f"Failed for {lang_code}")

    def test_missing_or_invalid_id(self):
        engine = MockEngine()
        resp = process_request(
            {"width": 10, "height": 10, "lang": "en"}, None, engine
        )
        self.assertFalse(resp.get("ok"))
        self.assertIn("Missing 'id'", resp.get("error", ""))

    def test_invalid_buffer_size(self):
        engine = MockEngine()
        # width 2, height 2 -> expected 12 bytes RGB
        resp = process_request(
            {"id": 2, "width": 2, "height": 2, "lang": "en"}, b"short", engine
        )
        self.assertFalse(resp.get("ok"))
        self.assertEqual(resp.get("id"), 2)
        self.assertIn("Invalid RGB buffer size", resp.get("error", ""))

    def test_extract_lines_tuple_format(self):
        # Format typical of RapidOCR 1.x / fallback:
        # [ [ [[10,10],[50,10],[50,30],[10,30]], "Hello", 0.95 ] ]
        mock_res = [
            [[[10, 10], [50, 10], [50, 30], [10, 30]], "Hello World", 0.96]
        ]
        lines = extract_lines_from_result(mock_res)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "Hello World")
        self.assertEqual(
            lines[0]["box"], [[10, 10], [50, 10], [50, 30], [10, 30]]
        )
        self.assertAlmostEqual(lines[0]["score"], 0.96)

    def test_extract_lines_object_format(self):
        # Format typical of RapidOCR 3.x RapidOCROutput
        class MockOutput:
            boxes = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
            txts = ("Start Game",)
            scores = (0.88,)

        lines = extract_lines_from_result(MockOutput())
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "Start Game")
        self.assertEqual(lines[0]["score"], 0.88)

    def test_successful_recognition(self):
        mock_res = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], "Test", 0.99]
        ]
        engine = MockEngine(mock_res)
        # 2x2 frame RGB = 12 bytes
        raw_rgb = bytes([255, 0, 0] * 4)
        resp = process_request(
            {"id": 100, "width": 2, "height": 2, "lang": "en"},
            raw_rgb,
            engine,
        )
        self.assertTrue(resp.get("ok"))
        self.assertEqual(resp.get("id"), 100)
        self.assertEqual(len(resp.get("lines", [])), 1)
        self.assertEqual(resp["lines"][0]["text"], "Test")
        self.assertIn("ms", resp)

    def test_run_server_full_session(self):
        """Simulates full communication cycle through stdin/stdout buffers."""
        mock_res = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], "OK", 0.9]
        ]
        engine = MockEngine(mock_res)

        input_data = io.BytesIO()
        # 1. Ping
        input_data.write(json.dumps({"cmd": "ping"}).encode("utf-8") + b"\n")
        # 2. Recognition request with 1x1 image = 3 bytes
        req = {"id": 10, "width": 1, "height": 1, "lang": "en"}
        input_data.write(json.dumps(req).encode("utf-8") + b"\n")
        input_data.write(b"\x10\x20\x30")  # 3 RGB bytes
        # 3. Shutdown
        input_data.write(json.dumps({"cmd": "shutdown"}).encode("utf-8") + b"\n")
        input_data.seek(0)

        output_data = io.BytesIO()

        run_server(stdin=input_data, stdout=output_data, engine=engine)

        output_data.seek(0)
        out_lines = [
            json.loads(line.decode("utf-8"))
            for line in output_data.readlines()
            if line.strip()
        ]

        # First line is {"event": "ready"}
        self.assertEqual(out_lines[0], {"event": "ready"})
        # Second line is ping response
        self.assertEqual(out_lines[1], {"ok": True, "cmd": "ping"})
        # Third line is OCR result
        self.assertEqual(out_lines[2]["id"], 10)
        self.assertTrue(out_lines[2]["ok"])
        self.assertEqual(out_lines[2]["lines"][0]["text"], "OK")
        # Fourth line is shutdown response
        self.assertEqual(out_lines[3], {"ok": True, "cmd": "shutdown"})

    def test_resolve_models_dir(self):
        models_dir = resolve_models_dir()
        if models_dir is not None:
            self.assertTrue(os.path.isdir(models_dir))

    def test_engine_manager_dispatch(self):
        eng_en = MockEngine([[[[0, 0], [10, 0], [10, 10], [0, 10]], "English", 0.99]])
        eng_ja = MockEngine([[[[0, 0], [10, 0], [10, 10], [0, 10]], "日本語", 0.99]])
        mgr = EngineManager()
        mgr.engines["en"] = eng_en
        mgr.engines["ja"] = eng_ja

        raw_bgr = bytes([0, 0, 0] * 4)
        resp_en = process_request(
            {"id": 1, "width": 2, "height": 2, "lang": "en"}, raw_bgr, mgr
        )
        self.assertTrue(resp_en["ok"])
        self.assertEqual(resp_en["lines"][0]["text"], "English")

        resp_ja = process_request(
            {"id": 2, "width": 2, "height": 2, "lang": "ja"}, raw_bgr, mgr
        )
        self.assertTrue(resp_ja["ok"])
        self.assertEqual(resp_ja["lines"][0]["text"], "日本語")

    def test_engine_manager_caches_instance_per_lang(self):
        mgr = EngineManager()
        mock_en = MockEngine([])
        mock_ja = MockEngine([])
        mgr.engines["en"] = mock_en
        mgr.engines["ja"] = mock_ja
        self.assertIs(mgr.get_engine("en"), mock_en)
        self.assertIs(mgr.get_engine("ja"), mock_ja)

    def test_engine_manager_uses_default_engine_if_provided(self):
        mock = MockEngine([])
        mgr = EngineManager(default_engine=mock)
        self.assertIs(mgr.get_engine("en"), mock)
        self.assertIs(mgr.get_engine("ja"), mock)
        self.assertIs(mgr.get_engine("ru"), mock)

    def test_rapidocr_lang_mapping(self):
        self.assertEqual(RAPIDOCR_LANG_MAP.get("ja"), "japan")
        self.assertEqual(RAPIDOCR_LANG_MAP.get("ko"), "korean")
        self.assertEqual(RAPIDOCR_LANG_MAP.get("ru"), "cyrillic")
        self.assertEqual(RAPIDOCR_LANG_MAP.get("zh"), "ch")

    def test_create_ocr_engine_passes_lang_type(self):
        with patch("rapidocr.RapidOCR") as mock_rapid:
            create_ocr_engine(None, lang="ja")
            mock_rapid.assert_called_once_with(params={"Rec.lang_type": "japan"})

        with patch("os.path.exists", return_value=True), patch("rapidocr.RapidOCR") as mock_rapid:
            create_ocr_engine("/fake/models", lang="ru")
            mock_rapid.assert_called_once_with(
                params={"Rec.lang_type": "cyrillic", "Global.model_root_dir": "/fake/models"}
            )

    def test_engine_manager_cache_limit_lru(self):
        def fake_create(models_dir, lang):
            return MockEngine([{"text": lang}])

        with patch("ocr_helper.server.create_ocr_engine", side_effect=fake_create):
            mgr = EngineManager()
            e_en = mgr.get_engine("en")
            e_ja = mgr.get_engine("ja")
            self.assertEqual(len(mgr.engines), 2)
            self.assertIn("en", mgr.engines)
            self.assertIn("ja", mgr.engines)

            # Touch 'en' so it becomes most recently used
            self.assertIs(mgr.get_engine("en"), e_en)

            # Loading third language 'ru' should evict 'ja'
            e_ru = mgr.get_engine("ru")
            self.assertEqual(len(mgr.engines), 2)
            self.assertIn("en", mgr.engines)
            self.assertIn("ru", mgr.engines)
            self.assertNotIn("ja", mgr.engines)

    def test_filter_ocr_lines(self):
        valid_box = [[0, 0], [100, 0], [100, 20], [0, 20]]
        tiny_box_w = [[0, 0], [4, 0], [4, 20], [0, 20]]
        tiny_box_h = [[0, 0], [100, 0], [100, 4], [0, 4]]

        # 1. "I@II#O*$" (0.9) -> отброшено ("symbols")
        kept, rej = filter_ocr_lines([(valid_box, "I@II#O*$", 0.9)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "symbols")

        # 2. "A DIVINE WEAPON MADE FROM YGGDRASIL," (0.95) -> оставлено
        kept, rej = filter_ocr_lines([(valid_box, "A DIVINE WEAPON MADE FROM YGGDRASIL,", 0.95)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rej), 0)

        # 3. "Chapter 126: The Divine Weapon" (0.9) -> оставлено
        kept, rej = filter_ocr_lines([(valid_box, "Chapter 126: The Divine Weapon", 0.9)])
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rej), 0)

        # 4. "?!" -> отброшено ("no_alnum")
        kept, rej = filter_ocr_lines([(valid_box, "?!", 0.9)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "no_alnum")

        # 5. "I" (0.9) -> оставлено
        kept, rej = filter_ocr_lines([(valid_box, "I", 0.9)])
        self.assertEqual(len(kept), 1)

        # 6. "I" (0.7) -> отброшено ("too_short")
        kept, rej = filter_ocr_lines([(valid_box, "I", 0.7)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "too_short")

        # 7. Русская строка "Это тест" (0.9) -> оставлено
        kept, rej = filter_ocr_lines([(valid_box, "Это тест", 0.9)])
        self.assertEqual(len(kept), 1)

        # 8. Японская строка "終末のワルキューレ" (0.9) -> оставлено
        kept, rej = filter_ocr_lines([(valid_box, "終末のワルキューレ", 0.9)])
        self.assertEqual(len(kept), 1)

        # 9. Оценка уверенности меньше 0.60 -> отброшено ("low_score")
        kept, rej = filter_ocr_lines([(valid_box, "Valid text", 0.55)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "low_score")

        # 10. Рамка меньше 6x6 -> отброшено ("tiny_box")
        kept, rej = filter_ocr_lines([(tiny_box_w, "Valid text", 0.9)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "tiny_box")

        kept, rej = filter_ocr_lines([(tiny_box_h, "Valid text", 0.9)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "tiny_box")

    def test_process_request_filters_garbage_lines(self):
        valid_box = [[0, 0], [100, 0], [100, 20], [0, 20]]
        mock_output = [
            {"box": valid_box, "text": "I@II#O*$", "score": 0.9},
            {"box": valid_box, "text": "Chapter 126", "score": 0.95},
            {"box": valid_box, "text": "?!", "score": 0.9},
        ]
        engine = MockEngine(mock_output)
        raw_bgr = b"\x00" * 12
        resp = process_request({"id": 1, "width": 2, "height": 2, "lang": "en"}, raw_bgr, engine)
        self.assertTrue(resp["ok"])
        self.assertEqual(len(resp["lines"]), 1)
        self.assertEqual(resp["lines"][0]["text"], "Chapter 126")

    def test_debug_dump_enabled(self):
        import shutil
        import tempfile

        test_dir = tempfile.mkdtemp()
        try:
            with patch.dict(os.environ, {SCREEN_TRANSLATOR_OCR_DEBUG_ENV: "1"}):
                valid_box = [[0, 0], [100, 0], [100, 20], [0, 20]]
                debug_lines = [
                    {
                        "index": 0,
                        "box": valid_box,
                        "text": "Hello World",
                        "score": 0.95,
                        "kept": True,
                        "reason": None,
                    }
                ]
                # Test save_ocr_debug_dump with patched server_dir
                with patch("ocr_helper.server.__file__", os.path.join(test_dir, "server.py")):
                    save_ocr_debug_dump(None, debug_lines, req_id=99)
                    dump_dir = os.path.join(test_dir, "ocr_debug")
                    self.assertTrue(os.path.isdir(dump_dir))
                    subdirs = os.listdir(dump_dir)
                    self.assertEqual(len(subdirs), 1)
                    json_path = os.path.join(dump_dir, subdirs[0], "result.json")
                    self.assertTrue(os.path.exists(json_path))
                    with open(json_path, "r", encoding="utf-8") as f:
                        saved_data = json.load(f)
                    self.assertEqual(len(saved_data), 1)
                    self.assertEqual(saved_data[0]["text"], "Hello World")
                    self.assertTrue(saved_data[0]["kept"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_merge_ocr_lines(self):
        box_a = [[0, 0], [100, 0], [100, 20], [0, 20]]
        # box_b overlaps strongly with box_a (shifted by 2px, IoU ~ 0.96)
        box_b = [[2, 0], [102, 0], [102, 20], [2, 20]]
        # box_c does not overlap with box_a or box_b
        box_c = [[200, 0], [300, 0], [300, 20], [200, 20]]

        # 1. Дубликаты с разной оценкой (сильное перекрытие): побеждает большая оценка
        line_low = (box_a, "Duplicate text", 0.75)
        line_high = (box_b, "Duplicate text", 0.95)
        merged = merge_ocr_lines([[line_low], [line_high]])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0][2], 0.95)

        # 2. Непересекающиеся рамки: сохраняются обе
        line_disjoint = (box_c, "Other text", 0.88)
        merged_two = merge_ocr_lines([[line_high], [line_disjoint]])
        self.assertEqual(len(merged_two), 2)
        texts = [item[1] for item in merged_two]
        self.assertIn("Duplicate text", texts)
        self.assertIn("Other text", texts)

        # 3. Пустые списки
        self.assertEqual(merge_ocr_lines([]), [])
        self.assertEqual(merge_ocr_lines([[], []]), [])

    def test_process_request_with_variants(self):
        box_a = [[0, 0], [50, 0], [50, 20], [0, 20]]
        engine = MockEngine([{"box": box_a, "text": "Variant recognized", "score": 0.92}])
        raw_bgr = b"\x00" * 12
        with patch.dict(os.environ, {SCREEN_TRANSLATOR_OCR_VARIANTS_ENV: "1"}):
            resp = process_request({"id": 1, "width": 2, "height": 2, "lang": "en"}, raw_bgr, engine)
            self.assertTrue(resp["ok"])
            self.assertEqual(len(resp["lines"]), 1)
            self.assertEqual(resp["lines"][0]["text"], "Variant recognized")


if __name__ == "__main__":
    unittest.main()
