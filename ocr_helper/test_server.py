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
    SMALL_FRAME_PIXELS,
    UPSCALE_SMALL_FRAMES,
    create_ocr_engine,
    extract_lines_from_result,
    filter_ocr_lines,
    is_condensed_lettering,
    lettering_variants,
    merge_ocr_lines,
    process_request,
    read_exact,
    refine_problematic_box,
    rescale_boxes_down,
    resolve_models_dir,
    score_ocr_candidate,
    run_server,
    save_ocr_debug_dump,
    should_upscale_frame,
)


class MockEngine:
    def __init__(self, return_val=None):
        self.return_val = return_val

    def __call__(self, img_bgr):
        return self.return_val


class TestOcrHelperProtocol(unittest.TestCase):
    def test_condensed_lettering_detection_is_limited_to_long_latin_text(self):
        box = [[10, 10], [80, 10], [80, 30], [10, 30]]
        self.assertTrue(is_condensed_lettering({"text": "WAIT FOR ME NOW", "box": box}))
        for text in ("III", "はい、待ってください", "Hello"):
            self.assertFalse(is_condensed_lettering({"text": text, "box": box}))
        self.assertFalse(is_condensed_lettering({"text": "WAIT FOR ME NOW",
                                               "box": [[0, 0], [300, 0], [300, 20], [0, 20]]}))

    def test_lettering_normalization_handles_polarity_and_measured_stroke_weight(self):
        from ocr_helper import server
        if server.cv2 is None or server.np is None:
            self.skipTest("OpenCV and numpy are required")
        cv, np = server.cv2, server.np
        for thickness in (1, 10):
            img = np.full((60, 160, 3), 205, dtype=np.uint8)
            for x in range(20, 140, 20):
                img[20:40, x:x + thickness] = 202
            before = img.copy()
            normal = list(lettering_variants(img))
            inverse = list(lettering_variants(255 - img))
            self.assertEqual(len(normal), 2)
            self.assertTrue(np.array_equal(img, before))
            for (a, sx, sy), (b, _, _) in zip(normal, inverse):
                self.assertTrue(np.array_equal(a, b))
                self.assertEqual(a.shape[:2], (round(60 * sy), round(160 * sx)))
                self.assertEqual(a.dtype, np.uint8)
            unadjusted = cv.resize(normal[0][0], (320, 120), interpolation=cv.INTER_CUBIC)
            adjusted_ink = int((normal[1][0][:, :, 0] < 128).sum())
            previous_ink = int((unadjusted[:, :, 0] < 128).sum())
            if thickness == 1:
                self.assertGreater(adjusted_ink, previous_ink)
            else:
                self.assertLess(adjusted_ink, previous_ink)

    def test_lettering_normalization_skips_blank_and_large_images(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("OpenCV and numpy are required")
        for shape in ((40, 120, 3), (1080, 1920, 3), (4, 100, 3)):
            self.assertEqual(list(lettering_variants(server.np.zeros(shape, dtype=server.np.uint8))), [])

    def test_lettering_size_limit_does_not_duplicate_confirmation_views(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("OpenCV and numpy are required")
        img = server.np.full((60, 2200, 3), 255, dtype=server.np.uint8)
        server.cv2.putText(img, "WAIT HERE", (20, 35), server.cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 0, 0), 1)
        variants = list(lettering_variants(img))
        self.assertEqual(len(variants), 1)
        self.assertLessEqual(max(variants[0][0].shape[:2]), 2000)

    def test_anisotropic_lettering_maps_line_and_word_boxes_without_mutating_input(self):
        box = [[36, 20], [180, 20], [180, 60], [36, 60]]
        line = {"text": "WAIT", "score": 0.97, "box": box,
                "words": [{"text": "WAIT", "score": 0.96, "box": box}]}
        mapped = rescale_boxes_down([line], 3.6, 2.0)[0]
        expected = [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]]
        self.assertEqual(mapped["box"], expected)
        self.assertEqual(mapped["words"][0]["box"], expected)
        self.assertEqual(mapped["words"][0]["score"], 0.96)
        self.assertEqual(line["box"], box)
        self.assertEqual(line["words"][0]["box"], box)

    def test_lettering_refinement_keeps_original_screen_geometry_and_rejects_neighbors(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("OpenCV and numpy are required")
        img = server.np.full((120, 180, 3), 255, dtype=server.np.uint8)
        line = {"text": "HE1P", "score": 0.7,
                "box": [[40, 50], [100, 50], [100, 65], [40, 65]]}
        local = [[72, 40], [288, 40], [288, 70], [72, 70]]
        corrected = {"text": "HELP", "score": 0.96, "box": local,
                     "words": [{"text": "HELP", "score": 0.95, "box": local}]}
        import copy
        for neighbor in (False, True):
            candidate = copy.deepcopy(corrected)
            if neighbor:
                candidate["box"] = [[72, 0], [288, 0], [288, 20], [72, 20]]
            with patch("ocr_helper.server.lettering_variants", return_value=[(img, 3.6, 2.0)]), \
                 patch("ocr_helper.server.call_engine", side_effect=[[], [], [candidate]]):
                result = refine_problematic_box(img, line, None, english_lettering=True)
            self.assertEqual(result["text"], "HE1P" if neighbor else "HELP")
            self.assertEqual(result["box"], line["box"])
            if not neighbor:
                self.assertEqual(result["words"][0]["box"], line["box"])

    def test_faint_lettering_recovery_requires_two_agreeing_high_confidence_views(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("OpenCV and numpy are required")
        img = server.np.full((60, 160, 3), 205, dtype=server.np.uint8)
        a = {"text": "WAIT HERE", "score": 0.96,
             "box": [[10, 10], [120, 10], [120, 30], [10, 30]]}
        import copy
        for other_text, score, shift, expected in (("WAIT HERE", 0.97, 0, True),
                                                  ("WHAT HERE", 0.99, 0, False),
                                                  ("WAIT HERE", 0.7, 0, False),
                                                  ("WAIT HERE", 0.99, 60, False)):
            b = copy.deepcopy(a)
            b.update(text=other_text, score=score)
            b["box"] = [[x * 2, (y + shift) * 2] for x, y in b["box"]]
            with self.subTest(text=other_text, score=score, shift=shift), \
                 patch("ocr_helper.server.lettering_variants", return_value=[(img, 1, 1), (img, 2, 2)]), \
                 patch("ocr_helper.server.call_engine", side_effect=[[], [], [copy.deepcopy(a)], [b]]):
                response = process_request(dict(id=1, width=160, height=60, lang="en"), img.tobytes(), None)
            self.assertTrue(response["ok"], response)
            self.assertEqual(len(response["lines"]), int(expected))
            if expected:
                self.assertEqual(response["lines"][0], a)

    def test_english_lettering_passes_are_bounded_and_leave_other_languages_alone(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("OpenCV and numpy are required")
        lines = [{"text": "UNCERTAIN", "score": 0.7,
                  "box": [[10, i * 22], [150, i * 22], [150, i * 22 + 20], [10, i * 22 + 20]]}
                 for i in range(10)]
        img = server.np.zeros((300, 200, 3), dtype=server.np.uint8)
        for lang, expected in (("en", 4), ("ja", 0)):
            with patch("ocr_helper.server.should_upscale_frame", return_value=False), \
                 patch("ocr_helper.server.call_engine", return_value=lines), \
                 patch("ocr_helper.server.refine_problematic_box", side_effect=lambda image, line, *a, **kw: line) as refine:
                response = process_request(dict(id=1, width=200, height=300, lang=lang), img.tobytes(), None)
            self.assertTrue(response["ok"], response)
            self.assertEqual(sum(call.kwargs["english_lettering"] for call in refine.call_args_list), expected)

    def test_ready_follows_model_warmup_and_keeps_stdout_clean(self):
        for configured, expected in (("ja-JP", "ja"), ("", "en")):
            output = io.BytesIO()
            stderr = io.StringIO()

            def load_engine(models_dir, lang):
                self.assertEqual(output.getvalue(), b"")
                self.assertEqual(lang, expected)
                print("loader output belongs on stderr")
                return MockEngine()

            with self.subTest(lang=configured), \
                 patch.dict(os.environ, {"SCREEN_TRANSLATOR_OCR_LANG": configured}), \
                 patch("ocr_helper.server.resolve_models_dir", return_value=None), \
                 patch("ocr_helper.server.create_ocr_engine", side_effect=load_engine), \
                 patch("sys.stderr", stderr):
                run_server(stdin=io.BytesIO(), stdout=output)
            self.assertEqual(json.loads(output.getvalue()), {"event": "ready"})
            self.assertIn("loader output", stderr.getvalue())

    def test_failed_warmup_does_not_report_ready(self):
        output = io.BytesIO()
        with patch("ocr_helper.server.resolve_models_dir", return_value=None), \
             patch("ocr_helper.server.create_ocr_engine", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                run_server(stdin=io.BytesIO(), stdout=output)
        self.assertEqual(output.getvalue(), b"")

    def test_ready_loads_lazy_rapidocr_models_before_notification(self):
        output = io.BytesIO()
        loaded = []

        def load_model(name):
            def load(config):
                self.assertEqual(output.getvalue(), b"")
                loaded.append(name)
                print("model initialization output belongs on stderr")
                return object()
            return load

        stderr = io.StringIO()
        with patch.dict(os.environ, {"SCREEN_TRANSLATOR_OCR_LANG": "en"}), \
             patch("ocr_helper.server.resolve_models_dir", return_value=None), \
             patch("rapidocr.main.TextDetector", side_effect=load_model("det")), \
             patch("rapidocr.main.TextClassifier", side_effect=load_model("cls")), \
             patch("rapidocr.main.TextRecognizer", side_effect=load_model("rec")), \
             patch("sys.stderr", stderr):
            run_server(stdin=io.BytesIO(), stdout=output)
        self.assertEqual(loaded, ["det", "cls", "rec"])
        self.assertEqual(json.loads(output.getvalue()), {"event": "ready"})
        self.assertEqual(stderr.getvalue().count("model initialization output"), 3)

    def test_lazy_model_failure_does_not_report_ready(self):
        output = io.BytesIO()
        with patch.dict(os.environ, {"SCREEN_TRANSLATOR_OCR_LANG": "en"}), \
             patch("ocr_helper.server.resolve_models_dir", return_value=None), \
             patch("rapidocr.main.TextDetector", side_effect=RuntimeError("model load failed")):
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                run_server(stdin=io.BytesIO(), stdout=output)
        self.assertEqual(output.getvalue(), b"")

    def test_short_cjk_dialogue_keeps_native_punctuation(self):
        box = [[0, 0], [40, 0], [40, 20], [0, 20]]
        for text in ("はい？", "何？", "是。", "不！", "「はい」", "你好，世界！"):
            with self.subTest(text=text):
                kept, rejected = filter_ocr_lines([(box, text, 0.99)])
                self.assertEqual(len(kept), 1)
                self.assertEqual(rejected, [])
        kept, rejected = filter_ocr_lines([(box, "I@II#O*$", 0.99)])
        self.assertEqual(kept, [])
        self.assertEqual(rejected[0][1], "symbols")

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

    @patch("ocr_helper.server.should_upscale_frame", lambda *_: False)
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
        eng_en = MockEngine([[[[0, 0], [20, 0], [20, 20], [0, 20]], "English", 0.99]])
        eng_ja = MockEngine([[[[0, 0], [20, 0], [20, 20], [0, 20]], "日本語", 0.99]])
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
        from rapidocr import ModelType, OCRVersion

        with patch("rapidocr.RapidOCR") as mock_rapid:
            create_ocr_engine(None, lang="en")
            mock_rapid.assert_called_once_with(params={
                "Rec.lang_type": "en", "Det.limit_type": "max", "Det.limit_side_len": 2000,
            })

        with patch("rapidocr.RapidOCR") as mock_rapid:
            create_ocr_engine(None, lang="ja")
            mock_rapid.assert_called_once_with(params={"Rec.lang_type": "japan"})

        with patch("os.path.exists", return_value=True), patch("rapidocr.RapidOCR") as mock_rapid:
            create_ocr_engine("/fake/models", lang="ru")
            mock_rapid.assert_called_once_with(
                params={"Rec.lang_type": "cyrillic", "Rec.ocr_version": OCRVersion.PPOCRV5,
                        "Rec.model_type": ModelType.MOBILE, "Global.model_root_dir": "/fake/models"}
            )

    def test_configured_languages_resolve_to_supported_recognition_models(self):
        from rapidocr.inference_engine.base import FileInfo, InferSession

        for lang in ("en", "ja", "zh", "ru", "ko"):
            with self.subTest(lang=lang):
                cfg = create_ocr_engine(lang=lang).cfg.Rec
                model = InferSession.get_model_url(FileInfo(
                    cfg.engine_type, cfg.ocr_version, cfg.task_type,
                    cfg.lang_type, cfg.model_type,
                ))
                self.assertTrue(model["model_dir"])

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

        # 9. Оценка уверенности меньше 0.50 -> отброшено ("low_score")
        kept, rej = filter_ocr_lines([(valid_box, "Valid text", 0.45)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "low_score")

        # 10. Рамка меньше 6x6 -> отброшено ("tiny_box")
        kept, rej = filter_ocr_lines([(tiny_box_w, "Valid text", 0.9)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "tiny_box")

        kept, rej = filter_ocr_lines([(tiny_box_h, "Valid text", 0.9)])
        self.assertEqual(len(kept), 0)
        self.assertEqual(rej[0][1], "tiny_box")

    @patch("ocr_helper.server.should_upscale_frame", lambda *_: False)
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

    @patch("ocr_helper.server.should_upscale_frame", lambda *_: False)
    def test_request_threshold_reaches_engine_and_filter(self):
        received = []

        def engine(img, **kwargs):
            received.append(kwargs["text_score"])
            return [{"box": [[0, 0], [100, 0], [100, 20], [0, 20]],
                     "text": "Hello world", "score": 0.55}]

        for threshold, count in ((0.5, 1), (0.6, 0), (0.3, 1)):
            received.clear()
            header = {"id": 1, "width": 2, "height": 2, "lang": "en",
                      "min_score": threshold}
            result = process_request(header, bytes(12), engine)
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["lines"]), count)
            self.assertTrue(received)
            self.assertTrue(all(score == threshold for score in received))
            if count:
                self.assertEqual(result["lines"][0]["score"], 0.55)

        for invalid in (-0.1, 1.1, float("nan"), float("inf"), "0.5", True):
            header["min_score"] = invalid
            self.assertFalse(process_request(header, bytes(12), engine)["ok"])

    def test_merge_preserves_lower_configured_threshold(self):
        line = {"box": [[0, 0], [100, 0], [100, 20], [0, 20]],
                "text": "Hello world", "score": 0.35}
        self.assertEqual(merge_ocr_lines([[line]], min_score=0.3), [line])
        self.assertEqual(merge_ocr_lines([[line]], min_score=0.4), [])

    def test_benchmark_quality_detects_missing_and_wrong_text(self):
        from ocr_helper.benchmark import edit_distance, quality
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        exact = quality(["Hello world", "Goodbye"], ["HELLO WORLD", "Goodbye"])
        self.assertEqual(exact["cer"], 0)
        self.assertEqual(exact["missing_lines"], 0)
        partial = quality(["Hello world", "Goodbye"], ["Hello world"])
        self.assertGreater(partial["cer"], 0)
        self.assertEqual(partial["missing_lines"], 1)

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

    def test_merge_keeps_complete_negated_reading_over_confident_clipped_suffix(self):
        full = {"text": "DO NOT OPEN THE DOOR", "score": 0.95,
                "box": [[0, 0], [300, 0], [300, 30], [0, 30]],
                "words": [{"text": "NOT", "score": 0.94,
                           "box": [[40, 0], [80, 0], [80, 30], [40, 30]]}]}
        clipped = {"text": "OPEN THE DOOR", "score": 0.99,
                   "box": [[95, 0], [300, 0], [300, 30], [95, 30]]}
        for groups in ([[full], [clipped]], [[clipped], [full]]):
            with self.subTest(groups=groups):
                merged = merge_ocr_lines(groups)
                self.assertEqual(merged, [full])
                self.assertEqual(merged[0]["words"], full["words"])

    def test_merge_does_not_guess_word_completion_or_combine_unique_fragments(self):
        same_box = [[0, 0], [100, 0], [100, 30], [0, 30]]
        # A substring inside a word is not evidence of a clipped phrase.
        short = (same_box, "IN", 0.99)
        long = (same_box, "BEGINNER", 0.91)
        self.assertEqual(merge_ocr_lines([[long], [short]]), [short])
        left = ([[0, 0], [300, 0], [300, 30], [0, 30]], "PLEASE FOLLOW THE", 0.96)
        right = ([[70, 0], [370, 0], [370, 30], [70, 30]], "FOLLOW THE INSTRUCTIONS", 0.99)
        for groups in ([[left], [right]], [[right], [left]]):
            merged = merge_ocr_lines(groups)
            self.assertEqual(len(merged), 2)
            self.assertCountEqual(merged, [left, right])
        nearly_same_region = ([[10, 0], [310, 0], [310, 30], [10, 30]],
                              "FOLLOW THE INSTRUCTIONS", 0.99)
        self.assertCountEqual(merge_ocr_lines([[left], [nearly_same_region]]),
                              [left, nearly_same_region])

    def test_merge_compares_confidence_only_for_competing_same_region_readings(self):
        box = [[0, 0], [200, 0], [200, 30], [0, 30]]
        weaker = (box, "OPEN THE FLOOR", 0.87)
        stronger = (box, "OPEN THE DOOR", 0.97)
        self.assertEqual(merge_ocr_lines([[weaker], [stronger]]), [stronger])
        # Different content in partially overlapping boxes can be two real
        # handwritten lines; neither reading covers the other's text region.
        top = (box, "THIS IS REAL", 0.95)
        lower = ([[0, 8], [200, 8], [200, 38], [0, 38]], "ANOTHER LINE", 0.99)
        self.assertCountEqual(merge_ocr_lines([[top], [lower]]), [top, lower])
        # Narrow lettering can lose a space after punctuation without changing
        # any glyph. Differently padded variants remain one reading.
        joined = (box, "WAIT!WE HAVE TO GO", 0.93)
        spaced = ([[0, -4], [200, -4], [200, 36], [0, 36]], "WAIT! WE HAVE TO GO", 0.99)
        self.assertEqual(merge_ocr_lines([[joined], [spaced]]), [spaced])

    def test_process_request_with_variants(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for image variants")
        box_a = [[0, 0], [50, 0], [50, 20], [0, 20]]
        engine = MockEngine([{"box": box_a, "text": "Variant recognized", "score": 0.92}])
        width, height = 1600, 1000
        raw_bgr = bytes(width * height * 3)
        with patch.dict(os.environ, {SCREEN_TRANSLATOR_OCR_VARIANTS_ENV: "1"}):
            resp = process_request({"id": 1, "width": width, "height": height, "lang": "en"}, raw_bgr, engine)
            self.assertTrue(resp["ok"])
            self.assertEqual(len(resp["lines"]), 1)
            self.assertEqual(resp["lines"][0]["text"], "Variant recognized")

    def test_candidate_ranking_does_not_reward_extra_nonsense_words(self):
        self.assertGreater(score_ocr_candidate("HELP", 0.95),
                           score_ocr_candidate("qzxv bnm klm", 0.74))
        for text in ("don't", "state-of-the-art", "iPhone", "McDonald", "Gungnir"):
            self.assertAlmostEqual(score_ocr_candidate(text, 0.91), 0.91)
        self.assertLess(score_ocr_candidate("Jdlk;@#k4", 0.91), 0.91)
        self.assertEqual(score_ocr_candidate("HELLO", float("nan")), 0.0)

    def test_suspicious_ending_needs_two_strong_matching_crops_and_keeps_word_geometry(self):
        from ocr_helper import server
        import copy
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required")
        img = server.np.zeros((60, 120, 3), dtype=server.np.uint8)
        word_box = [[40, 20], [95, 20], [95, 40], [40, 40]]
        line = {"text": "Next mission.n.", "score": 0.99,
                "box": [[10, 20], [95, 20], [95, 40], [10, 40]],
                "words": [{"text": "Next", "score": 0.99,
                           "box": [[10, 20], [35, 20], [35, 40], [10, 40]]},
                          {"text": "mission.n.", "score": 0.99, "box": word_box}]}

        def reading(text, scale, score=0.98, shift=0):
            return [{"text": text, "score": score,
                     "box": [[(x - 32 + shift) * scale, (y - 12) * scale] for x, y in word_box]}]

        variants = [(img, 2, 2), (img, 3, 3)]
        cases = [([reading("mission.", 2), reading("mission.", 3)], 2, True),
                 ([reading("mission.", 2), reading("mission!", 3)], 2, False),
                 ([reading("mission.", 2, score=0.94)], 2, False),
                 ([reading("mission.", 2, shift=100)], 2, False),
                 ([reading("missing.", 2)], 2, False),
                 ([reading("mission.", 2), []], 2, False),
                 ([reading("mission.", 2)], 1, False),
                 ([reading("mission.n.", 2)], 2, False)]
        for responses, calls, accepted in cases:
            original = copy.deepcopy(line)
            budget = server._RetryBudget(calls=calls)
            with self.subTest(responses=responses, calls=calls), \
                    patch("ocr_helper.server.lettering_variants", return_value=variants), \
                    patch("ocr_helper.server.call_engine", side_effect=responses) as recognize:
                result = server.refine_suspicious_ending(img, original, None, 0.5, budget)
                self.assertEqual(result["text"], "Next mission." if accepted else line["text"])
                self.assertEqual(original, line, "the input reading must remain unchanged")
                self.assertEqual(result["box"], line["box"])
                self.assertEqual(result["words"][0], line["words"][0])
                self.assertLessEqual(recognize.call_count, calls)
                if accepted:
                    self.assertEqual(result["words"][-1]["text"], "mission.")
                    self.assertEqual(result["words"][-1]["box"], word_box)

    def test_normal_endings_acronyms_and_domains_do_not_trigger_word_retries(self):
        from ocr_helper import server
        if server.np is None:
            self.skipTest("numpy is required")
        img = server.np.zeros((60, 120, 3), dtype=server.np.uint8)
        for word in ("mission.", "U.S.", "Ph.D.", "example.com", "iPhone", "O'Neill"):
            line = {"text": word, "score": 0.99, "words": [{"text": word,
                    "box": [[10, 10], [90, 10], [90, 30], [10, 30]]}]}
            with self.subTest(word=word), patch("ocr_helper.server.call_engine") as recognize:
                self.assertEqual(server.refine_suspicious_ending(img, line, None, 0.5, server._RetryBudget()), line)
                recognize.assert_not_called()

    def test_detected_fragments_follow_confirmed_text_without_splitting_names(self):
        from ocr_helper import server
        line = {"text": "TO VERAMONTE.", "words": [
            {"text": "TO", "score": 0.99, "box": [[0, 0], [20, 0], [20, 20], [0, 20]]},
            {"text": "VERAM", "score": 0.98, "box": [[30, 0], [80, 0], [80, 20], [30, 20]]},
            {"text": "ONTE.", "score": 0.97, "box": [[90, 0], [140, 0], [140, 20], [90, 20]]}]}
        result = server.align_detected_words(line)
        self.assertEqual(result["text"], line["text"])
        self.assertEqual([w["text"] for w in result["words"]], ["TO", "VERAMONTE."])
        self.assertEqual(result["words"][1]["box"], [[30, 0], [140, 0], [140, 20], [30, 20]])
        self.assertEqual(result["words"][1]["score"], 0.97)
        damaged = {**line, "text": "TO OTHERNAME."}
        self.assertEqual(server.align_detected_words(damaged), damaged)

    def test_word_consensus_corrects_only_agreeing_letters_and_preserves_punctuation(self):
        from ocr_helper import server
        import copy
        if server.np is None:
            self.skipTest("numpy is required")
        img = server.np.zeros((80, 240, 3), dtype=server.np.uint8)
        boxes = [[[20, 20], [100, 20], [100, 40], [20, 40]],
                 [[120, 20], [200, 20], [200, 40], [120, 40]]]
        line = {"text": "CAUGHT GLARD!", "score": 0.96, "box": [[20, 20], [200, 20], [200, 40], [20, 40]],
                "words": [{"text": text, "score": 0.999, "box": box} for text, box in zip(["CAUGHT", "GLARD!"], boxes)]}

        def view(texts):
            return [{"text": " ".join(texts), "score": 0.96, "box": [[12, 12], [192, 12], [192, 32], [12, 32]],
                     "words": [{"text": text, "score": 0.99,
                                "box": [[x - 8, y - 8] for x, y in box]} for text, box in zip(texts, boxes)]}]

        for second, expected in [(["CALGHT", "GUARD!"], "CAUGHT GUARD!"),
                                  (["CAUGHT", "GLARD!"], "CAUGHT GLARD!")]:
            index = 0 if second[0] == "CALGHT" else 1
            x1 = boxes[index][0][0] - 4
            def word_view(text):
                return [{"text": text, "score": 0.99,
                         "box": [[x - x1, y - 16] for x, y in boxes[index]]}]
            extra = [word_view("CAUGHT"), word_view("CAUGHT")] if index == 0 else [word_view("GUARD!"), word_view("GLARD!")]
            with self.subTest(second=second), \
                    patch("ocr_helper.server.lettering_variants", return_value=[(img, 1, 1), (img, 1, 1)]), \
                    patch("ocr_helper.server.call_engine", side_effect=[view(["CAUGHT", "GUARD!"]), view(second), *extra]):
                result = server.refine_confirmed_words(img, copy.deepcopy(line), None, 0.5, server._RetryBudget())
                self.assertEqual(result["text"], expected)

        punctuation = {**line, "text": "CAUGHT GLARD!"}
        with patch("ocr_helper.server.lettering_variants", return_value=[(img, 1, 1), (img, 1, 1)]), \
                patch("ocr_helper.server.call_engine", side_effect=[view(["CAUGHT", "GLARD"]), view(["CAUGHT", "GLARD"])]):
            self.assertEqual(server.refine_confirmed_words(img, punctuation, None, 0.5, server._RetryBudget())["text"], "CAUGHT GLARD!")

    def test_refinement_cannot_replace_clear_text_with_longer_low_confidence_noise(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        original = {"text": "HELP", "score": 0.90,
                    "box": [[40, 50], [100, 50], [100, 65], [40, 65]]}
        noisy = {"text": "qzxv bnm klm", "score": 0.74,
                 "box": [[20, 20], [80, 20], [80, 35], [20, 35]]}
        with patch("ocr_helper.server.call_engine", side_effect=[[noisy], []]):
            result = refine_problematic_box(server.np.zeros((120, 180, 3), dtype=server.np.uint8),
                                            original, None)
        self.assertEqual(result, original)

    def test_all_refinement_variants_keep_complete_negated_reading_and_word_metadata(self):
        from ocr_helper import server
        import copy
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        img = server.np.zeros((120, 400, 3), dtype=server.np.uint8)
        original = {"text": "DO NOT OPEN THE DOOR", "score": 0.80,
                    "box": [[40, 50], [340, 50], [340, 80], [40, 80]],
                    "words": [{"text": "NOT", "score": 0.90,
                               "box": [[80, 50], [120, 50], [120, 80], [80, 80]]}]}
        for variant in range(4):
            for text, left in (("OPEN THE DOOR", 115), ("OPEN THE DOOR", 20),
                               ("OPEN THE FLOOR", 115)):
                with self.subTest(variant=variant, text=text, left=left):
                    truncated = {"text": text, "score": 0.99,
                                 "box": [[left, 20], [320, 20], [320, 50], [left, 50]]}
                    responses = [[], [], [], []]
                    responses[variant] = [copy.deepcopy(truncated)]
                    with patch("ocr_helper.server.lettering_variants", return_value=[(img, 1, 1), (img, 1, 1)]), \
                         patch("ocr_helper.server.call_engine", side_effect=responses):
                        result = refine_problematic_box(img, original, None, english_lettering=True)
                    self.assertEqual(result, original)
                    self.assertEqual(result["words"][0]["text"], "NOT")
        # Raising the final acceptance threshold must not authorize stripping a
        # credible negation merely to obtain a higher-scored suffix.
        cropped = {"text": "OPEN THE DOOR", "score": 0.99,
                   "box": [[115, 20], [320, 20], [320, 50], [115, 50]]}
        with patch("ocr_helper.server.call_engine", side_effect=[[cropped], []]):
            result = refine_problematic_box(img, original, None, min_score=0.95)
        self.assertEqual(result, original)

    def test_later_refinement_does_not_truncate_a_newly_recovered_complete_reading(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        img = server.np.zeros((120, 400, 3), dtype=server.np.uint8)
        original = {"text": "Unreadable", "score": 0.30,
                    "box": [[40, 50], [340, 50], [340, 80], [40, 80]]}
        recovered = {"text": "DO NOT OPEN THE DOOR", "score": 0.80,
                     "box": [[20, 20], [320, 20], [320, 50], [20, 50]]}
        truncated = {"text": "OPEN THE DOOR", "score": 0.99,
                     "box": [[115, 20], [320, 20], [320, 50], [115, 50]]}
        with patch("ocr_helper.server.call_engine", side_effect=[[recovered], [truncated]]):
            result = refine_problematic_box(img, original, None)
        self.assertEqual(result["text"], "DO NOT OPEN THE DOOR")
        self.assertEqual(result["box"], original["box"])

    def test_process_refines_symbol_corruption_before_final_filtering(self):
        from ocr_helper import server
        import copy
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        img = server.np.zeros((80, 160, 3), dtype=server.np.uint8)
        clear = {"text": "HELLO", "score": 0.99,
                 "box": [[10, 10], [110, 10], [110, 30], [10, 30]]}
        damaged = {"text": "H@#$P", "score": 0.8,
                   "box": [[10, 40], [90, 40], [90, 65], [10, 65]]}
        for corrected_text, corrected_score, expected in (("HELP", 0.99, ["HELLO", "HELP"]),
                                                           ("H@#$P", 0.99, ["HELLO"]),
                                                           ("HELP", 0.4, ["HELLO"])):
            corrected = {"text": corrected_text, "score": corrected_score,
                         "box": [[10, 20], [90, 20], [90, 45], [10, 45]]}
            calls = []

            def recognize(engine, frame, min_score):
                calls.append(frame.shape)
                if len(calls) <= 2:
                    lines = copy.deepcopy([clear, damaged])
                    if len(calls) == 2:
                        for line in lines:
                            line["box"] = [[x * 2, y * 2] for x, y in line["box"]]
                    return lines
                return [copy.deepcopy(corrected)]

            with self.subTest(text=corrected_text, score=corrected_score), \
                    patch("ocr_helper.server.call_engine", side_effect=recognize):
                response = process_request(dict(id=1, width=160, height=80, lang="en"),
                                           img.tobytes(), None)
                self.assertTrue(response["ok"], response)
                self.assertEqual([line["text"] for line in response["lines"]], expected)
                self.assertGreater(len(calls), 2, "symbol corruption must reach crop refinement")
                self.assertLessEqual(len(calls), 2 + server.MAX_RETRY_CALLS)
                if len(expected) == 2:
                    self.assertEqual(response["lines"][1]["box"], damaged["box"])

    def test_dense_weak_lines_are_deduplicated_before_shared_retry_call_limit(self):
        from ocr_helper import server
        import copy
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        img = server.np.zeros((600, 800, 3), dtype=server.np.uint8)
        lines = [{"text": f"Weak line {i}", "score": 0.70,
                  "box": [[30, 15 + i * 26], [300, 15 + i * 26],
                          [300, 35 + i * 26], [30, 35 + i * 26]]}
                 for i in range(20)]
        calls = []

        def recognize(engine, frame, min_score):
            calls.append(frame.shape)
            if len(calls) > 2:
                return []
            result = copy.deepcopy(lines)
            if len(calls) == 2:
                for line in result:
                    line["box"] = [[x * 2, y * 2] for x, y in line["box"]]
            return result

        with patch("ocr_helper.server.call_engine", side_effect=recognize):
            response = process_request(dict(id=1, width=800, height=600, lang="en"), img.tobytes(), None)
        self.assertTrue(response["ok"], response)
        self.assertEqual(len(calls), 2 + server.MAX_RETRY_CALLS)
        self.assertEqual(len(response["lines"]), 20)
        self.assertCountEqual([line["text"] for line in response["lines"]], [line["text"] for line in lines])

    def test_retry_time_limit_returns_existing_lines_and_keeps_score_threshold(self):
        from ocr_helper import server
        import copy
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        img = server.np.zeros((600, 800, 3), dtype=server.np.uint8)
        original = {"text": "Existing weak text", "score": 0.70,
                    "box": [[30, 40], [300, 40], [300, 60], [30, 60]]}
        for threshold, expected in ((0.5, 1), (0.9, 0)):
            clock = [0.0]
            calls = []

            def recognize(engine, frame, min_score):
                calls.append(frame.shape)
                if len(calls) > 2:
                    clock[0] = server.MAX_RETRY_SECONDS + 0.1
                    return []
                line = copy.deepcopy(original)
                if len(calls) == 2:
                    line["box"] = [[x * 2, y * 2] for x, y in line["box"]]
                return [line]

            with self.subTest(threshold=threshold), \
                 patch("ocr_helper.server.time.perf_counter", side_effect=lambda: clock[0]), \
                 patch("ocr_helper.server.call_engine", side_effect=recognize):
                response = process_request(dict(id=1, width=800, height=600, lang="en", min_score=threshold),
                                           img.tobytes(), None)
            self.assertTrue(response["ok"], response)
            self.assertEqual(len(calls), 3, "no new retry may begin after the shared deadline")
            self.assertEqual(len(response["lines"]), expected)

    def test_refinement_only_replaces_the_original_line(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required for crop refinement")
        img = server.np.zeros((120, 180, 3), dtype=server.np.uint8)
        original = {"text": "HE1P", "score": 0.8,
                    "box": [[40, 50], [100, 50], [100, 65], [40, 65]]}
        neighbor = {"text": "GOODBYE", "score": 0.99,
                    "box": [[20, 0], [80, 0], [80, 15], [20, 15]]}
        corrected = {"text": "HELP", "score": 0.95,
                     "box": [[20, 20], [80, 20], [80, 35], [20, 35]],
                     "words": [{"text": "HELP", "score": 0.95,
                                "box": [[20, 20], [80, 20], [80, 35], [20, 35]]}]}
        import copy
        for variant in (0, 1):
            for include_target in (False, True):
                with self.subTest(variant=variant, include_target=include_target):
                    candidates = [neighbor, corrected] if include_target else [neighbor]
                    responses = [[], []]
                    responses[variant] = copy.deepcopy(candidates)
                    with patch("ocr_helper.server.call_engine", side_effect=responses):
                        result = refine_problematic_box(img, original, None)
                    self.assertEqual(result["text"], "HELP" if include_target else "HE1P")
                    self.assertEqual(result["box"], original["box"])
                    if include_target:
                        self.assertEqual(result["words"][0]["box"], original["box"])


    def test_should_upscale_frame(self):
        self.assertTrue(should_upscale_frame(1000, 360, cv2_available=True))
        self.assertTrue(should_upscale_frame(800, 600, cv2_available=True))
        # 1920 * 1080 = 2_073_600 > 1_500_000
        self.assertFalse(should_upscale_frame(1920, 1080, cv2_available=True))
        # If cv2 is not available, never upscale
        self.assertFalse(should_upscale_frame(1000, 360, cv2_available=False))

    def test_rescale_boxes_down(self):
        lines = [
            {
                "text": "Sample",
                "box": [[20.0, 40.0], [100.0, 40.0], [100.0, 80.0], [20.0, 80.0]],
                "score": 0.95,
            }
        ]
        rescaled = rescale_boxes_down(lines, 2.0)
        expected_box = [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
        self.assertEqual(rescaled[0]["box"], expected_box)
        self.assertEqual(rescaled[0]["text"], "Sample")
        self.assertEqual(rescaled[0]["score"], 0.95)

        # Scale factor <= 1.0 or empty lines
        self.assertEqual(rescale_boxes_down([], 2.0), [])
        self.assertEqual(rescale_boxes_down(lines, 1.0), lines)

    def test_process_request_small_frame_upscale(self):
        # 4x4 frame <= SMALL_FRAME_PIXELS -> should trigger upscale 2x and return rescaled box
        received_shapes = []

        class RecordingEngine:
            def __call__(self, img_bgr):
                if hasattr(img_bgr, "shape"):
                    received_shapes.append(img_bgr.shape)
                    if img_bgr.shape[0] == 4:
                        return []
                box_up = [[20.0, 20.0], [60.0, 20.0], [60.0, 40.0], [20.0, 40.0]]
                return [{"box": box_up, "text": "Upscaled text", "score": 0.90}]

        engine = RecordingEngine()
        raw_bgr = b"\x00" * (4 * 4 * 3)
        resp = process_request({"id": 10, "width": 4, "height": 4, "lang": "en"}, raw_bgr, engine)
        self.assertTrue(resp["ok"])
        self.assertEqual(len(resp["lines"]), 1)
        # Coordinates must be divided by 2
        expected_box = [[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]]
        self.assertEqual(resp["lines"][0]["box"], expected_box)
        # Verify engine received upscaled 8x8 image (4*2, 4*2)
        if received_shapes:
            self.assertEqual([shape[:2] for shape in received_shapes], [(4, 4), (8, 8)])

    def test_upscale_preserves_original_only_line_and_prefers_confidence(self):
        from ocr_helper import server
        if server.np is None or server.cv2 is None:
            self.skipTest("numpy and OpenCV are required")

        def engine(img, **kwargs):
            if img.shape[0] == 100:
                return [
                    {"text": "Original only", "score": 0.95,
                     "box": [[10, 10], [100, 10], [100, 30], [10, 30]]},
                    {"text": "READ", "score": 0.86,
                     "box": [[10, 60], [100, 60], [100, 80], [10, 80]]}]
            return [{"text": "READY", "score": 0.99,
                     "box": [[20, 120], [200, 120], [200, 160], [20, 160]]}]

        result = process_request({"id": 1, "width": 120, "height": 100, "lang": "en"},
                                 bytes(120 * 100 * 3), engine)
        self.assertTrue(result["ok"])
        self.assertEqual({line["text"] for line in result["lines"]}, {"Original only", "READY"})


if __name__ == "__main__":
    unittest.main()
