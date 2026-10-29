"""
Tests for OCR Helper protocol server using mock engines (no models needed).
Can be run via pytest (`python -m pytest ocr_helper`) or unittest.
"""

import io
import json
import unittest

from ocr_helper.server import (
    extract_lines_from_result,
    process_request,
    read_exact,
    run_server,
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
            {"id": 1, "width": 10, "height": 10, "lang": "ru"}, None, engine
        )
        self.assertFalse(resp.get("ok"))
        self.assertEqual(resp.get("id"), 1)
        self.assertIn("not supported yet", resp.get("error", ""))

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
            [[[0, 0], [2, 0], [2, 2], [0, 2]], "Test", 0.99]
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
            [[[0, 0], [1, 0], [1, 1], [0, 1]], "OK", 0.9]
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


if __name__ == "__main__":
    unittest.main()
