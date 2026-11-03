import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from service import create_app


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.app = create_app({"TESTING": True, "ENVIRONMENT": "production", "API_TOKEN": "x" * 32,
                               "UPLOAD_ROOT": self.directory.name})
        self.client = self.app.test_client()
        self.addCleanup(self.app.extensions["store"].engine.dispose)
        self.headers = {"Authorization": "Bearer " + "x" * 32}

    def upload(self, name="sample.mp4", content=b"test"):
        return self.client.post("/api/v1/recognize", data={"video": (io.BytesIO(content), name)}, headers=self.headers)

    def test_english_engine_does_not_load_original_checkpoint(self):
        result = {'engine': 'auto-avsr', 'text': 'hello', 'usable_text': 'hello', 'audio_used': False}
        with patch('general_bridge.available', return_value=True), patch('general_bridge.recognize_general', return_value=result), patch('lip_reader.get_model', side_effect=AssertionError('Unrelated model loaded')), patch('audiovisual.attach_lip_expressions'):
            response = self.client.post('/api/v1/recognize', headers=self.headers,
                data={'video': (io.BytesIO(b'test'), 'clip.mp4'), 'engine': 'auto-avsr'})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json['text'], 'hello')
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_production_requires_secret(self):
        with self.assertRaises(ValueError):
            create_app({"ENVIRONMENT": "production", "API_TOKEN": ""})

    def test_health_and_unauthorized(self):
        self.assertEqual(self.client.get("/health/live").status_code, 200)
        self.assertEqual(self.client.get("/health/ready").status_code, 503)
        self.assertEqual(self.client.post("/api/v1/recognize").status_code, 401)
        self.assertEqual(self.client.get("/absent").status_code, 404)

    def test_upload_limits_and_validation(self):
        self.assertEqual(self.upload(name="bad.exe").status_code, 400)
        self.app.config["MAX_CONTENT_LENGTH"] = 1024
        self.assertEqual(self.upload(content=b"x" * 2048).status_code, 413)

    def test_file_crossing_multipart_buffer_boundary(self):
        boundary = "curl-upload-boundary"
        prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="video"; '
                  'filename="grid.mpg"\r\nContent-Type: video/mpeg\r\n\r\n').encode()
        # CR at the end of a 64 KiB read must be retained while the next chunk
        # arrives. A form-memory limit equal to the read size rejects this file.
        video = b"x" * (65536 - len(prefix) - 1) + b"\r" + b"x" * 300000
        body = prefix + video + (f'\r\n--{boundary}\r\nContent-Disposition: form-data; '
                                  f'name="mode"\r\n\r\ngrid\r\n--{boundary}--\r\n').encode()
        def recognize(path, mode, *args, **kwargs):
            self.assertEqual(Path(path).read_bytes(), video)
            self.assertEqual(mode, "grid")
            return {"usable_text": None}
        with patch("lip_reader.get_model"), patch("preprocessing.LipDetector"), patch("recognition.recognize_file", side_effect=recognize):
            response = self.client.post("/api/v1/recognize", data=body,
                content_type=f"multipart/form-data; boundary={boundary}", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_oversized_text_field_still_rejected(self):
        response = self.client.post("/api/v1/recognize", headers=self.headers,
            data={"video": (io.BytesIO(b"test"), "grid.mpg"),
                  "mode": "x" * (128 * 1024 + 1)})
        self.assertEqual(response.status_code, 413)

    def test_busy_is_bounded(self):
        slot = self.app.extensions["inference_slot"]
        slot.acquire()
        try:
            response = self.upload()
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["Retry-After"], "5")
        finally:
            slot.release()

    def test_cleanup_and_no_server_path_disclosure(self):
        result = {"text": "hi", "usable_text": "hi", "source": "/private/input.mp4", "weights": "/private/weights.h5",
                  "confidence": None, "review_required": True}
        with patch("lip_reader.get_model"), patch("preprocessing.LipDetector"), patch("recognition.recognize_file", return_value=result):
            response = self.upload(name="../../sample.mp4")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("source", response.json)
        self.assertNotIn("weights", response.json)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        self.assertEqual(self.client.get("/health/ready").status_code, 200)

    def test_failure_cleans_up_and_releases_slot(self):
        with patch("lip_reader.get_model"), patch("preprocessing.LipDetector"), patch("recognition.recognize_file", side_effect=ValueError("private path")):
            self.assertEqual(self.upload().status_code, 422)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])
        slot = self.app.extensions["inference_slot"]
        self.assertTrue(slot.acquire(blocking=False))
        slot.release()


if __name__ == "__main__":
    unittest.main()
