import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.web_app import create_app


class WebAppTests(unittest.TestCase):
    def test_app_serves_homepage(self):
        app = create_app()
        client = app.test_client()
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"BisayaHub", response.data)

    def test_demo_endpoint_returns_summary(self):
        app = create_app()
        client = app.test_client()
        response = client.post(
            "/api/demo",
            json={"text": "Where are you going?", "audio_path": "clip.wav"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("translation", payload["summary"].lower())

    def test_translate_endpoint_returns_phonetic_guide(self):
        app = create_app()
        client = app.test_client()
        response = client.post(
            "/api/translate",
            json={"text": "Hain kaw pasingud?", "backend": "nllb"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("phonetic_guide", payload)
        self.assertIn("ʔ", payload["phonetic_guide"])


if __name__ == "__main__":
    unittest.main()
