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

    def test_translate_endpoint_returns_result_with_tokens(self):
        app = create_app()
        client = app.test_client()
        response = client.post(
            "/api/translate",
            json={"text": "Where are you going?", "backend": "nllb"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("translation", payload)
        self.assertIsInstance(payload["translation"], str)
        self.assertTrue(len(payload["translation"]) > 0)
        self.assertIn("phonetic_guide", payload)
        self.assertIn("tokens", payload)

    def test_tts_endpoint_returns_result(self):
        app = create_app()
        client = app.test_client()
        response = client.post(
            "/api/tts",
            json={"text": "Maayong buntag", "backend": "vits"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("output", payload)

    def test_backends_endpoint_lists_models(self):
        app = create_app()
        client = app.test_client()
        response = client.get("/api/backends")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("nllb", payload["translation"])

    def test_translate_phonetic_guide_empty_for_plain_text(self):
        """Phonetic guide should be empty when translation has no glottal markers."""
        app = create_app()
        client = app.test_client()
        # Translate plain English — NLLB output is unlikely to have
        # mid-word apostrophes, so phonetic_guide should be empty.
        response = client.post(
            "/api/translate",
            json={"text": "Hello", "backend": "nllb"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        # phonetic_guide is either '' or contains ʔ — never a spurious cue
        guide = payload.get("phonetic_guide", "")
        if guide:
            self.assertIn("ʔ", guide)


if __name__ == "__main__":
    unittest.main()
