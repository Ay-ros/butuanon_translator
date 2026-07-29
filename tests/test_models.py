import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.models import HuggingFaceAdapter, TranslationModel, SpeechModel


class ModelIntegrationTests(unittest.TestCase):
    def test_translation_model_returns_placeholder_translation(self):
        model = TranslationModel(model_name="nllb")
        result = model.translate("Where are you going?")

        self.assertIn("Butuanon", result)
        self.assertIn("output", result.lower())

    def test_speech_model_returns_placeholder_pipeline(self):
        model = SpeechModel(model_name="whisper")
        result = model.transcribe("clip.wav")

        self.assertIn("placeholder", result.lower())

        tts_model = SpeechModel(model_name="vits")
        audio_result = tts_model.synthesize("Hain kaw pasingud?")
        self.assertIn("placeholder", audio_result.lower())

    def test_translation_model_reports_runtime_status(self):
        adapter = HuggingFaceAdapter("facebook/nllb-200-distilled-600M", task="translation")
        model = TranslationModel(model_name="nllb", use_huggingface=True, adapter=adapter)
        runtime = model.get_runtime_status()

        self.assertEqual(runtime["model_name"], "nllb")
        self.assertIn(runtime["status"], {"ready", "fallback"})


if __name__ == "__main__":
    unittest.main()
