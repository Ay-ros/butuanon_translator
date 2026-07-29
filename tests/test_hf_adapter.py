import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.models import HuggingFaceAdapter, TranslationModel, SpeechModel


class HuggingFaceAdapterTests(unittest.TestCase):
    def test_adapter_builds_model_metadata(self):
        adapter = HuggingFaceAdapter("nllb", task="translation")
        metadata = adapter.describe()

        self.assertEqual(metadata["task"], "translation")
        self.assertEqual(metadata["name"], "nllb")
        self.assertIn("huggingface", metadata["source"].lower())

    def test_translation_and_speech_models_can_use_adapter(self):
        translator = TranslationModel(model_name="nllb")
        translator.adapter = HuggingFaceAdapter("nllb", task="translation")
        result = translator.translate("Where are you going?")

        self.assertIn("butuanon", result.lower())

        recognizer = SpeechModel(model_name="whisper")
        recognizer.adapter = HuggingFaceAdapter("whisper", task="asr")
        transcribed = recognizer.transcribe("clip.wav")
        self.assertIn("whisper", transcribed.lower())


if __name__ == "__main__":
    unittest.main()
