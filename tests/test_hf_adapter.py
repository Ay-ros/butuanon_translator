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

    def test_translation_model_can_use_adapter_offline(self):
        """With use_huggingface=False the adapter is attached but the model
        returns an offline placeholder — no download required."""
        translator = TranslationModel(model_name="nllb", use_huggingface=False)
        translator.adapter = HuggingFaceAdapter("nllb", task="translation")
        result = translator.translate("Where are you going?")

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_speech_model_can_use_adapter_offline(self):
        recognizer = SpeechModel(model_name="whisper", use_huggingface=False)
        recognizer.adapter = HuggingFaceAdapter("whisper", task="asr")
        transcribed = recognizer.transcribe("clip.wav")

        self.assertIsInstance(transcribed, str)
        self.assertTrue(len(transcribed) > 0)


if __name__ == "__main__":
    unittest.main()
