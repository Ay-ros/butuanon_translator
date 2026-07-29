import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.models import HuggingFaceAdapter, TranslationModel, SpeechModel


class ModelIntegrationTests(unittest.TestCase):
    def test_translation_model_returns_nonempty_output(self):
        """When use_huggingface=False, the model returns an offline placeholder.
        When use_huggingface=True and transformers is installed, it returns
        a real translation.  Either way the output must be a non-empty string.
        """
        model = TranslationModel(model_name="nllb", use_huggingface=False)
        result = model.translate("Where are you going?")

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_speech_model_returns_nonempty_output(self):
        model = SpeechModel(model_name="whisper", use_huggingface=False)
        result = model.transcribe("clip.wav")

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

        tts_model = SpeechModel(model_name="vits", use_huggingface=False)
        audio_result = tts_model.synthesize("Hain kaw pasingud?")
        self.assertIsInstance(audio_result, str)
        self.assertTrue(len(audio_result) > 0)

    def test_translation_model_reports_runtime_status(self):
        adapter = HuggingFaceAdapter("facebook/nllb-200-distilled-600M", task="translation")
        model = TranslationModel(model_name="nllb", use_huggingface=True, adapter=adapter)
        runtime = model.get_runtime_status()

        self.assertEqual(runtime["model_name"], "nllb")
        self.assertIn(runtime["status"], {"ready", "fallback"})
        self.assertIn("device", runtime)

    def test_translation_model_has_configurable_target_lang(self):
        model = TranslationModel(model_name="nllb", target_lang="ceb_Latn")
        self.assertEqual(model.target_lang, "ceb_Latn")

        model2 = TranslationModel(model_name="nllb", target_lang="fil_Latn")
        self.assertEqual(model2.target_lang, "fil_Latn")


if __name__ == "__main__":
    unittest.main()
