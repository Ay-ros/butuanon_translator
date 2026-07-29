import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.models import ModelRegistry, TranslationModel, SpeechModel


class ModelRegistryTests(unittest.TestCase):
    def test_registry_registers_translation_and_speech_backends(self):
        registry = ModelRegistry()
        registry.register("nllb", TranslationModel(model_name="nllb"))
        registry.register("whisper", SpeechModel(model_name="whisper"))
        registry.register("vits", SpeechModel(model_name="vits"))

        self.assertEqual(registry.get("nllb").model_name, "nllb")
        self.assertEqual(registry.get("whisper").model_name, "whisper")
        self.assertEqual(registry.get("vits").model_name, "vits")
        self.assertEqual(len(registry.list_models()), 3)

    def test_registry_supports_lookup_by_task(self):
        registry = ModelRegistry()
        registry.register("nllb", TranslationModel(model_name="nllb"))
        registry.register("whisper", SpeechModel(model_name="whisper"))

        self.assertEqual(registry.get_for_task("translation")[0].model_name, "nllb")
        self.assertEqual(registry.get_for_task("asr")[0].model_name, "whisper")


if __name__ == "__main__":
    unittest.main()
