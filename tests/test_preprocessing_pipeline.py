import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.preprocessing import build_preprocessing_plan, phoneticize_text


class PreprocessingPipelineTests(unittest.TestCase):
    def test_build_preprocessing_plan_returns_pdf_phases(self):
        plan = build_preprocessing_plan()

        self.assertEqual(plan[0]["name"], "Digitization and transcription")
        self.assertEqual(plan[1]["name"], "Audio slicing")
        self.assertEqual(plan[2]["name"], "Sentence alignment")
        self.assertEqual(plan[3]["name"], "Native speaker validation")

    def test_phoneticize_text_replaces_glottal_markers_with_ipa_symbol(self):
        self.assertEqual(phoneticize_text("Hain'ka"), "Hainʔka")
        self.assertEqual(phoneticize_text("pasingud"), "pasingud [glottal cue: ʔ]")
        self.assertEqual(phoneticize_text("báy-bay"), "báyʔbay")


if __name__ == "__main__":
    unittest.main()
