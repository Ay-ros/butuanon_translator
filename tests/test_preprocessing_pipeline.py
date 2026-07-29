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

    def test_phoneticize_text_annotates_midword_glottal_markers(self):
        # Mid-word apostrophe → glottal stop
        result = phoneticize_text("Hain'ka")
        self.assertIn("ʔ", result)
        self.assertIn("Hainʔka", result)

        # Mid-word hyphen → glottal stop
        result = phoneticize_text("dal-a")
        self.assertIn("ʔ", result)
        self.assertIn("dalʔa", result)

    def test_phoneticize_text_returns_empty_when_no_glottal_markers(self):
        # Plain text without any apostrophes/hyphens between letters
        result = phoneticize_text("Maayong buntag")
        self.assertEqual(result, "")

    def test_phoneticize_text_returns_empty_for_empty_input(self):
        self.assertEqual(phoneticize_text(""), "")

    def test_phoneticize_shows_transformation_guide(self):
        result = phoneticize_text("Palihug dal-a ako")
        # Should include both the IPA version and the word transformation
        self.assertIn("dalʔa", result)
        self.assertIn("dal-a", result)
        self.assertIn("→", result)


if __name__ == "__main__":
    unittest.main()
