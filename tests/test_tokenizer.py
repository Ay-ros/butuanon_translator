import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.tokenizer import GlottalAwareTokenizer, normalize_glottal, GLOTTAL_CANONICAL


class TokenizerTests(unittest.TestCase):
    def test_tokenizer_preserves_glottal_markers_and_punctuation(self):
        tokenizer = GlottalAwareTokenizer()

        # ASCII apostrophe is normalised to the canonical glottal marker
        tokens = tokenizer.tokenize("Hain'ka, pasingud?")

        expected_first = f"Hain{GLOTTAL_CANONICAL}ka"
        self.assertEqual(tokens[0], expected_first)
        self.assertIn(",", tokens)
        self.assertIn("pasingud", tokens)
        self.assertIn("?", tokens)

    def test_encode_decode_roundtrip(self):
        tokenizer = GlottalAwareTokenizer()
        tokens = tokenizer.encode(f"Hain{GLOTTAL_CANONICAL}ka")
        self.assertEqual(len(tokens), 1)
        decoded = tokenizer.decode(tokens)
        self.assertIn("Hain", decoded)

    def test_normalize_glottal_replaces_all_variants(self):
        raw = "it\u2018s it\u2019s it's it\u02BCs"
        result = normalize_glottal(raw)
        self.assertNotIn("'", result)
        self.assertNotIn("\u2018", result)
        self.assertNotIn("\u02BC", result)
        # all should be canonical
        self.assertEqual(result.count(GLOTTAL_CANONICAL), 4)


if __name__ == "__main__":
    unittest.main()
