import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.tokenizer import GlottalAwareTokenizer


class TokenizerTests(unittest.TestCase):
    def test_tokenizer_preserves_glottal_markers_and_punctuation(self):
        tokenizer = GlottalAwareTokenizer()

        tokens = tokenizer.tokenize("Hain'ka, pasingud?")

        self.assertEqual(tokens, ["Hain'ka", ",", "pasingud", "?"])
        self.assertEqual(tokenizer.encode("Hain'ka"), ["Hain'ka"])
        self.assertEqual(tokenizer.decode(["Hain'ka", "pasingud"]), "Hain'ka pasingud")


if __name__ == "__main__":
    unittest.main()
