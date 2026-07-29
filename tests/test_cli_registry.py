import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.cli import build_parser


class CliRegistryTests(unittest.TestCase):
    def test_parser_accepts_backend_selection(self):
        parser = build_parser()
        args = parser.parse_args(["--backend", "whisper"])

        self.assertEqual(args.backend, "whisper")


if __name__ == "__main__":
    unittest.main()
