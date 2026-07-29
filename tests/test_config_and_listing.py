import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.config import ProjectConfig
from butuanon_nlp.cli import build_parser


class ConfigAndListingTests(unittest.TestCase):
    def test_config_has_default_backend(self):
        config = ProjectConfig.from_workspace(ROOT)
        self.assertEqual(config.default_backend, "nllb")

    def test_parser_supports_listing_backends(self):
        parser = build_parser()
        args = parser.parse_args(["--list-backends"])

        self.assertTrue(args.list_backends)


if __name__ == "__main__":
    unittest.main()
