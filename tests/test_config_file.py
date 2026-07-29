import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.config import load_config, save_config


class ConfigFileTests(unittest.TestCase):
    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            save_config(config_path, {"default_backend": "whisper"})
            loaded = load_config(config_path)

            self.assertEqual(loaded["default_backend"], "whisper")


if __name__ == "__main__":
    unittest.main()
