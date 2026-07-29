import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.data_loader import load_samples


class DataLoaderTests(unittest.TestCase):
    def test_load_samples_preserves_glottal_markers_and_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = Path(tmp_dir) / "samples.jsonl"
            sample_path.write_text(
                json.dumps({"id": "0001", "en": "Where are you going?", "btw": "Hain kaw pasingud?", "audio_file": "clip_0001.wav"}) + "\n"
                + json.dumps({"id": "0002", "en": "I am eating.", "btw": "Gakaon ako.", "audio_file": "clip_0002.wav"}) + "\n",
                encoding="utf-8",
            )

            samples = load_samples(sample_path)

            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]["btw"], "Hain kaw pasingud?")
            self.assertTrue(samples[1]["audio_file"].endswith(".wav"))
            self.assertTrue(samples[0].get("btw"))


if __name__ == "__main__":
    unittest.main()
