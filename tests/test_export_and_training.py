import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.exporter import export_samples
from butuanon_nlp.training import simulate_training_run


class ExportAndTrainingTests(unittest.TestCase):
    def test_export_samples_to_jsonl_and_tsv(self):
        samples = [
            {"id": "0001", "en": "Where are you going?", "btw": "Hain kaw pasingud?", "audio_file": "clip_0001.wav"},
            {"id": "0002", "en": "I am eating.", "btw": "Gakaon ako.", "audio_file": "clip_0002.wav"},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir)
            jsonl_path = temp_path / "samples.jsonl"
            tsv_path = temp_path / "samples.tsv"

            export_samples(samples, jsonl_path, format_name="jsonl")
            export_samples(samples, tsv_path, format_name="tsv")

            jsonl_content = jsonl_path.read_text(encoding="utf-8")
            self.assertIn('"id": "0001"', jsonl_content)

            tsv_content = tsv_path.read_text(encoding="utf-8")
            self.assertIn("0001\tWhere are you going?\tHain kaw pasingud?\tclip_0001.wav", tsv_content)

    def test_simulate_training_run_returns_metrics(self):
        run = simulate_training_run(samples_count=3, epochs=2)

        self.assertEqual(run["dataset_size"], 3)
        self.assertEqual(run["epochs"], 2)
        self.assertEqual(run["status"], "completed")
        self.assertGreaterEqual(run["metrics"]["train_loss"], 0.0)
        self.assertGreaterEqual(run["metrics"]["val_bleu"], 0.0)


if __name__ == "__main__":
    unittest.main()
