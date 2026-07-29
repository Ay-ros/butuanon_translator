import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from butuanon_nlp.training import (
    build_task_training_workflow,
    build_training_plan,
    summarize_training_plan,
    summarize_training_workflow,
)


class TrainingPipelineTests(unittest.TestCase):
    def test_build_training_plan_returns_expected_steps(self):
        plan = build_training_plan(max_samples=5)

        self.assertEqual(plan["dataset_size"], 5)
        self.assertEqual(plan["phases"][0]["name"], "Data validation")
        self.assertIn("Tokenizer", plan["phases"][1]["name"])
        self.assertIn("Fine-tuning", plan["phases"][2]["name"])

    def test_summarize_training_plan_is_readable(self):
        plan = build_training_plan(max_samples=3)
        summary = summarize_training_plan(plan)

        self.assertIn("Phase 1", summary)
        self.assertIn("3 samples", summary)

    def test_build_task_training_workflow_includes_all_model_families(self):
        workflow = build_task_training_workflow(task="all", max_samples=5, epochs=2)

        self.assertEqual(workflow["task"], "all")
        self.assertEqual(len(workflow["models"]), 3)
        self.assertEqual([model["name"] for model in workflow["models"]], ["nllb", "whisper", "vits"])

    def test_summarize_training_workflow_lists_started_models(self):
        workflow = build_task_training_workflow(task="translation", max_samples=3, epochs=2)
        summary = summarize_training_workflow(workflow)

        self.assertIn("translation", summary.lower())
        self.assertIn("nllb", summary.lower())


if __name__ == "__main__":
    unittest.main()
