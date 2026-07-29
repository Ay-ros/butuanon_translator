from __future__ import annotations

from typing import Any, Dict, List


def build_training_plan(max_samples: int = 100) -> Dict[str, Any]:
    """Create a simple phased training plan for the seed corpus."""
    return {
        "dataset_size": max_samples,
        "phases": [
            {
                "name": "Data validation",
                "goal": "Ensure every sentence pair has English, Butuanon, and audio metadata.",
            },
            {
                "name": "Tokenizer adaptation",
                "goal": "Use a glottal-aware tokenizer and preserve apostrophes/hyphens in the corpus.",
            },
            {
                "name": "Fine-tuning",
                "goal": "Fine-tune a baseline translation model with the curated Butuanon seed dataset.",
            },
            {
                "name": "Evaluation",
                "goal": "Measure translation quality and prepare the next expansion cycle.",
            },
        ],
    }


def summarize_training_plan(plan: Dict[str, Any]) -> str:
    """Return a readable description of the training plan."""
    phases = plan.get("phases", [])
    lines = [f"Training plan for {plan.get('dataset_size', 0)} samples:"]
    for index, phase in enumerate(phases, start=1):
        lines.append(f"Phase {index}: {phase['name']} - {phase['goal']}")
    return "\n".join(lines)


def build_task_training_workflow(task: str = "all", max_samples: int = 100, epochs: int = 3) -> Dict[str, Any]:
    """Create a structured workflow definition for translation, ASR, or TTS training."""
    task = (task or "all").lower()

    model_specs: List[Dict[str, Any]] = []
    if task in {"all", "translation"}:
        model_specs.append(
            {
                "name": "nllb",
                "family": "translation",
                "goal": "Fine-tune a multilingual encoder-decoder for Butuanon translation.",
                "dataset": "parallel en <-> btw",
                "epochs": epochs,
            }
        )
    if task in {"all", "asr", "speech", "transcription"}:
        model_specs.append(
            {
                "name": "whisper",
                "family": "asr",
                "goal": "Adapt Whisper for Butuanon speech recognition with audio-text pairs.",
                "dataset": "audio + transcript",
                "epochs": epochs,
            }
        )
    if task in {"all", "tts", "speech"}:
        model_specs.append(
            {
                "name": "vits",
                "family": "tts",
                "goal": "Train or fine-tune a TTS model for natural Butuanon voice synthesis.",
                "dataset": "text + speech samples",
                "epochs": epochs,
            }
        )

    if not model_specs:
        raise ValueError(f"Unsupported training task: {task}")

    return {
        "task": task,
        "dataset_size": max_samples,
        "epochs": epochs,
        "models": model_specs,
        "status": "initialized",
    }


def summarize_training_workflow(workflow: Dict[str, Any]) -> str:
    """Render a readable summary for a task-specific training workflow."""
    lines = [f"Workflow for task '{workflow.get('task', 'all')}' with {workflow.get('dataset_size', 0)} samples:"]
    for model in workflow.get("models", []):
        lines.append(f"- {model['name']} ({model['family']}): {model['goal']}")
    return "\n".join(lines)


def simulate_training_run(samples_count: int, epochs: int = 3) -> Dict[str, Any]:
    """Return a deterministic training summary that can be used as a prototype before real fine-tuning."""
    train_loss = max(1.6 - (samples_count / 100) - (epochs * 0.1), 0.05)
    val_bleu = min(0.65 + (samples_count / 200) + (epochs * 0.03), 0.95)

    return {
        "dataset_size": samples_count,
        "epochs": epochs,
        "status": "completed",
        "metrics": {
            "train_loss": round(train_loss, 4),
            "val_bleu": round(val_bleu, 4),
        },
    }
