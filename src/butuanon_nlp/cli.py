import argparse
from pathlib import Path

from .config import ProjectConfig
from .data_loader import load_samples
from .exporter import export_samples
from .models import (
    TranslationModel,
    SpeechModel,
    create_default_model_registry,
)
from .tokenizer import GlottalAwareTokenizer
from .training import (
    build_task_training_workflow,
    simulate_training_run,
    summarize_training_workflow,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect, export, and prototype Butuanon seed training")
    parser.add_argument("--data", type=Path, default=None, help="Path to the JSONL dataset")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to preview")
    parser.add_argument("--export", type=Path, default=None, help="Export the samples to a JSONL or TSV file")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "tsv"], help="File format for export")
    parser.add_argument("--simulate-train", action="store_true", help="Run a lightweight training simulation")
    parser.add_argument("--task", type=str, default="all", help="Training task to scaffold: all, translation, asr, tts")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch count for the training simulation")
    parser.add_argument("--translate", type=str, default=None, help="Run a placeholder translation call")
    parser.add_argument("--transcribe", type=str, default=None, help="Run a placeholder transcription call")
    parser.add_argument("--synthesize", type=str, default=None, help="Run a placeholder speech synthesis call")
    parser.add_argument("--backend", type=str, default="nllb", help="Select a registered backend such as nllb, whisper, or vits")
    parser.add_argument("--list-backends", action="store_true", help="List the available registered backends")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = ProjectConfig.from_workspace(Path(__file__).resolve().parents[2])
    data_path = args.data or config.data_path
    samples = load_samples(data_path)
    tokenizer = GlottalAwareTokenizer()

    if args.list_backends:
        registry = create_default_model_registry()
        print("Available backends:")
        for name in registry.list_models():
            print(f"- {name}")
        return 0

    print(f"Loaded {len(samples)} samples from {data_path}")
    for sample in samples[: args.limit]:
        tokens = tokenizer.tokenize(sample["btw"])
        print(f"{sample['id']}: {sample['btw']} -> {tokens}")

    if args.export:
        output_path = export_samples(samples, args.export, format_name=args.format)
        print(f"Exported {len(samples)} samples to {output_path}")

    if args.simulate_train:
        run = simulate_training_run(samples_count=len(samples), epochs=args.epochs)
        print(f"Training simulation: {run}")

    workflow = build_task_training_workflow(task=args.task, max_samples=len(samples), epochs=args.epochs)
    print(summarize_training_workflow(workflow))

    registry = create_default_model_registry()

    if args.translate:
        backend_name = args.backend if args.backend in {"nllb", "whisper", "vits"} else "nllb"
        backend = registry.get(backend_name)
        if isinstance(backend, TranslationModel):
            print(backend.translate(args.translate))
        else:
            print(f"Backend '{backend_name}' is not a translation backend")

    if args.transcribe:
        backend_name = args.backend if args.backend in {"nllb", "whisper", "vits"} else "whisper"
        backend = registry.get(backend_name)
        if isinstance(backend, SpeechModel):
            print(backend.transcribe(args.transcribe))
        else:
            print(f"Backend '{backend_name}' is not a speech backend")

    if args.synthesize:
        backend_name = args.backend if args.backend in {"nllb", "whisper", "vits"} else "vits"
        backend = registry.get(backend_name)
        if isinstance(backend, SpeechModel):
            print(backend.synthesize(args.synthesize))
        else:
            print(f"Backend '{backend_name}' is not a speech backend")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
