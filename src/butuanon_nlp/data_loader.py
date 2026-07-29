import json
from pathlib import Path
from typing import Any, Dict, List, Union

REQUIRED_FIELDS = ("id", "en", "btw", "audio_file")


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a sample contains the required translation fields."""
    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        raise ValueError(f"Sample is missing required field(s): {', '.join(missing)}")

    if not isinstance(sample["id"], str) or not sample["id"].strip():
        raise ValueError("Sample field 'id' must be a non-empty string")

    if not isinstance(sample["en"], str) or not sample["en"].strip():
        raise ValueError("Sample field 'en' must be a non-empty string")

    if not isinstance(sample["btw"], str) or not sample["btw"].strip():
        raise ValueError("Sample field 'btw' must be a non-empty string")

    if not isinstance(sample["audio_file"], str) or not sample["audio_file"].strip():
        raise ValueError("Sample field 'audio_file' must be a non-empty string")

    return sample


def load_samples(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load aligned English-Butuanon samples from a JSONL file."""
    sample_path = Path(path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    samples: List[Dict[str, Any]] = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on line {line_number}, got {type(payload).__name__}")

            samples.append(validate_sample(payload))

    return samples
