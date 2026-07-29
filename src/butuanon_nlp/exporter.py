import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union


def export_samples(samples: Iterable[Dict[str, Any]], output_path: Union[str, Path], format_name: str = "jsonl") -> Path:
    """Export aligned samples to JSONL or TSV for downstream training and evaluation."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    format_name = (format_name or "jsonl").lower()
    if format_name == "jsonl":
        with output.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    elif format_name == "tsv":
        with output.open("w", encoding="utf-8") as handle:
            handle.write("id\ten\tbtw\taudio_file\n")
            for sample in samples:
                handle.write(
                    "\t".join(
                        [
                            str(sample.get("id", "")),
                            str(sample.get("en", "")),
                            str(sample.get("btw", "")),
                            str(sample.get("audio_file", "")),
                        ]
                    )
                    + "\n"
                )
    else:
        raise ValueError(f"Unsupported format: {format_name}")

    return output
