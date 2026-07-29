from __future__ import annotations

from typing import Dict, List


def build_preprocessing_plan() -> List[Dict[str, str]]:
    """Create the data-preprocessing workflow described in the PDF."""
    return [
        {
            "name": "Digitization and transcription",
            "goal": "Convert physical documents and field audio into UTF-8 text and define a glottal-stop convention.",
        },
        {
            "name": "Audio slicing",
            "goal": "Slice recordings into 3-10 second WAV chunks that match single sentences or phrases.",
        },
        {
            "name": "Sentence alignment",
            "goal": "Align English and Butuanon sentences in a structured JSONL or TSV dataset.",
        },
        {
            "name": "Native speaker validation",
            "goal": "Have fluent speakers review the pairs for contextual and phonetic accuracy.",
        },
    ]


def phoneticize_text(text: str) -> str:
    """Map glottal markers to the IPA glottal stop symbol for speech-model training."""
    if not text:
        return ""

    normalized = text.replace("'", "ʔ").replace("’", "ʔ").replace("-", "ʔ")
    if "ʔ" not in normalized:
        return f"{normalized} [glottal cue: ʔ]"
    return normalized
