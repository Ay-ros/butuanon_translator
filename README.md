# Butuanon NLP Translation Platform

This workspace contains a working prototype scaffold for a low-resource Butuanon NLP platform. It provides:

- text-to-text translation (adapter for Hugging Face / NLLB)
- speech-to-text (Whisper)
- text-to-speech (lightweight pyttsx3 fallback)
- a Flask web UI and CLI for quick demos and training scaffolds

## Project Vision

The platform uses transfer learning and careful data curation to build a practical baseline for Butuanon NLP even when a large digital corpus is not yet available. The first milestone is to create a small, validated seed dataset and a repeatable workflow for future expansion.

## Core Requirements

- Local fine-tuning on consumer-grade GPUs such as the NVIDIA RTX 4070 Super
- Baseline models from Hugging Face, including NLLB for translation and Whisper for transcription
- A custom tokenization strategy that preserves glottal markers and apostrophes
- A strict JSONL-based data format for parallel translation samples

## Seed Data Format

Each sample contains:

- id: a unique identifier
- en: the English sentence
- btw: the Butuanon translation
- audio_file: the associated audio clip path

Example:

```json
{"id": "0001", "en": "Where are you going?", "btw": "Hain kaw pasingud?", "audio_file": "clip_0001.wav"}
```

## Workspace Structure

- src/butuanon_nlp: Python package for the project
- data: sample data and future corpus files
- tests: validation tests for the loader and future pipeline logic

## Quick Start

1. Activate the virtual environment (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install or update Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Start the web prototype:

```powershell
python run_web_app.py
```

4. Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Tests

- **Run unit tests:**

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

All tests were updated to reflect adapter wiring; the suite should pass after installing model dependencies.

## Demo scripts

- **Whisper ASR test:** [scripts/whisper_test.py](scripts/whisper_test.py)
- **Programmatic web demo:** [scripts/demo_requests.py](scripts/demo_requests.py)

Run them from the project root with the virtualenv active:

```powershell
python scripts\whisper_test.py
python scripts\demo_requests.py
```

## Current Milestones

- Data ingestion and validation are working through the JSONL loader
- A glottal-aware tokenizer is available for preserving apostrophes and hyphens
- A preprocessing workflow now mirrors the PDF's stages: transcription, audio slicing, alignment, and native-speaker validation
- A simple training plan module now describes the next phases for fine-tuning and evaluation
- The CLI can export the seed corpus to JSONL or TSV and run a lightweight training simulation before real fine-tuning

## PDF-aligned workflow

The scaffold now reflects the main structure described in the PDF:

1. Build a seed corpus from paired English and Butuanon sentences
2. Preserve glottal stops in text and phonetic representation
3. Prepare the data with a validation and alignment workflow
4. Move into tokenizer adaptation and fine-tuning for translation

## Model & runtime notes

- **Hugging Face (NLLB):** the translation adapter uses HF models (example: `facebook/nllb-200-distilled-600M`). Large weights are downloaded on first use (several GB).
- **Whisper:** `openai-whisper` is used for ASR (CPU-friendly `tiny` in tests). Whisper expects float32 audio arrays when called programmatically.
- **TTS:** a lightweight `pyttsx3` fallback writes WAV files to the system temp directory when a real TTS backend is selected.
- **HF_TOKEN:** set a `HF_TOKEN` environment variable (Hugging Face access token) to avoid unauthenticated rate limits when downloading models.

## Files of interest

- **Code:** [src/butuanon_nlp/models.py](src/butuanon_nlp/models.py) — adapter wiring for translation/ASR/TTS
- **Web app:** [src/butuanon_nlp/web_app.py](src/butuanon_nlp/web_app.py)
- **Sample data:** [data/sample_data.jsonl](data/sample_data.jsonl)
- **Run script:** [run_web_app.py](run_web_app.py)

## Troubleshooting

- **Large downloads:** HF models are large; ensure you have disk space and a stable connection.
- **Missing binaries (ffmpeg):** Whisper file-based transcribe may call ffmpeg; the repo uses numpy/soundfile paths in scripts to avoid that.
- **If you see HF rate-limit warnings:** export your token:

```powershell
$env:HF_TOKEN = 'your_token_here'
```

If you want, I can also add a short section showing how to point the UI at hosted models or cache model checkpoints locally.
