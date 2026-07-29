# Butuanon NLP Translation Platform

This workspace contains a working prototype scaffold for a low-resource Butuanon NLP platform. It provides:

- **Text-to-Text Translation**: NLLB-200 (`facebook/nllb-200-distilled-600M`) for English to Cebuano (closest available to Butuanon).
- **Speech-to-Text (ASR)**: Whisper (`openai/whisper-tiny`) for transcribing audio (supports direct microphone recording in the browser).
- **Text-to-Speech (TTS)**: Facebook MMS-TTS (`facebook/mms-tts-ceb`) for Cebuano-accented voice synthesis.
- A beautiful, modern Flask web UI for quick demos.

## Project Vision

The platform uses transfer learning and careful data curation to build a practical baseline for Butuanon NLP even when a large digital corpus is not yet available. The first milestone is to create a small, validated seed dataset and a repeatable workflow for future expansion.

## Core Features

- **Local Inference**: All models run locally on consumer-grade hardware (CPU or GPU).
- **Glottal-Aware Tokenization**: A custom tokenization strategy that preserves glottal markers and apostrophes specific to Bisaya/Butuanon phonetics.
- **Web Interface**:
  - Direct microphone recording for Whisper transcription.
  - Automatic voice synthesis when translating text.
  - A clean, modern, responsive UI.

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

## Troubleshooting

- **Large downloads:** Hugging Face models are large and download on first use. Ensure you have disk space and a stable connection. The UI will indicate when models are being downloaded.
- **Microphone Recording (ffmpeg):** The platform uses `imageio-ffmpeg` to automatically decode `.webm` audio recorded from the browser without needing a system-wide `ffmpeg` installation on Windows. 
- **If you see HF rate-limit warnings:** export your token:

```powershell
$env:HF_TOKEN = 'your_token_here'
```

## Workspace Structure

- `src/butuanon_nlp`: Python package containing the model adapters (`models.py`), tokenization logic (`tokenizer.py`, `preprocessing.py`), and the Flask UI (`web_app.py`).
- `data`: Sample data and future corpus files.
- `tests`: Validation tests for the loader and NLP pipeline logic.

## Tests

- **Run unit tests:**

```powershell
python -m unittest discover -s tests -p 'test_*.py'
```

All tests are configured to validate the new adapter wiring, regex rules for glottal markers, and web UI integration.
