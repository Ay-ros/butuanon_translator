# BisayaHub — Butuanon NLP Translation Platform

> **A low-resource NLP platform for Butuanon language preservation and translation, built with transfer learning from closely related languages.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green)]()
[![Status: Prototype](https://img.shields.io/badge/status-prototype-orange)]()

---

## Table of Contents

- [Overview](#overview)
- [Important Disclaimers](#important-disclaimers)
- [Project Vision](#project-vision)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Data Format](#data-format)
- [Data Collection Strategy](#data-collection-strategy)
- [CLI Reference](#cli-reference)
- [Testing](#testing)
- [Demo Scripts](#demo-scripts)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

BisayaHub is a working prototype for a **Butuanon (btw) NLP platform** that provides text-to-text translation, speech-to-text transcription (ASR), and text-to-speech synthesis (TTS) — all running locally on consumer-grade hardware. The platform uses **transfer learning** from closely related and higher-resourced languages as a practical starting point while the Butuanon corpus is being built.

---

## Important Disclaimers

> [!CAUTION]
> ### Butuanon Is Not Directly Supported by Any Model Used Here
>
> **Butuanon** (ISO 639-3: `btw`) is an endangered Bisayan language spoken in the Butuan City area and surrounding regions of Agusan del Norte, Philippines. It is **not available** as a language option in any of the pre-trained models this platform currently uses. All models are operating on **proxy languages** — primarily **Cebuano (`ceb`)** — as the closest available relative.

### Model-Specific Limitations

| Model | What It Actually Does | Why It's Not Butuanon |
|---|---|---|
| **NLLB-200** (`facebook/nllb-200-distilled-600M`) | Translates English ↔ **Cebuano** (`ceb_Latn`) | NLLB supports 200 languages, but Butuanon is not one of them. Cebuano is the closest Bisayan language in NLLB's language set. Translations will be in **standard Cebuano**, not Butuanon. |
| **Whisper** (`openai/whisper-tiny`) | Transcribes speech to text | Whisper **does not have a Butuanon language model** and does not list Butuanon among its supported languages. When transcribing Butuanon speech, Whisper will attempt to match it to whatever language it detects — typically Cebuano, Tagalog, or English — producing inaccurate transcriptions. |
| **MMS-TTS** (`facebook/mms-tts-ceb`) | Synthesises speech in **Cebuano** | The MMS-TTS model used here is the Cebuano voice model. It will speak with a Cebuano accent and phonology, which differs from Butuanon pronunciation patterns. |

> [!WARNING]
> ### Output Quality
>
> Because all models currently operate on **Cebuano** (a related but distinct language), the outputs should be treated as **rough approximations**, not accurate Butuanon translations. The platform is a **scaffolding tool** for future fine-tuning, not a production Butuanon translator.

### What This Means in Practice

- **Translation output is Cebuano**, not Butuanon. Cebuano and Butuanon share significant vocabulary and grammar but have important phonological and lexical differences.
- **Speech transcription** will not accurately recognise Butuanon-specific words or phonemes (such as the glottal stop patterns unique to Butuanon).
- **Voice synthesis** sounds Cebuano, not Butuanon. The intonation, stress patterns, and phoneme inventory differ.

### Why We're Doing It This Way

Building NLP tools for a low-resource language requires starting *somewhere*. Cebuano is the best available proxy because:

1. **Mutual intelligibility** — Butuanon and Cebuano share a large portion of their vocabulary and grammatical structures.
2. **Transfer learning potential** — Models fine-tuned on Cebuano can be adapted to Butuanon with relatively small amounts of Butuanon-specific data.
3. **Practical baseline** — Having a working pipeline (even with proxy models) lets us iterate on the data collection, UI, and tooling while the Butuanon corpus grows.

The long-term goal is to **fine-tune each model on real Butuanon data** so that translations, transcriptions, and speech synthesis become genuinely Butuanon.

---

## Project Vision

The platform uses transfer learning and careful data curation to build a practical baseline for Butuanon NLP even when a large digital corpus is not yet available. The approach follows four phases:

1. **Scaffold** — Build the complete pipeline with proxy models (Cebuano) so the infrastructure is ready.
2. **Collect** — Gather validated Butuanon–English parallel data through community contributions (see [Data Collection Strategy](#data-collection-strategy)).
3. **Fine-Tune** — Adapt each model (NLLB, Whisper, MMS-TTS) using the collected Butuanon data.
4. **Iterate** — Continuously improve the models as the corpus grows, with native speaker validation at every stage.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Flask Web UI (BisayaHub)               │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │
│  │ Translate   │  │ Transcribe │  │ Text-to-Speech     │  │
│  │ (Text→Text) │  │ (Audio→Text│  │ (Text→Audio)       │  │
│  └──────┬─────┘  └──────┬─────┘  └──────────┬─────────┘  │
│         │               │                    │            │
├─────────┼───────────────┼────────────────────┼────────────┤
│         ▼               ▼                    ▼            │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Model Registry                       │     │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐    │     │
│  │  │  NLLB    │ │ Whisper  │ │   MMS-TTS      │    │     │
│  │  │ (nllb)   │ │(whisper) │ │   (vits)       │    │     │
│  │  └──────────┘ └──────────┘ └────────────────┘    │     │
│  └──────────────────────────────────────────────────┘     │
│                                                           │
│  ┌──────────────────────────────────────────────────┐     │
│  │         Glottal-Aware Tokenizer Layer             │     │
│  │  • normalize_glottal() — canonical ʔ markers      │     │
│  │  • GlottalAwareTokenizer — display/analysis       │     │
│  │  • NLLBTokenizerWrapper — model inference          │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Model ID | Size | Task | Proxy Language |
|---|---|---|---|---|
| Translation | `facebook/nllb-200-distilled-600M` | ~600 MB | `eng_Latn → ceb_Latn` | Cebuano |
| ASR | `openai/whisper-tiny` | ~75 MB | Speech → Text | Auto-detect |
| TTS | `facebook/mms-tts-ceb` | ~50 MB | Text → Speech | Cebuano |

All models are **lazily loaded** on first use and cached for the lifetime of the process. Device selection is automatic — CUDA (GPU) when available, CPU otherwise.

---

## Features

### Core Capabilities

- **Text-to-Text Translation** — NLLB-200 for English to Cebuano (closest available proxy for Butuanon).
- **Speech-to-Text (ASR)** — Whisper for transcribing uploaded audio or direct microphone recordings in the browser.
- **Text-to-Speech (TTS)** — Facebook MMS-TTS for Cebuano-accented voice synthesis, with a `pyttsx3` offline fallback.
- **Glottal-Aware Tokenization** — A custom tokenization strategy that preserves glottal markers (apostrophes, hyphens) specific to Bisaya/Butuanon phonetics.
- **Phonetic Annotation** — Automatic IPA glottal-stop guides for translated text (e.g., `dal-a → dalʔa`).

### Platform Features

- **Local Inference** — All models run locally; no data leaves your machine.
- **Web Interface** — A modern, responsive Flask UI for interactive demos.
- **CLI Tools** — Command-line interface for batch processing, data inspection, and training simulation.
- **Model Registry** — Pluggable backend system that makes it easy to swap or add models.
- **Data Pipeline** — JSONL-based data loader with validation, export (JSONL/TSV), and preprocessing workflows.

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **~2 GB free disk space** (for model downloads on first run)
- **A stable internet connection** (for first-time model downloads from Hugging Face)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ay-ros/butuanon_translator.git
   cd butuanon_translator
   ```

2. **Create and activate a virtual environment:**

   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   ```bash
   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Start the web app:**

   ```bash
   python run_web_app.py
   ```

5. **Open in your browser:**

   ```
   http://127.0.0.1:5000
   ```

> [!NOTE]
> On the very first run, Hugging Face models will download automatically. This can take several minutes depending on your connection. The UI will show status messages while models load.

### Running with Hugging Face Spaces (Optional)

The app can also be deployed to Hugging Face Spaces. Use the alternate entry point which defaults to port 7860:

```bash
python app.py
```

---

## Configuration

The platform is configured via `config.json` at the project root:

```json
{
  "default_backend": "nllb",
  "target_lang": "ceb_Latn",
  "source_lang": "eng_Latn",
  "device": "auto"
}
```

| Key | Default | Description |
|---|---|---|
| `default_backend` | `"nllb"` | Default translation backend. |
| `target_lang` | `"ceb_Latn"` | NLLB target language code (Cebuano). |
| `source_lang` | `"eng_Latn"` | NLLB source language code (English). |
| `device` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"`. Auto selects GPU if available. |

### Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face API token to avoid download rate-limits. |
| `PORT` | Override the default port (used by `app.py` for HF Spaces). |

---

## Project Structure

```
butuanon_translator/
├── app.py                     # Entry point for Hugging Face Spaces (port 7860)
├── run_web_app.py             # Local development entry point (port 5000)
├── config.json                # Platform configuration
├── requirements.txt           # Python dependencies
├── CHANGELOG.md               # Version history
├── README.md                  # This file
│
├── src/
│   └── butuanon_nlp/          # Main Python package
│       ├── __init__.py        # Package exports
│       ├── models.py          # TranslationModel, SpeechModel, ModelRegistry
│       ├── tokenizer.py       # GlottalAwareTokenizer, NLLBTokenizerWrapper
│       ├── preprocessing.py   # Phonetic annotation, data preprocessing plans
│       ├── data_loader.py     # JSONL dataset loader with validation
│       ├── exporter.py        # Export samples to JSONL or TSV
│       ├── config.py          # ProjectConfig, load/save config
│       ├── cli.py             # Command-line interface
│       ├── training.py        # Training plan scaffolding and simulation
│       └── web_app.py         # Flask web application with embedded UI
│
├── data/
│   └── sample_data.jsonl      # Seed dataset (6 English–Butuanon pairs)
│
├── scripts/
│   ├── demo_requests.py       # Hit API endpoints via Flask test client
│   └── whisper_test.py        # Standalone Whisper transcription test
│
├── tests/                     # Unit and integration tests
│   ├── test_data_loader.py
│   ├── test_tokenizer.py
│   ├── test_models.py
│   ├── test_preprocessing_pipeline.py
│   ├── test_web_app.py
│   ├── test_hf_adapter.py
│   ├── test_model_registry.py
│   ├── test_config_file.py
│   ├── test_config_and_listing.py
│   ├── test_cli_registry.py
│   ├── test_export_and_training.py
│   └── test_training_pipeline.py
│
└── outputs/
    └── exported_samples.tsv   # Exported seed data
```

### Module Reference

#### `models.py` — Model Adapters

The core inference layer. Contains:

- **`TranslationModel`** — Wraps NLLB-200 for seq2seq translation. Supports lazy loading, configurable source/target languages, and an offline fallback mode.
- **`SpeechModel`** — Unified ASR + TTS model:
  - **ASR**: Primary backend is the `transformers` Whisper pipeline; falls back to `openai-whisper`.
  - **TTS**: Primary backend is MMS-TTS (`facebook/mms-tts-ceb`); falls back to `pyttsx3` for system voice synthesis.
- **`ModelRegistry`** — A registry for managing multiple model backends. Supports lookup by name or by task (`translation`, `asr`, `tts`).
- **`HuggingFaceAdapter`** — A lightweight dataclass for model metadata and runtime selection.

#### `tokenizer.py` — Glottal-Aware Tokenization

Two-layer tokenization system:

- **`GlottalAwareTokenizer`** — A regex-based tokenizer that preserves apostrophes and hyphens as word-internal characters (these mark the Butuanon glottal stop). Used for display, corpus analysis, and phonetic preprocessing.
- **`NLLBTokenizerWrapper`** — Wraps the Hugging Face `AutoTokenizer` and applies glottal-stop normalisation **before** SentencePiece encoding. This is what the translation model uses at inference time.
- **`normalize_glottal()`** — Canonicalises various glottal-stop markers (`'`, `'`, `'`, `ʼ`) to a single form (`'`).

#### `preprocessing.py` — Phonetic Annotation

- **`phoneticize_text()`** — Detects mid-word apostrophes and hyphens (glottal stop markers) and produces an IPA reading guide. Example: `"Palihug dal-a ako"` → `"Palihug dalʔa ako  ·  dal-a → dalʔa"`.
- **`build_preprocessing_plan()`** — Returns the four-stage data preprocessing workflow (digitisation, audio slicing, sentence alignment, native speaker validation).

#### `data_loader.py` — Dataset Loader

- **`load_samples()`** — Loads English–Butuanon aligned sentence pairs from a JSONL file.
- **`validate_sample()`** — Ensures each sample has the required fields: `id`, `en`, `btw`, `audio_file`.

#### `exporter.py` — Data Export

- **`export_samples()`** — Exports validated samples to JSONL or TSV format for downstream training or review.

#### `config.py` — Configuration

- **`ProjectConfig`** — Dataclass that resolves workspace paths and loads settings from `config.json`.
- **`load_config()` / `save_config()`** — JSON config file I/O.

#### `training.py` — Training Scaffolding

- **`build_training_plan()`** — Creates a phased training plan (data validation → tokenizer adaptation → fine-tuning → evaluation).
- **`build_task_training_workflow()`** — Creates structured workflow definitions for translation, ASR, or TTS training tasks.
- **`simulate_training_run()`** — Returns deterministic mock training metrics for prototyping.

#### `cli.py` — Command-Line Interface

Full CLI for data inspection, export, translation, transcription, synthesis, and training simulation. See [CLI Reference](#cli-reference).

#### `web_app.py` — Flask Web Application

The complete web application with an embedded HTML/CSS/JS UI. Provides:

- A modern, responsive single-page interface ("BisayaHub")
- REST API endpoints for all NLP tasks
- Direct microphone recording for ASR
- Automatic TTS playback after translation
- Phonetic glottal-stop annotations
- Token-level display with chips

---

## API Reference

All API endpoints accept and return JSON. The web app runs on `http://127.0.0.1:5000` by default.

### `GET /api/backends`

List available backends for each task.

**Response:**
```json
{
  "translation": ["nllb"],
  "asr": ["whisper"],
  "tts": ["vits"]
}
```

### `POST /api/translate`

Translate text using the specified backend.

**Request:**
```json
{
  "text": "Good morning",
  "backend": "nllb"
}
```

**Response:**
```json
{
  "task": "translation",
  "backend": "nllb",
  "translation": "Maayong buntag",
  "phonetic_guide": "",
  "tokens": ["Maayong", "buntag"]
}
```

### `POST /api/asr/upload`

Upload an audio file for Whisper transcription. Uses `multipart/form-data`.

**Request:** Form field `audio` with an audio file (`.wav`, `.webm`, `.mp3`, etc.)

**Response:**
```json
{
  "transcription": "Hain kaw pasingud?",
  "translation": "Where are you going?"
}
```

### `POST /api/tts`

Synthesise speech from text.

**Request:**
```json
{
  "text": "Maayong buntag",
  "backend": "vits"
}
```

**Response:**
```json
{
  "task": "tts",
  "backend": "vits",
  "audio_url": "/api/tts/audio/butuanon_tts_abc123.wav",
  "output": "Audio generated: butuanon_tts_abc123.wav"
}
```

### `GET /api/tts/audio/<filename>`

Serve a generated TTS audio file (WAV).

### `POST /api/asr`

Transcribe from a local file path (for programmatic use).

**Request:**
```json
{
  "audio_path": "/path/to/audio.wav",
  "backend": "whisper"
}
```

### `POST /api/train`

Run a training simulation.

**Request:**
```json
{
  "task": "translation",
  "max_samples": 6,
  "epochs": 3
}
```

### `POST /api/demo`

Run a combined demo (translation + ASR + TTS in one call).

---

## Data Format

### JSONL Dataset

The platform uses JSONL (JSON Lines) files for aligned English–Butuanon datasets. Each line is a JSON object with the following **required fields**:

```json
{
  "id": "0001",
  "en": "Where are you going?",
  "btw": "Hain kaw pasingud?",
  "audio_file": "clip_0001.wav"
}
```

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier for the sentence pair. |
| `en` | `string` | English text. |
| `btw` | `string` | Butuanon text. Must use consistent glottal-stop markers. |
| `audio_file` | `string` | Filename of the associated audio recording. |

### Glottal-Stop Convention

Butuanon uses glottal stops (ʔ) extensively. In written text, these are marked with:

- **Hyphens** between letters: `dal-a` (dalʔa)
- **Apostrophes** between letters: `Hain'ka` (Hainʔka)

The tokenizer normalises all variants (`'`, `'`, `'`, `ʼ`) to the canonical form `'` (U+2019 RIGHT SINGLE QUOTATION MARK) before processing.

---

## Data Collection Strategy

> [!IMPORTANT]
> **The single biggest bottleneck for Butuanon NLP is data.** Without a substantial corpus of validated Butuanon–English parallel text and audio, the models cannot be fine-tuned beyond their Cebuano proxy behaviour.

### Inspiration: Mozilla Common Voice

We recommend following the model pioneered by [**Mozilla Common Voice**](https://commonvoice.mozilla.org/) — a community-driven, open-source platform for collecting voice data in hundreds of languages. Key principles we adopt:

1. **Community-Driven Collection** — Native Butuanon speakers record, transcribe, and validate sentences through a simple web or mobile interface.
2. **Open Data** — All collected data should be openly licensed (e.g., CC-0 or CC-BY) so that researchers and developers can use it freely.
3. **Validation by Consensus** — Each contribution is reviewed by multiple native speakers to ensure accuracy.
4. **Incremental Growth** — Start small (dozens of sentences) and grow the corpus over time. Even 100 validated sentence pairs can meaningfully improve fine-tuned model quality.

### Proposed Data Collection Workflow

```
┌───────────────────────────────────────────────────────┐
│  Phase 1: Seed Sentences                               │
│  • Curate 100–500 common phrases in Butuanon          │
│  • Focus on daily conversation, greetings, directions  │
│  • Have native speakers validate and correct           │
├───────────────────────────────────────────────────────┤
│  Phase 2: Audio Recording                              │
│  • Native speakers record each sentence (3–10 sec)    │
│  • Multiple speakers per sentence for diversity        │
│  • Use consistent recording settings (16kHz mono WAV) │
├───────────────────────────────────────────────────────┤
│  Phase 3: Alignment & Validation                       │
│  • Pair audio clips with their transcripts             │
│  • Cross-validate with 2+ speakers per pair            │
│  • Export to JSONL format for the pipeline              │
├───────────────────────────────────────────────────────┤
│  Phase 4: Community Platform (Future)                  │
│  • Build a Common Voice–style web app for Butuanon    │
│  • Allow remote community members to contribute       │
│  • Track contribution stats, quality metrics           │
└───────────────────────────────────────────────────────┘
```

### What You Can Do to Help

If you are a **native Butuanon speaker** or know someone who is:

- **Record sentences** — Even a few recorded sentences help. Use the built-in microphone recorder in the BisayaHub web app.
- **Validate translations** — Review existing English–Butuanon pairs in `data/sample_data.jsonl` and correct any errors.
- **Add new sentences** — Contribute everyday phrases, idioms, and conversational sentences.
- **Spread the word** — Share this project with Butuanon-speaking communities in Butuan, Agusan del Norte, and the diaspora.

### Data Quality Guidelines

| Criterion | Requirement |
|---|---|
| **Audio format** | 16kHz mono WAV, 3–10 seconds per clip |
| **Text encoding** | UTF-8, consistent glottal-stop markers |
| **Speaker diversity** | Aim for multiple speakers, ages, and genders |
| **Validation** | At least 2 native speakers approve each pair |
| **Licensing** | All contributions must be CC-0 or CC-BY-4.0 |

---

## CLI Reference

The CLI provides batch access to all platform features:

```bash
python -m butuanon_nlp.cli [OPTIONS]
```

Or run from the project root:

```bash
python -c "from butuanon_nlp.cli import main; main()" [OPTIONS]
```

### Options

| Flag | Description |
|---|---|
| `--data PATH` | Path to the JSONL dataset (default: `data/sample_data.jsonl`). |
| `--limit N` | Number of samples to preview (default: 5). |
| `--export PATH` | Export samples to a file. |
| `--format {jsonl,tsv}` | Export format (default: `jsonl`). |
| `--simulate-train` | Run a lightweight training simulation. |
| `--task TASK` | Training task: `all`, `translation`, `asr`, `tts` (default: `all`). |
| `--epochs N` | Epoch count for training simulation (default: 3). |
| `--translate TEXT` | Translate text using the selected backend. |
| `--transcribe PATH` | Transcribe an audio file. |
| `--synthesize TEXT` | Synthesise speech from text. |
| `--backend NAME` | Select backend: `nllb`, `whisper`, or `vits`. |
| `--list-backends` | List all registered backends. |

### Examples

```bash
# Preview the first 3 samples from the dataset
python -m butuanon_nlp.cli --limit 3

# Translate English to Cebuano
python -m butuanon_nlp.cli --translate "Good morning"

# Export dataset as TSV
python -m butuanon_nlp.cli --export outputs/data.tsv --format tsv

# Run a training simulation
python -m butuanon_nlp.cli --simulate-train --epochs 5

# List available backends
python -m butuanon_nlp.cli --list-backends
```

---

## Testing

The project has 12 test files covering the data loader, tokenizer, models, preprocessing, web app, config, CLI, training pipeline, and Hugging Face adapter.

### Run All Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Run a Specific Test

```bash
python -m unittest tests.test_tokenizer
python -m unittest tests.test_web_app
```

### What the Tests Cover

| Test File | Scope |
|---|---|
| `test_data_loader.py` | JSONL loading, validation, missing/malformed field handling |
| `test_tokenizer.py` | Glottal-aware tokenization, normalisation, encode/decode |
| `test_models.py` | TranslationModel and SpeechModel instantiation and fallbacks |
| `test_preprocessing_pipeline.py` | Phonetic annotation, glottal-stop detection |
| `test_web_app.py` | Flask routes, API endpoints, upload handling |
| `test_hf_adapter.py` | HuggingFaceAdapter metadata, describe() |
| `test_model_registry.py` | ModelRegistry register/get/list/get_for_task |
| `test_config_file.py` | Config load/save, ProjectConfig.from_workspace |
| `test_config_and_listing.py` | Config integration with model listing |
| `test_cli_registry.py` | CLI parser construction and backend listing |
| `test_export_and_training.py` | JSONL/TSV export, training plan generation |
| `test_training_pipeline.py` | Training workflow, simulation, summarisation |

---

## Demo Scripts

Two demo scripts are included in `scripts/` for quick validation:

### `scripts/demo_requests.py`

Hits the `/api/demo`, `/api/translate`, and `/api/tts` endpoints via Flask's test client (no server needed):

```bash
python scripts/demo_requests.py
```

### `scripts/whisper_test.py`

Standalone Whisper transcription test — loads a WAV file and runs `openai-whisper` directly:

```bash
python scripts/whisper_test.py
```

> [!NOTE]
> `whisper_test.py` contains a hard-coded file path. Update the `path` variable to point to an actual WAV file on your system.

---

## Troubleshooting

### Large Model Downloads

Hugging Face models are large and download on first use. The NLLB model alone is ~600 MB. Ensure you have:

- At least **2 GB of free disk space**
- A **stable internet connection**

The UI will indicate when models are being downloaded.

### Hugging Face Rate-Limit Warnings

If you see rate-limit warnings, set your Hugging Face token:

```powershell
# PowerShell
$env:HF_TOKEN = "your_token_here"
```

```bash
# Bash
export HF_TOKEN="your_token_here"
```

### Microphone Recording (ffmpeg)

The platform uses `imageio-ffmpeg` to decode `.webm` audio recorded from the browser. This bundles its own `ffmpeg` binary — no system-wide `ffmpeg` installation is needed on Windows.

If you encounter audio decoding issues:

```bash
pip install imageio-ffmpeg --upgrade
```

### CUDA / GPU Issues

If you have an NVIDIA GPU but models are running on CPU:

1. Ensure you have the correct version of PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. Verify CUDA is detected:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

### pyttsx3 Fallback on Linux

The `pyttsx3` TTS fallback requires `espeak` on Linux:

```bash
sudo apt install espeak
```

---

## Roadmap

- [x] Flask web UI with translation, ASR, and TTS
- [x] Glottal-aware tokenization for Butuanon phonetics
- [x] Phonetic annotation (IPA glottal-stop guides)
- [x] Microphone recording in the browser
- [x] Model registry with pluggable backends
- [x] CLI for batch processing
- [x] Training simulation and scaffolding
- [ ] Fine-tune NLLB on real Butuanon–English parallel data
- [ ] Fine-tune Whisper on Butuanon audio–transcript pairs
- [ ] Train a Butuanon-specific MMS-TTS voice
- [ ] Build a Common Voice–style community data collection platform
- [ ] Bidirectional translation (Butuanon → English)
- [ ] Mobile-friendly progressive web app (PWA)
- [ ] Offline mode with cached models
- [ ] BLEU / ChrF evaluation against native speaker translations

---

## Contributing

Contributions are welcome! There are three main ways to contribute:

### 1. Language Data (Most Needed)

If you speak Butuanon, you can help by:

- Adding sentence pairs to `data/sample_data.jsonl`
- Recording audio for existing sentences
- Validating and correcting existing data

### 2. Code

- Fork the repo, make your changes, and submit a pull request.
- Run the test suite before submitting: `python -m unittest discover -s tests -p "test_*.py"`

### 3. Documentation & Outreach

- Improve this README or add inline documentation.
- Help connect the project with Butuanon-speaking communities.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | PyTorch — tensor computation and model inference |
| `transformers` | Hugging Face Transformers — NLLB, Whisper, MMS-TTS |
| `datasets` | Hugging Face Datasets — data loading utilities |
| `tokenizers` | Fast tokenizer implementations |
| `sentencepiece` | SentencePiece — subword tokenization for NLLB |
| `librosa` | Audio analysis and feature extraction |
| `soundfile` | Audio file I/O |
| `scipy` | Scientific computing — WAV file writing |
| `ffmpeg-python` | FFmpeg bindings for audio processing |
| `flask` | Web framework for the UI and API |
| `openai-whisper` | OpenAI Whisper — fallback ASR engine |
| `pyttsx3` | Offline TTS fallback (system voice engine) |
| `imageio-ffmpeg` | Bundled ffmpeg binary for `.webm` decoding on Windows |

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <em>Built with ❤️ for the Butuanon-speaking community of Butuan City and Agusan del Norte.</em>
</p>
