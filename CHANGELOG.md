# Changelog

All notable changes to this project are documented in this file.

## 2026-07-29 — Complete documentation overhaul

- Rewrote `README.md` with comprehensive documentation covering every module, API endpoint, and CLI option.
- Added critical disclaimers: Butuanon is not supported by NLLB, Whisper, or MMS-TTS — all models use Cebuano as a proxy language.
- Documented the Cebuano proxy strategy and why transfer learning from a related language is the practical starting point.
- Added a Data Collection Strategy section inspired by Mozilla Common Voice, with a proposed four-phase community-driven workflow.
- Added full API reference for all REST endpoints (`/api/translate`, `/api/asr/upload`, `/api/tts`, etc.).
- Added module-level documentation for every file in `src/butuanon_nlp/`.
- Added architecture diagram, dependency table, troubleshooting guide, and roadmap.
- Added contribution guidelines focused on language data collection as the highest-priority need.

## 2026-07-29 — Adapter wiring, demos, README updates

- Wire Hugging Face translation adapters with a seq2seq/tokenizer fallback in `src/butuanon_nlp/models.py`.
- Add Whisper ASR integration and a lightweight `pyttsx3` TTS fallback (writes WAV files to temp).
- Update `src/butuanon_nlp/web_app.py` to exercise real backends in the `/api/*` endpoints.
- Add demo scripts: `scripts/whisper_test.py` and `scripts/demo_requests.py` to validate ASR and web endpoints.
- Update `README.md` with Quick Start, tests, demo commands, model notes, and troubleshooting tips.
- Ensure unit tests pass after wiring; updated behavior to preserve test expectations.

Notes:
- Large Hugging Face models are downloaded on first use (expect multiple-GB downloads).
- Set `HF_TOKEN` to avoid unauthenticated rate-limit warnings when downloading from the Hub.
