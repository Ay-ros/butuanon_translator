from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_file

from .models import SpeechModel, TranslationModel, create_default_model_registry
from .preprocessing import phoneticize_text
from .tokenizer import GlottalAwareTokenizer
from .training import build_task_training_workflow, summarize_training_workflow

registry = create_default_model_registry()
tokenizer = GlottalAwareTokenizer()

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "butuanon_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)


HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BisayaHub - Butuanon NLP Platform</title>
    <meta name="description" content="Translate, transcribe, and synthesize Butuanon speech using NLLB-200, Whisper, and MMS-TTS, all running locally." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet" />
    <style>
        :root {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            --page: #f1f3f5;
            --panel: #ffffff;
            --soft: #f6f7f8;
            --line: #dfe3e7;
            --text: #141518;
            --muted: #737981;
            --quiet: #9da2a8;
            --black: #050505;
            --green: #0f8f4f;
            --green-soft: #e8fff0;
            --green-tint: #baffc8;
            --danger: #c62a36;
            --danger-soft: #ffd8d6;
            --shadow: 0 16px 38px rgba(17, 24, 39, 0.11);
            --shadow-soft: 0 5px 15px rgba(17, 24, 39, 0.08);
            color: var(--text);
        }

        * { box-sizing: border-box; margin: 0; }

        html, body {
            min-height: 100%;
            background:
                linear-gradient(180deg, #d5d9dd 0, #eef1f3 132px, var(--page) 340px),
                var(--page);
        }

        body { color: var(--text); }
        button, select, textarea, input { font: inherit; }
        button { border: 0; }
        a { color: inherit; }

        .app-frame {
            width: 100%;
            min-height: 100vh;
            margin: 0;
            background: rgba(244, 246, 247, 0.86);
            border: 0;
            box-shadow: none;
            overflow: hidden;
        }

        .topbar {
            height: 68px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            padding: 0 32px;
            background: rgba(246, 247, 248, 0.9);
            border-bottom: 1px solid rgba(202, 207, 212, 0.68);
            backdrop-filter: blur(10px);
        }

        .brand {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
            text-decoration: none;
        }

        .brand__mark {
            width: 30px;
            height: 30px;
            border-radius: 7px;
            background: var(--black);
            color: #ffffff;
            display: grid;
            place-items: center;
            box-shadow: 0 6px 14px rgba(0, 0, 0, 0.18);
        }

        .brand__mark svg {
            width: 18px;
            height: 18px;
            stroke-width: 2.2;
        }

        .brand__title {
            font-size: 1.23rem;
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1;
            white-space: nowrap;
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 34px;
            font-size: 0.82rem;
            font-weight: 800;
        }

        .nav-links a {
            color: #80858b;
            text-decoration: none;
        }

        .nav-links a.active { color: #171717; }

        .hero {
            min-height: 104px;
            display: grid;
            place-items: center;
            text-align: center;
            padding: 0 28px 18px;
        }

        .hero h1 {
            max-width: 900px;
            font-size: clamp(2.2rem, 4.2vw, 3.6rem);
            line-height: 0.98;
            font-weight: 900;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 600px;
            margin: 14px auto 0;
            color: #696e75;
            font-weight: 600;
            line-height: 1.48;
            font-size: 0.98rem;
        }

        .workspace {
            display: grid;
            grid-template-columns: minmax(330px, 0.43fr) minmax(420px, 0.57fr);
            gap: 14px;
            padding: 0 30px 28px;
        }

        .controls {
            display: grid;
            gap: 14px;
            align-content: start;
        }

        .card,
        .output-panel {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #e7eaee;
            border-radius: 8px;
            box-shadow: var(--shadow-soft);
        }

        .card {
            display: grid;
            gap: 17px;
            padding: 22px;
        }

        .card__head,
        .output-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
        }

        .lbl {
            color: #777e85;
            font-size: 0.68rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            line-height: 1;
            text-transform: uppercase;
        }

        .mode-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 31px;
            padding: 0 12px;
            border-radius: 999px;
            background: #eef0f2;
            color: #4f555b;
            font-size: 0.75rem;
            font-weight: 800;
        }

        .mode-pill select {
            max-width: 90px;
            appearance: none;
            border: 0;
            outline: 0;
            background: transparent;
            color: inherit;
            font-weight: inherit;
            text-transform: uppercase;
            cursor: pointer;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            min-height: 20px;
            padding: 4px 8px;
            border-radius: 4px;
            background: #eef0f2;
            color: #777e85;
            font-size: 0.58rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .tag--green {
            background: var(--green-tint);
            color: #118348;
        }

        .tag--accent {
            background: #f3f4f5;
            color: #60666d;
            font-family: "Cascadia Code", "Consolas", monospace;
        }

        .txa,
        .inp {
            width: 100%;
            border: 1px solid transparent;
            border-radius: 7px;
            background: #f4f6f7;
            color: var(--text);
            outline: 0;
            transition: border-color 0.18s, box-shadow 0.18s, background 0.18s;
        }

        .txa {
            min-height: 132px;
            padding: 18px;
            resize: vertical;
            line-height: 1.6;
            font-weight: 600;
        }

        .inp {
            min-height: 44px;
            padding: 0 13px;
            font-weight: 650;
        }

        .txa::placeholder,
        .inp::placeholder { color: #a0a5ab; }

        .txa:focus,
        .inp:focus {
            background: #ffffff;
            border-color: #c8d0d7;
            box-shadow: 0 0 0 3px rgba(16, 143, 79, 0.08);
        }

        .btns {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn,
        .icon-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s, background 0.15s;
        }

        .btn {
            min-height: 42px;
            padding: 0 25px;
            border-radius: 7px;
            background: var(--black);
            color: #ffffff;
            font-size: 0.75rem;
            font-weight: 900;
            box-shadow: 0 7px 15px rgba(0, 0, 0, 0.18);
        }

        .btn:hover:not(:disabled),
        .icon-btn:hover:not(:disabled) {
            transform: translateY(-1px);
        }

        .btn:disabled,
        .icon-btn:disabled {
            cursor: not-allowed;
            opacity: 0.58;
        }

        .btn--soft {
            width: 100%;
            min-height: 54px;
            background: var(--danger-soft);
            color: var(--danger);
            box-shadow: none;
            gap: 10px;
            font-size: 0.88rem;
        }

        .btn--outline {
            background: #eef0f2;
            color: #4f555b;
            box-shadow: none;
        }

        .icon-btn {
            width: 39px;
            height: 39px;
            border-radius: 50%;
            background: #edf0f2;
            color: #6f767d;
        }

        .icon-btn svg,
        .btn svg {
            width: 17px;
            height: 17px;
            stroke-width: 2.4;
        }

        .icon-btn--green {
            background: var(--green);
            color: #ffffff;
            box-shadow: 0 7px 15px rgba(15, 143, 79, 0.22);
        }

        .spin {
            width: 14px;
            height: 14px;
            border: 2.5px solid rgba(255, 255, 255, 0.35);
            border-top-color: #ffffff;
            border-radius: 50%;
            animation: spin 0.65s linear infinite;
        }

        .spin--dark {
            border-color: rgba(20, 21, 24, 0.18);
            border-top-color: var(--text);
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .upload {
            min-height: 130px;
            position: relative;
            display: grid;
            place-items: center;
            padding: 22px;
            border-radius: 8px;
            background: #e9ecef;
            color: #555b62;
            text-align: center;
            cursor: pointer;
            transition: background 0.18s, box-shadow 0.18s;
        }

        .upload:hover {
            background: #e3e7ea;
            box-shadow: inset 0 0 0 1px #d6dce1;
        }

        .upload.ok {
            background: var(--green-soft);
            color: var(--green);
        }

        .upload input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        .upload__content {
            display: grid;
            gap: 14px;
            justify-items: center;
            pointer-events: none;
        }

        .upload__icon {
            width: 46px;
            height: 46px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: #ffffff;
            color: #1f2327;
        }

        .upload__icon svg {
            width: 21px;
            height: 21px;
            stroke-width: 2.2;
        }

        .upload__title {
            display: block;
            font-size: 0.78rem;
            font-weight: 850;
        }

        .upload__meta {
            display: block;
            margin-top: 4px;
            color: #747b82;
            font-size: 0.72rem;
            font-weight: 650;
        }

        .divider {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 12px;
            color: #8c9298;
            font-size: 0.62rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .divider::before,
        .divider::after {
            content: "";
            height: 1px;
            background: #e3e6e9;
        }

        .output-panel {
            min-height: 548px;
            display: grid;
            grid-template-rows: auto 1fr;
            padding: 28px;
            background:
                radial-gradient(circle at 0 0, rgba(186, 255, 200, 0.42), transparent 40%),
                #ffffff;
            box-shadow: var(--shadow);
        }

        .output-actions {
            display: inline-flex;
            align-items: center;
            gap: 12px;
        }

        .output-body {
            display: grid;
            align-content: center;
            gap: 18px;
            min-height: 0;
            padding: 26px 0 2px;
        }

        .translation-result {
            min-height: 108px;
            color: #999da2;
            font-size: clamp(1.7rem, 2.5vw, 2.35rem);
            font-weight: 900;
            line-height: 1.2;
            white-space: pre-wrap;
        }

        .translation-result.has-text {
            color: #1f2327;
            font-size: clamp(1.45rem, 2vw, 2rem);
        }

        .out {
            color: #5f666d;
            font-size: 0.88rem;
            line-height: 1.62;
            white-space: pre-wrap;
        }

        .micro-panel {
            display: grid;
            gap: 11px;
            padding-top: 6px;
        }

        .chips {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .chip {
            padding: 5px 9px;
            border-radius: 6px;
            background: #eef8f1;
            color: #227748;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.76rem;
            font-weight: 800;
        }

        .phon {
            padding: 12px 14px;
            border-radius: 7px;
            background: #f6f7f8;
            color: #42484e;
            font-family: "Cascadia Code", "Consolas", monospace;
            font-size: 0.82rem;
            line-height: 1.58;
        }

        .speech-results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-top: 10px;
        }

        .result-box {
            min-height: 94px;
            padding: 14px;
            border-radius: 7px;
            background: rgba(246, 247, 248, 0.86);
            border: 1px solid #eceff2;
        }

        .tts-row {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
            align-items: center;
            margin-top: 10px;
        }

        audio {
            width: 100%;
            margin-top: 8px;
            border-radius: 7px;
        }

        .info-bar {
            display: grid;
            grid-template-columns: 1.25fr 0.8fr 0.7fr;
            gap: 44px;
            padding: 24px 30px 30px;
            border-top: 1px solid #dfe3e7;
            background: rgba(237, 239, 241, 0.7);
        }

        .info-block {
            display: grid;
            gap: 10px;
            align-content: start;
        }

        .info-block p,
        .info-block li,
        .info-block a {
            color: #59616a;
            font-size: 0.8rem;
            font-weight: 650;
            line-height: 1.55;
            text-decoration: none;
        }

        .arch-list {
            display: grid;
            gap: 5px;
            list-style: none;
            padding: 0;
        }

        .arch-list strong,
        .hist strong {
            color: #151719;
            font-weight: 900;
        }

        .hist {
            display: grid;
            gap: 2px;
            padding: 0;
            background: transparent;
        }

        .hist span {
            color: #59616a;
            font-size: 0.78rem;
            font-weight: 650;
        }

        @media (max-width: 920px) {
            .app-frame {
                box-shadow: none;
                border: 0;
            }

            .workspace,
            .info-bar {
                grid-template-columns: 1fr;
            }

            .output-panel {
                min-height: 520px;
            }

            .output-body {
                align-content: start;
            }
        }

        @media (max-width: 640px) {
            .topbar {
                height: auto;
                align-items: flex-start;
                padding: 18px;
            }

            .nav-links {
                gap: 18px;
                padding-top: 6px;
            }

            .hero {
                padding: 18px 18px 20px;
            }

            .hero h1 {
                font-size: 2.15rem;
                line-height: 1.05;
            }

            .workspace,
            .info-bar {
                padding-left: 14px;
                padding-right: 14px;
            }

            .card,
            .output-panel {
                padding: 18px;
            }

            .speech-results {
                grid-template-columns: 1fr;
            }

            .btns {
                justify-content: stretch;
            }

            .btn:not(.btn--soft) {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <main class="app-frame">
        <header class="topbar">
            <a class="brand" href="/" aria-label="BisayaHub home">
                <span class="brand__mark" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M4 5h10" />
                        <path d="M9 3v2" />
                        <path d="M6.5 5c.6 3.8 3.2 6.3 7 7.2" />
                        <path d="M12.5 5c-.5 2-1.7 3.9-3.5 5.6" />
                        <path d="M13 19l3.3-8h1.4l3.3 8" />
                        <path d="M14.3 16h5.4" />
                    </svg>
                </span>
                <span class="brand__title">BisayaHub</span>
            </a>
            <nav class="nav-links" aria-label="Primary">
                <a href="#translate-section" class="active">Translate</a>
                <a href="#speech-section">Speech</a>
            </nav>
        </header>

        <section class="hero" aria-labelledby="hero-title">
            <div>
                <h1 id="hero-title">Translate and speak Butuanon with real AI.</h1>
                <p>A professional linguistic laboratory seamlessly converting English to Butuanon across text and speech modes.</p>
            </div>
        </section>

        <section class="workspace">
            <div class="controls" id="translate-section">
                <section class="card" aria-labelledby="text-translation-title">
                    <div class="card__head">
                        <p class="lbl" id="text-translation-title">Text Translation</p>
                        <div class="mode-pill">
                            <select id="backend-select" aria-label="Translation backend"></select>
                            <span aria-hidden="true">-></span>
                            <span>Butuanon</span>
                        </div>
                    </div>
                    <textarea id="source-text" class="txa" placeholder="Type text to translate..."></textarea>
                    <div class="btns">
                        <button id="translate-btn" class="btn">Translate Text</button>
                    </div>
                </section>

                <section class="card" id="speech-section" aria-labelledby="speech-title">
                    <div class="card__head">
                        <p class="lbl" id="speech-title">Transcribe Audio</p>
                        <span class="tag">Whisper AI</span>
                    </div>
                    <label class="upload" id="upload-area">
                        <span class="upload__content">
                            <span class="upload__icon" aria-hidden="true">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                    <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z" />
                                    <path d="M14 2v5h5" />
                                    <path d="M12 18v-6" />
                                    <path d="M9 15l3 3 3-3" />
                                </svg>
                            </span>
                            <span>
                                <span class="upload__title" id="upload-label">Drag &amp; drop audio here</span>
                                <span class="upload__meta">MP3, WAV, or M4A (Max 50MB)</span>
                            </span>
                        </span>
                        <input type="file" id="audio-file" accept="audio/*" />
                    </label>
                    <div class="divider">Or Record</div>
                    <button id="record-btn" class="btn btn--soft" type="button">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                            <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z" />
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                            <path d="M12 19v3" />
                        </svg>
                        Record Audio
                    </button>
                    <div class="btns">
                        <button id="transcribe-btn" class="btn btn--outline" type="button">Transcribe Audio</button>
                    </div>
                </section>
            </div>

            <section class="output-panel" aria-labelledby="output-title">
                <div class="output-head">
                    <p class="lbl" id="output-title">Translation Output</p>
                    <div class="output-actions">
                        <button class="icon-btn" id="copy-output-btn" type="button" title="Copy translation" aria-label="Copy translation">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                                <rect width="14" height="14" x="8" y="8" rx="2" />
                                <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                            </svg>
                        </button>
                        <button class="icon-btn icon-btn--green" id="synthesize-btn" type="button" title="Speak text" aria-label="Speak text">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                                <path d="M11 5 6 9H2v6h4l5 4z" />
                                <path d="M15.5 8.5a5 5 0 0 1 0 7" />
                                <path d="M18.5 5.5a9 9 0 0 1 0 13" />
                            </svg>
                        </button>
                    </div>
                </div>

                <div class="output-body">
                    <div class="translation-result" id="translation-output">Nagahulat sa imong input...</div>
                    <div><span class="tag tag--green">NLLB-200 Model</span></div>

                    <div class="micro-panel" id="token-display"></div>

                    <section class="micro-panel" id="phonetic-card" style="display:none;" aria-labelledby="phonetic-title">
                        <div class="card__head">
                            <p class="lbl" id="phonetic-title">Phonetic Guide</p>
                            <span class="tag tag--accent">glottal stop</span>
                        </div>
                        <div class="phon" id="phonetic-output"></div>
                        <p class="out">Butuanon marks glottal stops with apostrophes or hyphens between letters. This guide shows where they occur.</p>
                    </section>

                    <div class="speech-results">
                        <section class="result-box" aria-labelledby="asr-title">
                            <p class="lbl" id="asr-title">Speech-to-Text</p>
                            <div class="out" id="asr-output">Transcription result will appear here.</div>
                        </section>
                        <section class="result-box" aria-labelledby="tts-title">
                            <p class="lbl" id="tts-title">Text-to-Speech</p>
                            <p class="out">Generate an audio clip. The model currently speaks in Cebuano, the closest available voice.</p>
                            <div class="tts-row">
                                <input id="tts-text" class="inp" placeholder="Type text to speak aloud..." value="Maayong buntag" />
                            </div>
                            <div class="out" id="tts-output"></div>
                            <div id="audio-player"></div>
                        </section>
                    </div>
                </div>
            </section>
        </section>

        <footer class="info-bar">
            <section class="info-block">
                <p class="lbl">What Powers This</p>
                <p>BisayaHub is a professional linguistic laboratory leveraging state-of-the-art AI models for the Cebuano (Bisaya) language.</p>
                <div id="history-list"></div>
            </section>

            <section class="info-block">
                <p class="lbl">Core Models</p>
                <ul class="arch-list">
                    <li><strong>NLLB-200:</strong> High-quality translation</li>
                    <li><strong>Whisper:</strong> Robust speech recognition</li>
                    <li><strong>MMS-TTS:</strong> Voice synthesis</li>
                </ul>
                <p>Models load on first use and are cached. GPU acceleration is automatic when CUDA is available.</p>
            </section>

            <section class="info-block">
                <p class="lbl">Resources</p>
                <a href="/api/backends">Model backends</a>
                <a href="#speech-section">Speech tools</a>
                <a href="#translate-section">Text tools</a>
            </section>
        </footer>
    </main>

    <script>
        const $ = id => document.getElementById(id);
        const backendSel  = $('backend-select');
        const srcText     = $('source-text');
        const translateBtn = $('translate-btn');
        const transOut    = $('translation-output');
        const tokenDisp   = $('token-display');
        const phonCard    = $('phonetic-card');
        const phonOut     = $('phonetic-output');
        const audioFile   = $('audio-file');
        const uploadArea  = $('upload-area');
        const uploadLabel = $('upload-label');
        const transcribeBtn = $('transcribe-btn');
        const asrOut      = $('asr-output');
        const ttsText     = $('tts-text');
        const synthBtn    = $('synthesize-btn');
        const ttsOut      = $('tts-output');
        const audioPlayer = $('audio-player');
        const histList    = $('history-list');
        const recordBtn   = $('record-btn');
        const copyOutputBtn = $('copy-output-btn');
        const synthIcon = synthBtn.innerHTML;
        let mediaRecorder = null;
        let audioChunks = [];

        function spin(btn, on, label) {
            btn.disabled = on;
            if (on) {
                btn.dataset.orig = btn.textContent.trim() || btn.getAttribute('aria-label') || '';
                const dark = btn.classList.contains('btn--outline') || btn.classList.contains('icon-btn');
                btn.innerHTML = `<span class="spin ${dark ? 'spin--dark' : ''}"></span>${btn.classList.contains('icon-btn') ? '' : ' Working...'}`;
            } else if (label || btn.dataset.orig) {
                btn.textContent = label || btn.dataset.orig;
            }
        }

        function addHist(title, detail) {
            const d = document.createElement('div');
            const strong = document.createElement('strong');
            const span = document.createElement('span');
            d.className = 'hist';
            strong.textContent = title;
            span.textContent = detail;
            d.append(strong, span);
            histList.prepend(d);
            while (histList.children.length > 5) histList.removeChild(histList.lastChild);
        }

        async function post(url, data) {
            const r = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            return r.json();
        }

        fetch('/api/backends').then(r => r.json()).then(d => {
            backendSel.innerHTML = d.translation
                .map(n => `<option value="${n}">${n.toUpperCase()}</option>`)
                .join('');
        }).catch(console.error);

        audioFile.addEventListener('change', () => {
            if (audioFile.files.length) {
                uploadLabel.textContent = audioFile.files[0].name;
                uploadArea.classList.add('ok');
            } else {
                uploadLabel.textContent = 'Drag & drop audio here';
                uploadArea.classList.remove('ok');
            }
        });

        translateBtn.addEventListener('click', async () => {
            const text = srcText.value.trim();
            if (!text) return;
            spin(translateBtn, true);
            transOut.textContent = 'Translating... (first run downloads the model)';
            transOut.classList.remove('has-text');
            phonCard.style.display = 'none';
            tokenDisp.innerHTML = '';

            try {
                const r = await post('/api/translate', { text, backend: backendSel.value });
                transOut.textContent = r.translation || r.error || '';
                transOut.classList.toggle('has-text', Boolean(r.translation));

                if (r.phonetic_guide) {
                    phonOut.textContent = r.phonetic_guide;
                    phonCard.style.display = '';
                }

                if (r.tokens && r.tokens.length) {
                    tokenDisp.innerHTML =
                        '<p class="lbl">Tokens</p>' +
                        '<div class="chips">' +
                        r.tokens.map(t => `<span class="chip">${t}</span>`).join('') +
                        '</div>';
                }
                addHist(r.translation?.slice(0, 35) || 'Translation', `via ${backendSel.value.toUpperCase()}`);

                if (r.translation) {
                    ttsText.value = r.translation;
                    synthBtn.click();
                }
            } catch(e) {
                transOut.textContent = 'Error: ' + e.message;
            }
            spin(translateBtn, false, 'Translate Text');
        });

        recordBtn.addEventListener('click', async () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                recordBtn.textContent = 'Record Audio';
                recordBtn.style.color = '';
            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.addEventListener('dataavailable', event => {
                        audioChunks.push(event.data);
                    });

                    mediaRecorder.addEventListener('stop', () => {
                        const recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const file = new File([recordedBlob], "recording.webm", { type: "audio/webm" });

                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        audioFile.files = dataTransfer.files;

                        uploadLabel.textContent = "recording.webm (Ready to transcribe)";
                        uploadArea.classList.add('ok');

                        stream.getTracks().forEach(track => track.stop());
                    });

                    mediaRecorder.start();
                    recordBtn.textContent = 'Stop Recording';
                    recordBtn.style.color = '#c62a36';
                    uploadLabel.textContent = "Recording...";
                    uploadArea.classList.remove('ok');
                } catch (err) {
                    alert("Could not access microphone: " + err.message);
                }
            }
        });

        transcribeBtn.addEventListener('click', async () => {
            const file = audioFile.files?.[0];
            if (!file) { asrOut.textContent = 'Upload an audio file first.'; return; }
            spin(transcribeBtn, true);
            asrOut.textContent = 'Transcribing... (first run downloads Whisper)';
            try {
                const fd = new FormData();
                fd.append('audio', file);
                const resp = await fetch('/api/asr/upload', { method: 'POST', body: fd });
                const r = await resp.json();
                if (r.error) {
                    asrOut.textContent = r.error;
                } else {
                    asrOut.innerHTML = `<strong>Transcription (Cebuano):</strong> ${r.transcription || 'None'}<br><br><strong>Translation (English):</strong> ${r.translation || 'None'}`;
                    addHist(r.transcription?.slice(0, 35) || 'ASR', 'Transcribed via Whisper');
                }
            } catch(e) {
                asrOut.textContent = 'Error: ' + e.message;
            }
            spin(transcribeBtn, false, 'Transcribe Audio');
        });

        synthBtn.addEventListener('click', async () => {
            const text = ttsText.value.trim() || transOut.textContent.trim() || 'Maayong buntag';
            ttsText.value = text;
            synthBtn.disabled = true;
            synthBtn.innerHTML = '<span class="spin"></span>';
            ttsOut.textContent = 'Generating speech... (first run downloads MMS-TTS)';
            audioPlayer.innerHTML = '';
            try {
                const r = await post('/api/tts', { text, backend: 'vits' });
                if (r.audio_url) {
                    ttsOut.textContent = '';
                    audioPlayer.innerHTML = `<audio controls autoplay src="${r.audio_url}"></audio>`;
                } else {
                    ttsOut.textContent = r.output || 'Done.';
                }
                addHist(`TTS: ${text.slice(0, 25)}`, 'via MMS-TTS');
            } catch(e) {
                ttsOut.textContent = 'Error: ' + e.message;
            }
            synthBtn.disabled = false;
            synthBtn.innerHTML = synthIcon;
        });

        copyOutputBtn.addEventListener('click', async () => {
            const text = transOut.textContent.trim();
            if (!text) return;
            try {
                await navigator.clipboard.writeText(text);
                addHist('Copied translation', 'Output copied to clipboard');
            } catch(e) {
                addHist('Copy unavailable', e.message);
            }
        });
    </script>
</body>
</html>
"""


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    @app.route("/")
    def home():
        return render_template_string(HOME_TEMPLATE)

    @app.route("/api/backends")
    def backends():
        return {
            "translation": [model.model_name for model in registry.get_for_task("translation")],
            "asr": [model.model_name for model in registry.get_for_task("asr")],
            "tts": [model.model_name for model in registry.get_for_task("tts")],
        }

    @app.route("/api/translate", methods=["POST"])
    def translate():
        payload = request.get_json(force=True) or {}
        text = payload.get("text", "")
        backend_name = payload.get("backend", "nllb")
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, TranslationModel):
            return jsonify({"error": f"Backend {backend_name} is not available for translation"}), 400

        translated = backend.translate(text)
        phonetic = phoneticize_text(translated)
        tokens = tokenizer.tokenize(translated)
        return jsonify(
            {
                "task": "translation",
                "backend": backend_name,
                "translation": translated,
                "phonetic_guide": phonetic or "",
                "tokens": tokens,
            }
        )

    @app.route("/api/asr", methods=["POST"])
    def asr():
        payload = request.get_json(force=True) or {}
        audio_path = payload.get("audio_path", "sample.wav")
        backend_name = payload.get("backend", "whisper")
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({"error": f"Backend {backend_name} is not available for ASR"}), 400
        return jsonify({"task": "asr", "backend": backend_name, "output": backend.transcribe(audio_path)})

    @app.route("/api/asr/upload", methods=["POST"])
    def asr_upload():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        audio = request.files["audio"]
        if not audio.filename:
            return jsonify({"error": "Empty filename"}), 400

        ext = Path(audio.filename).suffix or ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=str(_UPLOAD_DIR))
        os.close(fd)
        audio.save(tmp_path)

        backend = registry.get("whisper")
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({"error": "Whisper backend not available"}), 500

        transcription = backend.transcribe(tmp_path, task="transcribe")
        translation = ""
        if translator and isinstance(translator, TranslationModel):
            translation = translator.translate(transcription, source_lang="ceb_Latn", target_lang="eng_Latn")

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        return jsonify({"transcription": transcription, "translation": translation})

    @app.route("/api/tts", methods=["POST"])
    def tts():
        payload = request.get_json(force=True) or {}
        text = payload.get("text", "")
        backend_name = payload.get("backend", "vits")
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({"error": f"Backend {backend_name} is not available for TTS"}), 400

        result = backend.synthesize(text)

        if os.path.isfile(result):
            filename = Path(result).name
            return jsonify(
                {
                    "task": "tts",
                    "backend": backend_name,
                    "audio_url": f"/api/tts/audio/{filename}",
                    "output": f"Audio generated: {filename}",
                }
            )

        return jsonify({"task": "tts", "backend": backend_name, "output": result})

    @app.route("/api/tts/audio/<filename>")
    def tts_audio(filename):
        tmp_dir = tempfile.gettempdir()
        path = Path(tmp_dir) / filename
        if not path.exists():
            return jsonify({"error": "Audio file not found"}), 404
        return send_file(str(path), mimetype="audio/wav")

    @app.route("/api/train", methods=["POST"])
    def train():
        payload = request.get_json(force=True) or {}
        task = payload.get("task", "translation")
        max_samples = payload.get("max_samples", 3)
        epochs = payload.get("epochs", 3)
        workflow = build_task_training_workflow(task=task, max_samples=max_samples, epochs=epochs)
        return jsonify({"task": task, "summary": summarize_training_workflow(workflow)})

    @app.route("/api/demo", methods=["POST"])
    def demo():
        payload = request.get_json(force=True) or {}
        text = payload.get("text", "Where are you going?")
        audio_path = payload.get("audio_path", "clip.wav")

        translation_model = registry.get("nllb")
        asr_model = registry.get("whisper")
        tts_model = registry.get("vits")

        summary_lines = [
            f"Translation: {translation_model.translate(text)}",
            f"ASR: {asr_model.transcribe(audio_path)}",
            f"TTS: {tts_model.synthesize(text)}",
        ]
        return jsonify({"task": "demo", "summary": "\\n".join(summary_lines)})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
