from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, render_template_string, send_file

from .models import (
    TranslationModel,
    SpeechModel,
    HuggingFaceAdapter,
    create_default_model_registry,
)
from .preprocessing import phoneticize_text
from .tokenizer import GlottalAwareTokenizer
from .training import build_task_training_workflow, summarize_training_workflow

registry = create_default_model_registry()
tokenizer = GlottalAwareTokenizer()

# Directory for uploaded audio files
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "butuanon_uploads"
_UPLOAD_DIR.mkdir(exist_ok=True)


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / 'templates'))
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB upload limit

    @app.route('/')
    def home():
        return render_template_string(
            '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>BisayaHub — Butuanon NLP Platform</title>
                <meta name="description" content="Translate, transcribe, and synthesise Butuanon speech using NLLB-200, Whisper, and MMS-TTS — all running locally." />
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
                <style>
                    :root {
                        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                        color: #191c1e;
                        --surface: rgba(255, 255, 255, 0.88);
                        --surface-strong: #ffffff;
                        --surface-muted: #eef1f5;
                        --border: rgba(207, 217, 229, 0.85);
                        --shadow: 0 20px 60px rgba(15, 23, 42, 0.07);
                        --accent: #400010;
                        --accent-soft: #f6e5eb;
                        --secondary: #505f76;
                        --success: #0d7d4d;
                        --success-bg: #e6f7ef;
                        --warn: #b45309;
                        --warn-bg: #fef3c7;
                    }

                    * { box-sizing: border-box; margin: 0; }

                    html, body {
                        min-height: 100%;
                        background: radial-gradient(ellipse at 20% 0%, rgba(64,0,16,0.09), transparent 50%),
                                    radial-gradient(ellipse at 80% 5%, rgba(34,45,69,0.06), transparent 40%),
                                    #f7f9fb;
                    }

                    body { color: #191c1e; }

                    button, select, textarea, input { font: inherit; }

                    /* ── Shell ────────────────────────────────────────── */
                    .shell { max-width: 1140px; margin: 0 auto; padding: 28px 22px 56px; }

                    /* ── Top bar ──────────────────────────────────────── */
                    .topbar {
                        display: flex; flex-wrap: wrap; justify-content: space-between;
                        gap: 16px; align-items: center;
                        padding: 18px 24px; border-radius: 24px;
                        background: rgba(255,255,255,0.82); border: 1px solid var(--border);
                        box-shadow: var(--shadow); backdrop-filter: blur(14px);
                        margin-bottom: 28px;
                    }
                    .brand { display: flex; align-items: center; gap: 14px; }
                    .brand__mark {
                        width: 46px; height: 46px; border-radius: 14px;
                        background: var(--accent); color: white;
                        display: grid; place-items: center; font-weight: 800; font-size: 1.1rem;
                    }
                    .brand__title { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em; }
                    .brand__sub { font-size: 0.88rem; color: var(--secondary); font-weight: 500; margin-top: 2px; }
                    .nav-links { display: flex; gap: 20px; flex-wrap: wrap; }
                    .nav-links a { text-decoration: none; color: var(--secondary); font-weight: 700; font-size: 0.92rem; }
                    .nav-links a.active { color: var(--accent); }

                    /* ── Hero ─────────────────────────────────────────── */
                    .hero {
                        padding: 36px 38px 34px; border-radius: 28px;
                        background: linear-gradient(170deg, rgba(255,255,255,0.93), rgba(238,241,245,0.96));
                        border: 1px solid var(--border); box-shadow: var(--shadow);
                        margin-bottom: 28px;
                    }
                    .hero h1 {
                        font-size: clamp(2.4rem, 4vw, 3.8rem);
                        line-height: 1.02; letter-spacing: -0.05em; margin-bottom: 14px;
                    }
                    .hero p { max-width: 700px; font-size: 1rem; color: var(--secondary); line-height: 1.7; }

                    /* ── Badges ───────────────────────────────────────── */
                    .tag {
                        display: inline-flex; padding: 8px 14px; border-radius: 999px;
                        font-weight: 700; font-size: 0.82rem; width: fit-content;
                    }
                    .tag--live { background: var(--success-bg); color: var(--success); }
                    .tag--accent { background: var(--accent-soft); color: var(--accent); }
                    .tag--muted { background: #eef1f5; color: var(--secondary); }

                    /* ── Grid ─────────────────────────────────────────── */
                    .grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 22px; }
                    .col { display: grid; gap: 20px; align-content: start; }

                    /* ── Card ─────────────────────────────────────────── */
                    .card {
                        background: var(--surface); border-radius: 24px;
                        border: 1px solid var(--border); box-shadow: var(--shadow);
                        padding: 24px; display: grid; gap: 16px;
                    }
                    .card__head { display: flex; justify-content: space-between; align-items: start; gap: 12px; }
                    .card__head h2 { font-size: 1.15rem; letter-spacing: -0.02em; }

                    /* ── Form elements ────────────────────────────────── */
                    .lbl { text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.76rem; color: var(--secondary); font-weight: 700; }
                    .sel, .inp, .txa {
                        width: 100%; border-radius: 16px; border: 1px solid #d8dadc;
                        background: #f7f9fb; color: #191c1e; padding: 14px 16px;
                        transition: border-color 0.2s, box-shadow 0.2s;
                    }
                    .sel { appearance: none; }
                    .txa { min-height: 170px; resize: vertical; line-height: 1.8; }
                    .sel:focus, .inp:focus, .txa:focus {
                        outline: none; border-color: var(--accent);
                        box-shadow: 0 0 0 3px rgba(64,0,16,0.07);
                    }

                    /* ── Buttons ──────────────────────────────────────── */
                    .btns { display: flex; gap: 12px; flex-wrap: wrap; }
                    .btn {
                        display: inline-flex; align-items: center; justify-content: center; gap: 7px;
                        padding: 13px 22px; border-radius: 16px; border: none;
                        background: var(--accent); color: white; font-weight: 700;
                        cursor: pointer; transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
                        font-size: 0.92rem;
                    }
                    .btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(64,0,16,0.16); }
                    .btn:disabled { opacity: 0.55; cursor: not-allowed; }
                    .btn--outline {
                        background: white; color: var(--accent);
                        border: 1.5px solid rgba(64,0,16,0.2);
                    }
                    .btn--outline:hover:not(:disabled) { background: var(--accent-soft); }

                    /* ── Spinner ──────────────────────────────────────── */
                    .spin {
                        display: inline-block; width: 14px; height: 14px;
                        border: 2.5px solid rgba(255,255,255,0.3); border-top-color: white;
                        border-radius: 50%; animation: spin 0.65s linear infinite;
                    }
                    .spin--dark { border-color: rgba(64,0,16,0.15); border-top-color: var(--accent); }
                    @keyframes spin { to { transform: rotate(360deg); } }

                    /* ── Output panels ────────────────────────────────── */
                    .out { color: var(--secondary); line-height: 1.7; white-space: pre-wrap; font-size: 0.95rem; }
                    .out--result { color: #191c1e; font-size: 1.08rem; font-weight: 500; min-height: 32px; }

                    /* ── Token chips ──────────────────────────────────── */
                    .chips { display: flex; flex-wrap: wrap; gap: 5px; }
                    .chip {
                        padding: 4px 9px; border-radius: 7px;
                        background: var(--accent-soft); color: var(--accent);
                        font-size: 0.82rem; font-weight: 600;
                        font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
                    }

                    /* ── File upload ──────────────────────────────────── */
                    .upload {
                        border: 2px dashed rgba(64,0,16,0.15); border-radius: 16px;
                        padding: 18px; text-align: center; cursor: pointer;
                        transition: border-color 0.2s, background 0.2s;
                        color: var(--secondary); font-weight: 600; font-size: 0.9rem;
                        position: relative;
                    }
                    .upload:hover { border-color: var(--accent); background: rgba(64,0,16,0.02); }
                    .upload.ok { border-color: var(--success); background: var(--success-bg); color: var(--success); }
                    .upload input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

                    /* ── Audio player ─────────────────────────────────── */
                    audio { width: 100%; border-radius: 10px; margin-top: 6px; }

                    /* ── Phonetic annotation ──────────────────────────── */
                    .phon {
                        padding: 14px 18px; border-radius: 14px;
                        background: #fdf6f0; border: 1px solid #f0e0d0;
                        font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
                        font-size: 0.9rem; color: #7c4a1a; line-height: 1.6;
                    }

                    /* ── History ──────────────────────────────────────── */
                    .hist {
                        padding: 14px 16px; border-radius: 14px;
                        background: #f9fafb; border: 1px solid rgba(227,232,240,0.95);
                    }
                    .hist strong { display: block; margin-bottom: 4px; font-size: 0.92rem; }
                    .hist span { color: var(--secondary); font-size: 0.86rem; }

                    /* ── Architecture compact ─────────────────────────── */
                    .arch-list { list-style: none; padding: 0; display: grid; gap: 8px; }
                    .arch-list li {
                        display: flex; align-items: baseline; gap: 8px;
                        font-size: 0.88rem; color: var(--secondary); line-height: 1.5;
                    }
                    .arch-list li strong { color: #191c1e; white-space: nowrap; }

                    /* ── Responsive ───────────────────────────────────── */
                    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
                    @media (max-width: 640px) { .shell { padding: 18px 14px 32px; } .hero { padding: 24px; } }
                </style>
            </head>
            <body>
                <div class="shell">
                    <!-- Top bar -->
                    <header class="topbar">
                        <div class="brand">
                            <div class="brand__mark">B</div>
                            <div>
                                <div class="brand__title">BisayaHub</div>
                                <div class="brand__sub">Butuanon NLP Platform</div>
                            </div>
                        </div>
                        <nav class="nav-links">
                            <a href="#translate-section" class="active">Translate</a>
                            <a href="#speech-section">Speech</a>
                        </nav>
                    </header>

                    <!-- Hero -->
                    <section class="hero">
                        <span class="tag tag--live" style="margin-bottom:14px;">&#x2713; Live Models</span>
                        <h1>Translate and speak Butuanon with real AI.</h1>
                        <p>Type English text to get a Cebuano translation from NLLB-200. Upload audio for Whisper transcription. Generate speech with MMS-TTS. Everything runs locally on your machine.</p>
                    </section>

                    <!-- Main grid -->
                    <div class="grid">

                        <!-- LEFT COLUMN: Translation -->
                        <div class="col" id="translate-section">

                            <!-- Input card -->
                            <section class="card">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Translation</p>
                                        <h2>English → Cebuano</h2>
                                    </div>
                                    <span class="tag tag--live">NLLB-200</span>
                                </div>
                                <select id="backend-select" class="sel"></select>
                                <textarea id="source-text" class="txa" placeholder="Type or paste English text here…"></textarea>
                                <div class="btns">
                                    <button id="translate-btn" class="btn">Translate</button>
                                </div>
                            </section>

                            <!-- Translation output -->
                            <section class="card">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Result</p>
                                        <h2>Translation output</h2>
                                    </div>
                                    <span class="tag tag--muted">Live</span>
                                </div>
                                <div class="out out--result" id="translation-output">Translation will appear here…</div>
                                <div id="token-display"></div>
                            </section>

                            <!-- Phonetic guide -->
                            <section class="card" id="phonetic-card" style="display:none;">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Phonetic guide</p>
                                        <h2>Glottal stop annotations</h2>
                                    </div>
                                    <span class="tag tag--accent">ʔ</span>
                                </div>
                                <div class="phon" id="phonetic-output"></div>
                                <p class="out" style="font-size:0.82rem;">
                                    Butuanon marks glottal stops with apostrophes or hyphens between letters
                                    (e.g. <em>dal-a</em> → <em>dalʔa</em>). This guide shows where they occur.
                                </p>
                            </section>
                        </div>

                        <!-- RIGHT COLUMN: Speech + Info -->
                        <div class="col" id="speech-section">

                            <!-- Speech-to-Text -->
                            <section class="card">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Speech-to-Text</p>
                                        <h2>Transcribe audio</h2>
                                    </div>
                                    <span class="tag tag--muted">Whisper</span>
                                </div>
                                <p class="out">Upload an audio file or record from your microphone, and Whisper will transcribe the speech to text.</p>
                                <div class="upload" id="upload-area">
                                    <span id="upload-label">Drop an audio file here or click to browse</span>
                                    <input type="file" id="audio-file" accept="audio/*" />
                                </div>
                                <div class="btns">
                                    <button id="record-btn" class="btn--outline btn">Record Mic</button>
                                    <button id="transcribe-btn" class="btn--outline btn">Transcribe</button>
                                </div>
                                <div class="out" id="asr-output">Transcription result will appear here.</div>
                            </section>

                            <!-- Text-to-Speech -->
                            <section class="card">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Text-to-Speech</p>
                                        <h2>Generate voice</h2>
                                    </div>
                                    <span class="tag tag--muted">MMS-TTS</span>
                                </div>
                                <p class="out">Type text below and generate an audio clip. The model currently speaks in Cebuano (closest available) — Butuanon voice training is a future milestone.</p>
                                <input id="tts-text" class="inp" placeholder="Type text to speak aloud…" value="Maayong buntag" />
                                <div class="btns">
                                    <button id="synthesize-btn" class="btn--outline btn">Speak</button>
                                </div>
                                <div class="out" id="tts-output"></div>
                                <div id="audio-player"></div>
                            </section>

                            <!-- Architecture (compact) -->
                            <section class="card">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Models</p>
                                        <h2>What powers this</h2>
                                    </div>
                                </div>
                                <ul class="arch-list">
                                    <li><strong>NLLB-200</strong> Translates English → Cebuano (closest to Butuanon)</li>
                                    <li><strong>Whisper</strong> Transcribes audio to text (speech recognition)</li>
                                    <li><strong>MMS-TTS</strong> Converts text to spoken audio (voice synthesis)</li>
                                </ul>
                                <p class="out" style="font-size:0.82rem;">Models load on first use and are cached. GPU acceleration is automatic when CUDA is available.</p>
                            </section>

                            <!-- Activity -->
                            <section class="card">
                                <div class="card__head">
                                    <div>
                                        <p class="lbl">Activity</p>
                                        <h2>Recent</h2>
                                    </div>
                                </div>
                                <div id="history-list">
                                    <div class="hist">
                                        <strong>Ready</strong>
                                        <span>Waiting for first interaction…</span>
                                    </div>
                                </div>
                            </section>
                        </div>
                    </div>
                </div>

                <script>
                    // -- DOM refs --
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
                    let mediaRecorder = null;
                    let audioChunks = [];

                    // -- Helpers --
                    function spin(btn, on, label) {
                        btn.disabled = on;
                        if (on) {
                            btn.dataset.orig = btn.textContent;
                            const dark = btn.classList.contains('btn--outline');
                            btn.innerHTML = `<span class="spin ${dark?'spin--dark':''}"></span> Working…`;
                        } else {
                            btn.textContent = label || btn.dataset.orig || 'Done';
                        }
                    }

                    function addHist(title, detail) {
                        const d = document.createElement('div');
                        d.className = 'hist';
                        d.innerHTML = `<strong>${title}</strong><span>${detail}</span>`;
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

                    // -- Load backends --
                    fetch('/api/backends').then(r => r.json()).then(d => {
                        backendSel.innerHTML = d.translation
                            .map(n => `<option value="${n}">${n.toUpperCase()}</option>`)
                            .join('');
                    }).catch(console.error);

                    // -- File upload visual --
                    audioFile.addEventListener('change', () => {
                        if (audioFile.files.length) {
                            uploadLabel.textContent = audioFile.files[0].name;
                            uploadArea.classList.add('ok');
                        } else {
                            uploadLabel.textContent = 'Drop an audio file here or click to browse';
                            uploadArea.classList.remove('ok');
                        }
                    });

                    // -- Translate --
                    translateBtn.addEventListener('click', async () => {
                        const text = srcText.value.trim();
                        if (!text) return;
                        spin(translateBtn, true);
                        transOut.textContent = 'Translating… (first run downloads the model)';
                        phonCard.style.display = 'none';
                        tokenDisp.innerHTML = '';

                        try {
                            const r = await post('/api/translate', { text, backend: backendSel.value });
                            transOut.textContent = r.translation || r.error || '';

                            // Phonetic guide — only show if there are actual glottal annotations
                            if (r.phonetic_guide) {
                                phonOut.textContent = r.phonetic_guide;
                                phonCard.style.display = '';
                            }

                            // Token chips
                            if (r.tokens && r.tokens.length) {
                                tokenDisp.innerHTML =
                                    '<p class="lbl" style="margin-bottom:6px;">Tokens</p>' +
                                    '<div class="chips">' +
                                    r.tokens.map(t => `<span class="chip">${t}</span>`).join('') +
                                    '</div>';
                            }
                            addHist(r.translation?.slice(0,35) || 'Translation', `via ${backendSel.value.toUpperCase()}`);
                            
                            if (r.translation) {
                                ttsText.value = r.translation;
                                synthBtn.click();
                            }
                        } catch(e) {
                            transOut.textContent = 'Error: ' + e.message;
                        }
                        spin(translateBtn, false, 'Translate');
                    });
                    
                    // -- Record (Mic) --
                    recordBtn.addEventListener('click', async () => {
                        if (mediaRecorder && mediaRecorder.state === 'recording') {
                            mediaRecorder.stop();
                            recordBtn.textContent = 'Record Mic';
                            recordBtn.style.color = '';
                            recordBtn.style.borderColor = '';
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
                                recordBtn.style.color = '#ef4444';
                                recordBtn.style.borderColor = '#ef4444';
                                uploadLabel.textContent = "Recording...";
                                uploadArea.classList.remove('ok');
                            } catch (err) {
                                alert("Could not access microphone: " + err.message);
                            }
                        }
                    });

                    // -- Transcribe (ASR) --
                    transcribeBtn.addEventListener('click', async () => {
                        const file = audioFile.files?.[0];
                        if (!file) { asrOut.textContent = 'Upload an audio file first.'; return; }
                        spin(transcribeBtn, true);
                        asrOut.textContent = 'Transcribing… (first run downloads Whisper)';
                        try {
                            const fd = new FormData();
                            fd.append('audio', file);
                            const resp = await fetch('/api/asr/upload', { method: 'POST', body: fd });
                            const r = await resp.json();
                            asrOut.textContent = r.output || r.error || 'No output.';
                            addHist(r.output?.slice(0,35) || 'ASR', 'Transcribed via Whisper');
                        } catch(e) {
                            asrOut.textContent = 'Error: ' + e.message;
                        }
                        spin(transcribeBtn, false, 'Transcribe');
                    });

                    // -- Synthesize (TTS) --
                    synthBtn.addEventListener('click', async () => {
                        const text = ttsText.value.trim() || 'Maayong buntag';
                        spin(synthBtn, true);
                        ttsOut.textContent = 'Generating speech… (first run downloads MMS-TTS)';
                        audioPlayer.innerHTML = '';
                        try {
                            const r = await post('/api/tts', { text, backend: 'vits' });
                            if (r.audio_url) {
                                ttsOut.textContent = '';
                                audioPlayer.innerHTML = `<audio controls autoplay src="${r.audio_url}"></audio>`;
                            } else {
                                ttsOut.textContent = r.output || 'Done.';
                            }
                            addHist(`TTS: ${text.slice(0,25)}`, 'via MMS-TTS');
                        } catch(e) {
                            ttsOut.textContent = 'Error: ' + e.message;
                        }
                        spin(synthBtn, false, 'Speak');
                    });
                </script>
            </body>
            </html>
            '''
        )

    # ── API routes ──────────────────────────────────────────────────────

    @app.route('/api/backends')
    def backends():
        return {
            'translation': [model.model_name for model in registry.get_for_task('translation')],
            'asr': [model.model_name for model in registry.get_for_task('asr')],
            'tts': [model.model_name for model in registry.get_for_task('tts')],
        }

    @app.route('/api/translate', methods=['POST'])
    def translate():
        payload = request.get_json(force=True) or {}
        text = payload.get('text', '')
        backend_name = payload.get('backend', 'nllb')
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, TranslationModel):
            return jsonify({'error': f'Backend {backend_name} is not available for translation'}), 400

        translated = backend.translate(text)
        phonetic = phoneticize_text(translated)
        tokens = tokenizer.tokenize(translated)
        return jsonify({
            'task': 'translation',
            'backend': backend_name,
            'translation': translated,
            'phonetic_guide': phonetic or '',
            'tokens': tokens,
        })

    @app.route('/api/asr', methods=['POST'])
    def asr():
        payload = request.get_json(force=True) or {}
        audio_path = payload.get('audio_path', 'sample.wav')
        backend_name = payload.get('backend', 'whisper')
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({'error': f'Backend {backend_name} is not available for ASR'}), 400
        return jsonify({'task': 'asr', 'backend': backend_name, 'output': backend.transcribe(audio_path)})

    @app.route('/api/asr/upload', methods=['POST'])
    def asr_upload():
        """Accept an uploaded audio file and run Whisper on it."""
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        audio = request.files['audio']
        if not audio.filename:
            return jsonify({'error': 'Empty filename'}), 400

        ext = Path(audio.filename).suffix or '.wav'
        fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=str(_UPLOAD_DIR))
        os.close(fd)
        audio.save(tmp_path)

        backend = registry.get('whisper')
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({'error': 'Whisper backend not available'}), 500

        result_text = backend.transcribe(tmp_path)

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        return jsonify({'task': 'asr', 'backend': 'whisper', 'output': result_text})

    @app.route('/api/tts', methods=['POST'])
    def tts():
        payload = request.get_json(force=True) or {}
        text = payload.get('text', '')
        backend_name = payload.get('backend', 'vits')
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({'error': f'Backend {backend_name} is not available for TTS'}), 400

        result = backend.synthesize(text)

        if os.path.isfile(result):
            filename = Path(result).name
            return jsonify({
                'task': 'tts',
                'backend': backend_name,
                'audio_url': f'/api/tts/audio/{filename}',
                'output': f'Audio generated: {filename}',
            })

        return jsonify({'task': 'tts', 'backend': backend_name, 'output': result})

    @app.route('/api/tts/audio/<filename>')
    def tts_audio(filename):
        """Serve a generated TTS WAV file."""
        tmp_dir = tempfile.gettempdir()
        path = Path(tmp_dir) / filename
        if not path.exists():
            return jsonify({'error': 'Audio file not found'}), 404
        return send_file(str(path), mimetype='audio/wav')

    @app.route('/api/train', methods=['POST'])
    def train():
        payload = request.get_json(force=True) or {}
        task = payload.get('task', 'translation')
        max_samples = payload.get('max_samples', 3)
        epochs = payload.get('epochs', 3)
        workflow = build_task_training_workflow(task=task, max_samples=max_samples, epochs=epochs)
        return jsonify({'task': task, 'summary': summarize_training_workflow(workflow)})

    @app.route('/api/demo', methods=['POST'])
    def demo():
        payload = request.get_json(force=True) or {}
        text = payload.get('text', 'Where are you going?')
        audio_path = payload.get('audio_path', 'clip.wav')

        translation_model = registry.get('nllb')
        asr_model = registry.get('whisper')
        tts_model = registry.get('vits')

        summary_lines = [
            f"Translation: {translation_model.translate(text)}",
            f"ASR: {asr_model.transcribe(audio_path)}",
            f"TTS: {tts_model.synthesize(text)}",
        ]
        return jsonify({'task': 'demo', 'summary': '\n'.join(summary_lines)})

    return app


app = create_app()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
