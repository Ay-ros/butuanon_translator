from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, render_template_string

from .models import (
    TranslationModel,
    SpeechModel,
    create_default_model_registry,
)
from .preprocessing import phoneticize_text
from .tokenizer import GlottalAwareTokenizer
from .training import build_task_training_workflow, summarize_training_workflow
from .models import HuggingFaceAdapter

registry = create_default_model_registry()
tokenizer = GlottalAwareTokenizer()


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / 'templates'))

    @app.route('/')
    def home():
        return render_template_string(
            '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>BisayaHub</title>
                <style>
                    :root {
                        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                        color: #191c1e;
                        background: #f7f9fb;
                        --surface: rgba(255, 255, 255, 0.88);
                        --surface-strong: #ffffff;
                        --surface-muted: #eef1f5;
                        --border: rgba(207, 217, 229, 0.85);
                        --shadow: 0 28px 80px rgba(15, 23, 42, 0.08);
                        --accent: #400010;
                        --accent-soft: #f6e5eb;
                        --secondary: #505f76;
                    }

                    * {
                        box-sizing: border-box;
                    }

                    html, body {
                        margin: 0;
                        min-height: 100%;
                        background: radial-gradient(circle at top left, rgba(64, 0, 16, 0.12), transparent 25%), radial-gradient(circle at 80% 10%, rgba(34, 45, 69, 0.08), transparent 22%), #f7f9fb;
                    }

                    body {
                        color: #191c1e;
                    }

                    button, select, textarea, input {
                        font: inherit;
                    }

                    .site-shell {
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 32px 24px 48px;
                    }

                    .topbar {
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                        gap: 18px;
                        align-items: center;
                        padding: 22px 26px;
                        border-radius: 28px;
                        background: rgba(255, 255, 255, 0.84);
                        border: 1px solid var(--border);
                        box-shadow: var(--shadow);
                        backdrop-filter: blur(15px);
                        margin-bottom: 32px;
                    }

                    .brand {
                        display: flex;
                        align-items: center;
                        gap: 16px;
                    }

                    .brand__mark {
                        width: 52px;
                        height: 52px;
                        border-radius: 18px;
                        background: var(--accent);
                        color: white;
                        display: grid;
                        place-items: center;
                        font-weight: 800;
                        letter-spacing: -0.05em;
                    }

                    .brand__title {
                        margin: 0;
                        font-size: 1.8rem;
                        letter-spacing: -0.03em;
                    }

                    .brand__subtitle {
                        margin: 4px 0 0;
                        font-size: 0.95rem;
                        color: var(--secondary);
                        font-weight: 500;
                    }

                    .nav-links {
                        display: flex;
                        align-items: center;
                        gap: 24px;
                        flex-wrap: wrap;
                    }

                    .nav-links a {
                        text-decoration: none;
                        color: var(--secondary);
                        font-weight: 700;
                    }

                    .nav-links a.active {
                        color: #131b2e;
                    }

                    .hero {
                        padding: 42px 42px 40px;
                        border-radius: 32px;
                        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(240,244,249,0.95));
                        border: 1px solid var(--border);
                        box-shadow: var(--shadow);
                        margin-bottom: 32px;
                    }

                    .hero h1 {
                        margin: 0 0 16px;
                        font-size: clamp(3rem, 4.5vw, 4.8rem);
                        line-height: 0.98;
                        letter-spacing: -0.06em;
                    }

                    .hero p {
                        margin: 0;
                        max-width: 760px;
                        font-size: 1.05rem;
                        color: var(--secondary);
                        line-height: 1.8;
                    }

                    .badge {
                        display: inline-flex;
                        padding: 12px 16px;
                        border-radius: 999px;
                        background: var(--accent-soft);
                        color: var(--accent);
                        font-weight: 700;
                        margin-top: 22px;
                        width: fit-content;
                    }

                    .main-grid {
                        display: grid;
                        grid-template-columns: 1.5fr 0.9fr;
                        gap: 26px;
                    }

                    .card {
                        background: var(--surface);
                        border-radius: 28px;
                        border: 1px solid var(--border);
                        box-shadow: var(--shadow);
                        padding: 26px;
                    }

                    .card h2 {
                        margin: 0 0 18px;
                        font-size: 1.25rem;
                        letter-spacing: -0.02em;
                    }

                    .section-grid {
                        display: grid;
                        gap: 22px;
                    }

                    .field-group {
                        display: grid;
                        gap: 16px;
                    }

                    .select-field,
                    .text-field,
                    .textarea-field {
                        width: 100%;
                        border-radius: 20px;
                        border: 1px solid #d8dadc;
                        background: #f7f9fb;
                        color: #191c1e;
                        padding: 16px 18px;
                    }

                    .select-field {
                        appearance: none;
                    }

                    .textarea-field {
                        min-height: 210px;
                        resize: vertical;
                        line-height: 1.85;
                    }

                    .textarea-field:focus,
                    .select-field:focus,
                    .text-field:focus {
                        outline: none;
                        border-color: var(--accent);
                        box-shadow: 0 0 0 4px rgba(64, 0, 16, 0.08);
                    }

                    .button-row {
                        display: flex;
                        gap: 16px;
                        flex-wrap: wrap;
                        align-items: center;
                        margin-top: 6px;
                    }

                    .button {
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        padding: 16px 24px;
                        border-radius: 18px;
                        border: none;
                        background: var(--accent);
                        color: white;
                        font-weight: 700;
                        cursor: pointer;
                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                    }

                    .button:hover {
                        transform: translateY(-1px);
                        box-shadow: 0 18px 36px rgba(64, 0, 16, 0.18);
                    }

                    .pill-group {
                        display: flex;
                        gap: 12px;
                        flex-wrap: wrap;
                    }

                    .pill {
                        padding: 10px 16px;
                        border-radius: 999px;
                        border: 1px solid rgba(115, 125, 145, 0.18);
                        background: white;
                        color: #191c1e;
                        font-weight: 700;
                        cursor: pointer;
                    }

                    .pill.active {
                        background: var(--accent);
                        color: white;
                        border-color: transparent;
                    }

                    .info-card {
                        display: grid;
                        gap: 18px;
                    }

                    .info-card__row {
                        display: flex;
                        justify-content: space-between;
                        gap: 12px;
                        flex-wrap: wrap;
                    }

                    .info-card__badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 8px;
                        padding: 10px 14px;
                        border-radius: 16px;
                        background: #eef1f5;
                        color: var(--secondary);
                        font-weight: 700;
                    }

                    .history-card {
                        display: grid;
                        gap: 18px;
                    }

                    .secondary-button {
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        padding: 14px 20px;
                        border-radius: 18px;
                        border: 1px solid rgba(64, 0, 16, 0.2);
                        background: #ffffff;
                        color: var(--accent);
                        font-weight: 700;
                        cursor: pointer;
                    }

                    .history-item {
                        padding: 18px 20px;
                        border-radius: 20px;
                        background: #f9fafb;
                        border: 1px solid rgba(227, 232, 240, 0.95);
                    }

                    .history-item strong {
                        display: block;
                        margin-bottom: 8px;
                    }

                    .history-item span {
                        color: var(--secondary);
                        font-size: 0.96rem;
                    }

                    .panel__content {
                        color: var(--secondary);
                        line-height: 1.75;
                        white-space: pre-wrap;
                    }

                    .meta-label {
                        text-transform: uppercase;
                        letter-spacing: 0.16em;
                        font-size: 0.79rem;
                        color: var(--secondary);
                        font-weight: 700;
                    }

                    @media (max-width: 960px) {
                        .main-grid {
                            grid-template-columns: 1fr;
                        }
                    }

                    @media (max-width: 680px) {
                        .site-shell {
                            padding: 22px 18px 32px;
                        }

                        .hero {
                            padding: 28px;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="site-shell">
                    <div class="topbar">
                        <div class="brand">
                            <div class="brand__mark">B</div>
                            <div>
                                <p class="brand__title">BisayaHub</p>
                                <p class="brand__subtitle">Refined translation, ASR, and speech tooling for Butuanon.</p>
                            </div>
                        </div>
                        <div class="nav-links">
                            <a href="#" class="active">Translate</a>
                            <a href="#">Dictionary</a>
                            <a href="#">Voice Lab</a>
                        </div>
                    </div>

                    <section class="hero">
                        <span class="badge">Linguistic Precision</span>
                        <h1>Translate, transcribe, and speak Bisaya with a single toolkit.</h1>
                        <p>Explore a premium language workspace designed for clarity, accuracy, and modern editorial presentation. The UI routes tasks to translation, speech-to-text, and text-to-speech models through a shared backend registry.</p>
                    </section>

                    <div class="main-grid">
                        <div class="section-grid">
                            <section class="card">
                                <div class="info-card">
                                    <div class="info-card__row">
                                        <div>
                                            <p class="meta-label">Translation</p>
                                            <h2>Text translation</h2>
                                        </div>
                                        <span class="info-card__badge">NLLB</span>
                                    </div>
                                    <div class="field-group">
                                        <select id="backend-select" class="select-field"></select>
                                        <textarea id="source-text" class="textarea-field" placeholder="Type English or Bisaya here..."></textarea>
                                    </div>
                                    <div class="button-row">
                                        <button id="translate-button" class="button">Translate</button>
                                    </div>
                                </div>
                            </section>

                            <section class="card">
                                <div class="info-card">
                                    <div class="info-card__row">
                                        <div>
                                            <p class="meta-label">Output</p>
                                            <h2>Hubad result</h2>
                                        </div>
                                        <span class="info-card__badge">Live</span>
                                    </div>
                                    <div class="panel__content" id="translation-output">Your translated output appears here.</div>
                                </div>
                            </section>

                            <section class="card">
                                <div class="info-card">
                                    <div class="info-card__row">
                                        <div>
                                            <p class="meta-label">Phonetic guide</p>
                                            <h2>Glottal and sound hints</h2>
                                        </div>
                                        <span class="info-card__badge">Preview</span>
                                    </div>
                                    <div class="panel__content" id="phonetic-output">Glottal markers, phonetic hints, and reading guidance will appear here after translation.</div>
                                </div>
                            </section>
                        </div>

                        <aside class="section-grid">
                            <section class="card history-card">
                                <div>
                                    <p class="meta-label">Voice Lab</p>
                                    <h2>Whisper + VITS</h2>
                                </div>
                                <div class="panel__content">
                                    <p>Use Whisper for transcription and VITS for speech synthesis. This panel lets you simulate ASR/TTS flows from the same page.</p>
                                </div>
                                <div class="field-group">
                                    <label class="meta-label" for="audio-path">Audio source</label>
                                    <input id="audio-path" class="text-field" placeholder="audio/sample.wav" />
                                </div>
                                <div class="field-group">
                                    <label class="meta-label" for="task-select">Training task</label>
                                    <select id="task-select" class="select-field">
                                        <option value="translation">Translation</option>
                                        <option value="asr">ASR</option>
                                        <option value="tts">TTS</option>
                                    </select>
                                </div>
                                <div class="button-row">
                                    <button id="transcribe-button" class="secondary-button">Transcribe</button>
                                    <button id="synthesize-button" class="secondary-button">Synthesize</button>
                                    <button id="demo-button" class="button">Run Demo</button>
                                </div>
                                <div class="panel__content" id="speech-output">Transcription and speech synthesis responses appear here.</div>
                            </section>

                            <section class="card history-card">
                                <div>
                                    <p class="meta-label">Training</p>
                                    <h2>Start a workflow</h2>
                                </div>
                                <div class="panel__content">
                                    Launch a task-specific training scaffold for translation, ASR, or TTS using the same backend registry that powers the app.
                                </div>
                                <div class="button-row">
                                    <button id="train-button" class="button">Train</button>
                                </div>
                                <div class="panel__content" id="training-output">Training workflow details will appear here.</div>
                            </section>

                            <section class="card history-card">
                                <div>
                                    <p class="meta-label">System</p>
                                    <h2>Architecture</h2>
                                </div>
                                <div class="panel__content">
                                    NLLB handles translation. Whisper is the ASR model. VITS is the TTS model.
                                    <br /><br />
                                    The web app routes each UI action to the appropriate backend through shared registry logic, so the browser can remain lightweight while the model stack stays organized.
                                </div>
                            </section>

                            <section class="card history-card">
                                <div>
                                    <p class="meta-label">History</p>
                                    <h2>Recent activity</h2>
                                </div>
                                <div class="history-item">
                                    <strong>Maayong buntag</strong>
                                    <span>Translated using NLLB</span>
                                </div>
                                <div class="history-item">
                                    <strong>Salamat</strong>
                                    <span>Text-to-speech generated with VITS</span>
                                </div>
                            </section>
                        </aside>
                    </div>
                </div>

                <script>
                    const backendSelect = document.getElementById('backend-select');
                    const translateButton = document.getElementById('translate-button');
                    const sourceText = document.getElementById('source-text');
                    const translationOutput = document.getElementById('translation-output');
                    const phoneticOutput = document.getElementById('phonetic-output');
                    const audioPath = document.getElementById('audio-path');
                    const transcribeButton = document.getElementById('transcribe-button');
                    const synthesizeButton = document.getElementById('synthesize-button');
                    const speechOutput = document.getElementById('speech-output');
                    const trainButton = document.getElementById('train-button');
                    const trainingOutput = document.getElementById('training-output');
                    const taskSelect = document.getElementById('task-select');
                    const demoButton = document.getElementById('demo-button');

                    async function loadBackends() {
                        const response = await fetch('/api/backends');
                        const payload = await response.json();
                        backendSelect.innerHTML = payload.translation
                            .map(name => `<option value="${name}">${name.toUpperCase()}</option>`)
                            .join('');
                    }

                    async function postJson(path, data) {
                        const response = await fetch(path, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(data),
                        });
                        return response.json();
                    }

                    translateButton.addEventListener('click', async () => {
                        const text = sourceText.value.trim();
                        const backend = backendSelect.value;
                        const result = await postJson('/api/translate', { text, backend });
                        translationOutput.textContent = result.translation;
                        phoneticOutput.textContent = result.phonetic_guide || 'No phonetic guide available yet.';
                    });

                    transcribeButton.addEventListener('click', async () => {
                        const path = audioPath.value.trim();
                        const result = await postJson('/api/asr', { audio_path: path, backend: 'whisper' });
                        speechOutput.textContent = result.output;
                    });

                    synthesizeButton.addEventListener('click', async () => {
                        const text = sourceText.value.trim() || 'Maayong buntag';
                        const result = await postJson('/api/tts', { text, backend: 'vits' });
                        speechOutput.textContent = result.output;
                    });

                    trainButton.addEventListener('click', async () => {
                        const task = taskSelect.value;
                        const result = await postJson('/api/train', { task, max_samples: 3, epochs: 3 });
                        trainingOutput.textContent = result.summary;
                    });

                    demoButton.addEventListener('click', async () => {
                        const result = await postJson('/api/demo', {
                            text: sourceText.value.trim() || 'Where are you going?',
                            audio_path: audioPath.value.trim() || 'clip.wav',
                        });
                        speechOutput.textContent = result.summary;
                    });

                    loadBackends().catch(console.error);
                </script>
            </body>
            </html>
            '''
        )

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
        return jsonify({
            'task': 'translation',
            'backend': backend_name,
            'translation': translated,
            'phonetic_guide': phonetic or 'No phonetic guide available.',
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

    @app.route('/api/tts', methods=['POST'])
    def tts():
        payload = request.get_json(force=True) or {}
        text = payload.get('text', '')
        backend_name = payload.get('backend', 'vits')
        backend = registry.get(backend_name)
        if not backend or not isinstance(backend, SpeechModel):
            return jsonify({'error': f'Backend {backend_name} is not available for TTS'}), 400
        return jsonify({'task': 'tts', 'backend': backend_name, 'output': backend.synthesize(text)})

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

        translation_model = TranslationModel(
            model_name='nllb',
            use_huggingface=True,
            adapter=HuggingFaceAdapter('facebook/nllb-200-distilled-600M', task='translation'),
        )
        asr_model = SpeechModel(
            model_name='whisper',
            use_huggingface=True,
            adapter=HuggingFaceAdapter('openai/whisper-tiny', task='asr'),
        )
        tts_model = SpeechModel(
            model_name='vits',
            use_huggingface=True,
            adapter=HuggingFaceAdapter('facebook/mms-tts-eng', task='tts'),
        )

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
