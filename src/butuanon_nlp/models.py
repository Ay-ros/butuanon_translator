"""Model adapters for the Butuanon NLP translation platform.

Provides three inference backends:

* **TranslationModel** — NLLB-200 (``facebook/nllb-200-distilled-600M``)
* **SpeechModel (ASR)** — Whisper via ``transformers`` (primary) or
  ``openai-whisper`` (fallback)
* **SpeechModel (TTS)** — Facebook MMS-TTS (``facebook/mms-tts-ceb``) via
  ``transformers``, with ``pyttsx3`` as an offline fallback

All models are lazily loaded on first use and cached for the lifetime of the
process.  Device selection is automatic (CUDA when available, otherwise CPU).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

# Ensure ffmpeg is in PATH (required by Whisper for .webm decoding on Windows)
try:
    import imageio_ffmpeg
    import shutil
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    _target_exe = os.path.join(_ffmpeg_dir, 'ffmpeg.exe')
    if not os.path.exists(_target_exe):
        shutil.copyfile(_ffmpeg_exe, _target_exe)
    os.environ["PATH"] += os.pathsep + _ffmpeg_dir
except ImportError:
    pass

# Optional heavy imports — lazily loaded inside methods
try:
    import transformers
except Exception:
    transformers = None

try:
    import whisper as _openai_whisper
except Exception:
    _openai_whisper = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import scipy.io.wavfile as _scipy_wav
except Exception:
    _scipy_wav = None

try:
    from .tokenizer import NLLBTokenizerWrapper, normalize_glottal
except Exception:
    NLLBTokenizerWrapper = None  # type: ignore[misc,assignment]
    normalize_glottal = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────────

MODEL_FALLBACKS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "whisper": "openai/whisper-tiny",
    "vits": "facebook/mms-tts-ceb",
}


def _get_device() -> str:
    """Return ``'cuda'`` when a GPU is available, otherwise ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── Metadata dataclasses ───────────────────────────────────────────────────

@dataclass
class HuggingFaceAdapter:
    """A lightweight Hugging Face-style adapter for model metadata and runtime selection."""
    name: str
    task: str = "translation"
    source: str = "huggingface"
    revision: str = "main"
    metadata: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "source": self.source,
            "revision": self.revision,
            "metadata": self.metadata,
        }


@dataclass
class RuntimeStatus:
    """A light-weight runtime descriptor for model availability."""
    model_name: str
    status: str
    backend: str
    adapter_name: Optional[str] = None


# ── Translation ────────────────────────────────────────────────────────────

@dataclass
class TranslationModel:
    """NLLB-based translation model.

    When ``use_huggingface`` is *True* (the default) and the ``transformers``
    library is installed, this class loads ``facebook/nllb-200-distilled-600M``
    and performs real seq2seq translation.  The ``target_lang`` field controls
    which NLLB language token is forced (default: ``ceb_Latn`` for Cebuano).
    """

    model_name: str = "nllb"
    device: str = ""
    use_huggingface: bool = True
    adapter: Optional[HuggingFaceAdapter] = None
    target_lang: str = "ceb_Latn"
    source_lang: str = "eng_Latn"

    # Lazily initialised internals (not part of the dataclass signature)
    _hf_model: object = field(default=None, init=False, repr=False)
    _tok_wrapper: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.device:
            self.device = _get_device()

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load the NLLB model + tokenizer.  Returns *True* on success."""
        if self._hf_model is not None:
            return True
        if transformers is None:
            log.warning("transformers library not installed — cannot load NLLB")
            return False
        model_id = (
            self.adapter.name
            if self.adapter
            else MODEL_FALLBACKS.get(self.model_name, self.model_name)
        )
        try:
            log.info("Loading NLLB model %s on %s …", model_id, self.device)
            self._tok_wrapper = NLLBTokenizerWrapper(
                model_id=model_id,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )
            self._hf_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
            if self.device == "cuda":
                self._hf_model.to("cuda")
            log.info("NLLB model loaded successfully.")
            return True
        except Exception as exc:
            log.error("Failed to load NLLB model: %s", exc)
            self._hf_model = None
            self._tok_wrapper = None
            return False

    # -- public API ----------------------------------------------------------

    def translate(self, text: str) -> str:
        if not text:
            return ""

        if self.use_huggingface and self._ensure_loaded():
            try:
                inputs = self._tok_wrapper.encode(text)
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                forced_bos = self._tok_wrapper.get_target_token_id()
                outputs = self._hf_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=256,
                )
                decoded = self._tok_wrapper.decode(outputs[0])
                return decoded
            except Exception as exc:
                log.error("Translation failed: %s", exc)
                return f"Translation error: {exc}"

        # Offline / fallback placeholder
        return f"[offline] placeholder for: {text}"

    def get_runtime_status(self) -> dict:
        adapter_name = (
            self.adapter.name
            if self.adapter
            else MODEL_FALLBACKS.get(self.model_name, self.model_name)
        )
        status = "ready" if self.use_huggingface else "fallback"
        return {
            "model_name": self.model_name,
            "status": status,
            "backend": "huggingface" if self.use_huggingface else "prototype",
            "adapter_name": adapter_name,
            "device": self.device,
        }


# ── Speech (ASR + TTS) ────────────────────────────────────────────────────

@dataclass
class SpeechModel:
    """Whisper / MMS-TTS speech pipeline.

    * **ASR** — primary: ``transformers`` Whisper pipeline; fallback:
      ``openai-whisper``.
    * **TTS** — primary: ``facebook/mms-tts-eng`` via ``transformers``;
      fallback: ``pyttsx3``.
    """

    model_name: str = "whisper"
    device: str = ""
    use_huggingface: bool = True
    adapter: Optional[HuggingFaceAdapter] = None

    _asr_pipeline: object = field(default=None, init=False, repr=False)
    _tts_model: object = field(default=None, init=False, repr=False)
    _tts_tokenizer: object = field(default=None, init=False, repr=False)
    _whisper_fallback: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.device:
            self.device = _get_device()

    # ── ASR ────────────────────────────────────────────────────────────────

    def _ensure_asr_loaded(self) -> bool:
        if self._asr_pipeline is not None:
            return True
        if transformers is not None:
            try:
                model_id = MODEL_FALLBACKS.get("whisper", "openai/whisper-tiny")
                if self.adapter and "whisper" in self.adapter.name.lower():
                    model_id = self.adapter.name
                log.info("Loading Whisper ASR pipeline (%s) on %s …", model_id, self.device)
                device_idx = 0 if self.device == "cuda" else -1
                self._asr_pipeline = transformers.pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    device=device_idx,
                )
                log.info("Whisper ASR pipeline loaded.")
                return True
            except Exception as exc:
                log.warning("transformers ASR pipeline failed: %s — trying openai-whisper", exc)
        # fallback to openai-whisper
        if _openai_whisper is not None:
            try:
                log.info("Loading openai-whisper (tiny) on %s …", self.device)
                self._whisper_fallback = _openai_whisper.load_model("tiny", device=self.device)
                log.info("openai-whisper loaded.")
                return True
            except Exception as exc:
                log.error("openai-whisper load failed: %s", exc)
        return False

    def transcribe(self, audio_path: str) -> str:
        adapter_name = self.adapter.name if self.adapter else self.model_name

        if self.use_huggingface and self._ensure_asr_loaded():
            # primary: transformers pipeline
            if self._asr_pipeline is not None:
                try:
                    result = self._asr_pipeline(audio_path)
                    return result.get("text", "")
                except Exception as exc:
                    log.error("ASR pipeline error: %s", exc)
                    return f"ASR error: {exc}"
            # fallback: openai-whisper
            if self._whisper_fallback is not None:
                try:
                    result = self._whisper_fallback.transcribe(audio_path)
                    return result.get("text", "")
                except Exception as exc:
                    log.error("openai-whisper error: %s", exc)
                    return f"ASR error: {exc}"

        return f"[offline] placeholder transcription for {audio_path} using {adapter_name.upper()}"

    # ── TTS ────────────────────────────────────────────────────────────────

    def _ensure_tts_loaded(self) -> bool:
        if self._tts_model is not None:
            return True
        if transformers is not None:
            try:
                model_id = MODEL_FALLBACKS.get("vits", "facebook/mms-tts-ceb")
                if self.adapter and "tts" in self.adapter.name.lower():
                    model_id = self.adapter.name
                log.info("Loading MMS-TTS model (%s) …", model_id)
                self._tts_model = transformers.VitsModel.from_pretrained(model_id)
                self._tts_tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
                if self.device == "cuda":
                    self._tts_model.to("cuda")
                log.info("MMS-TTS model loaded.")
                return True
            except Exception as exc:
                log.warning("MMS-TTS load failed: %s", exc)
        return False

    def synthesize(self, text: str) -> str:
        """Synthesize speech and return the path to a WAV file."""
        adapter_name = self.adapter.name if self.adapter else self.model_name

        # Primary: MMS-TTS via transformers
        if self.use_huggingface and self._ensure_tts_loaded():
            try:
                inputs = self._tts_tokenizer(text, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    output = self._tts_model(**inputs)
                waveform = output.waveform[0].cpu().numpy()
                sample_rate = self._tts_model.config.sampling_rate

                fd, path = tempfile.mkstemp(suffix=".wav", prefix="butuanon_tts_")
                os.close(fd)

                if _scipy_wav is not None:
                    _scipy_wav.write(path, rate=sample_rate, data=waveform)
                else:
                    # Bare-bones WAV writer when scipy is not available
                    import struct, wave
                    import numpy as np
                    pcm = (waveform * 32767).astype(np.int16)
                    with wave.open(path, "w") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(pcm.tobytes())

                log.info("TTS audio written to %s", path)
                return path
            except Exception as exc:
                log.error("MMS-TTS error: %s", exc)

        # Fallback: pyttsx3 (system voice engine)
        if pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="butuanon_tts_")
                os.close(fd)
                engine.save_to_file(text, path)
                engine.runAndWait()
                return path
            except Exception as exc:
                log.error("pyttsx3 TTS error: %s", exc)

        return f"[offline] placeholder audio for '{text}' using {adapter_name.upper()}"


# ── Registry ───────────────────────────────────────────────────────────────

class ModelRegistry:
    """Registry for translation and speech backends supporting the Butuanon pipeline."""

    def __init__(self) -> None:
        self._models: Dict[str, Union[TranslationModel, SpeechModel]] = {}

    def register(self, name: str, model: Union[TranslationModel, SpeechModel]) -> None:
        self._models[name] = model

    def get(self, name: str) -> Optional[Union[TranslationModel, SpeechModel]]:
        return self._models.get(name)

    def list_models(self) -> List[str]:
        return sorted(self._models.keys())

    def get_for_task(self, task: str) -> List[Union[TranslationModel, SpeechModel]]:
        task = task.lower()
        return [
            model
            for model in self._models.values()
            if (isinstance(model, TranslationModel) and task == "translation")
            or (isinstance(model, SpeechModel) and task in {"asr", "tts"})
        ]


def create_default_model_registry(
    target_lang: str = "ceb_Latn",
    source_lang: str = "eng_Latn",
) -> ModelRegistry:
    """Create a registry pre-loaded with real HF model adapters.

    Models are lazily loaded on first inference call — registering them here
    is cheap.
    """
    registry = ModelRegistry()

    registry.register(
        "nllb",
        TranslationModel(
            model_name="nllb",
            use_huggingface=True,
            target_lang=target_lang,
            source_lang=source_lang,
            adapter=HuggingFaceAdapter(
                name=MODEL_FALLBACKS["nllb"],
                task="translation",
            ),
        ),
    )

    registry.register(
        "whisper",
        SpeechModel(
            model_name="whisper",
            use_huggingface=True,
            adapter=HuggingFaceAdapter(
                name=MODEL_FALLBACKS["whisper"],
                task="asr",
            ),
        ),
    )

    registry.register(
        "vits",
        SpeechModel(
            model_name="vits",
            use_huggingface=True,
            adapter=HuggingFaceAdapter(
                name=MODEL_FALLBACKS["vits"],
                task="tts",
            ),
        ),
    )

    return registry
