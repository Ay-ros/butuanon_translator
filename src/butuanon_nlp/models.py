from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import os
import tempfile

# Optional heavy imports — lazily loaded inside methods
try:
    import transformers
except Exception:
    transformers = None

try:
    import whisper
except Exception:
    whisper = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


MODEL_FALLBACKS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "whisper": "openai/whisper-tiny",
    "vits": "facebook/mms-tts-eng",
}


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


@dataclass
class TranslationModel:
    """Placeholder integration point for an NLLB-style translation model."""
    model_name: str = "nllb"
    device: str = "cpu"
    use_huggingface: bool = False
    adapter: Optional[HuggingFaceAdapter] = None

    def translate(self, text: str) -> str:
        if not text:
            return ""
        adapter_name = self.adapter.name if self.adapter else self.model_name
        # If Hugging Face is requested and available, run a real pipeline (lazy)
        if self.use_huggingface and self.adapter and transformers is not None:
            model_id = self.adapter.name or MODEL_FALLBACKS.get(self.model_name, self.model_name)
            # create a translation pipeline lazily
            pipeline_error = None
            if not hasattr(self, "_hf_pipeline") or self._hf_pipeline is None:
                try:
                    self._hf_pipeline = transformers.pipeline(
                        "text2text-generation",
                        model=model_id,
                        device=-1,  # CPU
                    )
                except Exception as e:
                    pipeline_error = e
                    self._hf_pipeline = None
            if hasattr(self, "_hf_pipeline") and self._hf_pipeline is not None:
                try:
                    result = self._hf_pipeline(text, max_length=512)
                    # pipeline may return 'translation_text' or 'generated_text' keys
                    if isinstance(result, list) and result:
                        translated = (
                            result[0].get("translation_text")
                            or result[0].get("generated_text")
                            or result[0].get("text")
                        )
                    else:
                        translated = str(result)
                    return translated or ""
                except Exception as e:
                    pipeline_error = e
            # If pipeline unavailable or failed, try a lower-level seq2seq fallback
            try:
                AutoTokenizer = transformers.AutoTokenizer
                AutoModelForSeq2SeqLM = transformers.AutoModelForSeq2SeqLM
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                # force CPU
                try:
                    model.to("cpu")
                except Exception:
                    pass
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model.generate(**inputs, max_new_tokens=200)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return decoded
            except Exception as e2:
                # prefer pipeline_error if present for diagnostics
                if pipeline_error:
                    return f"Translation failed: {pipeline_error}; fallback failed: {e2}"
                return f"Translation fallback failed: {e2}"

        # fallback prototype output
        return f"Butuanon-style output: 'Gikaon ka sa Butuanon?'"

    def get_runtime_status(self) -> dict:
        adapter_name = self.adapter.name if self.adapter else MODEL_FALLBACKS.get(self.model_name, self.model_name)
        status = "ready" if self.use_huggingface else "fallback"
        return {
            "model_name": self.model_name,
            "status": status,
            "backend": "huggingface" if self.use_huggingface else "prototype",
            "adapter_name": adapter_name,
        }


@dataclass
class SpeechModel:
    """Placeholder integration point for Whisper/VITS-style speech pipelines."""
    model_name: str = "whisper"
    device: str = "cpu"
    use_huggingface: bool = False
    adapter: Optional[HuggingFaceAdapter] = None

    def transcribe(self, audio_path: str) -> str:
        adapter_name = self.adapter.name if self.adapter else self.model_name
        # If Whisper is available, run real transcription (lazy)
        if self.use_huggingface and whisper is not None:
            # load model lazily
            if not hasattr(self, "_whisper_model") or self._whisper_model is None:
                try:
                    # use a small model for CPU friendliness
                    self._whisper_model = whisper.load_model("tiny", device=self.device)
                except Exception as e:
                    return f"Failed to load Whisper model: {e}"
            try:
                res = self._whisper_model.transcribe(audio_path)
                return res.get("text", "")
            except Exception as e:
                return f"Transcription failed: {e}"

        return f"Placeholder transcription for {audio_path} using {adapter_name.upper()} (ASR pipeline)."

    def synthesize(self, text: str) -> str:
        adapter_name = self.adapter.name if self.adapter else self.model_name
        # Only run the real TTS engine when the model is explicitly set to use_huggingface
        # or when an adapter is provided (indicating a configured backend).
        if (self.use_huggingface or (self.adapter is not None)) and pyttsx3 is not None:
            try:
                engine = pyttsx3.init()
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="butuanon_tts_")
                os.close(fd)
                engine.save_to_file(text, path)
                engine.runAndWait()
                return path
            except Exception as e:
                return f"TTS failed: {e}"

        return f"Placeholder audio synthesis for '{text}' using {adapter_name.upper()} (TTS pipeline)."


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


def create_default_model_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("nllb", TranslationModel(model_name="nllb"))
    registry.register("whisper", SpeechModel(model_name="whisper"))
    registry.register("vits", SpeechModel(model_name="vits"))
    return registry
