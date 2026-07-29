"""Glottal-aware tokenizer with optional NLLB SentencePiece integration.

This module provides two layers of tokenization:

1. **GlottalAwareTokenizer** — a lightweight regex tokenizer that preserves
   apostrophes and hyphens (common glottal-stop markers in Butuanon).  It is
   used for phonetic pre-processing, corpus analysis, and display.

2. **NLLBTokenizerWrapper** — wraps the real ``AutoTokenizer`` from Hugging
   Face so that glottal normalisation is applied *before* SentencePiece
   encoding.  This is what the translation model uses at inference time.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

# Lazy import — only needed when the NLLB wrapper is used.
try:
    import transformers as _transformers
except Exception:
    _transformers = None


# ── glottal marker normalisation ────────────────────────────────────────────

# Characters that communities use to mark the glottal stop in Butuanon text.
_GLOTTAL_MARKERS = ("'", "\u2018", "\u2019", "\u02BC")  # ' ' ' ʼ

# The canonical glottal-stop placeholder that the tokenizer preserves.  Using
# the typographic right-single-quote (') because NLLB's SentencePiece model
# keeps it as a visible character, unlike a plain ASCII apostrophe which can
# be absorbed into a larger subword.
GLOTTAL_CANONICAL = "\u2019"  # '


def normalize_glottal(text: str) -> str:
    """Replace all known glottal markers with the canonical form."""
    for marker in _GLOTTAL_MARKERS:
        text = text.replace(marker, GLOTTAL_CANONICAL)
    return text


# ── GlottalAwareTokenizer (lightweight / display) ──────────────────────────

class GlottalAwareTokenizer:
    """A simple tokenizer that preserves apostrophes and hyphens inside words.

    This tokenizer is used for:
    - Displaying token-level breakdowns in the UI
    - Phonetic pre-processing (glottal cue insertion)
    - Corpus statistics (token counts, vocabulary extraction)

    It is **not** used for model inference — the ``NLLBTokenizerWrapper``
    handles that.
    """

    _TOKEN_PATTERN = re.compile(
        r"[A-Za-z0-9\u2019']+(?:['\u2019\-][A-Za-z0-9\u2019']+)*|[^\w\s]",
        re.UNICODE,
    )

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        normalized = normalize_glottal(text)
        return [tok for tok in self._TOKEN_PATTERN.findall(normalized) if tok.strip()]

    def encode(self, text: str) -> List[str]:
        return self.tokenize(text)

    def decode(self, tokens: Iterable[str]) -> str:
        return " ".join(tok for tok in tokens if self._is_word_token(tok))

    @staticmethod
    def _is_word_token(token: str) -> bool:
        return bool(token) and (
            token.isalnum()
            or "'" in token
            or "-" in token
            or GLOTTAL_CANONICAL in token
        )


# ── NLLBTokenizerWrapper (model inference) ─────────────────────────────────

class NLLBTokenizerWrapper:
    """Thin wrapper around the NLLB ``AutoTokenizer`` that applies
    glottal-stop normalisation before SentencePiece encoding.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier, e.g. ``facebook/nllb-200-distilled-600M``.
    source_lang : str
        BCP-47 source language code, default ``eng_Latn``.
    target_lang : str
        BCP-47 target language code, default ``ceb_Latn`` (Cebuano — closest
        to Butuanon in NLLB's supported set).
    """

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        source_lang: str = "eng_Latn",
        target_lang: str = "ceb_Latn",
    ) -> None:
        if _transformers is None:
            raise RuntimeError(
                "The `transformers` library is required for NLLBTokenizerWrapper. "
                "Install it with: pip install transformers sentencepiece"
            )
        self.model_id = model_id
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._hf_tokenizer = _transformers.AutoTokenizer.from_pretrained(
            model_id, src_lang=source_lang
        )
        self._glottal = GlottalAwareTokenizer()

    # -- public helpers -------------------------------------------------------

    @property
    def hf_tokenizer(self):
        """Access the underlying HF tokenizer (for ``forced_bos_token_id`` etc.)."""
        return self._hf_tokenizer

    def get_target_token_id(self) -> int:
        """Return the token ID that NLLB uses for the configured target language."""
        return self._hf_tokenizer.convert_tokens_to_ids(self.target_lang)

    def glottal_tokens(self, text: str) -> List[str]:
        """Return the lightweight glottal-aware token list (for display)."""
        return self._glottal.tokenize(text)

    # -- encode / decode wrappers --------------------------------------------

    def encode(self, text: str, **kwargs) -> Dict:
        """Normalise glottal markers, then run the HF tokenizer.

        Returns the dictionary expected by ``model.generate()``.
        """
        text = normalize_glottal(text)
        return self._hf_tokenizer(text, return_tensors="pt", **kwargs)

    def decode(self, token_ids, **kwargs) -> str:
        """Decode token IDs back to text."""
        return self._hf_tokenizer.decode(token_ids, skip_special_tokens=True, **kwargs)
