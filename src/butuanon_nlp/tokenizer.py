import re
from typing import Iterable, List


class GlottalAwareTokenizer:
    """A simple tokenizer that preserves apostrophes and hyphens inside words."""

    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[^\w\s]", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [token for token in self._TOKEN_PATTERN.findall(text) if token.strip()]

    def encode(self, text: str) -> List[str]:
        return self.tokenize(text)

    def decode(self, tokens: Iterable[str]) -> str:
        return " ".join(token for token in tokens if self._is_word_token(token))

    @staticmethod
    def _is_word_token(token: str) -> bool:
        return bool(token) and (token.isalnum() or "'" in token or "-" in token or "’" in token)
