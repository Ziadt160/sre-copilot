"""A deterministic, model-free embedder for tests.

Hashing trick over lowercase word tokens, so texts that share words get similar vectors and a
query equal to a chunk's text scores 1.0. It is not a quality signal; it only makes dense-leg
tests meaningful without loading a model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from sre_shared.embed.base import SPECS, EmbeddingSpec, Vectors, l2_normalize
from sre_shared.models import EmbeddingModelName

_WORD = re.compile(r"[A-Za-z0-9_.-]+")


class FakeEmbedder:
    def __init__(self, name: EmbeddingModelName = "gte-small") -> None:
        self.spec: EmbeddingSpec = replace(SPECS[name], revision="fake")

    @property
    def revision(self) -> str:
        return "fake"

    def count_tokens(self, text: str) -> int:
        return len(_WORD.findall(text))

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        return self._embed(list(texts))

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        return self._embed(list(texts))

    def _embed(self, texts: list[str]) -> Vectors:
        dims = self.spec.dims
        out = np.zeros((len(texts), dims), dtype=np.float32)
        for row, text in enumerate(texts):
            words = _WORD.findall(text.lower())
            if not words:
                out[row, 0] = 1.0
                continue
            for w in words:
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                out[row, h % dims] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(out)
