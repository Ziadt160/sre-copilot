"""Embedding model specs and the Embedder protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from sre_shared.models import EmbeddingModelName

Vectors = NDArray[np.float32]


@dataclass(frozen=True)
class EmbeddingSpec:
    name: EmbeddingModelName
    hf_id: str
    revision: str | None
    dims: int
    column: str
    """Column in `chunks` holding this model's vectors."""
    query_prefix: str
    """Prepended to queries only (BGE v1.5 retrieval instruction); documents are embedded as is."""
    max_tokens: int
    """Model input limit including special tokens."""
    backend: Literal["onnx", "sentence-transformers"]


SPECS: dict[EmbeddingModelName, EmbeddingSpec] = {
    "gte-small": EmbeddingSpec(
        name="gte-small",
        hf_id="Supabase/gte-small",
        revision="93b36ff09519291b77d6000d2e86bd8565378086",
        dims=384,
        column="emb_gte_small",
        query_prefix="",
        max_tokens=512,
        backend="onnx",
    ),
    "bge-large": EmbeddingSpec(
        name="bge-large",
        hf_id="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        dims=1024,
        column="emb_bge_large",
        query_prefix="Represent this sentence for searching relevant passages: ",
        max_tokens=512,
        backend="sentence-transformers",
    ),
    "bge-m3": EmbeddingSpec(
        name="bge-m3",
        hf_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        dims=1024,
        column="emb_bge_m3",
        query_prefix="",
        max_tokens=8192,
        backend="sentence-transformers",
    ),
}

EMBEDDING_COLUMNS: frozenset[str] = frozenset(s.column for s in SPECS.values())


@runtime_checkable
class Embedder(Protocol):
    spec: EmbeddingSpec

    @property
    def revision(self) -> str:
        """Resolved model revision (commit SHA), recorded in version stamps."""
        ...

    def embed_documents(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        """L2-normalized float32 vectors, shape (len(texts), dims)."""
        ...

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> Vectors:
        """Like embed_documents, with the model's query prefix applied."""
        ...

    def count_tokens(self, text: str) -> int:
        """Tokens of `text` in this model's tokenizer, special tokens excluded, never truncated."""
        ...


def l2_normalize(x: NDArray[np.floating]) -> Vectors:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vectors = (x / norms).astype(np.float32)
    return out
