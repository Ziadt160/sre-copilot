"""Interfaces inside retrieval that more than one module implements or calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from sre_shared.models import RetrievedChunk


class RerankerUnavailable(RuntimeError):
    """The reranker could not score this request; callers keep the fused order (design doc,
    degradation ladder step 1)."""


class Reranker(Protocol):
    """Scores (query, chunk) pairs; higher is more relevant. Scores are only compared within
    one call. Implementations raise `RerankerUnavailable` instead of returning partial scores."""

    name: str

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]: ...
