"""Interfaces inside the answer path that more than one module implements or calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from sre_shared.models import ANONYMOUS, AnswerRequest, AnswerResult, Claims, RetrievedChunk


class AnswererLike(Protocol):
    """What callers (the API, the answer eval runner) need from the answer path. The concrete
    answerer comes from `sre_core.answer.build_answerer(...)`."""

    async def answer(self, req: AnswerRequest, claims: Claims = ANONYMOUS) -> AnswerResult: ...


class Compressor(Protocol):
    """Shrinks the retrieved chunks to the context budget before generation (design doc:
    drop sentences unrelated to the question, merge overlapping chunks, fill a fixed token
    share). Returns chunks in prompt order; a chunk keeps its `chunk_id`, so citations still
    resolve, while its `body` may be cut down."""

    name: str

    async def compress(
        self, query: str, chunks: Sequence[RetrievedChunk], token_budget: int
    ) -> list[RetrievedChunk]: ...
