"""Context compression (design doc): drop sentences unrelated to the question, group chunks of
one page section, then fill a fixed token share in rank order.

Token counts are `len(text) // 4`, the same estimate the quota guard in `sre_shared.llm` uses;
the embedder's exact tokenizer would count gte-small tokens, not the chat model's, so it is no
more exact for the budget that matters.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sre_shared.embed import Embedder
from sre_shared.llm import CHARS_PER_TOKEN
from sre_shared.models import RetrievedChunk

_FENCE = re.compile(r"(^```.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(\[])")


def count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def chunk_tokens(chunk: RetrievedChunk) -> int:
    return count_tokens(chunk.header) + count_tokens(chunk.body)


@dataclass(frozen=True)
class _Unit:
    text: str
    sep: str  # what precedes this unit when units are joined back together


def split_units(body: str) -> list[_Unit]:
    """Sentences, with each fenced code block kept whole as one unit."""
    units: list[_Unit] = []
    sep = ""
    for part in _FENCE.split(body):
        if not part.strip():
            continue
        if part.startswith("```"):
            units.append(_Unit(part.strip("\n"), sep or "\n\n"))
            sep = "\n\n"
            continue
        for para in re.split(r"\n\s*\n", part):
            para = para.strip()
            if not para:
                continue
            first = True
            for sent in _SENTENCE_END.split(para):
                sent = sent.strip()
                if sent:
                    units.append(_Unit(sent, (sep or "\n\n") if first and units else " "))
                    first = False
            sep = "\n\n"
    if units:
        units[0] = _Unit(units[0].text, "")
    return units


def _join(units: Sequence[_Unit]) -> str:
    out = ""
    for i, u in enumerate(units):
        out += (u.sep if i else "") + u.text
    return out


SectionKey = tuple[str, str, tuple[str, ...]]


def section_key(c: RetrievedChunk) -> SectionKey:
    return (c.source_path, c.k8s_version, tuple(c.section_path))


def group_adjacent(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Make chunks of the same page version and section adjacent, at the rank of their best
    member and in rank order. Chunks are not fused: each keeps its own `chunk_id` and body, so a
    citation traces to the chunk that holds the text. The prompt renders each run of
    same-section chunks as one source (`prompts.group_sources`)."""
    out: list[RetrievedChunk] = []
    done: set[SectionKey] = set()
    for c in chunks:
        key = section_key(c)
        if key in done:
            continue
        done.add(key)
        out.extend(m for m in chunks if section_key(m) == key)
    return out


def fill_budget(chunks: Sequence[RetrievedChunk], token_budget: int) -> list[RetrievedChunk]:
    """Take chunks in rank order while they fit. A first chunk larger than the whole budget is
    cut to it (header first, then body); a later chunk that does not fit is skipped (smaller
    ones after it may still fit)."""
    out: list[RetrievedChunk] = []
    used = 0
    for c in chunks:
        n = chunk_tokens(c)
        if used + n <= token_budget:
            out.append(c)
            used += n
        elif not out:
            room = token_budget * CHARS_PER_TOKEN
            header = c.header[:room]
            body = c.body[: max(room - len(header), 0)]
            out.append(c.model_copy(update={"header": header, "body": body}))
            used = token_budget
    return out


class NoCompressor:
    """Pass chunks through; the token budget still applies."""

    name = "none"

    async def compress(
        self, query: str, chunks: Sequence[RetrievedChunk], token_budget: int
    ) -> list[RetrievedChunk]:
        return fill_budget(chunks, token_budget)


class SentenceCompressor:
    """Keep sentences whose embedding similarity to the question is at least `threshold`, plus
    the best `min_per_chunk` of each chunk, in original order; group same-section chunks; fill
    the budget in rank order."""

    name = "sentence"

    def __init__(self, embedder: Embedder, threshold: float = 0.80, min_per_chunk: int = 2) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.min_per_chunk = min_per_chunk

    def _compress_sync(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        per_chunk = [split_units(c.body) for c in chunks]
        flat = [u.text for units in per_chunk for u in units]
        if flat:
            qv = self.embedder.embed_queries([query])[0]
            sims = self.embedder.embed_documents(flat) @ qv
        else:
            sims = np.zeros(0, dtype=np.float32)
        out: list[RetrievedChunk] = []
        pos = 0
        for c, units in zip(chunks, per_chunk, strict=True):
            s = sims[pos : pos + len(units)]
            pos += len(units)
            if not units:
                out.append(c)
                continue
            best = set(np.argsort(-s, kind="stable")[: self.min_per_chunk].tolist())
            keep = [u for i, u in enumerate(units) if s[i] >= self.threshold or i in best]
            out.append(c.model_copy(update={"body": _join(keep)}))
        return out

    async def compress(
        self, query: str, chunks: Sequence[RetrievedChunk], token_budget: int
    ) -> list[RetrievedChunk]:
        shrunk = await asyncio.to_thread(self._compress_sync, query, list(chunks))
        return fill_budget(group_adjacent(shrunk), token_budget)
