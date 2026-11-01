"""Batch embedding of chunks that lack a vector, through the `embedding_cache`."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from pgvector import HalfVector

from sre_ingest import store
from sre_shared.embed.base import Embedder
from sre_shared.models import EmbeddingModelName


@dataclass
class EmbedStats:
    model: str
    revision: str
    embedded: int = 0
    cache_hits: int = 0
    truncated: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_pending(
    conn: store.Conn,
    cv: int,
    model: EmbeddingModelName,
    embedder: Embedder,
    *,
    batch_size: int = 32,
    commit_every: int = 512,
) -> EmbedStats:
    import time

    t0 = time.perf_counter()
    stats = EmbedStats(model=model, revision=embedder.revision)
    pending = store.pending_embeddings(conn, cv, model)
    truncated_before = int(getattr(embedder, "truncated_total", 0))
    for lo in range(0, len(pending), commit_every):
        part = pending[lo : lo + commit_every]
        hashes = [text_hash(t) for _, t in part]
        cached = store.cache_lookup(conn, model, embedder.revision, sorted(set(hashes)))
        missing: dict[str, str] = {}
        for (_, text), h in zip(part, hashes, strict=True):
            if h not in cached:
                missing[h] = text
        stats.cache_hits += len(part) - len(missing)
        if missing:
            keys = list(missing)
            vecs = embedder.embed_documents([missing[k] for k in keys], batch_size=batch_size)
            store.cache_store(conn, model, embedder.revision, list(zip(keys, vecs, strict=True)))
            for k, v in zip(keys, vecs, strict=True):
                cached[k] = HalfVector(v)
            if not hasattr(embedder, "truncated_total"):
                limit = embedder.spec.max_tokens - 2
                stats.truncated += sum(embedder.count_tokens(missing[k]) > limit for k in keys)
        store.set_embeddings(
            conn, model, [(cid, cached[h]) for (cid, _), h in zip(part, hashes, strict=True)]
        )
        conn.commit()
        stats.embedded += len(part)
    stats.truncated += int(getattr(embedder, "truncated_total", 0)) - truncated_before
    stats.seconds = round(time.perf_counter() - t0, 3)
    return stats
