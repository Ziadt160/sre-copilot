"""Retriever: dense + keyword candidates in one SQL call, weighted RRF in Python.

Row access is governed by RLS: every statement runs inside `as_user(conn, claims)`, so a caller
only sees chunks in its own access groups. Request values reach SQL only as bound parameters;
the vector column comes from `SPECS`, and the SQL function re-checks it against a whitelist.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from pgvector import HalfVector
from psycopg_pool import AsyncConnectionPool

from sre_core.retrieval.fusion import Candidate, fuse_rrf
from sre_shared.config import load_retrieval_defaults
from sre_shared.db import as_user
from sre_shared.embed import SPECS, Embedder, get_embedder
from sre_shared.models import (
    ANONYMOUS,
    Claims,
    EmbeddingModelName,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from sre_shared.settings import Environment, get_settings

_CANDIDATES_SQL = (
    "select chunk_id, dense_rank, keyword_rank, dense_similarity, keyword_score"
    " from public.search_candidates("
    "%(column)s, %(qvec)s::halfvec, %(query)s, %(ts)s, %(n)s, %(version)s,"
    " %(k8s)s::text[], %(doc)s::text[], %(dense)s, %(keyword)s, %(exact)s, %(kw_match)s)"
)

_HYDRATE_SQL = (
    "select id, chunk_key, source_path, k8s_version, doc_type, title, section_path, url,"
    " header, body from public.chunks where id = any(%(ids)s)"
)


class NoActiveCorpusError(LookupError):
    """No corpus version is active (or visible) and the request named none."""


class Retriever:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedders: Mapping[EmbeddingModelName, Embedder] | None = None,
    ) -> None:
        """`embedders` maps model names to query embedders; missing ones load on first use
        via `sre_shared.embed.get_embedder` (the Edge Function in the demo environment)."""
        self._pool = pool
        self._embedders = dict(embedders or {})

    def _embedder(self, name: EmbeddingModelName) -> Embedder:
        emb = self._embedders.get(name)
        if emb is None:
            if name == "gte-small" and get_settings().sre_env == Environment.DEMO:
                from sre_core.retrieval.edge import EdgeFunctionEmbedder

                emb = EdgeFunctionEmbedder()
            else:
                emb = get_embedder(name)
            self._embedders[name] = emb
        return emb

    async def retrieve(
        self,
        req: RetrievalRequest,
        claims: Claims = ANONYMOUS,
        *,
        exact: bool = False,
    ) -> RetrievalResult:
        """Hybrid retrieval under the caller's RLS context.

        `exact=True` runs the dense leg as an exact scan (the SQL orders by an expression the HNSW
        index cannot serve) so evals can measure
        ANN recall against it; production never sets it.
        """
        t_start = time.perf_counter()
        spec = SPECS[req.embedding_model]

        qvec: HalfVector | None = None
        if req.use_dense:
            embedder = self._embedder(req.embedding_model)
            vectors = await asyncio.to_thread(embedder.embed_queries, [req.query])
            qvec = HalfVector(vectors[0])
        t_embed = time.perf_counter()

        max_scan_tuples = load_retrieval_defaults().hnsw.max_scan_tuples
        params: dict[str, Any] = {
            "column": spec.column,
            "qvec": qvec,
            "query": req.query,
            "ts": req.ts_config,
            "n": req.candidates_per_leg,
            "k8s": req.filters.k8s_versions,
            "doc": [d.value for d in req.filters.doc_types] if req.filters.doc_types else None,
            "dense": req.use_dense,
            "keyword": req.use_keyword,
            "exact": exact,
            "kw_match": req.keyword_match,
        }

        async with self._pool.connection() as pooled, as_user(pooled, claims) as conn:
            version = req.corpus_version_id
            if version is None:
                row = await (
                    await conn.execute("select sre.active_corpus_version() as v")
                ).fetchone()
                version = row["v"] if row else None
                if version is None:
                    raise NoActiveCorpusError("no active corpus version")
            params["version"] = version

            await conn.execute(
                "select set_config('hnsw.ef_search', %s, true),"
                " set_config('hnsw.iterative_scan', %s, true),"
                " set_config('hnsw.max_scan_tuples', %s, true)",
                [str(req.ef_search), req.iterative_scan, str(max_scan_tuples)],
            )

            cur = await conn.execute(_CANDIDATES_SQL, params)
            candidates = [
                Candidate(
                    chunk_id=r["chunk_id"],
                    dense_rank=r["dense_rank"],
                    keyword_rank=r["keyword_rank"],
                    dense_similarity=r["dense_similarity"],
                    keyword_score=r["keyword_score"],
                )
                for r in await cur.fetchall()
            ]
            t_sql = time.perf_counter()

            fused = fuse_rrf(
                candidates, w_dense=req.w_dense, w_keyword=req.w_keyword, rrf_k=req.rrf_k
            )[: req.k]
            t_fuse = time.perf_counter()

            by_id: dict[int, dict[str, Any]] = {}
            if fused:
                hyd = await conn.execute(
                    _HYDRATE_SQL, {"ids": [f.candidate.chunk_id for f in fused]}
                )
                by_id = {r["id"]: r for r in await hyd.fetchall()}
            t_hydrate = time.perf_counter()

        chunks = []
        for f in fused:
            r = by_id.get(f.candidate.chunk_id)
            if r is None:  # vanished between the two statements; cannot happen in one transaction
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=r["id"],
                    chunk_key=r["chunk_key"],
                    source_path=r["source_path"],
                    k8s_version=r["k8s_version"],
                    doc_type=r["doc_type"],
                    title=r["title"],
                    section_path=r["section_path"],
                    url=r["url"],
                    header=r["header"],
                    body=r["body"],
                    score=f.score,
                    dense_rank=f.candidate.dense_rank,
                    keyword_rank=f.candidate.keyword_rank,
                    dense_similarity=f.candidate.dense_similarity,
                    keyword_score=f.candidate.keyword_score,
                )
            )
        t_end = time.perf_counter()
        return RetrievalResult(
            request=req,
            corpus_version_id=version,
            chunks=chunks,
            timings_ms={
                "embed": (t_embed - t_start) * 1000,
                "sql": ((t_sql - t_embed) + (t_hydrate - t_fuse)) * 1000,
                "fuse": (t_fuse - t_sql) * 1000,
                "total": (t_end - t_start) * 1000,
            },
        )
