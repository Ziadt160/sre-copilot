"""Retriever: dense + keyword candidates in one SQL call, weighted RRF in Python.

Row access is governed by RLS: every statement runs inside `as_user(conn, claims)`, so a caller
only sees chunks in its own access groups. Request values reach SQL only as bound parameters;
the vector column comes from `SPECS`, and the SQL function re-checks it against a whitelist.

Optional parts (Phase 2): a query-type router (`route_by_query_type`), a reranker applied to the
top `rerank_candidates` after fusion and outside the transaction (on failure the fused order
stands), and a read pool on a lab replica (on a connection failure the read is retried on the
primary; retrieval is read-only, so that is always safe).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pgvector import HalfVector
from psycopg import OperationalError
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from sre_core.retrieval.base import Reranker, RerankerUnavailable
from sre_core.retrieval.fusion import Candidate, Fused, fuse_rrf
from sre_core.retrieval.router import route_request
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


log = logging.getLogger("sre_core.retrieval")


@dataclass
class _Found:
    version: int
    fused: list[Fused]
    by_id: dict[int, dict[str, Any]]
    t_sql: float
    t_fuse: float
    t_hydrate: float


class NoActiveCorpusError(LookupError):
    """No corpus version is active (or visible) and the request named none."""


class Retriever:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedders: Mapping[EmbeddingModelName, Embedder] | None = None,
        *,
        reranker: Reranker | None = None,
        read_pool: AsyncConnectionPool | None = None,
    ) -> None:
        """`embedders` maps model names to query embedders; missing ones load on first use
        via `sre_shared.embed.get_embedder` (the Edge Function in the demo environment).
        `reranker` serves requests that set `rerank_candidates`; `read_pool` (a lab replica)
        takes retrieval reads, with the primary `pool` as the fallback."""
        self._pool = pool
        self._read_pool = read_pool
        self._reranker = reranker
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
        route: str | None = None
        if req.route_by_query_type:
            req, route = route_request(req)
        spec = SPECS[req.embedding_model]

        qvec: HalfVector | None = None
        if req.use_dense:
            embedder = self._embedder(req.embedding_model)
            vectors = await asyncio.to_thread(embedder.embed_queries, [req.query])
            qvec = HalfVector(vectors[0])
        t_embed = time.perf_counter()

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

        n_fused = req.k
        if req.rerank_candidates and self._reranker is not None:
            n_fused = max(req.k, req.rerank_candidates)
        pools = [p for p in (self._read_pool, self._pool) if p is not None]
        found: _Found | None = None
        for attempt, pool in enumerate(pools):
            try:
                found = await self._search(pool, req, claims, params, n_fused)
                break
            except (OperationalError, PoolTimeout):
                if attempt == len(pools) - 1:
                    raise
                log.warning("read pool unavailable; retrieving from the primary")
        assert found is not None
        version, fused, by_id = found.version, found.fused, found.by_id
        t_sql, t_fuse, t_hydrate = found.t_sql, found.t_fuse, found.t_hydrate

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
        timings = {
            "embed": (t_embed - t_start) * 1000,
            "sql": ((t_sql - t_embed) + (t_hydrate - t_fuse)) * 1000,
            "fuse": (t_fuse - t_sql) * 1000,
        }
        reranked = False
        if req.rerank_candidates and self._reranker is not None and chunks:
            t_rerank = time.perf_counter()
            try:
                scores = await self._reranker.rerank(req.query, chunks)
                order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
                chunks = [chunks[i].model_copy(update={"score": float(scores[i])}) for i in order]
                reranked = True
            except RerankerUnavailable:
                log.warning("reranker unavailable; keeping the fused order")
            timings["rerank"] = (time.perf_counter() - t_rerank) * 1000
        chunks = chunks[: req.k]
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return RetrievalResult(
            request=req,
            corpus_version_id=version,
            chunks=chunks,
            timings_ms=timings,
            route=route,
            reranked=reranked,
        )

    async def _search(
        self,
        pool: AsyncConnectionPool,
        req: RetrievalRequest,
        claims: Claims,
        params: dict[str, Any],
        n_fused: int,
    ) -> _Found:
        """Candidates, fusion and hydration in one RLS-scoped transaction on `pool`."""
        max_scan_tuples = load_retrieval_defaults().hnsw.max_scan_tuples
        async with pool.connection() as pooled, as_user(pooled, claims) as conn:
            version = req.corpus_version_id
            if version is None:
                row = await (
                    await conn.execute("select sre.active_corpus_version() as v")
                ).fetchone()
                version = row["v"] if row else None
                if version is None:
                    raise NoActiveCorpusError("no active corpus version")
            params = {**params, "version": version}

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
            )[:n_fused]
            t_fuse = time.perf_counter()

            by_id: dict[int, dict[str, Any]] = {}
            if fused:
                hyd = await conn.execute(
                    _HYDRATE_SQL, {"ids": [f.candidate.chunk_id for f in fused]}
                )
                by_id = {r["id"]: r for r in await hyd.fetchall()}
            t_hydrate = time.perf_counter()
        return _Found(version, fused, by_id, t_sql, t_fuse, t_hydrate)
