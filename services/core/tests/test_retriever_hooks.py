"""Retriever hooks added for Phase 2: query-type router, reranker, read pool with fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.retrieval import Retriever
from sre_core.retrieval.base import RerankerUnavailable
from sre_core.retrieval.router import route_request
from sre_shared.config import RouteConfig, RouterConfig
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import ANONYMOUS, RetrievalRequest, RetrievedChunk

if TYPE_CHECKING:
    from conftest import Corpus

POD_Q = "smallest deployable unit pods run containers"


def test_router_sends_literals_to_the_literal_route() -> None:
    router = RouterConfig(
        literal=RouteConfig(use_keyword=True, w_keyword=0.5, rrf_k=10),
        default=RouteConfig(use_keyword=False),
    )
    req = RetrievalRequest(query="pod in CrashLoopBackOff", route_by_query_type=True)
    routed, name = route_request(req, router)
    assert (
        name == "literal" and routed.use_keyword and (routed.w_keyword, routed.rrf_k) == (0.5, 10)
    )
    assert not routed.route_by_query_type
    plain, name = route_request(RetrievalRequest(query="how do pods restart"), router)
    assert name == "default" and plain.use_dense and not plain.use_keyword


class ReverseReranker:
    name = "reverse"

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        self.seen = [c.chunk_id for c in chunks]
        return [float(i) for i in range(len(chunks))]  # last fused chunk scores highest


class DownReranker:
    name = "down"

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        raise RerankerUnavailable("gpu server down")


@pytest.mark.integration
async def test_reranker_reorders_the_top_candidates(
    pool: AsyncConnectionPool, fake_embedder: FakeEmbedder, corpus: Corpus
) -> None:
    plain = Retriever(pool, {"gte-small": fake_embedder})
    fused = await plain.retrieve(RetrievalRequest(query=POD_Q, k=3), ANONYMOUS)
    reranker = ReverseReranker()
    r = Retriever(pool, {"gte-small": fake_embedder}, reranker=reranker)
    res = await r.retrieve(RetrievalRequest(query=POD_Q, k=2, rerank_candidates=3), ANONYMOUS)
    assert reranker.seen == [c.chunk_id for c in fused.chunks]  # the 3 fused candidates
    assert [c.chunk_id for c in res.chunks] == [fused.chunks[2].chunk_id, fused.chunks[1].chunk_id]
    assert res.reranked and "rerank" in res.timings_ms


@pytest.mark.integration
async def test_reranker_failure_keeps_the_fused_order(
    pool: AsyncConnectionPool, fake_embedder: FakeEmbedder, corpus: Corpus
) -> None:
    plain = await Retriever(pool, {"gte-small": fake_embedder}).retrieve(
        RetrievalRequest(query=POD_Q, k=2), ANONYMOUS
    )
    r = Retriever(pool, {"gte-small": fake_embedder}, reranker=DownReranker())
    res = await r.retrieve(RetrievalRequest(query=POD_Q, k=2, rerank_candidates=3), ANONYMOUS)
    assert [c.chunk_id for c in res.chunks] == [c.chunk_id for c in plain.chunks]
    assert not res.reranked


@pytest.mark.integration
async def test_routing_is_reported(
    pool: AsyncConnectionPool, fake_embedder: FakeEmbedder, corpus: Corpus
) -> None:
    r = Retriever(pool, {"gte-small": fake_embedder})
    res = await r.retrieve(RetrievalRequest(query=POD_Q, route_by_query_type=True), ANONYMOUS)
    assert res.route == "default" and not res.request.use_keyword


@pytest.mark.integration
async def test_dead_read_pool_falls_back_to_the_primary(
    pool: AsyncConnectionPool, fake_embedder: FakeEmbedder, corpus: Corpus
) -> None:
    dead = AsyncConnectionPool(
        "postgresql://postgres:postgres@127.0.0.1:1/postgres",
        min_size=1,
        max_size=1,
        timeout=1.0,
        kwargs={"connect_timeout": 1},
        open=False,
    )
    await dead.open(wait=False)
    try:
        r = Retriever(pool, {"gte-small": fake_embedder}, read_pool=dead)
        res = await r.retrieve(RetrievalRequest(query=POD_Q, k=2), ANONYMOUS)
        assert res.chunks and res.chunks[0].source_path == "content/en/docs/concepts/pods.md"
    finally:
        await dead.close(timeout=1.0)
