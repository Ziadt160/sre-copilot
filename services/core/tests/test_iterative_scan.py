"""Filtered dense queries under HNSW: iterative scan keeps returning K rows; without it a
selective filter starves the result when ef_search < K."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.retrieval import Retriever
from sre_shared.db import SEARCH_PATH_OPTION, create_pool
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import ANONYMOUS, RetrievalFilters, RetrievalRequest

pytestmark = pytest.mark.integration

K = 20


@pytest.fixture
async def index_only_retriever(
    pg_dsn: str, big_corpus: int, db: object, fake_embedder: FakeEmbedder
) -> AsyncIterator[Retriever]:
    """A retriever whose connections cannot fall back to a seq scan, so the HNSW index decides."""
    # The 200-row table is tiny; make the planner prefer HNSW by removing the btree alternatives
    # (filter index, unique key) for the duration of the test; restored below.
    await db.execute("drop index public.chunks_filter_idx")  # type: ignore[attr-defined]
    await db.execute(  # type: ignore[attr-defined]
        "alter table public.chunks drop constraint chunks_corpus_version_id_chunk_key_key"
    )
    pool: AsyncConnectionPool = create_pool(pg_dsn, min_size=1, max_size=2)
    pool.kwargs["options"] = f"{SEARCH_PATH_OPTION} -c enable_seqscan=off -c enable_bitmapscan=off"
    await pool.open()
    try:
        yield Retriever(pool, {"gte-small": fake_embedder})
    finally:
        await pool.close()
        await db.execute(  # type: ignore[attr-defined]
            "alter table public.chunks add constraint chunks_corpus_version_id_chunk_key_key"
            " unique (corpus_version_id, chunk_key)"
        )
        await db.execute(  # type: ignore[attr-defined]
            "create index chunks_filter_idx on public.chunks (corpus_version_id, k8s_version, doc_type)"
        )


def query(iterative: str, exact: bool = False) -> RetrievalRequest:
    return RetrievalRequest(
        query="widget3 gadget3 sprocket3 flange3 common words here",
        filters=RetrievalFilters(k8s_versions=["1.31"]),
        use_keyword=False,
        candidates_per_leg=K,
        k=K,
        ef_search=10,  # below K
        iterative_scan=iterative,  # type: ignore[arg-type]
    )


async def test_iterative_scan_returns_k_where_off_returns_fewer(
    index_only_retriever: Retriever,
) -> None:
    on = await index_only_retriever.retrieve(query("relaxed_order"), ANONYMOUS)
    strict = await index_only_retriever.retrieve(query("strict_order"), ANONYMOUS)
    off = await index_only_retriever.retrieve(query("off"), ANONYMOUS)
    assert len(on.chunks) == K
    assert len(strict.chunks) == K
    assert len(off.chunks) < K
    assert {c.k8s_version for c in on.chunks} == {"1.31"}
    # relaxed_order is re-sorted exactly: ranks follow similarity.
    sims = [c.dense_similarity for c in on.chunks]
    assert sims == sorted(sims, reverse=True)


async def test_exact_scan_returns_k_regardless_of_ef_search(
    index_only_retriever: Retriever,
) -> None:
    res = await index_only_retriever.retrieve(query("off"), ANONYMOUS, exact=True)
    assert len(res.chunks) == K
