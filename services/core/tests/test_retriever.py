"""Retriever against a real Postgres: each leg, hybrid, filters, exact vs index."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sre_core.retrieval import NoActiveCorpusError, Retriever
from sre_shared.models import ANONYMOUS, DocType, RetrievalFilters, RetrievalRequest

if TYPE_CHECKING:
    from conftest import Corpus

pytestmark = pytest.mark.integration

POD_Q = "smallest deployable unit pods run containers"


def req(**kw: object) -> RetrievalRequest:
    return RetrievalRequest(query=kw.pop("query", POD_Q), **kw)  # type: ignore[arg-type]


async def test_hybrid_returns_both_legs_and_timings(retriever: Retriever, corpus: Corpus) -> None:
    res = await retriever.retrieve(req(k=3), ANONYMOUS)
    assert res.corpus_version_id == corpus.version_id
    top = res.chunks[0]
    assert top.chunk_id == corpus.one("content/en/docs/concepts/pods.md@1.31")
    assert top.dense_rank == 1 and top.keyword_rank == 1
    assert top.dense_similarity is not None and top.dense_similarity > 0.5
    assert top.keyword_score is not None and top.keyword_score > 0
    assert top.score == pytest.approx(2 / 61)
    assert set(res.timings_ms) == {"embed", "sql", "fuse", "total"}
    assert res.timings_ms["total"] >= res.timings_ms["sql"]


async def test_dense_only_has_no_keyword_fields(retriever: Retriever, corpus: Corpus) -> None:
    res = await retriever.retrieve(req(use_keyword=False, k=5), ANONYMOUS)
    assert res.chunks
    assert all(c.keyword_rank is None and c.keyword_score is None for c in res.chunks)
    assert [c.dense_rank for c in res.chunks] == list(range(1, len(res.chunks) + 1))
    assert res.chunks[0].score == pytest.approx(1 / 61)


async def test_keyword_only_has_no_dense_fields(retriever: Retriever, corpus: Corpus) -> None:
    res = await retriever.retrieve(req(use_dense=False, query="liveness probes"), ANONYMOUS)
    assert [c.source_path for c in res.chunks] == ["content/en/docs/tasks/probes.md"]
    c = res.chunks[0]
    assert c.dense_rank is None and c.dense_similarity is None
    assert c.keyword_rank == 1


async def test_ts_config_simple_does_not_stem(retriever: Retriever, corpus: Corpus) -> None:
    # The body says "restarts": the english config stems it to "restart", simple keeps it as is.
    english = await retriever.retrieve(req(use_dense=False, query="restart"), ANONYMOUS)
    simple = await retriever.retrieve(
        req(use_dense=False, query="restart", ts_config="simple"), ANONYMOUS
    )
    assert english.chunks  # "restarts" stems to "restart"
    assert not simple.chunks


async def test_version_filter(retriever: Retriever, corpus: Corpus) -> None:
    q = req(filters=RetrievalFilters(k8s_versions=["1.30"]), k=10)
    res = await retriever.retrieve(q, ANONYMOUS)
    assert res.chunks
    assert {c.k8s_version for c in res.chunks} == {"1.30"}


async def test_doc_type_filter(retriever: Retriever, corpus: Corpus) -> None:
    q = req(filters=RetrievalFilters(doc_types=[DocType.TASK]), k=10)
    res = await retriever.retrieve(q, ANONYMOUS)
    assert res.chunks
    assert {c.doc_type for c in res.chunks} == {DocType.TASK}


async def test_filter_with_no_match_is_empty(retriever: Retriever, corpus: Corpus) -> None:
    res = await retriever.retrieve(req(filters=RetrievalFilters(k8s_versions=["9.99"])), ANONYMOUS)
    assert res.chunks == []


async def test_explicit_corpus_version_and_missing_active(
    retriever: Retriever, corpus: Corpus, db: object
) -> None:
    res = await retriever.retrieve(req(corpus_version_id=corpus.version_id), ANONYMOUS)
    assert res.chunks
    other = await retriever.retrieve(req(corpus_version_id=corpus.version_id + 1000), ANONYMOUS)
    assert other.chunks == []
    await db.execute("update public.corpus_versions set status = 'retired'")  # type: ignore[attr-defined]
    with pytest.raises(NoActiveCorpusError):
        await retriever.retrieve(req(), ANONYMOUS)


async def test_exact_equals_index_path_on_small_table(retriever: Retriever, corpus: Corpus) -> None:
    q = req(use_keyword=False, k=5)
    approx = await retriever.retrieve(q, ANONYMOUS)
    exact = await retriever.retrieve(q, ANONYMOUS, exact=True)
    assert [c.chunk_id for c in approx.chunks] == [c.chunk_id for c in exact.chunks]
    for a, b in zip(approx.chunks, exact.chunks, strict=True):
        assert a.dense_similarity == pytest.approx(b.dense_similarity)


async def test_candidates_per_leg_caps_dense_leg(retriever: Retriever, corpus: Corpus) -> None:
    res = await retriever.retrieve(
        req(embedding_model="gte-small", use_keyword=False, candidates_per_leg=2), ANONYMOUS
    )
    assert len(res.chunks) == 2
