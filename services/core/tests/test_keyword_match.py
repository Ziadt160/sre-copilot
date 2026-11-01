"""Keyword leg match modes. Found in the lab: with every term required (websearch_to_tsquery),
natural questions matched nothing, so keyword-only and hybrid were not a fair comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psycopg
import pytest

from sre_core.retrieval import Retriever
from sre_shared.models import ANONYMOUS, RetrievalRequest

if TYPE_CHECKING:
    from conftest import Corpus

pytestmark = pytest.mark.integration

# "flaky" and "app" occur in no chunk; the other terms are in the probes task.
QUESTION = "How do I configure liveness probes for my flaky app?"


def kw(query: str, **extra: Any) -> RetrievalRequest:
    return RetrievalRequest(query=query, use_dense=False, k=3, **extra)


async def test_any_term_matches_a_natural_question(retriever: Retriever, corpus: Corpus) -> None:
    res = await retriever.retrieve(kw(QUESTION), ANONYMOUS)
    assert res.chunks and res.chunks[0].source_path == "content/en/docs/tasks/probes.md"
    strict = await retriever.retrieve(kw(QUESTION, keyword_match="all"), ANONYMOUS)
    assert strict.chunks == []


async def test_more_matching_terms_rank_higher(retriever: Retriever, corpus: Corpus) -> None:
    # pods.md has pods, network and storage; services.md has pods and network only.
    res = await retriever.retrieve(kw("pods network storage"), ANONYMOUS)
    paths = [c.source_path for c in res.chunks]
    assert paths.index("content/en/docs/concepts/pods.md") < paths.index(
        "content/en/docs/concepts/services.md"
    )


@pytest.mark.parametrize(
    "query", ["pods' | 'secret", "pods & !network", "quokka:*", "'", "a | b & (c", "the of and"]
)
async def test_query_syntax_is_never_interpreted(
    retriever: Retriever, corpus: Corpus, query: str
) -> None:
    """User text only reaches tsquery through plainto_tsquery; none of these may error, and RLS
    still hides the team chunks (quokka lives in a team-a chunk)."""
    res = await retriever.retrieve(kw(query), ANONYMOUS)
    assert all("internal/" not in c.source_path for c in res.chunks)


async def test_unknown_match_mode_is_rejected(db: Any, corpus: Corpus) -> None:
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        await db.execute(
            "select * from public.search_candidates('emb_gte_small', null, 'pods', 'english', 5,"
            " %s, null, null, false, true, false, 'some')",
            [corpus.version_id],
        )
