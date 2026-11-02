"""GET /search parameter handling and response shape."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx
import pytest

from sre_core.api.main import create_app
from sre_core.retrieval import Retriever

if TYPE_CHECKING:
    from conftest import Corpus

pytestmark = pytest.mark.integration


@pytest.fixture
async def client(retriever: Retriever) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(retriever)), base_url="http://test"
    ) as c:
        yield c


async def test_search_response_shape(client: httpx.AsyncClient, corpus: Corpus) -> None:
    r = await client.get("/search", params={"q": "smallest deployable unit pods", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["corpus_version_id"] == corpus.version_id
    assert len(body["chunks"]) <= 2
    top = body["chunks"][0]
    assert top["source_path"] == "content/en/docs/concepts/pods.md"
    assert {"dense_rank", "keyword_rank", "dense_similarity", "keyword_score", "score"} <= set(top)
    assert top["keyword_rank"] is None
    assert set(body["timings_ms"]) == {"embed", "sql", "fuse", "total"}
    # No mode given: config/retrieval.yaml's default (dense, decision 29).
    assert body["request"]["use_dense"] and not body["request"]["use_keyword"]


async def test_explicit_hybrid_uses_both_legs(client: httpx.AsyncClient, corpus: Corpus) -> None:
    r = await client.get("/search", params={"q": "smallest deployable unit pods", "mode": "hybrid"})
    req = r.json()["request"]
    assert req["use_dense"] and req["use_keyword"]


async def test_modes(client: httpx.AsyncClient, corpus: Corpus) -> None:
    dense = (await client.get("/search", params={"q": "pods", "mode": "dense"})).json()
    kw = (await client.get("/search", params={"q": "pods", "mode": "keyword"})).json()
    assert all(c["keyword_rank"] is None for c in dense["chunks"])
    assert all(c["dense_rank"] is None for c in kw["chunks"])
    assert kw["chunks"]


async def test_filters_repeat(client: httpx.AsyncClient, corpus: Corpus) -> None:
    r = await client.get(
        "/search", params={"q": "pods probes", "version": ["1.30"], "doc_type": ["task"]}
    )
    chunks = r.json()["chunks"]
    assert chunks and {c["k8s_version"] for c in chunks} == {"1.30"}
    assert {c["doc_type"] for c in chunks} == {"task"}


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "x", "k": 0},
        {"q": "x", "mode": "fuzzy"},
        {"q": "x", "doc_type": "nope"},
    ],
)
async def test_bad_params_are_422(client: httpx.AsyncClient, params: dict[str, object]) -> None:
    assert (await client.get("/search", params=params)).status_code == 422  # type: ignore[arg-type]


async def test_no_active_corpus_is_503(client: httpx.AsyncClient, db: object) -> None:
    assert (await client.get("/search", params={"q": "pods"})).status_code == 503


async def test_the_web_origin_may_call_the_api_and_others_may_not(
    client: httpx.AsyncClient,
) -> None:
    preflight = {"Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "authorization,content-type"}  # fmt: skip
    ok = await client.options("/chat", headers={"Origin": "http://localhost:3000", **preflight})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
    other = await client.options("/chat", headers={"Origin": "https://evil.test", **preflight})
    assert "access-control-allow-origin" not in other.headers
