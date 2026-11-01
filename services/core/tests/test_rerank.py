"""RemoteReranker against a mock transport: no network, no GPU server."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from sre_core.retrieval.base import RerankerUnavailable
from sre_core.retrieval.rerank import RemoteReranker
from sre_shared.models import DocType, RetrievedChunk


def chunk(i: int, header: str = "h", body: str = "b") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        chunk_key=f"k{i}",
        source_path="p.md",
        k8s_version="1.37",
        doc_type=DocType.CONCEPT,
        title="t",
        section_path=[],
        url="https://example.test/",
        header=f"{header}{i}",
        body=f"{body}{i}",
        score=1.0,
    )


def make(handler: Callable[[httpx.Request], httpx.Response], **kw: object) -> RemoteReranker:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return RemoteReranker(base_url="http://gpu.test/", client=client, **kw)  # type: ignore[arg-type]


async def test_success_returns_scores_in_order_and_sends_contract() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"scores": [0.1, 0.9]})

    rr = make(handler, revision="abc123", batch_size=8)
    scores = await rr.rerank("why pending", [chunk(1), chunk(2)])
    await rr.aclose()
    assert scores == [0.1, 0.9]
    assert str(seen[0].url) == "http://gpu.test/rerank"
    body = json.loads(seen[0].content)
    assert body == {
        "hf_id": "BAAI/bge-reranker-base",
        "revision": "abc123",
        "query": "why pending",
        "texts": ["h1\n\nb1", "h2\n\nb2"],
        "max_length": 512,
        "batch_size": 8,
    }


async def test_default_revision_is_main() -> None:
    bodies: list[dict[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"scores": [0.5]})

    rr = make(handler)
    await rr.rerank("q", [chunk(1)])
    assert bodies[0]["revision"] == "main"


async def test_empty_chunks_makes_no_request() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert await make(handler).rerank("q", []) == []


async def test_client_is_reused_across_calls() -> None:
    n = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(200, json={"scores": [1.0]})

    rr = make(handler)
    await rr.rerank("q", [chunk(1)])
    await rr.rerank("q", [chunk(1)])
    assert n == 2
    assert not rr._client.is_closed
    await rr.aclose()
    assert rr._client.is_closed


async def test_http_500_is_unavailable() -> None:
    rr = make(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(RerankerUnavailable):
        await rr.rerank("q", [chunk(1)])


async def test_timeout_is_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    with pytest.raises(RerankerUnavailable):
        await make(handler).rerank("q", [chunk(1)])


async def test_connection_error_is_unavailable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(RerankerUnavailable):
        await make(handler).rerank("q", [chunk(1)])


async def test_wrong_score_count_is_unavailable() -> None:
    rr = make(lambda req: httpx.Response(200, json={"scores": [0.1]}))
    with pytest.raises(RerankerUnavailable):
        await rr.rerank("q", [chunk(1), chunk(2)])


@pytest.mark.parametrize("payload", [{"nope": []}, {"scores": "x"}, {"scores": ["a"]}])
async def test_malformed_body_is_unavailable(payload: dict[str, object]) -> None:
    rr = make(lambda req: httpx.Response(200, json=payload))
    with pytest.raises(RerankerUnavailable):
        await rr.rerank("q", [chunk(1)])


async def test_non_json_body_is_unavailable() -> None:
    rr = make(lambda req: httpx.Response(200, text="<html>"))
    with pytest.raises(RerankerUnavailable):
        await rr.rerank("q", [chunk(1)])


def test_default_reranker_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision 33: the lab reranks; the demo and CI do not (Workers AI parity is unchecked)."""
    from sre_core.retrieval.rerank import RemoteReranker, default_reranker
    from sre_shared.settings import get_settings

    for env, expected in (("lab", True), ("demo", False), ("ci", False)):
        monkeypatch.setenv("SRE_ENV", env)
        get_settings.cache_clear()
        got = default_reranker()
        assert isinstance(got, RemoteReranker) is expected, env
        if got is not None:
            assert got.revision == "2cfc18c9415c912f9d8155881c133215df768a70"
    get_settings.cache_clear()
