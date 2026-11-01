"""POST /answer over HTTP: JSON, SSE framing, error mapping, and RLS on the seeded corpus."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING

import httpx
import jwt
import pytest

from sre_core.answer import build_answerer
from sre_core.answer.testing import fake_chat
from sre_core.api.main import create_app
from sre_core.retrieval import NoActiveCorpusError, Retriever
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import AnswerConfig, Claims
from sre_shared.settings import get_settings

if TYPE_CHECKING:
    from conftest import Corpus

pytestmark = pytest.mark.integration

JWT_SECRET = "test-secret-test-secret-test-secret-0123"
CFG = AnswerConfig(model="ollama:fake", k=5, relevance_floor=0.0)


def client_for(retriever: Retriever, reply: str = "It is a unit [1].", tpd: int | None = None):
    chat, comps = fake_chat(reply, tpd=tpd)
    ans = build_answerer(retriever, config=CFG, chat=chat, embedder=FakeEmbedder())
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(retriever, ans)), base_url="http://test"
    ), comps


@pytest.fixture
async def client(retriever: Retriever) -> AsyncIterator[httpx.AsyncClient]:
    c, _ = client_for(retriever)
    async with c:
        yield c


async def test_answer_json(client: httpx.AsyncClient, corpus: Corpus) -> None:
    r = await client.post("/answer", json={"question": "smallest deployable unit pods"})
    assert r.status_code == 200
    body = r.json()
    assert body["corpus_version_id"] == corpus.version_id
    assert body["citations"][0]["chunk_id"] == body["context"][0]["chunk_id"]
    assert body["abstained"] is False and body["prompt_tokens"] == 123


async def test_answer_sse_frames(client: httpx.AsyncClient, corpus: Corpus) -> None:
    r = await client.post(
        "/answer", json={"question": "pods"}, headers={"Accept": "text/event-stream"}
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = [f for f in r.text.split("\n\n") if f]
    assert r.text.endswith("\n\n")
    names = []
    for f in frames:
        lines = f.split("\n")
        assert len(lines) == 2 and lines[0].startswith("event: ") and lines[1].startswith("data: ")
        names.append(lines[0][len("event: ") :])
        json.loads(lines[1][len("data: ") :])
    assert names[0] == "sources" and names[-1] == "done" and "token" in names
    done = json.loads(frames[-1].split("\n")[1][len("data: ") :])
    assert done["citations"] and done["answer"] == "It is a unit [1]."


async def test_quota_maps_to_503(retriever: Retriever, corpus: Corpus) -> None:
    c, comps = client_for(retriever, tpd=10)
    async with c:
        r = await c.post("/answer", json={"question": "pods"})
        assert r.status_code == 503 and "tokens" in r.json()["detail"]
        r = await c.post(
            "/answer", json={"question": "pods"}, headers={"Accept": "text/event-stream"}
        )
        assert r.status_code == 503
    assert comps.calls == []


async def test_no_active_corpus_maps_to_503() -> None:
    class NoCorpus:
        async def retrieve(self, *a: object, **k: object) -> object:
            raise NoActiveCorpusError("no active corpus")

    chat, _ = fake_chat()
    ans = build_answerer(NoCorpus(), config=CFG, chat=chat)  # type: ignore[arg-type]
    app = create_app(NoCorpus(), ans)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/answer", json={"question": "q"})).status_code == 503
        r = await c.post("/answer", json={"question": "q"}, headers={"Accept": "text/event-stream"})
        assert r.status_code == 503


@pytest.fixture
def jwt_secret(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_rls_user_b_never_sees_team_a(
    retriever: Retriever, corpus: Corpus, user_a: Claims, user_b: Claims, jwt_secret: None
) -> None:
    c, _ = client_for(retriever, reply="See [1][2][3][4][5].")
    async with c:
        q = {"question": "quokka rollout runbook credentials rotate nightly"}
        rb = await c.post("/answer", json=q, headers=_bearer(user_b))
        ra = await c.post("/answer", json=q, headers=_bearer(user_a))
    team_a, team_b = (
        corpus.one("content/en/docs/internal/alpha.md@1.31"),
        corpus.one("content/en/docs/internal/bravo.md@1.31"),
    )
    ids_b = {x["chunk_id"] for x in rb.json()["context"]} | {
        x["chunk_id"] for x in rb.json()["citations"]
    }
    assert rb.status_code == 200 and team_a not in ids_b
    ids_a = {x["chunk_id"] for x in ra.json()["context"]}
    assert team_a in ids_a and team_b not in ids_a


def _bearer(claims: Claims) -> dict[str, str]:
    payload = {
        "sub": claims.sub,
        "role": "authenticated",
        "exp": int(time.time()) + 3600,
        "app_metadata": {"groups": claims.groups},
    }
    return {"Authorization": f"Bearer {jwt.encode(payload, JWT_SECRET, algorithm='HS256')}"}
