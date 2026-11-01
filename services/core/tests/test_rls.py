"""RLS leak tests: a `team-a` chunk is visible to user_a and never to user_b or anonymous,
through the Retriever and through GET /search. A positive control and a deliberately broken
policy prove the tests can fail."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

import httpx
import jwt
import pytest

from sre_core.api.main import create_app
from sre_core.retrieval import Retriever
from sre_shared.models import ANONYMOUS, Claims, RetrievalRequest
from sre_shared.settings import get_settings

if TYPE_CHECKING:
    from conftest import Corpus

pytestmark = pytest.mark.integration

JWT_SECRET = "test-secret-0123456789abcdef0123456789abcdef"
A_PATH = "content/en/docs/internal/alpha.md"
B_PATH = "content/en/docs/internal/bravo.md"
QUERIES = [
    "quokka rollout runbook credentials rotate",
    "wombat escalation pager vault",
    "smallest deployable unit pods containers",  # matches a public chunk
]
POLICY = "access_group = any ((select sre.user_groups())::text[])"


def token(
    sub: str,
    groups: list[str],
    *,
    secret: str = JWT_SECRET,
    role: str = "authenticated",
    exp_in: int = 3600,
) -> str:
    payload = {
        "sub": sub,
        "role": role,
        "aud": "authenticated",
        "exp": int(time.time()) + exp_in,
        "app_metadata": {"groups": groups},
    }
    return jwt.encode(payload, secret, algorithm="HS256")


async def visible_paths(retriever: Retriever, claims: Claims, mode: str) -> set[str]:
    """Every source path any query can surface for `claims`, with the widest candidate net."""
    seen: set[str] = set()
    for q in QUERIES:
        res = await retriever.retrieve(
            RetrievalRequest(
                query=q,
                k=100,
                candidates_per_leg=500,
                use_dense=mode in ("hybrid", "dense"),
                use_keyword=mode in ("hybrid", "keyword"),
            ),
            claims,
        )
        seen |= {c.source_path for c in res.chunks}
    return seen


@pytest.mark.parametrize("mode", ["hybrid", "dense", "keyword"])
async def test_retriever_respects_access_groups(
    retriever: Retriever, corpus: Corpus, user_a: Claims, user_b: Claims, mode: str
) -> None:
    a = await visible_paths(retriever, user_a, mode)
    b = await visible_paths(retriever, user_b, mode)
    anon = await visible_paths(retriever, ANONYMOUS, mode)
    assert A_PATH in a and B_PATH not in a  # positive control: the data is findable
    assert B_PATH in b and A_PATH not in b
    assert A_PATH not in anon and B_PATH not in anon
    assert anon, "public chunks stay visible to anonymous callers"


async def test_leak_test_can_fail_with_a_permissive_policy(
    retriever: Retriever, corpus: Corpus, user_b: Claims, db: Any
) -> None:
    await db.execute("alter policy chunks_read on public.chunks using (true)")
    try:
        assert A_PATH in await visible_paths(retriever, user_b, "hybrid")
    finally:
        await db.execute(f"alter policy chunks_read on public.chunks using ({POLICY})")
    assert A_PATH not in await visible_paths(retriever, user_b, "hybrid")


async def test_leak_test_can_fail_with_the_policy_dropped(
    retriever: Retriever, corpus: Corpus, user_a: Claims, db: Any
) -> None:
    """With no policy RLS denies everything: the positive control (user_a sees A) fails."""
    await db.execute("drop policy chunks_read on public.chunks")
    try:
        assert A_PATH not in await visible_paths(retriever, user_a, "hybrid")
    finally:
        await db.execute(
            f"create policy chunks_read on public.chunks for select to anon, "
            f"authenticated using ({POLICY})"
        )
    assert A_PATH in await visible_paths(retriever, user_a, "hybrid")


# --- HTTP path -------------------------------------------------------------------------------


@pytest.fixture
def jwt_secret(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client(retriever: Retriever, jwt_secret: None) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(retriever)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def http_paths(
    client: httpx.AsyncClient, bearer: str | None, mode: str = "hybrid"
) -> set[str]:
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    seen: set[str] = set()
    for q in QUERIES:
        r = await client.get("/search", params={"q": q, "k": 100, "mode": mode}, headers=headers)
        assert r.status_code == 200, r.text
        seen |= {c["source_path"] for c in r.json()["chunks"]}
    return seen


@pytest.mark.parametrize("mode", ["hybrid", "dense", "keyword"])
async def test_http_search_respects_access_groups(
    client: httpx.AsyncClient, corpus: Corpus, mode: str
) -> None:
    a = await http_paths(client, token("user-a", ["team-a"]), mode)
    b = await http_paths(client, token("user-b", ["team-b"]), mode)
    anon = await http_paths(client, None, mode)
    assert A_PATH in a and B_PATH not in a
    assert B_PATH in b and A_PATH not in b
    assert A_PATH not in anon and B_PATH not in anon


async def test_http_anon_role_token_is_anonymous(client: httpx.AsyncClient, corpus: Corpus) -> None:
    paths = await http_paths(client, token("x", ["team-a"], role="anon"))
    assert A_PATH not in paths


async def test_http_bad_tokens_are_rejected_not_downgraded(
    client: httpx.AsyncClient, corpus: Corpus
) -> None:
    forged = token("user-b", ["team-a"], secret="not-the-real-secret-0123456789abcdef0123")
    for bearer in (forged, token("user-a", ["team-a"], exp_in=-10), "garbage"):
        r = await client.get(
            "/search", params={"q": QUERIES[0]}, headers={"Authorization": f"Bearer {bearer}"}
        )
        assert r.status_code == 401
    r = await client.get("/search", params={"q": "x"}, headers={"Authorization": "Basic abc"})
    assert r.status_code == 401


async def test_http_group_claim_outside_app_metadata_is_ignored(
    client: httpx.AsyncClient, corpus: Corpus
) -> None:
    """user_metadata is user-writable in Supabase; only app_metadata may grant groups."""
    payload = {
        "sub": "user-b",
        "role": "authenticated",
        "exp": int(time.time()) + 3600,
        "user_metadata": {"groups": ["team-a"]},
        "groups": ["team-a"],
    }
    bearer = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    assert A_PATH not in await http_paths(client, bearer)


@pytest.mark.parametrize("app_metadata", ["team-a", ["team-a"], 42])
async def test_http_malformed_app_metadata_is_401(
    client: httpx.AsyncClient, corpus: Corpus, app_metadata: object
) -> None:
    """Found by V01: a validly signed token whose app_metadata is not an object returned 500."""
    payload = {
        "sub": "user-b",
        "role": "authenticated",
        "exp": int(time.time()) + 3600,
        "app_metadata": app_metadata,
    }
    bearer = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    r = await client.get(
        "/search", params={"q": QUERIES[0]}, headers={"Authorization": f"Bearer {bearer}"}
    )
    assert r.status_code == 401
