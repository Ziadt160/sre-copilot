"""Answer cache: exact (fakeredis) and semantic (pgvector) tiers never cross permission scopes."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import fakeredis
import numpy as np
import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.cache.factory import (
    CacheRefused,
    TwoTierAnswerCache,
    literal_tokens,
    make_answer_cache,
    scope_hash,
)
from sre_core.ports import CachedAnswer
from sre_shared.config import CacheSettings
from sre_shared.models import Citation, Claims
from sre_shared.settings import Settings
from sre_shared.testing.seed import create_corpus_version

CFG = CacheSettings(
    exact_ttl_s=3600, semantic_ttl_s=86400, semantic_min_similarity=0.92, semantic_enabled=True
)
TEAM_A = Claims(sub="a", groups=["team-a"])
TEAM_B = Claims(sub="b", groups=["team-b"])
ANON = Claims(sub="anon", role="anon")
PROMPT = "p1"

CITE = Citation(chunk_id=1, source_path="x.md", k8s_version="1.31", url="https://x", title="X")


def answer(q: str, text: str = "the answer", route: str = "simple") -> CachedAnswer:
    return CachedAnswer.model_construct(
        question=q, answer=text, citations=[CITE], route=route, cache="exact"
    )


class VectorEmbedder:
    """Maps chosen questions to chosen directions; anything else gets its own axis."""

    def __init__(self, table: dict[str, int]) -> None:
        self.table = table
        self.mix: dict[str, float] = {}

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            axis = self.table.get(t, 300 + (hash(t) % 80))
            out[i, axis] = 1.0
            if t in self.mix:  # tilt away from the axis to control similarity
                out[i, 383] = self.mix[t]
        return out


@pytest.fixture
def clock() -> list[datetime]:
    return [datetime(2026, 11, 2, 12, 0, tzinfo=UTC)]


def exact_cache(clock: list[datetime]) -> TwoTierAnswerCache:
    return TwoTierAnswerCache(None, fakeredis.FakeAsyncRedis(), None, CFG, clock=lambda: clock[0])


async def lookup(
    c: TwoTierAnswerCache, q: str, claims: Claims, version: int = 1, prompt: str = PROMPT
) -> CachedAnswer | None:
    return await c.lookup(q, claims=claims, corpus_version_id=version, prompt_version=prompt)


async def put(
    c: TwoTierAnswerCache, a: CachedAnswer, claims: Claims, version: int = 1, prompt: str = PROMPT
) -> None:
    await c.store(a, claims=claims, corpus_version_id=version, prompt_version=prompt)


# --- scope ------------------------------------------------------------------------------


def test_scope_hash_is_order_free_includes_public_and_separates_groups() -> None:
    assert scope_hash(Claims(sub="x", groups=["a", "b"])) == scope_hash(
        Claims(sub="y", groups=["b", "a", "public", "a"])
    )
    assert scope_hash(ANON) == scope_hash(Claims(sub="z", groups=["public"]))
    assert len({scope_hash(c) for c in (TEAM_A, TEAM_B, ANON)}) == 3
    assert scope_hash(Claims(sub="x", groups=["a", "b"])) != scope_hash(TEAM_A)


# --- exact tier -------------------------------------------------------------------------


async def test_exact_hit_only_for_the_same_scope(clock: list[datetime]) -> None:
    c = exact_cache(clock)
    await put(c, answer("what is a pod?"), TEAM_A)
    hit = await lookup(c, "what is a pod?", TEAM_A)
    assert hit is not None and hit.cache == "exact" and hit.answer == "the answer"
    assert await lookup(c, "what is a pod?", TEAM_B) is None
    assert await lookup(c, "what is a pod?", ANON) is None
    # Same groups in another order, or one user with the same access, share the entry.
    twin = Claims(sub="other", groups=["public", "team-a"])
    assert await lookup(c, "what is a pod?", twin) is not None
    # More access is a different scope (it may have been answered from more documents).
    wider = Claims(sub="w", groups=["team-a", "team-b"])
    assert await lookup(c, "what is a pod?", wider) is None


async def test_exact_misses_on_other_corpus_or_prompt_version(clock: list[datetime]) -> None:
    c = exact_cache(clock)
    await put(c, answer("q"), ANON, version=1)
    assert await lookup(c, "q", ANON, version=1) is not None
    assert await lookup(c, "q", ANON, version=2) is None
    assert await lookup(c, "q", ANON, prompt="p2") is None


async def test_exact_normalises_whitespace_but_not_case(clock: list[datetime]) -> None:
    c = exact_cache(clock)
    await put(c, answer("why  is\nmy pod pending?"), ANON)
    assert await lookup(c, "  why is my pod pending? ", ANON) is not None
    assert await lookup(c, "Why is my pod pending?", ANON) is None


async def test_store_refuses_actions_and_skips_ungrounded(clock: list[datetime]) -> None:
    c = exact_cache(clock)
    for route in ("action", "escalate"):
        with pytest.raises(CacheRefused):
            await put(c, answer("scale it", route=route), TEAM_A)
    assert await lookup(c, "scale it", TEAM_A) is None
    ungrounded = CachedAnswer.model_construct(
        question="q", answer="I do not know", citations=[], route="simple", cache="exact"
    )
    await put(c, ungrounded, TEAM_A)
    assert await lookup(c, "q", TEAM_A) is None


async def test_exact_ttl(clock: list[datetime]) -> None:
    r = fakeredis.FakeAsyncRedis()
    c = TwoTierAnswerCache(None, r, None, CFG)
    await put(c, answer("q"), ANON)
    (key,) = [k async for k in r.scan_iter("sre:ans:v1:*")]
    assert 0 < await r.ttl(key) <= 3600


# --- semantic tier ----------------------------------------------------------------------


@pytest.fixture
async def version(db: object) -> int:
    return await create_corpus_version(db)  # type: ignore[arg-type]


def semantic(
    pool: AsyncConnectionPool, emb: VectorEmbedder, clock: list[datetime]
) -> TwoTierAnswerCache:
    return TwoTierAnswerCache(pool, None, emb, CFG, clock=lambda: clock[0])  # type: ignore[arg-type]


PARA = {
    "how do I restart a deployment?": 10,
    "what is the way to restart a deployment?": 10,  # same direction: a paraphrase
    "what is a service?": 20,
}


async def test_semantic_hit_for_paraphrase_in_scope_only(
    pool: AsyncConnectionPool, version: int, clock: list[datetime]
) -> None:
    c = semantic(pool, VectorEmbedder(PARA), clock)
    await put(c, answer("how do I restart a deployment?"), TEAM_A, version)
    hit = await lookup(c, "what is the way to restart a deployment?", TEAM_A, version)
    assert hit is not None and hit.cache == "semantic"
    for other in (TEAM_B, ANON, Claims(sub="w", groups=["team-a", "team-b"])):
        assert await lookup(c, "what is the way to restart a deployment?", other, version) is None
        assert await lookup(c, "how do I restart a deployment?", other, version) is None


async def test_semantic_misses_on_other_corpus_or_prompt_version_and_low_similarity(
    pool: AsyncConnectionPool, db: object, version: int, clock: list[datetime]
) -> None:
    c = semantic(pool, VectorEmbedder(PARA), clock)
    q, para = "how do I restart a deployment?", "what is the way to restart a deployment?"
    await put(c, answer(q), ANON, version)
    other_version = await create_corpus_version(db, label="other", status="retired")  # type: ignore[arg-type]
    assert await lookup(c, para, ANON, other_version) is None
    assert await lookup(c, para, ANON, version, prompt="p2") is None
    assert await lookup(c, "what is a service?", ANON, version) is None  # orthogonal
    assert await lookup(c, para, ANON, version) is not None


async def test_similarity_threshold_is_respected(
    pool: AsyncConnectionPool, version: int, clock: list[datetime]
) -> None:
    emb = VectorEmbedder({"a": 5, "b": 5})
    emb.mix["b"] = 0.6  # cos = 1/sqrt(1.36) = 0.857 < 0.92
    c = semantic(pool, emb, clock)
    await put(c, answer("a"), ANON, version)
    assert await lookup(c, "b", ANON, version) is None
    emb.mix["b"] = 0.2  # cos = 0.981
    assert await lookup(c, "b", ANON, version) is not None


async def test_literal_tokens_must_match_exactly(
    pool: AsyncConnectionPool, version: int, clock: list[datetime]
) -> None:
    dir_q = "what does --log-dir do in kubelet?"
    size_q = "what does --log-file-max-size do in kubelet?"
    dir_para = "explain the kubelet flag --log-dir"
    plain_q = "what does the kubelet log flag do?"
    emb = VectorEmbedder({dir_q: 7, size_q: 7, dir_para: 7, plain_q: 7})  # all equally "similar"
    c = semantic(pool, emb, clock)
    assert literal_tokens(dir_q) == ["--log-dir"]
    await put(c, answer(dir_q, "log dir answer"), ANON, version)
    assert await lookup(c, size_q, ANON, version) is None  # different flag: must miss
    assert await lookup(c, plain_q, ANON, version) is None  # no literal at all: must miss
    hit = await lookup(c, dir_para, ANON, version)  # same literal: hit
    assert hit is not None and hit.answer == "log dir answer"
    # And the other way round: a literal-free entry is not served to a literal question.
    await put(c, answer(plain_q, "generic"), TEAM_A, version)
    assert await lookup(c, dir_q, TEAM_A, version) is None


async def test_semantic_expiry_and_replacement(
    pool: AsyncConnectionPool, version: int, clock: list[datetime]
) -> None:
    c = semantic(pool, VectorEmbedder(PARA), clock)
    await put(c, answer("how do I restart a deployment?", "v1"), ANON, version)
    await put(c, answer("how do I restart a deployment?", "v2"), ANON, version)
    async with pool.connection() as conn:
        n = await (await conn.execute("select count(*) as n from public.semantic_cache")).fetchone()
    assert n["n"] == 1  # type: ignore[index]
    hit = await lookup(c, "what is the way to restart a deployment?", ANON, version)
    assert hit is not None and hit.answer == "v2"
    clock[0] += timedelta(seconds=CFG.semantic_ttl_s + 1)
    assert await lookup(c, "what is the way to restart a deployment?", ANON, version) is None


async def test_semantic_refuses_actions(
    pool: AsyncConnectionPool, version: int, clock: list[datetime]
) -> None:
    c = semantic(pool, VectorEmbedder(PARA), clock)
    with pytest.raises(CacheRefused):
        await put(c, answer("scale it", route="action"), TEAM_A, version)
    async with pool.connection() as conn:
        n = await (await conn.execute("select count(*) as n from public.semantic_cache")).fetchone()
    assert n["n"] == 0  # type: ignore[index]


async def test_the_factory_serves_only_the_exact_tier_by_default(
    pool: AsyncConnectionPool, version: int, clock: list[datetime]
) -> None:
    c = make_answer_cache(
        pool,
        Settings(_env_file=None, redis_url=None),
        redis=fakeredis.FakeAsyncRedis(),
        embedder=VectorEmbedder(PARA),  # type: ignore[arg-type]
        clock=lambda: clock[0],
    )
    await c.store(
        answer("how do I restart a deployment?"),
        claims=TEAM_A,
        corpus_version_id=version,
        prompt_version=PROMPT,
    )
    exact = await c.lookup(
        "how do I restart a deployment?",
        claims=TEAM_A,
        corpus_version_id=version,
        prompt_version=PROMPT,
    )
    sem = await c.lookup(
        "what is the way to restart a deployment?",
        claims=TEAM_A,
        corpus_version_id=version,
        prompt_version=PROMPT,
    )
    assert exact is not None and exact.cache == "exact"
    assert sem is None  # the semantic tier is off in config/agent.yaml (decision 48)
    for other in (TEAM_B, ANON):
        for q in ("how do I restart a deployment?", "what is the way to restart a deployment?"):
            assert (
                await c.lookup(q, claims=other, corpus_version_id=version, prompt_version=PROMPT)
                is None
            )
