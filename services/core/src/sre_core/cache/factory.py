"""Answer caches: exact (Redis) and semantic (pgvector `semantic_cache`), Phase 3 T14.

Isolation is the point. Every key and every SQL filter carries the permission scope (a hash of
the caller's sorted access groups, always including `public`), the corpus version and the
prompt version, so an answer built from team-a's documents is unreachable for any other scope,
and a re-ingested corpus or an edited prompt starts cold. The filter is in the SQL `where`, not
applied to the rows afterwards. A semantic hit also needs the same literal tokens (error names,
multi-word flags, field paths): embeddings call `--log-dir` and `--log-file-max-size` neighbours,
and their answers differ.

Only answers built from retrieved documents are cached (route simple or complex, with
citations); action and escalation turns carry tool output and live state and are refused.
A cache failure is a miss, never a failed turn.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
from psycopg_pool import AsyncConnectionPool

from sre_core.ports import AnswerCache, CachedAnswer
from sre_shared.config import CacheSettings, load_agent_config
from sre_shared.embed.base import Embedder
from sre_shared.literals import find_exact_strings
from sre_shared.models import Claims
from sre_shared.settings import Settings, get_settings

log = logging.getLogger(__name__)

PUBLIC = "public"
KEY_PREFIX = "sre:ans:v1:"
CACHEABLE_ROUTES = ("simple", "complex")
SEMANTIC_CANDIDATES = 5  # nearest rows examined; each must still pass the literal guard
_WS = re.compile(r"\s+")


class CacheRefused(ValueError):
    """`store` was given something that must never be cached (an action answer, say)."""


def scope_hash(claims: Claims) -> str:
    """sha256 of the sorted, de-duplicated access groups, `public` always included. Two callers
    who can read the same documents share a scope; anyone with another group set does not."""
    groups = sorted({*claims.groups, PUBLIC})
    return hashlib.sha256(json.dumps(groups).encode()).hexdigest()


def normalise_question(question: str) -> str:
    """Whitespace only: case matters in Kubernetes names, so it is kept."""
    return _WS.sub(" ", question).strip()


def literal_tokens(question: str) -> list[str]:
    return sorted({s for _, s in find_exact_strings(question)})


def _vector_text(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"


_LOOKUP_SQL = """
select id, answer, -(embedding <#> %(q)s::halfvec) as similarity
from public.semantic_cache
where scope_hash = %(scope)s
  and corpus_version_id = %(version)s
  and prompt_version = %(prompt)s
  and expires_at > %(now)s
  and literal_tokens = %(lits)s::text[]
order by embedding <#> %(q)s::halfvec
limit %(n)s
"""


class TwoTierAnswerCache:
    def __init__(
        self,
        pool: AsyncConnectionPool[Any] | None,
        redis: Any,
        embedder: Embedder | None,
        settings: CacheSettings,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._pool = pool
        self._redis = redis
        self._embedder = embedder
        self._cfg = settings
        self._clock = clock or (lambda: datetime.now(UTC))

    # -- exact tier ---------------------------------------------------------------------

    def _key(self, question: str, claims: Claims, version: int, prompt: str) -> str:
        digest = hashlib.sha256(normalise_question(question).encode()).hexdigest()
        return f"{KEY_PREFIX}{scope_hash(claims)}:{version}:{prompt}:{digest}"

    # -- AnswerCache --------------------------------------------------------------------

    async def lookup(
        self, question: str, *, claims: Claims, corpus_version_id: int, prompt_version: str
    ) -> CachedAnswer | None:
        try:
            if self._redis is not None:
                raw = await self._redis.get(
                    self._key(question, claims, corpus_version_id, prompt_version)
                )
                if raw is not None:
                    hit = CachedAnswer.model_validate_json(raw)
                    return hit.model_copy(update={"cache": "exact"})
            if self._semantic_on():
                return await self._semantic_lookup(
                    question, claims, corpus_version_id, prompt_version
                )
        except Exception:
            log.warning("answer cache lookup failed; treating as a miss", exc_info=True)
        return None

    async def store(
        self,
        answer: CachedAnswer,
        *,
        claims: Claims,
        corpus_version_id: int,
        prompt_version: str,
    ) -> None:
        if answer.route not in CACHEABLE_ROUTES:
            raise CacheRefused(f"route {answer.route!r} answers are never cached")
        if not answer.citations or not answer.answer.strip():
            return  # not built from retrieved documents (an abstention, say)
        try:
            if self._redis is not None and self._cfg.exact_ttl_s > 0:
                await self._redis.set(
                    self._key(answer.question, claims, corpus_version_id, prompt_version),
                    answer.model_dump_json(),
                    ex=self._cfg.exact_ttl_s,
                )
            if self._semantic_on() and self._cfg.semantic_ttl_s:
                await self._semantic_store(answer, claims, corpus_version_id, prompt_version)
        except Exception:
            log.warning("answer cache store failed", exc_info=True)

    # -- semantic tier ------------------------------------------------------------------

    def _semantic_on(self) -> bool:
        return self._cfg.semantic_enabled and self._pool is not None and self._embedder is not None

    async def _embed(self, question: str) -> list[float]:
        assert self._embedder is not None
        vecs = await asyncio.to_thread(self._embedder.embed_queries, [question])
        v = np.asarray(vecs[0], dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return [float(x) for x in (v / norm if norm else v)]  # inner product == cosine

    async def _semantic_lookup(
        self, question: str, claims: Claims, version: int, prompt: str
    ) -> CachedAnswer | None:
        assert self._pool is not None
        q = _vector_text(await self._embed(question))
        params = {
            "q": q,
            "scope": scope_hash(claims),
            "version": version,
            "prompt": prompt,
            "now": self._clock(),
            "lits": literal_tokens(question),
            "n": SEMANTIC_CANDIDATES,
        }
        async with self._pool.connection() as conn:
            cur = await conn.execute(_LOOKUP_SQL, params)
            rows = await cur.fetchall()
            for row in rows:
                if float(row["similarity"]) < self._cfg.semantic_min_similarity:
                    break  # ordered by distance: the rest are farther
                await conn.execute(
                    "update public.semantic_cache set hits = hits + 1 where id = %s", [row["id"]]
                )
                hit = CachedAnswer.model_validate(row["answer"])
                return hit.model_copy(update={"cache": "semantic"})
        return None

    async def _semantic_store(
        self, answer: CachedAnswer, claims: Claims, version: int, prompt: str
    ) -> None:
        assert self._pool is not None
        question = normalise_question(answer.question)
        vec = _vector_text(await self._embed(question))
        now = self._clock()
        async with self._pool.connection() as conn, conn.transaction():
            # One live row per (scope, version, prompt, question); expired rows go with it.
            await conn.execute(
                "delete from public.semantic_cache where expires_at <= %s or "
                "(scope_hash = %s and corpus_version_id = %s and prompt_version = %s "
                "and question = %s)",
                [now, scope_hash(claims), version, prompt, question],
            )
            await conn.execute(
                "insert into public.semantic_cache (scope_hash, corpus_version_id, "
                "prompt_version, route, question, literal_tokens, embedding, answer, expires_at) "
                "values (%s, %s, %s, %s, %s, %s, %s::halfvec, %s::jsonb, "
                "%s::timestamptz + make_interval(secs => %s))",
                [
                    scope_hash(claims),
                    version,
                    prompt,
                    answer.route,
                    question,
                    literal_tokens(question),
                    vec,
                    answer.model_dump_json(),
                    now,
                    self._cfg.semantic_ttl_s,
                ],
            )


def make_answer_cache(
    pool: AsyncConnectionPool[Any] | None,
    settings: Settings | None = None,
    *,
    redis: Any = None,
    embedder: Embedder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AnswerCache:
    """Exact tier when Redis is available (`redis_url`, or an injected async client), semantic
    tier when a pool is given. The embedder defaults to the production query model."""
    settings = settings or get_settings()
    cfg = load_agent_config(settings.config_dir).cache
    if redis is None and settings.redis_url:
        import redis.asyncio as aioredis

        redis = aioredis.Redis.from_url(settings.redis_url)
    if pool is not None and embedder is None:
        from sre_shared.embed import get_embedder

        # The table's column is halfvec(384): gte-small, whatever the lab experiments with.
        embedder = get_embedder("gte-small")
    return TwoTierAnswerCache(pool, redis, embedder, cfg, clock=clock)
