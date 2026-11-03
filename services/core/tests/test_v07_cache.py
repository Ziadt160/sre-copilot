"""V07 (adversarial verification): the semantic cache when other scopes hold the NEAREST rows.

test_cache_answer.py (T14) shows a single foreign row is never returned. Here the closest
neighbours of the query all belong to other scopes (identical vectors), with the caller's own row
slightly farther away: the scope filter must still win over distance, and the caller's own row
must still be found."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.cache.factory import TwoTierAnswerCache
from sre_core.ports import CachedAnswer
from sre_shared.config import CacheSettings
from sre_shared.models import Citation, Claims
from sre_shared.testing.seed import create_corpus_version

pytestmark = pytest.mark.integration

CFG = CacheSettings(
    exact_ttl_s=3600, semantic_ttl_s=86400, semantic_min_similarity=0.92, semantic_enabled=True
)
CITE = Citation(chunk_id=1, source_path="x.md", k8s_version="1.31", url="https://x", title="X")
PROMPT = "p1"
QUERY = "what is the way to restart a deployment?"
OWN = "how can a deployment be restarted?"
FOREIGN = [f"how do I restart a deployment, take {n}?" for n in range(20)]


class AxisEmbedder:
    """The query and the foreign questions share one direction; the caller's own question is
    tilted slightly away from it (cosine about 0.995), everything else is orthogonal."""

    def embed_queries(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            if t == QUERY or t in FOREIGN:
                out[i, 10] = 1.0
            elif t == OWN:
                out[i, 10], out[i, 11] = 1.0, 0.1
            else:
                out[i, 200 + i % 100] = 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def answer(q: str, text: str) -> CachedAnswer:
    return CachedAnswer.model_construct(
        question=q, answer=text, citations=[CITE], route="simple", cache="semantic"
    )


async def test_nearer_rows_of_other_scopes_never_leak_and_never_hide_the_callers_row(
    db: object, pool: AsyncConnectionPool
) -> None:
    version = await create_corpus_version(db)  # type: ignore[arg-type]
    now = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    cache = TwoTierAnswerCache(pool, None, AxisEmbedder(), CFG, clock=lambda: now)  # type: ignore[arg-type]
    for n, q in enumerate(FOREIGN):
        other = Claims(sub=f"u{n}", groups=["team-b"] if n % 2 else [f"team-{n}"])
        await cache.store(
            answer(q, f"secret of {other.groups[0]}"),
            claims=other, corpus_version_id=version, prompt_version=PROMPT,
        )  # fmt: skip
    team_a = Claims(sub="a", groups=["team-a"])
    await cache.store(
        answer(OWN, "team-a's answer"), claims=team_a, corpus_version_id=version,
        prompt_version=PROMPT,
    )  # fmt: skip

    hit = await cache.lookup(QUERY, claims=team_a, corpus_version_id=version, prompt_version=PROMPT)
    assert hit is not None and hit.answer == "team-a's answer"
    own_b = await cache.lookup(
        QUERY, claims=Claims(sub="b", groups=["team-b"]), corpus_version_id=version,
        prompt_version=PROMPT,
    )  # fmt: skip
    assert own_b is not None and own_b.answer == "secret of team-b"  # the near rows are real
    for caller in (
        Claims(sub="nobody"),  # no groups: public only
        Claims(sub="anon", role="anon"),
        Claims(sub="c", groups=["team-c"]),
        Claims(sub="w", groups=["team-a", "team-c"]),  # wider than team-a: another scope
    ):
        got = await cache.lookup(
            QUERY, claims=caller, corpus_version_id=version, prompt_version=PROMPT
        )
        assert got is None, caller
