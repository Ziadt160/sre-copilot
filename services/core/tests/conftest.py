"""Fixtures for core tests: a small seeded corpus across access groups, a pool, a Retriever."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.retrieval import Retriever
from sre_shared.db import create_pool
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import DocType
from sre_shared.testing.seed import FixtureDoc, create_corpus_version, insert_docs

# Distinct vocabulary per doc so the hashing fake embedder and the keyword leg both have
# an unambiguous best match.
PUBLIC_POD = (
    "A Pod is the smallest deployable unit. Pods run containers sharing network and storage."
)
PUBLIC_SVC = "A Service exposes pods as a network service with a stable virtual IP address."
PUBLIC_TASK = "Configure liveness probes with kubectl and a probe handler for container restarts."
TEAM_A_SECRET = "Quokka rollout runbook for team alpha: the quokka credentials rotate nightly."
TEAM_B_SECRET = "Wombat escalation notes for team bravo: the wombat pager lives in vault."


@dataclass
class Corpus:
    version_id: int
    ids: dict[str, list[int]]

    def one(self, key: str) -> int:
        return self.ids[key][0]


def make_docs() -> list[FixtureDoc]:
    return [
        FixtureDoc(
            "content/en/docs/concepts/pods.md",
            "1.31",
            "Pods",
            [(["Overview"], PUBLIC_POD)],
        ),
        FixtureDoc(
            "content/en/docs/concepts/services.md",
            "1.31",
            "Service",
            [(["Overview"], PUBLIC_SVC)],
        ),
        FixtureDoc(
            "content/en/docs/tasks/probes.md",
            "1.30",
            "Probes",
            [(["Liveness"], PUBLIC_TASK)],
            doc_type=DocType.TASK,
        ),
        FixtureDoc(
            "content/en/docs/internal/alpha.md",
            "1.31",
            "Alpha runbook",
            [(["Rotation"], TEAM_A_SECRET)],
            access_group="team-a",
        ),
        FixtureDoc(
            "content/en/docs/internal/bravo.md",
            "1.31",
            "Bravo notes",
            [(["Pager"], TEAM_B_SECRET)],
            access_group="team-b",
        ),
    ]


@pytest.fixture
async def corpus(db: object, fake_embedder: FakeEmbedder) -> Corpus:
    vid = await create_corpus_version(db)  # type: ignore[arg-type]
    ids = await insert_docs(
        db,  # type: ignore[arg-type]
        make_docs(),
        corpus_version_id=vid,
        embedders=[fake_embedder],
    )
    return Corpus(vid, ids)


@pytest.fixture
async def big_corpus(db: object, fake_embedder: FakeEmbedder) -> int:
    """200 chunks; version 1.31 holds 20 of them (10%)."""
    vid = await create_corpus_version(db)  # type: ignore[arg-type]
    docs = []
    for version, n in (("1.30", 180), ("1.31", 20)):
        sections = [
            ([f"S{i}"], f"widget{i} gadget{i % 7} sprocket{i % 13} flange{i % 5} common words here")
            for i in range(n)
        ]
        docs.append(
            FixtureDoc(f"content/en/docs/big-{version}.md", version, f"Big {version}", sections)
        )
    await insert_docs(db, docs, corpus_version_id=vid, embedders=[fake_embedder])  # type: ignore[arg-type]
    return vid


@pytest.fixture
async def pool(pg_dsn: str) -> AsyncIterator[AsyncConnectionPool]:
    p = create_pool(pg_dsn, min_size=1, max_size=3)
    await p.open()
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
def retriever(pool: AsyncConnectionPool, fake_embedder: FakeEmbedder) -> Retriever:
    return Retriever(pool, {"gte-small": fake_embedder})
