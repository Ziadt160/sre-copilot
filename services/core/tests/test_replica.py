"""Read routing against a real streaming replica (the `replica` compose profile).

Skipped unless both URLs are set (the database must already hold the schema, see
infra/postgres/schema.py):

    SRE_TEST_PRIMARY_URL=postgresql://postgres:postgres@127.0.0.1:54410/sre
    SRE_TEST_REPLICA_URL=postgresql://postgres:postgres@127.0.0.1:54411/sre

The tests write a tiny corpus to the lab primary and truncate it afterwards.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import pytest
from psycopg_pool import AsyncConnectionPool

from sre_core.retrieval import Retriever
from sre_shared.db import connect, create_pool
from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import RetrievalRequest
from sre_shared.testing.plugin import DATA_TABLES
from sre_shared.testing.seed import FixtureDoc, create_corpus_version, insert_docs

PRIMARY_URL = os.environ.get("SRE_TEST_PRIMARY_URL")
REPLICA_URL = os.environ.get("SRE_TEST_REPLICA_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (PRIMARY_URL and REPLICA_URL),
        reason="set SRE_TEST_PRIMARY_URL and SRE_TEST_REPLICA_URL (compose profile `replica`)",
    ),
]

QUERY = "Pods run containers sharing network and storage"
DEAD_URL = "postgresql://postgres:postgres@127.0.0.1:1/sre"  # nothing listens on port 1


class SpyPool(AsyncConnectionPool):
    """Records `pg_is_in_recovery()` of every connection the retriever takes from the pool."""

    recovery_flags: list[bool]

    def connection(self, *args: Any, **kwargs: Any) -> AbstractAsyncContextManager[Any]:
        outer = super().connection(*args, **kwargs)

        @asynccontextmanager
        async def spied() -> AsyncIterator[Any]:
            async with outer as conn:
                row = await (await conn.execute("select pg_is_in_recovery() as r")).fetchone()
                self.recovery_flags.append(bool(row["r"]))
                yield conn

        return spied()


def _spy_pool(dsn: str) -> SpyPool:
    real = create_pool(dsn, min_size=1, max_size=2)
    spy = SpyPool(
        dsn,
        min_size=1,
        max_size=2,
        kwargs=real.kwargs,
        configure=real._configure,  # noqa: SLF001
        timeout=2,
        open=False,
    )
    spy.recovery_flags = []
    return spy


@pytest.fixture
async def seeded(fake_embedder: FakeEmbedder) -> AsyncIterator[int]:
    """A tiny active corpus written on the primary and visible on the replica."""
    assert PRIMARY_URL and REPLICA_URL
    async with await connect(PRIMARY_URL, autocommit=True) as primary:
        vid = await create_corpus_version(primary)
        doc = FixtureDoc(
            "content/en/docs/concepts/pods.md",
            "1.31",
            "Pods",
            [(["Overview"], f"A Pod is the smallest deployable unit. {QUERY}.")],
        )
        await insert_docs(primary, [doc], corpus_version_id=vid, embedders=[fake_embedder])
        async with await connect(REPLICA_URL, autocommit=True) as replica:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                row = await (
                    await replica.execute(
                        "select count(*) as n from public.chunks where corpus_version_id = %s",
                        [vid],
                    )
                ).fetchone()
                if row and row["n"]:
                    break
                await asyncio.sleep(0.05)
            else:
                pytest.fail("the row did not reach the replica within 5 s")
        try:
            yield vid
        finally:
            await primary.execute(
                "truncate "
                + ", ".join(f"public.{t}" for t in DATA_TABLES)
                + " restart identity cascade"
            )


async def test_reads_go_to_the_replica(seeded: int, fake_embedder: FakeEmbedder) -> None:
    assert PRIMARY_URL and REPLICA_URL
    primary_pool = _spy_pool(PRIMARY_URL)
    replica_pool = _spy_pool(REPLICA_URL)
    await primary_pool.open()
    await replica_pool.open()
    try:
        retriever = Retriever(primary_pool, {"gte-small": fake_embedder}, read_pool=replica_pool)
        result = await retriever.retrieve(RetrievalRequest(query=QUERY, k=3))
    finally:
        await replica_pool.close()
        await primary_pool.close()
    assert result.chunks, "the replica returned no chunks"
    assert result.chunks[0].source_path == "content/en/docs/concepts/pods.md"
    assert replica_pool.recovery_flags == [True]  # the read connection was a standby
    assert primary_pool.recovery_flags == []  # and the primary was not touched


async def test_dead_replica_falls_back_to_the_primary(
    seeded: int, fake_embedder: FakeEmbedder
) -> None:
    assert PRIMARY_URL
    primary_pool = _spy_pool(PRIMARY_URL)
    dead_pool = _spy_pool(DEAD_URL)
    await primary_pool.open()
    await dead_pool.open()
    try:
        retriever = Retriever(primary_pool, {"gte-small": fake_embedder}, read_pool=dead_pool)
        result = await retriever.retrieve(RetrievalRequest(query=QUERY, k=3))
    finally:
        await dead_pool.close()
        await primary_pool.close()
    assert result.chunks
    assert dead_pool.recovery_flags == []
    assert primary_pool.recovery_flags == [False]  # served by the primary
