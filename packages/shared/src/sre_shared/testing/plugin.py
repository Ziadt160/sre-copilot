"""pytest plugin with shared fixtures. Loaded from the repo-root conftest.py.

Fixtures
- pg_admin_dsn (session): admin DSN of the test Postgres server (see testing.pg).
- pg_dsn (session): a fresh, migrated database for this test session.
- db: an async connection to pg_dsn as the table owner; tables are truncated after each test.
- fake_embedder: deterministic model-free embedder (gte-small dims).
- user_a / user_b: Claims for RLS tests; user_a can read group "team-a", user_b cannot.

Integration tests carry `@pytest.mark.integration`; they are skipped when neither Docker nor
SRE_TEST_DATABASE_URL is available.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Iterator

import pytest

from sre_shared.embed.fake import FakeEmbedder
from sre_shared.models import Claims

if sys.platform == "win32":
    # psycopg async needs a selector loop; pytest-asyncio creates loops from the policy.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DATA_TABLES = (
    # Phase 3 agent tables (truncate cascades through their foreign keys)
    "semantic_cache",
    "review_labels",
    "ui_events",
    "feedback_sources",
    "feedback",
    "outbox",
    "pages",
    "incidents",
    "operations",
    "approvals",
    "messages",
    "threads",
    # Phases 1-2
    "eval_scores",
    "eval_runs",
    "embedding_cache",
    "chunks",
    "documents",
    "corpus_versions",
)


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def pg_admin_dsn() -> Iterator[str]:
    from sre_shared.testing.pg import admin_server

    if not os.environ.get("SRE_TEST_DATABASE_URL") and not _docker_available():
        pytest.skip("integration tests need Docker or SRE_TEST_DATABASE_URL")
    with admin_server() as dsn:
        yield dsn


@pytest.fixture(scope="session")
def pg_dsn(pg_admin_dsn: str) -> Iterator[str]:
    from sre_shared.testing.pg import session_database

    with session_database(pg_admin_dsn) as dsn:
        yield dsn


@pytest.fixture
async def db(pg_dsn: str) -> AsyncIterator[object]:
    from sre_shared.db import connect

    conn = await connect(pg_dsn, autocommit=True)
    try:
        yield conn
    finally:
        await conn.execute(
            "truncate "
            + ", ".join(f"public.{t}" for t in DATA_TABLES)
            + " restart identity cascade"
        )
        await conn.close()


@pytest.fixture(scope="session")
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder("gte-small")


@pytest.fixture
def user_a() -> Claims:
    return Claims(sub="user-a", groups=["team-a"])


@pytest.fixture
def user_b() -> Claims:
    return Claims(sub="user-b", groups=["team-b"])
