"""Postgres access: psycopg 3 (async) with pgvector types, and the RLS-scoped user context."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from psycopg import AsyncConnection, OperationalError, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from sre_shared.models import Claims
from sre_shared.settings import get_settings

# Supabase installs pgvector in the `extensions` schema; keep it on the path so halfvec resolves.
SEARCH_PATH_OPTION = "-c search_path=public,extensions"


async def _configure(conn: AsyncConnection[Any]) -> None:
    await register_vector_async(conn)


# New connections are retried, queries never are. The lab's Docker port forward drops about 1 in
# 25 connects (measured 2026-11-01), and a hosted database's network blips look the same.
CONNECT_ATTEMPTS = 4
CONNECT_TIMEOUT_S = 5


def is_transient_connect_error(exc: OperationalError) -> bool:
    """A dropped or timed-out connect is worth retrying; a server rejection (bad password,
    missing database) is final. psycopg sets no SQLSTATE on connect errors, so read the text."""
    return "FATAL:" not in str(exc)


def _backoff(attempt: int) -> float:
    return 0.25 * float(2**attempt)  # 0.5 s, 1 s, 2 s


async def connect(dsn: str | None = None, *, autocommit: bool = False) -> AsyncConnection[DictRow]:
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            conn = await AsyncConnection.connect(
                dsn or get_settings().database_url,
                autocommit=autocommit,
                options=SEARCH_PATH_OPTION,
                row_factory=dict_row,
                connect_timeout=CONNECT_TIMEOUT_S,
            )
            break
        except OperationalError as exc:
            if attempt == CONNECT_ATTEMPTS or not is_transient_connect_error(exc):
                raise
            await asyncio.sleep(_backoff(attempt))
    await _configure(conn)
    return conn


def connect_sync(
    dsn: str | None = None, *, autocommit: bool = False
) -> psycopg.Connection[DictRow]:
    """The synchronous twin of `connect`, for batch jobs such as ingest; same retry rule."""
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            conn = psycopg.connect(
                dsn or get_settings().database_url,
                autocommit=autocommit,
                options=SEARCH_PATH_OPTION,
                row_factory=dict_row,
                connect_timeout=CONNECT_TIMEOUT_S,
            )
            break
        except OperationalError as exc:
            if attempt == CONNECT_ATTEMPTS or not is_transient_connect_error(exc):
                raise
            time.sleep(_backoff(attempt))
    register_vector(conn)
    return conn


def create_pool(
    dsn: str | None = None,
    *,
    min_size: int | None = None,
    max_size: int | None = None,
    timeout: float = 30.0,
) -> AsyncConnectionPool:
    """An unopened pool; call `await pool.open()` at startup (FastAPI lifespan). `timeout` is
    how long a request waits for a connection before `PoolTimeout`; a pool with a fallback
    (the lab replica) should give up quickly."""
    settings = get_settings()
    return AsyncConnectionPool(
        dsn or settings.database_url,
        min_size=min_size or settings.db_pool_min,
        max_size=max_size or settings.db_pool_max,
        timeout=timeout,
        # The pool retries failed connects itself; the timeout keeps a swallowed one from
        # hanging a worker (seen with the lab's Docker port forward).
        kwargs={
            "options": SEARCH_PATH_OPTION,
            "row_factory": dict_row,
            "connect_timeout": CONNECT_TIMEOUT_S,
        },
        configure=_configure,
        open=False,
    )


@asynccontextmanager
async def as_user(
    conn: AsyncConnection[Any], claims: Claims
) -> AsyncIterator[AsyncConnection[Any]]:
    """Run statements inside a transaction as `claims.role` with `request.jwt.claims` set,
    so RLS policies apply exactly as they would for a Supabase API request."""
    async with conn.transaction():
        await conn.execute(sql.SQL("set local role {}").format(sql.Identifier(claims.role)))
        await conn.execute(
            "select set_config('request.jwt.claims', %s, true)",
            [json.dumps(claims.to_jwt_claims())],
        )
        yield conn
