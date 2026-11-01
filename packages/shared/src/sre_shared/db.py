"""Postgres access: psycopg 3 (async) with pgvector types, and the RLS-scoped user context."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from sre_shared.models import Claims
from sre_shared.settings import get_settings

# Supabase installs pgvector in the `extensions` schema; keep it on the path so halfvec resolves.
SEARCH_PATH_OPTION = "-c search_path=public,extensions"


async def _configure(conn: AsyncConnection[Any]) -> None:
    await register_vector_async(conn)


async def connect(dsn: str | None = None, *, autocommit: bool = False) -> AsyncConnection[DictRow]:
    conn = await AsyncConnection.connect(
        dsn or get_settings().database_url,
        autocommit=autocommit,
        options=SEARCH_PATH_OPTION,
        row_factory=dict_row,
    )
    await _configure(conn)
    return conn


def create_pool(
    dsn: str | None = None, *, min_size: int | None = None, max_size: int | None = None
) -> AsyncConnectionPool:
    """An unopened pool; call `await pool.open()` at startup (FastAPI lifespan)."""
    settings = get_settings()
    return AsyncConnectionPool(
        dsn or settings.database_url,
        min_size=min_size or settings.db_pool_min,
        max_size=max_size or settings.db_pool_max,
        kwargs={"options": SEARCH_PATH_OPTION, "row_factory": dict_row},
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
