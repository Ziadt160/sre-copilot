"""Postgres access for mcp-ops: the operations table (audit log + idempotency), incidents, pages.

Every unit of work runs `set local role mcp_ops` first. That role can touch only operations,
incidents and pages (and read approvals), so a bug or an injected query here cannot reach threads,
messages or feedback: the database enforces the boundary, not this module.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from sre_shared.contracts.ops import TERMINAL_OPERATION

Row = dict[str, Any]

# A refusal or a superseded row must not hold the idempotency key: the same call, properly
# approved, has to be able to run later. The real key stays as a prefix so the audit log can
# still be searched by it.
REFUSED_SUFFIX = ":refused:"
REPLAY_SUFFIX = ":replay:"


class OpsStore:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self._pool = pool

    @asynccontextmanager
    async def tx(self) -> AsyncIterator[AsyncConnection[Row]]:
        async with self._pool.connection() as conn, conn.transaction():
            await conn.execute("set local role mcp_ops")
            yield conn

    # --- operations ---------------------------------------------------------------------------

    async def insert_first(self, row: Row) -> tuple[Row, bool]:
        """Insert-first idempotency: (row, created). On a key conflict the stored row comes back
        and nothing is written, so a replay can never create a second operation."""
        async with self.tx() as conn:
            cur = await conn.execute(
                """
                insert into public.operations
                    (id, idempotency_key, kind, tool, args, namespace, target, subject, scopes,
                     thread_id, turn, approval_id, status)
                values (%(id)s, %(idempotency_key)s, %(kind)s, %(tool)s, %(args)s, %(namespace)s,
                        %(target)s, %(subject)s, %(scopes)s, %(thread_id)s, %(turn)s,
                        %(approval_id)s, %(status)s)
                on conflict (idempotency_key) where kind <> 'read' do nothing
                returning *
                """,
                {**row, "args": Jsonb(row["args"])},
            )
            made = await cur.fetchone()
            if made is not None:
                return made, True
            cur = await conn.execute(
                "select * from public.operations where idempotency_key = %s and kind <> 'read'",
                [row["idempotency_key"]],
            )
            stored = await cur.fetchone()
        assert stored is not None  # the conflict proves it exists; rows are never deleted
        return stored, False

    async def insert_audit(self, row: Row) -> Row:
        """A row that is not an idempotent write: a refusal, or a read about to run."""
        async with self.tx() as conn:
            cur = await conn.execute(
                """
                insert into public.operations
                    (id, idempotency_key, kind, tool, args, namespace, target, subject, scopes,
                     thread_id, turn, approval_id, status, error, latency_ms, finished_at)
                values (%(id)s, %(idempotency_key)s, %(kind)s, %(tool)s, %(args)s, %(namespace)s,
                        %(target)s, %(subject)s, %(scopes)s, %(thread_id)s, %(turn)s,
                        %(approval_id)s, %(status)s, %(error)s, %(latency_ms)s,
                        case when %(status)s = 'running' then null else now() end)
                returning *
                """,
                {**row, "args": Jsonb(row["args"])},
            )
            made = await cur.fetchone()
        assert made is not None
        return made

    async def finish_read(
        self, op_id: uuid.UUID, status: str, result: Row | None, error: str | None, ms: float
    ) -> None:
        async with self.tx() as conn:
            await conn.execute(
                """
                update public.operations
                   set status = %s, result = %s, error = %s, latency_ms = %s, finished_at = now()
                 where id = %s and kind = 'read' and status = 'running'
                """,
                [status, Jsonb(result) if result is not None else None, error, ms, op_id],
            )

    async def get(self, op_id: uuid.UUID) -> Row | None:
        async with self.tx() as conn:
            cur = await conn.execute("select * from public.operations where id = %s", [op_id])
            return await cur.fetchone()

    async def get_by_key(self, key: str) -> Row | None:
        async with self.tx() as conn:
            cur = await conn.execute(
                "select * from public.operations where idempotency_key = %s and kind <> 'read'",
                [key],
            )
            return await cur.fetchone()

    async def claim(self, op_id: uuid.UUID, task_id: str | None) -> Row | None:
        """approved -> running, atomically; None when the row is not (or no longer) approved.
        The approval is checked again here, in the same statement, so an approval that expired
        between admission and execution never runs. Exactly one caller can win."""
        async with self.tx() as conn:
            cur = await conn.execute(
                """
                update public.operations o
                   set status = 'running', started_at = now(), task_id = %s
                 where o.id = %s and o.status = 'approved'
                   and (o.approval_id is null or exists (
                        select 1 from public.approvals a
                         where a.id = o.approval_id and a.status = 'approved'
                           and a.expires_at > now()))
                returning o.*
                """,
                [task_id, op_id],
            )
            return await cur.fetchone()

    async def refuse_unclaimed(self, op: Row, reason: str) -> Row | None:
        """An approved row whose approval lapsed before it could start: refuse it and free the
        key. Only touches rows still `approved`."""
        async with self.tx() as conn:
            cur = await conn.execute(
                """
                update public.operations
                   set status = 'refused', error = %s, finished_at = now(),
                       idempotency_key = idempotency_key || %s || id::text
                 where id = %s and status = 'approved'
                returning *
                """,
                [reason, REFUSED_SUFFIX, op["id"]],
            )
            return await cur.fetchone()

    async def finish(
        self,
        op_id: uuid.UUID,
        status: str,
        result: Row | None,
        error: str | None,
        *,
        extra: tuple[str, list[Any]] | None = None,
    ) -> Row | None:
        """running -> succeeded|failed; latency is measured by the database from `started_at`.
        `extra` is one more statement to run in the same transaction (an incident row), so the
        side effect and the status change commit together."""
        assert status in TERMINAL_OPERATION
        async with self.tx() as conn:
            if extra is not None:
                await conn.execute(extra[0], extra[1])
            cur = await conn.execute(
                """
                update public.operations
                   set status = %s, result = %s, error = %s, finished_at = now(),
                       latency_ms = extract(epoch from (clock_timestamp() - started_at)) * 1000
                 where id = %s and status in ('running', 'unknown')
                returning *
                """,
                [status, Jsonb(result) if result is not None else None, error, op_id],
            )
            return await cur.fetchone()

    async def running(self, min_age_s: float = 0.0) -> list[Row]:
        """`running` rows that started at least `min_age_s` ago: younger ones may belong to a
        live instance (a rolling deploy starts the new pod while the old one finishes)."""
        async with self.tx() as conn:
            cur = await conn.execute(
                """
                select * from public.operations
                 where status = 'running'
                   and coalesce(started_at, created_at) <= now() - make_interval(secs => %s)
                 order by created_at
                """,
                [min_age_s],
            )
            return await cur.fetchall()

    async def mark_unknown(self, op_id: uuid.UUID) -> Row | None:
        async with self.tx() as conn:
            cur = await conn.execute(
                """
                update public.operations set status = 'unknown'
                 where id = %s and status = 'running' returning *
                """,
                [op_id],
            )
            return await cur.fetchone()

    async def settle_unknown(
        self, op_id: uuid.UUID, status: str, result: Row | None, error: str | None
    ) -> None:
        async with self.tx() as conn:
            await conn.execute(
                """
                update public.operations
                   set status = %s, result = %s, error = %s, finished_at = now()
                 where id = %s and status = 'unknown'
                """,
                [status, Jsonb(result) if result is not None else None, error, op_id],
            )

    # --- approvals (read only) ---------------------------------------------------------------

    async def get_approval(self, approval_id: uuid.UUID) -> Row | None:
        """The approval plus `live`: whether it has not expired, by the database clock."""
        async with self.tx() as conn:
            cur = await conn.execute(
                "select *, expires_at > now() as live from public.approvals where id = %s",
                [approval_id],
            )
            return await cur.fetchone()

    # --- escalation ---------------------------------------------------------------------------

    async def get_incident(self, incident_id: uuid.UUID) -> Row | None:
        async with self.tx() as conn:
            cur = await conn.execute("select * from public.incidents where id = %s", [incident_id])
            return await cur.fetchone()

    async def incident_for_operation(self, op_id: uuid.UUID) -> Row | None:
        async with self.tx() as conn:
            cur = await conn.execute(
                "select * from public.incidents where operation_id = %s", [op_id]
            )
            return await cur.fetchone()

    async def page_for_operation(self, op_id: uuid.UUID) -> Row | None:
        async with self.tx() as conn:
            cur = await conn.execute("select * from public.pages where operation_id = %s", [op_id])
            return await cur.fetchone()
