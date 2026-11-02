"""Threads and messages in Postgres (the `ThreadStore` port).

Reads of a user's rows run `as_user`, so RLS decides what is visible exactly as for the REST API.
Writes by the service (messages, status) run as the table owner after the caller was checked.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from sre_core.agent.ports import (
    MessageRecord,
    ThreadBusy,
    ThreadNotFound,
    ThreadRecord,
    ThreadStatus,
)
from sre_shared.db import as_user
from sre_shared.models import Claims

_THREAD_COLS = "id, owner_sub, title, status, next_turn, summary, created_at, updated_at"
_MESSAGE_COLS = "thread_id, turn, role, content, route, citations, trace_run_id"

# A client_message_id is remembered in the outbox table (a row born `done`, so no drain picks
# it up): its unique dedupe_key is the only place a durable unique constraint exists without a
# migration, and it gives the claim below an atomic "first writer wins".
LEDGER_KIND = "client_message"


@dataclass(frozen=True)
class ClaimedTurn:
    thread: ThreadRecord
    turn: int
    replay: bool  # True when a retried send found its turn already claimed


def _is_user(claims: Claims) -> bool:
    return claims.role == "authenticated"


class PgThreadStore:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self.pool = pool

    async def create(self, claims: Claims, title: str | None = None) -> ThreadRecord:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"insert into public.threads (owner_sub, title) values (%s, %s) "
                f"returning {_THREAD_COLS}",
                [claims.sub, title],
            )
            return ThreadRecord.model_validate(await cur.fetchone())

    async def get(self, claims: Claims, thread_id: uuid.UUID) -> ThreadRecord | None:
        if not _is_user(claims):
            return None
        async with self.pool.connection() as conn, as_user(conn, claims):
            cur = await conn.execute(
                f"select {_THREAD_COLS} from public.threads where id = %s",
                [thread_id],
            )
            row = await cur.fetchone()
        return ThreadRecord.model_validate(row) if row else None

    async def list_threads(self, claims: Claims, limit: int = 50) -> list[ThreadRecord]:
        if not _is_user(claims):
            return []
        async with self.pool.connection() as conn, as_user(conn, claims):
            cur = await conn.execute(
                f"select {_THREAD_COLS} from public.threads order by updated_at desc, id limit %s",
                [limit],
            )
            return [ThreadRecord.model_validate(r) for r in await cur.fetchall()]

    async def claim_turn(self, claims: Claims, thread_id: uuid.UUID) -> int:
        """One conditional update: the row lock orders concurrent claims, RLS limits it to the
        owner, and `status = 'active'` refuses a thread that waits on an approval or a human."""
        if not _is_user(claims):
            raise ThreadNotFound(str(thread_id))
        async with self.pool.connection() as conn, as_user(conn, claims):
            cur = await conn.execute(
                "update public.threads set next_turn = next_turn + 1, updated_at = now() "
                "where id = %s and status = 'active' returning next_turn - 1 as turn",
                [thread_id],
            )
            row = await cur.fetchone()
            if row is not None:
                return int(row["turn"])
            cur = await conn.execute("select status from public.threads where id = %s", [thread_id])
            existing = await cur.fetchone()
        if existing is None:
            raise ThreadNotFound(str(thread_id))
        raise ThreadBusy(f"thread {thread_id} is {existing['status']}")

    async def claim_message(
        self,
        claims: Claims,
        thread_id: uuid.UUID | None,
        message: str,
        *,
        client_message_id: str | None = None,
        title: str | None = None,
    ) -> ClaimedTurn:
        """Start a turn atomically: create the thread when `thread_id` is None, claim the next
        turn and save the user message in one transaction. A retried `client_message_id`
        returns the turn it already claimed instead of a new one."""
        if not _is_user(claims):
            raise ThreadNotFound(str(thread_id))
        key = f"chat:{claims.sub}:{client_message_id}" if client_message_id else None
        async with self.pool.connection() as conn:
            if key is not None:
                # Serialises concurrent sends of the same id; the second sees the first's row.
                await conn.execute("select pg_advisory_xact_lock(hashtextextended(%s, 0))", [key])
                cur = await conn.execute(
                    "select payload from public.outbox where dedupe_key = %s", [key]
                )
                seen = await cur.fetchone()
                if seen is not None:
                    prior = uuid.UUID(seen["payload"]["thread_id"])
                    if thread_id is not None and thread_id != prior:
                        raise ValueError("client_message_id was used on another thread")
                    cur = await conn.execute(
                        f"select {_THREAD_COLS} from public.threads "
                        "where id = %s and owner_sub = %s",
                        [prior, claims.sub],
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise ThreadNotFound(str(prior))
                    return ClaimedTurn(
                        ThreadRecord.model_validate(row), int(seen["payload"]["turn"]), True
                    )
            if thread_id is None:
                cur = await conn.execute(
                    "insert into public.threads (owner_sub, title, next_turn) values (%s, %s, 2) "
                    "returning id",
                    [claims.sub, title],
                )
                created = await cur.fetchone()
                assert created is not None
                thread_id, turn = created["id"], 1
            else:
                cur = await conn.execute(
                    "update public.threads set next_turn = next_turn + 1, updated_at = now() "
                    "where id = %s and owner_sub = %s and status = 'active' "
                    "returning next_turn - 1 as turn",
                    [thread_id, claims.sub],
                )
                claimed = await cur.fetchone()
                if claimed is None:
                    cur = await conn.execute(
                        "select status from public.threads where id = %s and owner_sub = %s",
                        [thread_id, claims.sub],
                    )
                    existing = await cur.fetchone()
                    if existing is None:
                        raise ThreadNotFound(str(thread_id))
                    raise ThreadBusy(f"thread {thread_id} is {existing['status']}")
                turn = int(claimed["turn"])
            await conn.execute(
                "insert into public.messages (thread_id, turn, role, content) "
                "values (%s, %s, 'user', %s)",
                [thread_id, turn, message],
            )
            if key is not None:
                await conn.execute(
                    "insert into public.outbox (kind, dedupe_key, payload, status, processed_at) "
                    "values (%s, %s, %s, 'done', now())",
                    [LEDGER_KIND, key, Jsonb({"thread_id": str(thread_id), "turn": turn})],
                )
            cur = await conn.execute(
                f"select {_THREAD_COLS} from public.threads where id = %s",
                [thread_id],
            )
            return ClaimedTurn(ThreadRecord.model_validate(await cur.fetchone()), turn, False)

    async def add_message(self, msg: MessageRecord) -> None:
        """Insert, or complete the row for (thread, turn, role): the agent and the API both
        persist the assistant message, and whichever comes second must not fail or lose the
        trace id."""
        params: list[Any] = [
            msg.thread_id,
            msg.turn,
            msg.role,
            msg.content,
            msg.route,
            Jsonb([c.model_dump(mode="json") for c in msg.citations]),
            msg.trace_run_id,
        ]
        async with self.pool.connection() as conn:
            await conn.execute(
                f"insert into public.messages ({_MESSAGE_COLS}) "
                "values (%s, %s, %s, %s, %s, %s, %s) "
                "on conflict (thread_id, turn, role) do update set "
                "content = excluded.content, route = excluded.route, "
                "citations = excluded.citations, "
                "trace_run_id = coalesce(excluded.trace_run_id, public.messages.trace_run_id)",
                params,
            )

    async def messages(
        self, claims: Claims, thread_id: uuid.UUID, *, last: int = 20
    ) -> list[MessageRecord]:
        if not _is_user(claims):
            raise ThreadNotFound(str(thread_id))
        async with self.pool.connection() as conn, as_user(conn, claims):
            cur = await conn.execute("select 1 from public.threads where id = %s", [thread_id])
            if await cur.fetchone() is None:
                raise ThreadNotFound(str(thread_id))
            cur = await conn.execute(
                f"select {_MESSAGE_COLS} from ("
                f"select id, {_MESSAGE_COLS} from public.messages where thread_id = %s "
                "order by turn desc, id desc limit %s) recent order by turn, id",
                [thread_id, last],
            )
            return [MessageRecord.model_validate(r) for r in await cur.fetchall()]

    async def set_status(self, thread_id: uuid.UUID, status: ThreadStatus) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "update public.threads set status = %s, updated_at = now() where id = %s",
                [status, thread_id],
            )

    async def set_summary(self, thread_id: uuid.UUID, summary: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "update public.threads set summary = %s, updated_at = now() where id = %s",
                [summary, thread_id],
            )
