"""Feedback and implicit UI events, stored as the user so RLS enforces ownership."""

from __future__ import annotations

from typing import Any

from psycopg_pool import AsyncConnectionPool

from sre_shared.contracts.agent import FeedbackRequest, UiEventRequest
from sre_shared.db import as_user
from sre_shared.models import Claims


class PgFeedbackStore:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self.pool = pool

    async def save_feedback(self, claims: Claims, req: FeedbackRequest) -> None:
        """Upsert the turn's rating and the per-source marks. A row for a thread the caller
        does not own is refused by the `with check` of the RLS policy (psycopg raises)."""
        async with self.pool.connection() as conn, as_user(conn, claims):
            if req.rating is not None:
                await conn.execute(
                    "insert into public.feedback (thread_id, turn, owner_sub, rating, reason) "
                    "values (%s, %s, %s, %s, %s) "
                    "on conflict (thread_id, turn, owner_sub) do update set "
                    "rating = excluded.rating, reason = excluded.reason",
                    [req.thread_id, req.turn, claims.sub, req.rating, req.reason],
                )
            for chunk_id, useful in req.sources.items():
                await conn.execute(
                    "insert into public.feedback_sources "
                    "(thread_id, turn, chunk_id, owner_sub, useful) values (%s, %s, %s, %s, %s) "
                    "on conflict (thread_id, turn, chunk_id, owner_sub) do update set "
                    "useful = excluded.useful",
                    [req.thread_id, req.turn, chunk_id, claims.sub, useful],
                )

    async def save_event(self, claims: Claims, req: UiEventRequest) -> None:
        async with self.pool.connection() as conn, as_user(conn, claims):
            await conn.execute(
                "insert into public.ui_events (thread_id, turn, owner_sub, kind) "
                "values (%s, %s, %s, %s)",
                [req.thread_id, req.turn, claims.sub, req.kind],
            )
