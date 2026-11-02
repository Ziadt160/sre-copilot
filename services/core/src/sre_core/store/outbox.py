"""The transactional outbox: side work (feedback sync, audit fan-out) queued in Postgres.

Delivery is at-least-once: a row is marked `done` only after its handler returned, so a crash in
between runs the handler again. Handlers must therefore be idempotent. A kind without a
registered handler is left pending (its module may simply not be loaded in this process).

    python -m sre_core.store.outbox drain [--handlers pkg.module,...]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from sre_shared import aio
from sre_shared.db import create_pool

log = logging.getLogger(__name__)

Handler = Callable[[Mapping[str, Any]], Awaitable[None]]
_HANDLERS: dict[str, Handler] = {}

MAX_ATTEMPTS = 5
HANDLER_TIMEOUT_S = 30.0


def register_outbox_handler(kind: str, fn: Handler) -> None:
    """Register `fn(payload)` for `kind`. Registering a kind again replaces the handler."""
    _HANDLERS[kind] = fn


def registered_kinds() -> list[str]:
    return sorted(_HANDLERS)


class PgOutbox:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self.pool = pool

    async def enqueue(self, kind: str, dedupe_key: str, payload: Mapping[str, Any]) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "insert into public.outbox (kind, dedupe_key, payload) values (%s, %s, %s) "
                "on conflict (dedupe_key) do nothing",
                [kind, dedupe_key, Jsonb(dict(payload))],
            )

    async def drain(self, limit: int = 50) -> int:
        """Run handlers for pending rows; returns how many finished. Each row is processed in
        its own transaction holding the row lock (`skip locked`), so two drainers never run the
        same row at once."""
        done = 0
        failed: list[int] = []  # not retried within this drain: a retry waits for the next one
        kinds = list(_HANDLERS)
        if not kinds:
            return 0
        for _ in range(limit):
            async with self.pool.connection() as conn:
                cur = await conn.execute(
                    "select id, kind, payload, attempts from public.outbox "
                    "where status = 'pending' and kind = any(%s) and id <> all(%s) "
                    "order by id limit 1 for update skip locked",
                    [kinds, failed],
                )
                row = await cur.fetchone()
                if row is None:
                    break
                try:
                    async with asyncio.timeout(HANDLER_TIMEOUT_S):
                        await _HANDLERS[row["kind"]](row["payload"])
                except Exception as exc:  # a failing handler must not stop the others
                    attempts = row["attempts"] + 1
                    status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
                    log.warning("outbox %s attempt %d failed: %s", row["kind"], attempts, exc)
                    failed.append(row["id"])
                    await conn.execute(
                        "update public.outbox set attempts = %s, status = %s, last_error = %s "
                        "where id = %s",
                        [attempts, status, str(exc)[:500], row["id"]],
                    )
                    continue
                await conn.execute(
                    "update public.outbox set status = 'done', attempts = attempts + 1, "
                    "processed_at = now(), last_error = null where id = %s",
                    [row["id"]],
                )
                done += 1
        return done


class OutboxDrainer:
    """Best-effort background draining: `kick()` after a request starts one drain unless one is
    already running. The tasks are held here so they are not garbage collected mid-run."""

    def __init__(self, outbox: PgOutbox) -> None:
        self.outbox = outbox
        self._task: asyncio.Task[None] | None = None

    def kick(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        try:
            await self.outbox.drain()
        except Exception:
            log.exception("outbox drain failed")

    async def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


async def _drain_cli(handlers: list[str]) -> int:
    for module in handlers:
        importlib.import_module(module)
    pool = create_pool(min_size=1, max_size=2)
    await pool.open()
    try:
        return await PgOutbox(pool).drain()
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sre_core.store.outbox")
    sub = parser.add_subparsers(dest="cmd", required=True)
    drain = sub.add_parser("drain", help="run handlers for pending outbox rows once")
    drain.add_argument(
        "--handlers",
        default="",
        help="comma-separated modules that call register_outbox_handler on import",
    )
    args = parser.parse_args(argv)
    modules = [m for m in args.handlers.split(",") if m]
    print(f"drained {aio.run(_drain_cli(modules))} outbox rows")


if __name__ == "__main__":
    main()
