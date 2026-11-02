"""LangGraph checkpointer on the primary database, in schema `agent_ckpt`.

`AsyncPostgresSaver` needs autocommit connections with dict rows; its tables are created
unqualified, so the search path points at the schema the migration made. `setup()` runs
`CREATE INDEX CONCURRENTLY`, which cannot run inside a transaction: call it from the CLI
(`python -m sre_core.store.checkpoints setup`), not per request.
"""

from __future__ import annotations

import argparse
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from sre_shared import aio
from sre_shared.settings import get_settings

CHECKPOINT_SCHEMA = "agent_ckpt"


def make_checkpoint_pool(
    dsn: str | None = None, *, min_size: int = 1, max_size: int = 4
) -> AsyncConnectionPool[Any]:
    """An unopened autocommit pool for the saver; the caller opens and closes it."""
    return AsyncConnectionPool(
        dsn or get_settings().database_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": f"-c search_path={CHECKPOINT_SCHEMA}",
            "connect_timeout": 5,
        },
        open=False,
    )


def make_checkpointer(pool: AsyncConnectionPool[Any]) -> AsyncPostgresSaver:
    return AsyncPostgresSaver(pool)


async def setup_checkpointer(dsn: str | None = None) -> None:
    pool = make_checkpoint_pool(dsn, max_size=1)
    await pool.open()
    try:
        await make_checkpointer(pool).setup()
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sre_core.store.checkpoints")
    parser.add_argument("cmd", choices=["setup"])
    parser.parse_args(argv)
    aio.run(setup_checkpointer())
    print(f"checkpoint tables ready in schema {CHECKPOINT_SCHEMA}")


if __name__ == "__main__":
    main()
