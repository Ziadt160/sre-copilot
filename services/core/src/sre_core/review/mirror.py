"""Pull human labels from the LangSmith review queue into `review_labels`.

LangSmith keeps traces for 14 days on the free plan; the labels are the durable part, so they live
in our database. One row per LangSmith feedback item, upserted on (source, external_id): running
the mirror again changes nothing unless a reviewer edited a label.

    uv run python -m sre_core.review.mirror [--queue sre-copilot-review]

Needs LANGSMITH_API_KEY and DATABASE_URL (a role that bypasses RLS: this is an operator job).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection

from sre_shared import aio
from sre_shared.db import connect
from sre_shared.settings import get_settings

from .tracing import default_client

QUEUE_NAME = "sre-copilot-review"
SOURCE = "langsmith"
# Annotations typed in the LangSmith UI arrive as source type "app"; model/API feedback
# (our own user_rating thumbs) is not a review label.
HUMAN_SOURCES = ["app"]

_UPSERT = """
insert into public.review_labels (source, external_id, run_id, thread_id, turn, reviewer, label)
values (%(source)s, %(external_id)s, %(run_id)s, %(thread_id)s, %(turn)s, %(reviewer)s, %(label)s)
on conflict (source, external_id) do update
   set label = excluded.label, reviewer = excluded.reviewer
returning (xmax = 0) as inserted
"""

_FIND_MESSAGE = "select thread_id, turn from public.messages where trace_run_id = %s limit 1"


@dataclass
class MirrorResult:
    runs: int = 0
    inserted: int = 0
    updated: int = 0


def _label(fb: Any) -> dict[str, Any]:
    return {
        "key": fb.key,
        "score": fb.score,
        "value": fb.value,
        "comment": fb.comment,
        "correction": fb.correction,
    }


def _reviewer(fb: Any) -> str | None:
    meta = getattr(getattr(fb, "feedback_source", None), "metadata", None) or {}
    who = meta.get("user_id") or meta.get("user_name")
    return str(who) if who else None


def _sync_fetch(client: Any, queue_name: str) -> list[tuple[str, list[Any]]]:
    """(run id, human feedback) for every completed queue item. Synchronous SDK calls."""
    queue = next(iter(client.list_annotation_queues(name=queue_name)), None)
    if queue is None:
        raise SystemExit(f"annotation queue {queue_name!r} not found; run tools/langsmith_queues.py")
    out: list[tuple[str, list[Any]]] = []
    for run in client.list_runs_from_annotation_queue(queue.id, status="completed"):
        feedback = list(
            client.list_feedback(run_ids=[run.id], feedback_source_type=HUMAN_SOURCES)
        )
        out.append((str(run.id), feedback))
    return out


async def mirror_labels(
    conn: AsyncConnection[Any],
    client: Any,
    *,
    queue_name: str = QUEUE_NAME,
) -> MirrorResult:
    items = await asyncio.to_thread(_sync_fetch, client, queue_name)
    result = MirrorResult(runs=len(items))
    for run_id, feedback in items:
        located = await (await conn.execute(_FIND_MESSAGE, (run_id,))).fetchone()
        for fb in feedback:
            params = {
                "source": SOURCE,
                "external_id": str(fb.id),
                "run_id": run_id,
                "thread_id": located["thread_id"] if located else None,
                "turn": located["turn"] if located else None,
                "reviewer": _reviewer(fb),
                "label": json.dumps(_label(fb), default=str),
            }
            row = await (await conn.execute(_UPSERT, params)).fetchone()
            if row and row["inserted"]:
                result.inserted += 1
            else:
                result.updated += 1
    await conn.commit()
    return result


async def _main(queue_name: str, client_factory: Callable[[str], Any]) -> MirrorResult:
    settings = get_settings()
    key = settings.langsmith_api_key
    if key is None:
        raise SystemExit("LANGSMITH_API_KEY is not set")
    conn = await connect()
    try:
        return await mirror_labels(conn, client_factory(key.get_secret_value()), queue_name=queue_name)
    finally:
        await conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--queue", default=QUEUE_NAME)
    args = parser.parse_args(argv)
    result = aio.run(_main(args.queue, default_client))
    print(f"{result.runs} runs: {result.inserted} labels added, {result.updated} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
