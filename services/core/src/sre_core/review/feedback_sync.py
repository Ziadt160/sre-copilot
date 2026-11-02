"""The `feedback_sync` outbox handler: a thumbs on a turn becomes LangSmith feedback on its trace.

The UI feedback row is already in Postgres; this copies it to the trace so reviewers see it and
the automation rule can route thumbs-down runs to the review queue. The outbox may deliver a job
more than once, so the LangSmith feedback id is derived from (trace run, user, key): a second
delivery finds that id and updates it in place instead of adding a row.

Outbox payload (enqueued by the feedback endpoint with dedupe key `feedback:<thread>:<turn>:<sub>`):

    {"trace_run_id": str | None, "owner_sub": str, "rating": -1 | 1, "reason": str | None}

Returns a short outcome string; a message with no trace (tracing was off when it was answered) or
no API key is a skip, not an error, so the outbox does not retry it forever.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from sre_shared.settings import Settings, get_settings

from .tracing import ClientFactory, default_client, tracing_enabled

FEEDBACK_KEY = "user_rating"
KIND = "feedback_sync"

# Fixed namespace: ids must be stable across processes and releases.
_NAMESPACE = uuid.UUID("0b8c4a1e-6f23-4d0a-9a5e-3c1f7d2e8b64")


def feedback_id(trace_run_id: str, owner_sub: str, key: str = FEEDBACK_KEY) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, f"{trace_run_id}:{owner_sub}:{key}")


def _send(client: Any, payload: Mapping[str, Any]) -> str:
    run_id = str(payload["trace_run_id"])
    fid = feedback_id(run_id, str(payload["owner_sub"]))
    score = 1 if int(payload["rating"]) > 0 else 0
    comment = payload.get("reason") or None
    existing = {
        str(f.id): f
        for f in client.list_feedback(run_ids=[run_id], feedback_key=[FEEDBACK_KEY])
    }
    if str(fid) in existing:
        old = existing[str(fid)]
        if old.score == score and (old.comment or None) == comment:
            return "unchanged"
        client.update_feedback(fid, score=score, comment=comment)
        return "updated"
    # extend_trace_retention defaults to True in the SDK, and extended retention is billed.
    client.create_feedback(
        run_id,
        FEEDBACK_KEY,
        score=score,
        comment=comment,
        feedback_id=fid,
        extend_trace_retention=False,
    )
    return "created"


async def feedback_sync(
    payload: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    client_factory: Callable[[str], Any] | None = None,
) -> str:
    settings = settings or get_settings()
    if not payload.get("trace_run_id"):
        return "skipped: no trace"
    if not tracing_enabled(settings):
        return "skipped: tracing off"
    key = settings.langsmith_api_key
    assert key is not None
    factory: ClientFactory | Callable[[str], Any] = client_factory or default_client
    client = factory(key.get_secret_value())
    # The SDK client is synchronous; keep the event loop free.
    return await asyncio.to_thread(_send, client, payload)


HANDLERS: dict[str, Callable[..., Any]] = {KIND: feedback_sync}
