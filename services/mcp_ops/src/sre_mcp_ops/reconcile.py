"""Settling operations nobody is running: at startup, periodically, and after a cancelled body.

A crash (or a cancel) between "claimed" and "recorded" leaves a row that says running while
nobody runs it. Each such row becomes `unknown`, then the backend's (or the database's) end
state decides `succeeded` or `failed`. A row whose end state cannot be read stays `unknown`:
honest beats guessed. Nothing is ever re-run here, so settling cannot duplicate a change.

Startup and the periodic pass only touch rows that started more than `min_age_s` ago (the task
redelivery timeout): a younger `running` row may belong to a live instance, as in a rolling
deploy, and must be left to finish.
"""

from __future__ import annotations

import logging
from typing import Any

from sre_mcp_ops.backends.base import ClusterBackend
from sre_mcp_ops.store import OpsStore, Row

log = logging.getLogger(__name__)


async def settle_row(
    store: OpsStore, backend: ClusterBackend, row: Row, why: str = "interrupted"
) -> dict[str, Any] | None:
    """running -> unknown -> succeeded|failed. None when the row is no longer `running`
    (somebody finished it) or its end state could not be read (left `unknown`)."""
    if await store.mark_unknown(row["id"]) is None:
        return None
    result: dict[str, Any] = {"reconciled": True}
    try:
        if row["tool"] == "create_incident":
            inc = await store.incident_for_operation(row["id"])
            ok = inc is not None
            if inc is not None:
                result["incident_id"] = str(inc["id"])
        elif row["tool"] == "page_oncall":
            ok = await store.page_for_operation(row["id"]) is not None
        else:
            ok = await backend.applied(
                row["tool"],
                row["args"],
                key=row["idempotency_key"],
                at=row["created_at"].isoformat(),
            )
    except Exception:
        log.exception("could not settle operation %s; left unknown", row["id"])
        return None
    status = "succeeded" if ok else "failed"
    error = None if ok else f"{why} before the change was applied"
    await store.settle_unknown(row["id"], status, result if ok else None, error)
    return {"id": str(row["id"]), "status": status}


async def reconcile(
    store: OpsStore, backend: ClusterBackend, *, min_age_s: float = 120.0
) -> list[dict[str, Any]]:
    settled: list[dict[str, Any]] = []
    for row in await store.running(min_age_s):
        if row["kind"] == "read":
            await store.finish_read(row["id"], "failed", None, "interrupted by a restart", 0.0)
            continue
        item = await settle_row(store, backend, row)
        if item is not None:
            settled.append(item)
    return settled
