"""Startup reconciler: settle operations a previous process left `running`.

A crash between "claimed" and "recorded" leaves a row that says running while nobody is running
it. Each such row becomes `unknown`, then the backend's (or the database's) end state decides
`succeeded` or `failed`. A row whose end state cannot be read stays `unknown`: honest beats
guessed. Nothing is ever re-run here, so reconciling cannot duplicate a change.
"""

from __future__ import annotations

import logging
from typing import Any

from sre_mcp_ops.backends.base import ClusterBackend
from sre_mcp_ops.store import OpsStore

log = logging.getLogger(__name__)


async def reconcile(store: OpsStore, backend: ClusterBackend) -> list[dict[str, Any]]:
    settled: list[dict[str, Any]] = []
    for row in await store.running():
        if row["kind"] == "read":
            await store.finish_read(row["id"], "failed", None, "interrupted by a restart", 0.0)
            continue
        await store.mark_unknown(row["id"])
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
            log.exception("could not reconcile operation %s; left unknown", row["id"])
            continue
        status = "succeeded" if ok else "failed"
        error = None if ok else "interrupted before the change was applied"
        await store.settle_unknown(row["id"], status, result if ok else None, error)
        settled.append({"id": str(row["id"]), "status": status})
    return settled
