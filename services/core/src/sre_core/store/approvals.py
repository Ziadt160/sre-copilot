"""Approvals in Postgres (the `ApprovalStore` port).

The gate's guarantees are single conditional updates, so they hold under concurrency and across
processes: `decide` succeeds once and only while the row is pending and unexpired (`expires_at`
is compared with the database clock, never the caller's), and `expire_due` moves every overdue
pending row to `expired`. `claim_execution` lets exactly one resume execute an approval.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any, Literal

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from sre_shared.contracts import ops
from sre_shared.db import as_user
from sre_shared.models import Claims

_INSERT_COLS = (
    "id, thread_id, turn, tool, args, args_hash, namespace, target, risk, required_scope, "
    "expected_effect, status, requested_by, decided_by, decided_at, expires_at, created_at"
)
_COLS = _INSERT_COLS + ", execution_claimed_at"


def _record(row: Mapping[str, Any]) -> ops.ApprovalRecord:
    return ops.ApprovalRecord.model_validate(dict(row))


class PgApprovalStore:
    def __init__(
        self, pool: AsyncConnectionPool[Any], policy: ops.NamespacePolicy | None = None
    ) -> None:
        self.pool = pool
        # Needed to re-rate an edited call (risk, required level); set by the runtime.
        self.policy = policy

    async def create(self, approval: ops.ApprovalRecord) -> ops.ApprovalRecord:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"insert into public.approvals ({_INSERT_COLS}) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "coalesce(%s, now())) "
                f"returning {_COLS}",
                [
                    approval.id, approval.thread_id, approval.turn, approval.tool,
                    Jsonb(approval.args), approval.args_hash, approval.namespace, approval.target,
                    approval.risk, approval.required_scope, approval.expected_effect,
                    approval.status, approval.requested_by, approval.decided_by,
                    approval.decided_at, approval.expires_at, approval.created_at,
                ],
            )  # fmt: skip
            return _record(await cur.fetchone() or {})

    async def get(self, approval_id: uuid.UUID) -> ops.ApprovalRecord | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"select {_COLS} from public.approvals where id = %s",
                [approval_id],
            )
            row = await cur.fetchone()
        return _record(row) if row else None

    async def decide(
        self,
        approval_id: uuid.UUID,
        *,
        decision: Literal["approved", "rejected"],
        decided_by: str,
        args: Mapping[str, Any] | None = None,
    ) -> ops.ApprovalRecord | None:
        """The decided record, or None when it was not pending or has expired. Edited args are
        re-validated and stored with their hash, namespace, target, risk and effect, and the
        required level becomes the higher of the old and the new (the row must describe what
        will run)."""
        sets = ["status = %s", "decided_by = %s", "decided_at = now()"]
        params: list[Any] = [decision, decided_by]
        if args is not None:
            current = await self.get(approval_id)
            if current is None:
                return None
            norm = ops.normalize_args(current.tool, args)
            ns, target = ops.target_of(current.tool, ops.parse_args(current.tool, norm))
            parsed = ops.parse_args(current.tool, norm)
            policy = self.policy or ops.NamespacePolicy(allowed=frozenset())
            level = ops.required_approval(current.tool, parsed, policy, route="action")
            scope = "admin" if "admin" in (level, current.required_scope) else "operator"
            sets += [
                "args = %s", "args_hash = %s", "namespace = %s", "target = %s",
                "risk = %s", "expected_effect = %s", "required_scope = %s",
            ]  # fmt: skip
            params += [
                Jsonb(norm),
                ops.args_hash(current.tool, norm),
                ns or current.namespace,
                target or current.target,
                ops.risk_of(current.tool, parsed, policy),
                ops.expected_effect(current.tool, parsed),
                scope,
            ]
        params.append(approval_id)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"update public.approvals set {', '.join(sets)} "
                "where id = %s and status = 'pending' and expires_at > now() "
                f"returning {_COLS}",
                params,
            )
            row = await cur.fetchone()
        return _record(row) if row else None

    async def claim_execution(self, approval_id: uuid.UUID) -> ops.ApprovalRecord | None:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "update public.approvals set execution_claimed_at = now() "
                "where id = %s and status = 'approved' and execution_claimed_at is null "
                f"and expires_at > now() returning {_COLS}",
                [approval_id],
            )
            row = await cur.fetchone()
        return _record(row) if row else None

    async def mark_executed(self, approval_id: uuid.UUID) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                "update public.approvals set status = 'executed' "
                "where id = %s and status = 'approved'",
                [approval_id],
            )

    async def expire_due(self) -> list[ops.ApprovalRecord]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "update public.approvals set status = 'expired' where expires_at <= now() and "
                "(status = 'pending' or (status = 'approved' and execution_claimed_at is null)) "
                f"returning {_COLS}"
            )
            return [_record(r) for r in await cur.fetchall()]

    # --- reads that apply RLS (the API's views; not part of the port) -------------------------

    async def get_visible(
        self, claims: Claims, approval_id: uuid.UUID
    ) -> ops.ApprovalRecord | None:
        """The approval if the caller may see it: the thread's owner, or an admin for the ones
        that need an admin (the `approvals_read` policy)."""
        if claims.role != "authenticated":
            return None
        async with self.pool.connection() as conn, as_user(conn, claims):
            cur = await conn.execute(
                f"select {_COLS} from public.approvals where id = %s",
                [approval_id],
            )
            row = await cur.fetchone()
        return _record(row) if row else None

    async def list_visible(
        self,
        claims: Claims,
        *,
        status: str = "pending",
        thread_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> list[ops.ApprovalRecord]:
        if claims.role != "authenticated":
            return []
        where = ["status = %s"]
        params: list[Any] = [status]
        if status == "pending":
            where.append("expires_at > now()")  # overdue rows wait for the sweeper
        if thread_id is not None:
            where.append("thread_id = %s")
            params.append(thread_id)
        params.append(limit)
        async with self.pool.connection() as conn, as_user(conn, claims):
            cur = await conn.execute(
                f"select {_COLS} from public.approvals "
                f"where {' and '.join(where)} order by created_at desc limit %s",
                params,
            )
            return [_record(r) for r in await cur.fetchall()]
