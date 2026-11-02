"""Factories at the paths fixed in `sre_core.agent.ports`."""

from __future__ import annotations

from typing import Any

from psycopg_pool import AsyncConnectionPool

from sre_core.store.approvals import PgApprovalStore
from sre_core.store.feedback import PgFeedbackStore
from sre_core.store.outbox import PgOutbox
from sre_core.store.threads import PgThreadStore
from sre_shared.contracts.ops import NamespacePolicy


def make_thread_store(pool: AsyncConnectionPool[Any]) -> PgThreadStore:
    return PgThreadStore(pool)


def make_approval_store(
    pool: AsyncConnectionPool[Any], policy: NamespacePolicy | None = None
) -> PgApprovalStore:
    return PgApprovalStore(pool, policy)


def make_outbox(pool: AsyncConnectionPool[Any]) -> PgOutbox:
    return PgOutbox(pool)


def make_feedback_store(pool: AsyncConnectionPool[Any]) -> PgFeedbackStore:
    return PgFeedbackStore(pool)
