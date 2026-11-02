"""Everything the chat API needs, built once at startup (Phase 3).

`Runtime` is the shape the routes depend on; `build_runtime` wires the real implementations from
the factory paths listed in `sre_core.agent.ports` (T13 owns this module). Tests build a
`Runtime` directly from the fakes in `sre_core.agent.testing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sre_core.agent.ports import (
    AnswerCache,
    ApprovalStore,
    BudgetGuard,
    ChatAgent,
    ModelRouter,
    OpsClient,
    Outbox,
    ThreadStore,
)
from sre_shared.contracts.ops import NamespacePolicy


@dataclass
class Runtime:
    threads: ThreadStore
    approvals: ApprovalStore
    ops: OpsClient
    cache: AnswerCache
    budget: BudgetGuard
    models: ModelRouter
    outbox: Outbox
    policy: NamespacePolicy
    agent: ChatAgent | None = None  # built last: the agent depends on the rest
    extras: dict[str, Any] | None = None  # pools, checkpointer, background tasks (T13)


async def build_runtime(settings: Any) -> Runtime:
    """Wire the real implementations (T13). Until then the chat routes are not mounted."""
    raise NotImplementedError("build_runtime is implemented by T13 (Phase 3 Wave 1)")
