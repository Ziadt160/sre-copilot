"""Everything the chat API needs, built once at startup (Phase 3).

`Runtime` is the shape the routes depend on; `build_runtime` wires the real implementations from
the factory paths listed in `sre_core.agent.ports` (T13 owns this module). Tests build a
`Runtime` directly from the fakes in `sre_core.agent.testing`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sre_core.ports import (
    AnswerCache,
    ApprovalStore,
    BudgetGuard,
    ChatAgent,
    ModelRouter,
    OpsClient,
    Outbox,
    ThreadStore,
)
from sre_core.store.checkpoints import make_checkpoint_pool, make_checkpointer
from sre_core.store.factory import (
    make_approval_store,
    make_feedback_store,
    make_outbox,
    make_thread_store,
)
from sre_core.store.outbox import register_outbox_handler
from sre_shared.config import load_agent_config, load_mcp_ops_config
from sre_shared.contracts.ops import NamespacePolicy
from sre_shared.db import create_pool


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
    retriever: Any = None  # sre_core.retrieval.Retriever (the simple and complex routes)
    answerer: Any = None  # sre_core.answer.Answerer (the simple route reuses it)
    agent: ChatAgent | None = None  # built last: the agent depends on the rest
    extras: dict[str, Any] | None = None  # pools, checkpointer, "chat" services, "feedback"


def _factory(module: str, name: str) -> Callable[..., Any]:
    """Import a component another task owns by the path fixed in `agent.ports`, so a missing one
    fails with a message that names it instead of an ImportError from deep in startup."""
    try:
        fn: Callable[..., Any] = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"chat runtime needs {module}.{name}: {exc}") from exc
    return fn


def register_default_handlers() -> None:
    """Outbox handlers other tasks own (T16's `feedback_sync`); idempotent."""
    from sre_core.review.feedback_sync import HANDLERS

    for kind, fn in HANDLERS.items():
        register_outbox_handler(kind, fn)


async def build_runtime(settings: Any, *, retriever: Any = None, answerer: Any = None) -> Runtime:
    """Wire the real implementations. A `retriever`/`answerer` already built by the app (its
    pool and replica) are reused; otherwise they are built here. Pools opened here are listed in
    `extras["pools"]` for `close_runtime`."""
    register_default_handlers()
    pool = create_pool(min_size=1, max_size=8)
    await pool.open()
    pools = [pool]
    try:
        if retriever is None:
            from sre_core.answer import build_answerer
            from sre_core.retrieval import Retriever
            from sre_core.retrieval.rerank import default_reranker

            retriever = Retriever(pool, reranker=default_reranker())
            answerer = answerer or build_answerer(retriever)
        ckpt_pool = make_checkpoint_pool()
        await ckpt_pool.open()
        pools.append(ckpt_pool)
        checkpointer = make_checkpointer(ckpt_pool)

        config = load_agent_config()
        policy = load_mcp_ops_config().policy()
        threads = make_thread_store(pool)
        approvals = make_approval_store(pool)
        outbox = make_outbox(pool)
        cache = _factory("sre_core.cache.factory", "make_answer_cache")(pool, settings)
        budget = _factory("sre_core.budget.factory", "make_budget_guard")(settings)
        from sre_core.review.tracing import client_wrapper

        models = _factory("sre_core.agent.models", "make_model_router")(
            settings, client_wrapper=client_wrapper(settings)
        )
        ops_client = _factory("sre_core.agent.ops_client", "make_ops_client")(settings)
        agent = _factory("sre_core.agent.factory", "make_chat_agent")(
            threads=threads, approvals=approvals, ops=ops_client, cache=cache, budget=budget,
            models=models, outbox=outbox, policy=policy, retriever=retriever,
            answerer=answerer, checkpointer=checkpointer, config=config, settings=settings,
        )  # fmt: skip
    except BaseException:
        for p in reversed(pools):
            await p.close()
        raise
    return Runtime(
        threads=threads, approvals=approvals, ops=ops_client, cache=cache, budget=budget,
        models=models, outbox=outbox, policy=policy, retriever=retriever, answerer=answerer,
        agent=agent,
        extras={
            "pools": pools,
            "checkpointer": checkpointer,
            "feedback": make_feedback_store(pool),
        },
    )  # fmt: skip


async def close_runtime(rt: Runtime) -> None:
    """Close the pools `build_runtime` opened (the chat tasks are stopped by the API first)."""
    for p in reversed((rt.extras or {}).get("pools", [])):
        await p.close()
