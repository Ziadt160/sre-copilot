"""The lab rig: the real agent graph (T12) over the real mcp-ops server (T11, simulator backend).

`sre-eval agent run --rig sre_evals.agent.live:lab` (local Ollama model for every agent role and
the answerer: no quota) or `--rig sre_evals.agent.live:hosted` (config/agent.yaml as it is, with
its fallback chain: the headline runs).

Everything runs in this process. mcp-ops is served over HTTP on a local port, so tokens,
admission and tasks go through the same stack as in production. The agent's tables and mcp-ops'
tables live in a fresh migrated database made for the run and dropped after it; retrieval reads
the eval corpus (`SRE_EVAL_DATABASE_URL`, default `settings.database_url`).
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, get_args

from fastmcp.utilities.tests import asgi_server
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from sre_core.agent.factory import make_chat_agent
from sre_core.agent.models import make_model_router
from sre_core.agent.ops_client import FastMcpOpsClient
from sre_core.agent.ports import ModelRole, OpsClient
from sre_core.agent.testing import FakeBudgetGuard, MemoryAnswerCache, MemoryOutbox
from sre_core.answer.service import build_answerer
from sre_core.retrieval.rerank import default_reranker
from sre_core.retrieval.service import Retriever
from sre_core.store.approvals import PgApprovalStore
from sre_core.store.threads import PgThreadStore
from sre_evals.agent.fakes import DroppedReplyOps
from sre_evals.agent.items import AgentItem
from sre_evals.agent.rig import AgentRig, FaultPlan, RigFactory
from sre_mcp_ops.backends.simulator import SimulatorBackend
from sre_mcp_ops.server import build_server
from sre_mcp_ops.service import CallBudget, OpsService
from sre_mcp_ops.store import OpsStore
from sre_shared.config import (
    AgentConfig,
    load_agent_config,
    load_answer_config,
    load_mcp_ops_config,
)
from sre_shared.contracts.ops import McpTokenClaims
from sre_shared.db import create_pool
from sre_shared.models import (
    ANONYMOUS,
    Claims,
    DocType,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from sre_shared.settings import Settings, get_settings
from sre_shared.testing.pg import session_database, with_dbname


class SimCluster:
    """`ClusterState` over the simulator, in the shapes the runner reads (T15-1)."""

    def __init__(self, sim: SimulatorBackend) -> None:
        self.sim = sim

    def reset(self, scenario: str, seed: int) -> None:
        self.sim.reset(scenario, seed)

    def snapshot(self) -> dict[str, Any]:
        return self.sim.snapshot()

    def mutations(self) -> list[dict[str, Any]]:
        keep = ("tool", "namespace", "target", "args", "idempotency_key", "changed")
        return [{k: m[k] for k in keep} for m in self.sim.mutation_log()]


class RecordingOpsService(OpsService):
    """Remembers the last task body it ran, so the `redeliver` fault can deliver it again."""

    last: tuple[McpTokenClaims, str, dict[str, Any]] | None = None

    async def execute(
        self, claims: McpTokenClaims, tool: str, raw: dict[str, Any], task_id: str | None
    ) -> dict[str, Any]:
        self.last = (claims, tool, dict(raw))
        return await super().execute(claims, tool, raw, task_id)

    async def redeliver(self) -> None:
        if self.last is not None:
            claims, tool, raw = self.last
            await super().execute(claims, tool, raw, "redelivered")


class PlantingRetriever(Retriever):
    """The real retriever, with planted passages (injection items) ranked first."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.planted: list[str] = []

    async def retrieve(
        self, req: RetrievalRequest, claims: Claims = ANONYMOUS, *, exact: bool = False
    ) -> RetrievalResult:
        res = await super().retrieve(req, claims, exact=exact)
        if not self.planted:
            return res
        fake = [_planted_chunk(-(i + 1), text) for i, text in enumerate(self.planted)]
        return res.model_copy(update={"chunks": [*fake, *res.chunks]})


def _planted_chunk(chunk_id: int, body: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, chunk_key=f"planted{-chunk_id}", source_path="planted/notes.md",
        k8s_version="v1.31", doc_type=DocType.CONCEPT, title="Operations notes", section_path=[],
        url="https://example.test/planted", header="Operations notes", body=body, score=1.0,
    )  # fmt: skip


def lab_config(cfg: AgentConfig) -> AgentConfig:
    """Every role on the lab model and no hosted fallback (the extractive answer stays last)."""
    roles: tuple[ModelRole, ...] = get_args(ModelRole)
    models = cfg.models.model_copy(update=dict.fromkeys(roles, cfg.models.lab))
    fallback = cfg.fallback.model_copy(update={"providers": []})
    return cfg.model_copy(update={"models": models, "fallback": fallback})


@asynccontextmanager
async def live_rig(models: str = "lab", *, seed: int = 7) -> AsyncIterator[RigFactory]:
    base = get_settings()
    secret = secrets.token_hex(32)  # this run's core -> mcp-ops signing key, never stored
    settings = base.model_copy(update={"mcp_token_secret": SecretStr(secret)})
    corpus_dsn = os.environ.get("SRE_EVAL_DATABASE_URL") or base.database_url
    cfg = load_agent_config(settings.config_dir)
    answer_cfg = load_answer_config(settings.config_dir)
    if models == "lab":
        cfg = lab_config(cfg)
        # reasoning_effort is a gpt-oss knob; Ollama refuses it for qwen ("does not support thinking")
        answer_cfg = answer_cfg.model_copy(
            update={"model": cfg.models.lab, "reasoning_effort": None}
        )
    mcp_cfg = load_mcp_ops_config(settings.config_dir)
    policy = mcp_cfg.policy()

    with session_database(with_dbname(corpus_dsn, "postgres"), prefix="sre_agent_eval") as dsn:
        corpus_pool = create_pool(corpus_dsn, min_size=1, max_size=4)
        pool = create_pool(dsn, min_size=2, max_size=8)
        await corpus_pool.open(wait=True)
        await pool.open(wait=True)
        try:
            reranker = default_reranker()
            if reranker is not None:
                await reranker.warm()
            sim = SimulatorBackend("default", seed)
            service = RecordingOpsService(
                config=mcp_cfg, store=OpsStore(pool), backend=sim, budget=CallBudget()
            )
            server = build_server(
                Settings(mcp_token_secret=SecretStr(secret)),
                backend=sim, pool=pool, config=mcp_cfg, service=service,
            )  # fmt: skip
            router = make_model_router(settings, config=cfg)
            async with asgi_server(server) as running:
                http_ops = FastMcpOpsClient(lambda token: running.client(auth=token))

                async def factory(item: AgentItem, faults: FaultPlan) -> AgentRig:
                    retriever = PlantingRetriever(corpus_pool, reranker=reranker)
                    answerer = build_answerer(retriever, config=answer_cfg)
                    ops_client: OpsClient = (
                        DroppedReplyOps(http_ops) if faults.dup_write else http_ops
                    )
                    threads, approvals = PgThreadStore(pool), PgApprovalStore(pool, policy)
                    agent = make_chat_agent(
                        threads=threads, approvals=approvals, ops=ops_client,
                        cache=MemoryAnswerCache(), budget=FakeBudgetGuard(), models=router,
                        outbox=MemoryOutbox(), policy=policy, retriever=retriever,
                        answerer=answerer, checkpointer=InMemorySaver(), config=cfg,
                        settings=settings,
                    )  # fmt: skip
                    return AgentRig(
                        agent=agent, threads=threads, approvals=approvals,
                        cluster=SimCluster(sim), retriever=retriever, plant_log=sim.set_log,
                        plant_chunk=retriever.planted.append,
                        redeliver=service.redeliver if faults.redeliver else None,
                    )  # fmt: skip

                yield factory
        finally:
            await pool.close()
            await corpus_pool.close()


def lab() -> AbstractAsyncContextManager[RigFactory]:
    return live_rig("lab")


def hosted() -> AbstractAsyncContextManager[RigFactory]:
    return live_rig("router")
