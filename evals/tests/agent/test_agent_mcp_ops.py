"""The agent's mcp-ops client (T12) against the real mcp-ops server (T11), over HTTP in-process.

Each side passed its own tests on its own fakes; these tests hold the seam between them: tokens
minted the way the agent mints them, and results parsed the way the agent parses them.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastmcp.utilities.tests import ASGIServer, asgi_server
from psycopg import AsyncConnection

from sre_core.agent.ops_client import FastMcpOpsClient
from sre_mcp_ops.backends.simulator import SimulatorBackend
from sre_mcp_ops.server import build_server
from sre_shared.config import load_mcp_ops_config
from sre_shared.contracts import ops
from sre_shared.db import create_pool
from sre_shared.settings import Settings

SECRET = "agent-mcp-ops-test-secret-0123456789abcdef"

pytestmark = pytest.mark.integration


@dataclass
class Seam:
    server: ASGIServer
    sim: SimulatorBackend
    db: AsyncConnection[Any]
    client: FastMcpOpsClient

    def token(
        self,
        *,
        sub: str,
        scopes: list[ops.Scope],
        namespaces: list[str],
        thread_id: uuid.UUID,
        turn: int = 1,
        approval_id: uuid.UUID | None = None,
        tools: list[ops.ToolName],
        max_calls: int = 1,
    ) -> str:
        # The same call the agent's Ctx.mint makes (sre_core.agent.nodes.common).
        return ops.mint_mcp_token(
            secret=SECRET, sub=sub, scopes=scopes, namespaces=namespaces, thread_id=thread_id,
            turn=turn, approval_id=approval_id, tools=tools, max_calls=max_calls, ttl_s=120,
        )  # fmt: skip

    async def thread(self, owner: str = "alice") -> uuid.UUID:
        cur = await self.db.execute(
            "insert into public.threads (owner_sub) values (%s) returning id", [owner]
        )
        row = await cur.fetchone()
        assert row is not None
        return row["id"]  # type: ignore[no-any-return]

    async def approval(self, thread_id: uuid.UUID, tool: str, args: dict[str, Any]) -> uuid.UUID:
        norm = ops.normalize_args(tool, args)
        cur = await self.db.execute(
            """insert into public.approvals (thread_id, turn, tool, args, args_hash, namespace,
                   target, risk, required_scope, expected_effect, status, requested_by,
                   decided_by, expires_at)
               values (%s, 1, %s, %s::jsonb, %s, %s, %s, 'medium', 'operator', 'x', 'approved',
                       'alice', 'alice', now() + interval '15 minutes') returning id""",
            [thread_id, tool, json.dumps(norm), ops.args_hash(tool, norm),
             norm.get("namespace", ""), norm.get("deployment", "")],
        )  # fmt: skip
        row = await cur.fetchone()
        assert row is not None
        return row["id"]  # type: ignore[no-any-return]


@pytest.fixture
async def seam(db: AsyncConnection[Any], pg_dsn: str) -> AsyncIterator[Seam]:
    pool = create_pool(pg_dsn, min_size=1, max_size=4)
    await pool.open()
    sim = SimulatorBackend("default", 7)
    server = build_server(
        Settings(mcp_token_secret=SECRET),  # type: ignore[arg-type]
        backend=sim, pool=pool, config=load_mcp_ops_config(),
    )  # fmt: skip
    try:
        async with asgi_server(server) as running:
            yield Seam(running, sim, db, FastMcpOpsClient(lambda t: running.client(auth=t)))
    finally:
        await pool.close()


async def test_read_returns_the_tool_output(seam: Seam) -> None:
    thread_id = await seam.thread()
    token = seam.token(
        sub="alice", scopes=["viewer"], namespaces=["shop"], thread_id=thread_id,
        tools=["list_pods"],
    )  # fmt: skip
    out = await seam.client.read("list_pods", {"namespace": "shop"}, token)
    assert out["trust"] == "untrusted_data"
    assert out["data"]["pods"]


async def test_approved_write_runs_once_and_polls_to_the_end(seam: Seam) -> None:
    thread_id = await seam.thread()
    args = {"namespace": "shop", "deployment": "checkout"}
    approval_id = await seam.approval(thread_id, "restart_deployment", args)

    def requester(tool: ops.ToolName, approval: uuid.UUID | None = None) -> str:
        # requester_token in sre_core.agent.nodes.action: one call per token (O-2)
        return seam.token(
            sub="alice", scopes=["viewer", "operator"], namespaces=["shop"],
            thread_id=thread_id, approval_id=approval, tools=[tool],
        )  # fmt: skip

    rec = await seam.client.write(
        "restart_deployment", args, requester("restart_deployment", approval_id)
    )
    assert rec.tool == "restart_deployment"
    assert rec.approval_id == approval_id
    final = await seam.client.operation(rec.id, requester("get_operation"))
    assert final.id == rec.id
    assert final.status == "succeeded"
    assert [m["tool"] for m in seam.sim.mutation_log()] == ["restart_deployment"]


async def test_escalation_opens_an_incident_and_pages(seam: Seam) -> None:
    thread_id = await seam.thread()
    # The escalate node's token: the user's own namespaces, viewer only, two calls.
    token = seam.token(
        sub="alice", scopes=["viewer"], namespaces=["shop"], thread_id=thread_id,
        tools=["create_incident", "page_oncall"], max_calls=2,
    )  # fmt: skip
    args = ops.CreateIncidentArgs(
        severity="sev3", title="Escalated: checkout is down", summary="a person needs to look"
    ).model_dump(mode="json")
    created = await seam.client.escalate("create_incident", args, token)
    assert created.status == "succeeded"
    incident_id = (created.result or {})["incident_id"]
    paged = await seam.client.escalate("page_oncall", {"incident_id": str(incident_id)}, token)
    assert paged.status == "succeeded"


async def test_a_retried_write_with_the_same_token_gets_the_stored_operation(seam: Seam) -> None:
    """A reply lost on the way back makes the client send the same request again, with the same
    single-use token: it must get the stored operation, not a refusal (dup_write fault)."""
    thread_id = await seam.thread()
    args = {"namespace": "shop", "deployment": "checkout"}
    approval_id = await seam.approval(thread_id, "restart_deployment", args)
    token = seam.token(
        sub="alice", scopes=["viewer", "operator"], namespaces=["shop"], thread_id=thread_id,
        approval_id=approval_id, tools=["restart_deployment"],
    )  # fmt: skip
    first = await seam.client.write("restart_deployment", args, token)
    again = await seam.client.write("restart_deployment", args, token)
    assert again.id == first.id
    assert len(seam.sim.mutation_log()) == 1


async def test_logs_can_be_read_by_deployment(seam: Seam) -> None:
    """The agent never sees list_pods output when it proposes calls, so it asks for a
    deployment's logs; mcp-ops picks the pod (the one restarting most)."""
    thread_id = await seam.thread()
    token = seam.token(
        sub="alice", scopes=["viewer"], namespaces=["shop"], thread_id=thread_id,
        tools=["get_logs"],
    )  # fmt: skip
    out = await seam.client.read("get_logs", {"namespace": "shop", "deployment": "checkout"}, token)
    assert out["data"]["pod"].startswith("checkout-")
