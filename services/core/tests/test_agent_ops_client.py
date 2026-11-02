"""The FastMCP ops client against an in-memory server defined here (not T11's)."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp_tasks import TasksExtension

from sre_core.agent.ops_client import FastMcpOpsClient, OpsError, make_ops_client
from sre_core.agent.ports import OpsClient
from sre_shared.settings import Settings

OP_ID = uuid.uuid4()


def record(status: str, tool: str = "restart_deployment", **extra: Any) -> dict[str, Any]:
    return {
        "id": str(OP_ID), "idempotency_key": "k", "kind": "write", "tool": tool, "args": {},
        "subject": "alice", "scopes": ["operator"], "status": status, **extra,
    }  # fmt: skip


def server() -> FastMCP:
    mcp = FastMCP("ops-test")
    mcp.add_extension(TasksExtension(url="memory://"))

    @mcp.tool
    def list_pods(namespace: str) -> dict[str, Any]:
        return {"items": [f"{namespace}-pod"]}

    @mcp.tool
    def get_operation(operation_id: str) -> dict[str, Any]:
        return record("succeeded", result={"ok": True})

    @mcp.tool
    def create_incident(title: str) -> dict[str, Any]:
        return record("succeeded", tool="create_incident", kind="escalation")

    @mcp.tool(task=True)
    async def restart_deployment(namespace: str, deployment: str) -> dict[str, Any]:
        return record("succeeded", result={"restarted": deployment})

    return mcp


def client_for(srv: FastMCP, seen: list[str]) -> FastMcpOpsClient:
    def factory(token: str) -> Client[Any]:
        seen.append(token)
        return Client(srv)

    return FastMcpOpsClient(factory)


async def test_read_escalate_and_operation_calls_pass_the_token_per_call() -> None:
    seen: list[str] = []
    c = client_for(server(), seen)
    assert await c.read("list_pods", {"namespace": "shop"}, "tok-1") == {"items": ["shop-pod"]}
    esc = await c.escalate("create_incident", {"title": "x"}, "tok-2")
    assert esc.kind == "escalation" and esc.status == "succeeded"
    op = await c.operation(OP_ID, "tok-3")
    assert op.id == OP_ID and op.result == {"ok": True}
    assert seen == ["tok-1", "tok-2", "tok-3"]


async def test_a_write_is_submitted_as_a_task_and_returns_its_operation() -> None:
    seen: list[str] = []
    rec = await client_for(server(), seen).write(
        "restart_deployment", {"namespace": "shop", "deployment": "checkout"}, "tok-w"
    )
    assert rec.id == OP_ID and rec.status == "succeeded" and seen == ["tok-w"]


async def test_a_tool_that_does_not_run_as_a_task_is_an_error_not_a_silent_success() -> None:
    with pytest.raises(Exception, match="did not run as a task"):
        await client_for(server(), []).write("list_pods", {"namespace": "shop"}, "tok")


def test_the_real_client_targets_the_configured_url_with_a_bearer_token() -> None:
    client = make_ops_client(Settings(mcp_ops_url="http://127.0.0.1:9/mcp"))  # type: ignore[call-arg]
    assert isinstance(client, OpsClient) and isinstance(client, FastMcpOpsClient)
    built = client._factory("secret-token")
    assert str(built.transport.url) == "http://127.0.0.1:9/mcp"  # type: ignore[attr-defined]


def test_ops_error_is_a_runtime_error() -> None:
    assert issubclass(OpsError, RuntimeError)
