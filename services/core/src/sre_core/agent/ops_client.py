"""`OpsClient` over FastMCP: calls mcp-ops with a per-call bearer token.

Reads and escalation tools are plain tool calls. A write is submitted as an MCP task
(`fastmcp_tasks.call_tool_task`) and is NOT re-attached from another client: the agent polls
`get_operation` (Postgres) for its outcome, so a restart of the task store cannot lose it.

What this client needs from mcp-ops (T11): a write tool declared `task=True` whose result, or
whose task `statusMessage` at creation (`operation_id=<uuid>`), identifies the operations row.
If neither shows up within `submit_wait_s`, `write` raises `OpsError` rather than guess.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import fastmcp_tasks  # noqa: F401  (importing it registers client support for tasks)
from fastmcp import Client
from fastmcp_tasks import call_tool_task

from sre_core.agent.ports import OpsClient
from sre_shared.contracts import ops
from sre_shared.settings import Settings

ClientFactory = Callable[[str], Client[Any]]
_OPERATION_ID = re.compile(r"operation_id=([0-9a-fA-F-]{36})")


class OpsError(RuntimeError):
    """mcp-ops refused the call or returned something that is not a tool result."""


def _structured(result: Any) -> dict[str, Any]:
    data = result.structured_content
    if data is None and isinstance(result.data, dict):
        data = result.data
    if not isinstance(data, dict):
        raise OpsError("the tool returned no structured result")
    # FastMCP wraps non-object returns as {"result": ...}; objects arrive as they are.
    return data


class FastMcpOpsClient:
    def __init__(
        self, client_factory: ClientFactory, *, timeout_s: float = 30.0, submit_wait_s: float = 10.0
    ) -> None:
        self._factory = client_factory
        self._timeout_s = timeout_s
        self._submit_wait_s = submit_wait_s

    async def _call(self, tool: str, args: Mapping[str, Any], token: str) -> dict[str, Any]:
        async with self._factory(token) as client:
            result = await client.call_tool(tool, dict(args), timeout=self._timeout_s)
        return _structured(result)

    async def read(self, tool: str, args: Mapping[str, Any], token: str) -> dict[str, Any]:
        return await self._call(tool, args, token)

    async def escalate(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        return ops.OperationRecord.model_validate(await self._call(tool, args, token))

    async def operation(self, operation_id: uuid.UUID, token: str) -> ops.OperationRecord:
        data = await self._call("get_operation", {"operation_id": str(operation_id)}, token)
        return ops.OperationRecord.model_validate(data)

    async def write(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        async with self._factory(token) as client:
            task = await call_tool_task(client, tool, dict(args), timeout=self._timeout_s)
            found = _OPERATION_ID.search(task.create_result.status_message or "")
            if found:
                return _running_stub(uuid.UUID(found.group(1)), tool, args)
            try:
                final = await task.wait(timeout=self._submit_wait_s)
            except TimeoutError as exc:
                raise OpsError("the write task did not identify its operation in time") from exc
            if final.status != "completed":
                raise OpsError(f"the write task ended {final.status}")
            result = await task.result()
        return ops.OperationRecord.model_validate(_structured(result))


def _running_stub(op_id: uuid.UUID, tool: str, args: Mapping[str, Any]) -> ops.OperationRecord:
    """Only the id and status matter to the caller, which then polls `operation`."""
    return ops.OperationRecord(
        id=op_id, idempotency_key="", kind="write", tool=tool, args=dict(args), subject="",
        scopes=[], status="running",
    )  # fmt: skip


def make_ops_client(settings: Settings) -> OpsClient:
    """A client for `settings.mcp_ops_url` (streamable HTTP) sending `Authorization: Bearer`."""

    def factory(token: str) -> Client[Any]:
        return Client(settings.mcp_ops_url, auth=token)

    return FastMcpOpsClient(factory)
