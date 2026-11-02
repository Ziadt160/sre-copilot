"""Fixtures for mcp-ops tests: a real HTTP stack in-process (bearer auth and all), a migrated
Postgres, and a backend chosen with `--backend simulator|kind` (kind needs a cluster; CI only)."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.utilities.tests import ASGIServer, asgi_server
from fastmcp_tasks import call_tool_task
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from sre_mcp_ops.backends.base import ClusterBackend
from sre_mcp_ops.backends.simulator import SimulatorBackend
from sre_mcp_ops.server import build_server
from sre_mcp_ops.service import CallBudget, OpsService
from sre_mcp_ops.store import OpsStore
from sre_shared.config import McpOpsConfig, load_mcp_ops_config
from sre_shared.contracts import ops
from sre_shared.db import create_pool
from sre_shared.settings import Settings

SECRET = "mcp-test-secret-0123456789abcdef0123456789"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--backend", default="simulator", choices=["simulator", "kind"])


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "/mcp_ops/" in str(item.path).replace("\\", "/"):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def backend_name(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--backend"))


@pytest.fixture
def backend(backend_name: str) -> ClusterBackend:
    if backend_name == "kind":
        pytest.importorskip("kubernetes")
        from sre_mcp_ops.backends.kind import KindBackend

        try:
            return KindBackend()
        except Exception as exc:  # no cluster reachable
            pytest.skip(f"kind cluster unavailable: {type(exc).__name__}")
    return SimulatorBackend("default", 7)


@pytest.fixture
def sim(backend: ClusterBackend) -> SimulatorBackend:
    if not isinstance(backend, SimulatorBackend):
        pytest.skip("simulator-only test")
    return backend


@pytest.fixture
async def pool(pg_dsn: str) -> AsyncIterator[AsyncConnectionPool[Any]]:
    p = create_pool(pg_dsn, min_size=1, max_size=6)
    await p.open()
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
def config() -> McpOpsConfig:
    return load_mcp_ops_config()


@dataclass
class Rig:
    server: ASGIServer
    db: AsyncConnection[Any]
    service: OpsService
    store: OpsStore
    config: McpOpsConfig

    def token(
        self,
        *,
        sub: str = "alice",
        scopes: tuple[str, ...] = ("viewer", "operator"),
        namespaces: tuple[str, ...] = ("shop", "staging", "payments"),
        thread_id: uuid.UUID | None = None,
        turn: int | None = 1,
        approval_id: uuid.UUID | None = None,
        max_calls: int = 5,
        secret: str = SECRET,
        now: float | None = None,
        ttl_s: int = 120,
    ) -> str:
        return ops.mint_mcp_token(
            secret=secret, sub=sub, scopes=list(scopes), namespaces=list(namespaces),  # type: ignore[arg-type]
            thread_id=thread_id, turn=turn, approval_id=approval_id, max_calls=max_calls,
            ttl_s=ttl_s, now=now,
        )  # fmt: skip

    def client(self, token: str | None) -> Client:
        return self.server.client(auth=token)

    async def thread(self, owner: str = "alice") -> uuid.UUID:
        cur = await self.db.execute(
            "insert into public.threads (owner_sub) values (%s) returning id", [owner]
        )
        row = await cur.fetchone()
        assert row is not None
        return row["id"]  # type: ignore[no-any-return]

    async def approval(
        self,
        thread_id: uuid.UUID,
        tool: str,
        args: dict[str, Any],
        *,
        status: str = "approved",
        requested_by: str = "alice",
        required_scope: str = "operator",
        expires_in_s: int = 900,
        turn: int = 1,
        decided_by: str | None = "bob",
    ) -> uuid.UUID:
        norm = ops.normalize_args(tool, args)
        cur = await self.db.execute(
            """insert into public.approvals (thread_id, turn, tool, args, args_hash, namespace,
                   target, risk, required_scope, expected_effect, status, requested_by,
                   decided_by, expires_at)
               values (%s, %s, %s, %s::jsonb, %s, %s, %s, 'medium', %s, 'x', %s, %s, %s,
                       now() + make_interval(secs => %s)) returning id""",
            [thread_id, turn, tool, __import__("json").dumps(norm), ops.args_hash(tool, norm),
             norm.get("namespace", ""), norm.get("deployment", ""), required_scope, status,
             requested_by, decided_by, expires_in_s],
        )  # fmt: skip
        row = await cur.fetchone()
        assert row is not None
        return row["id"]  # type: ignore[no-any-return]

    async def rows(self, where: str = "true", *params: Any) -> list[dict[str, Any]]:
        cur = await self.db.execute(
            f"select * from public.operations where {where} order by created_at", list(params)
        )
        return await cur.fetchall()

    async def write(
        self, tool: str, args: dict[str, Any], token: str, *, wait: bool = True
    ) -> dict[str, Any]:
        """A write as the agent makes it: call_tool_task, then the task's result."""
        async with self.client(token) as c:
            task = await call_tool_task(c, tool, args)
            assert task.task_id
            res = await task.result()
        assert res.structured_content is not None
        return dict(res.structured_content)

    async def read(self, tool: str, args: dict[str, Any], token: str) -> dict[str, Any]:
        async with self.client(token) as c:
            res = await c.call_tool(tool, args)
        assert res.structured_content is not None
        return dict(res.structured_content)

    def now(self) -> float:
        return time.time()


@pytest.fixture
async def rig(
    db: AsyncConnection[Any],
    pool: AsyncConnectionPool[Any],
    backend: ClusterBackend,
    config: McpOpsConfig,
) -> AsyncIterator[Rig]:
    store = OpsStore(pool)
    service = OpsService(config=config, store=store, backend=backend, budget=CallBudget())
    server = build_server(
        Settings(mcp_token_secret=SECRET),  # type: ignore[arg-type]
        backend=backend, pool=pool, config=config, service=service,
    )  # fmt: skip
    async with asgi_server(server) as running:
        yield Rig(running, db, service, store, config)
