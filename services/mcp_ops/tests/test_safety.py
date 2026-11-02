"""Refusals: mcp-ops enforces tokens, scopes, namespaces, approvals and the call budget itself.
Every authenticated refusal must leave an audit row; an unauthenticated one has no subject and
is stopped at the HTTP layer before any tool code runs."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import jwt
import pytest
from fastmcp.exceptions import ToolError

from sre_shared.contracts import ops

if TYPE_CHECKING:
    from conftest import Rig

WEB = {"namespace": "shop", "deployment": "web"}
SCALE3 = {**WEB, "replicas": 3}


async def case(
    rig: Rig,
    tool: str,
    args: dict[str, Any],
    *,
    approval: dict[str, Any] | None = None,
    approval_tool: str | None = None,
    approval_args: dict[str, Any] | None = None,
    no_thread: bool = False,
    **token_kw: Any,
) -> tuple[str, dict[str, Any], str]:
    t = None if no_thread else await rig.thread()
    appr = None
    if approval is not None and t is not None:
        appr = await rig.approval(t, approval_tool or tool, approval_args or args, **approval)
    return tool, args, rig.token(thread_id=t, approval_id=appr, **token_kw)


CASES: dict[str, tuple[Any, str]] = {
    "viewer calls a write": (
        lambda r: case(r, "restart_deployment", WEB, approval={}, scopes=("viewer",)),
        "needs the operator scope",
    ),
    "namespace not on the allow-list": (
        lambda r: case(r, "list_pods", {"namespace": "kube-system"}, namespaces=("kube-system",)),
        "allow-list",
    ),
    "namespace not in the token": (
        lambda r: case(r, "list_pods", {"namespace": "shop"}, namespaces=("staging",)),
        "may not act in namespace shop",
    ),
    "write without approval": (
        lambda r: case(r, "restart_deployment", WEB),
        "needs an approval",
    ),
    "approval for different args": (
        lambda r: case(
            r, "scale_deployment", {**WEB, "replicas": 4}, approval={}, approval_args=SCALE3
        ),
        "different call",
    ),
    "approval for a different tool": (
        lambda r: case(
            r, "restart_deployment", WEB, approval={}, approval_tool="scale_deployment",
            approval_args=SCALE3,
        ),
        "different call",
    ),
    "expired approval": (
        lambda r: case(r, "restart_deployment", WEB, approval={"expires_in_s": -5}),
        "expired",
    ),
    "rejected approval": (
        lambda r: case(r, "restart_deployment", WEB, approval={"status": "rejected"}),
        "approval is rejected",
    ),
    "pending approval": (
        lambda r: case(r, "restart_deployment", WEB, approval={"status": "pending"}),
        "approval is pending",
    ),
    "approval from another turn": (
        lambda r: case(r, "restart_deployment", WEB, approval={"turn": 2}),
        "another turn",
    ),
    "approval requested by someone else": (
        lambda r: case(r, "restart_deployment", WEB, approval={"requested_by": "bob"}),
        "someone else",
    ),
    "operator writes in a protected namespace": (
        lambda r: case(r, "restart_deployment", {"namespace": "payments", "deployment": "api"},
                       approval={}),
        "admin's approval",
    ),
    "admin-level call approved without an admin decision": (
        lambda r: case(r, "restart_deployment", {"namespace": "payments", "deployment": "api"},
                       approval={"required_scope": "operator"}),
        "admin's approval",
    ),
    "operator scales to zero": (
        lambda r: case(r, "scale_deployment", {**WEB, "replicas": 0}, approval={}),
        "admin's approval",
    ),
    "write token not bound to a turn": (
        lambda r: case(r, "restart_deployment", WEB, no_thread=True),
        "bound to a turn",
    ),
    "extra argument": (
        lambda r: case(r, "list_pods", {"namespace": "shop", "all": True}),
        "invalid arguments",
    ),
    "bad namespace name": (
        lambda r: case(r, "list_pods", {"namespace": "Shop; rm -rf"}),
        "invalid arguments",
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", list(CASES))
async def test_refused_and_audited(rig: Rig, name: str) -> None:
    build, why = CASES[name]
    tool, args, token = await build(rig)
    async with rig.client(token) as c:
        with pytest.raises(ToolError, match=why):
            await c.call_tool(tool, args)
    rows = await rig.rows("tool = %s and status = 'refused'", tool)
    assert len(rows) == 1
    row = rows[0]
    assert row["subject"] == "alice" and why in row["error"]
    assert row["latency_ms"] is not None and row["scopes"]
    if tool in ops.WRITE_TOOLS:  # a refusal never holds the idempotency key
        assert ":refused:" in row["idempotency_key"]
    assert not await rig.rows("status in ('approved', 'running', 'succeeded') and kind <> 'read'")


def forged(rig: Rig, **override: Any) -> str:
    now = int(time.time())
    claims = {
        "iss": "sre-core", "aud": "mcp-ops", "sub": "alice", "scopes": ["viewer", "operator"],
        "namespaces": ["shop"], "iat": now, "exp": now + 100, "jti": "x", "max_calls": 5,
    }  # fmt: skip
    claims.update(override)
    return jwt.encode(claims, "mcp-test-secret-0123456789abcdef0123456789", algorithm="HS256")


async def bad_token_cases(rig: Rig) -> dict[str, str | None]:
    now = time.time()
    return {
        "no token": None,
        "expired": rig.token(now=now - 1000),
        "wrong secret": rig.token(secret="another-secret-0123456789abcdef012345678"),
        "wrong audience": forged(rig, aud="somebody-else"),
        "wrong issuer": forged(rig, iss="not-core"),
        "not a jwt": "garbage",
        "alg none": jwt.encode(
            {
                "iss": "sre-core",
                "aud": "mcp-ops",
                "sub": "alice",
                "scopes": ["admin"],
                "namespaces": ["shop"],
                "iat": int(now),
                "exp": int(now) + 60,
                "jti": "q",
            },
            None,
            algorithm="none",
        ),
        "lifetime beyond 300 s": forged(rig, exp=int(now) + 3000),
    }


@pytest.mark.parametrize(
    "name",
    ["no token", "expired", "wrong secret", "wrong audience", "wrong issuer", "not a jwt",
     "alg none", "lifetime beyond 300 s"],
)  # fmt: skip
async def test_unauthenticated_calls_never_reach_a_tool(rig: Rig, name: str) -> None:
    token = (await bad_token_cases(rig))[name]
    with pytest.raises(Exception):
        async with rig.client(token) as c:
            await c.call_tool("list_pods", {"namespace": "shop"})
    assert await rig.rows() == []


async def test_max_calls_is_enforced_per_token(rig: Rig) -> None:
    tok = rig.token(max_calls=2)
    async with rig.client(tok) as c:
        await c.call_tool("list_pods", {"namespace": "shop"})
        await c.call_tool("list_pods", {"namespace": "shop"})
        with pytest.raises(ToolError, match="max_calls"):
            await c.call_tool("list_pods", {"namespace": "shop"})
    refused = await rig.rows("status = 'refused'")
    assert len(refused) == 1 and "max_calls" in refused[0]["error"]
    async with rig.client(rig.token(max_calls=2)) as c:  # a new token has its own budget
        await c.call_tool("list_pods", {"namespace": "shop"})


async def test_a_second_subject_cannot_replay_another_callers_operation(rig: Rig) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    out = await rig.write("restart_deployment", WEB, rig.token(thread_id=t, approval_id=appr))
    assert out["status"] == "succeeded"
    # Same thread, turn and args under another subject: same key, must not return alice's result.
    appr2 = await rig.approval(t, "restart_deployment", WEB, requested_by="mallory")
    async with rig.client(rig.token(sub="mallory", thread_id=t, approval_id=appr2)) as c:
        with pytest.raises(ToolError, match="another caller"):
            await c.call_tool("restart_deployment", WEB)


INCIDENT = {"severity": "sev3", "title": "abc", "summary": "abc"}


async def test_a_read_bound_token_cannot_open_an_incident(rig: Rig) -> None:
    t = await rig.thread()
    async with rig.client(rig.token(thread_id=t, tools=("list_pods",))) as c:
        with pytest.raises(ToolError, match="not valid for create_incident"):
            await c.call_tool("create_incident", INCIDENT)
    assert not await rig.rows("kind = 'escalation' and status <> 'refused'")
    (row,) = await rig.rows("status = 'refused'")
    assert row["tool"] == "create_incident" and row["subject"] == "alice"


async def test_an_unbound_token_cannot_open_an_incident_without_an_approval(rig: Rig) -> None:
    t = await rig.thread()
    async with rig.client(rig.token(thread_id=t, tools=None)) as c:
        with pytest.raises(ToolError, match="needs a token bound to it"):
            await c.call_tool("create_incident", INCIDENT)
    assert len(await rig.rows("status = 'refused'")) == 1
    assert not await rig.rows("kind = 'escalation' and status <> 'refused'")


async def test_a_token_bound_to_one_read_cannot_call_another(rig: Rig) -> None:
    async with rig.client(rig.token(tools=("list_pods",))) as c:
        await c.call_tool("list_pods", {"namespace": "shop"})
        with pytest.raises(ToolError, match="not valid for get_logs"):
            await c.call_tool("get_logs", {"namespace": "shop", "pod": "web-abc"})
    assert len(await rig.rows("status = 'refused'")) == 1
