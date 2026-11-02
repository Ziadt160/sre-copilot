"""V07 (adversarial verification): the chat API and Postgres stores with the REAL agent graph.

The agent runs on fakes for everything outside the database (scripted model, fake mcp-ops,
in-memory checkpointer); threads, approvals and the outbox are Postgres with RLS. Attacks: the
approval gate through HTTP (twice, racing, expiry, the sweeper, the wrong decider, edits), a crash
between decision and resume, client_message_id replay, another user's thread, direct table
access as an API role (threads, approvals, the checkpoint schema) and the day budget. A test
marked `xfail(strict=True, reason="V07-<n>: ...")` reproduces a defect."""

from __future__ import annotations

import asyncio
import json
import time
import tomllib
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import jwt
import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from psycopg_pool import AsyncConnectionPool

from sre_core.agent import testing as fk
from sre_core.agent.factory import make_chat_agent
from sre_core.agent.ports import ModelRole
from sre_core.answer import build_answerer
from sre_core.answer.testing import StubRetriever, chunk, fake_chat
from sre_core.api.chat import services
from sre_core.api.main import create_app
from sre_core.api.runs import sweep_once
from sre_core.api.runtime import Runtime
from sre_core.store.approvals import PgApprovalStore
from sre_core.store.feedback import PgFeedbackStore
from sre_core.store.outbox import PgOutbox
from sre_core.store.threads import PgThreadStore
from sre_shared.config import load_agent_config
from sre_shared.contracts import agent as ev
from sre_shared.contracts import ops
from sre_shared.db import as_user
from sre_shared.llm import ChatResult, Message
from sre_shared.models import Claims
from sre_shared.settings import Settings, get_settings

pytestmark = pytest.mark.integration

JWT_SECRET = "test-secret-test-secret-test-secret-0123"
MCP_SECRET = "m" * 40
SETTINGS = Settings(mcp_token_secret=MCP_SECRET)
POLICY = ops.NamespacePolicy(
    allowed=frozenset({"shop", "staging", "payments"}), protected=frozenset({"payments"})
)
RESTART = ops.normalize_args("restart_deployment", {"namespace": "shop", "deployment": "checkout"})
LEDGER = ops.normalize_args("restart_deployment", {"namespace": "payments", "deployment": "ledger"})
SCALE = ops.normalize_args(
    "scale_deployment", {"namespace": "shop", "deployment": "checkout", "replicas": 3}
)
# Tokens the scripted model maps to tool calls (nothing in the prompts contains them otherwise).
CALLS: dict[str, list[tuple[str, Mapping[str, Any]]]] = {
    "op-shop": [("restart_deployment", RESTART)],
    "op-ledger": [("restart_deployment", LEDGER)],
    "op-scale": [("scale_deployment", SCALE)],
}
REPO = Path(__file__).resolve().parents[3]


@dataclass
class RuleModel(fk.ScriptedModelRouter):
    """Rewrites to the operation token in the current message, routes it to `action` and proposes
    its calls; anything else is a documentation question."""

    async def complete(
        self,
        role: ModelRole,
        messages: Sequence[Message],
        *,
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        self.calls.append((role, list(messages)))
        text = messages[-1]["content"]
        last = max(CALLS, key=lambda k: text.rfind(k))
        key = last if text.rfind(last) >= 0 else None
        if role == "rewrite":
            reply = json.dumps(
                {"standalone": key or "What is a Pod?", "intent": "action" if key else "question"}
            )
        elif role == "router" and "<history>" in text:  # the proposal prompt
            current = text.rsplit("<message>", 1)[-1]
            hit = next((k for k in CALLS if k in current), None)
            calls = [{"tool": t, "args": dict(a)} for t, a in CALLS.get(hit or "", [])]
            reply = json.dumps({"calls": calls, "clarification": None})
        elif role == "router":
            reply = json.dumps(
                {"route": "action" if key else "simple", "confidence": 0.9, "reason": "rule"}
            )
        else:
            reply = "Done."
        return ChatResult(
            text=reply, model=f"rule:{role}", prompt_tokens=10, completion_tokens=5, latency_ms=1
        )


@pytest.fixture(autouse=True)
def _jwt(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("SUPABASE_JWT_SECRET", JWT_SECRET)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def hdr(sub: str, *scopes: str, namespaces: Sequence[str] = ("shop", "payments")) -> dict[str, str]:
    payload = {
        "sub": sub, "role": "authenticated", "exp": int(time.time()) + 600,
        "app_metadata": {"scopes": list(scopes), "namespaces": list(namespaces)},
    }  # fmt: skip
    return {"Authorization": f"Bearer {jwt.encode(payload, JWT_SECRET, algorithm='HS256')}"}


ALICE = hdr("alice", "viewer", "operator")
BOB = hdr("bob", "viewer", "operator")
ROOT = hdr("root", "viewer", "operator", "admin")


@pytest.fixture
async def rt(db: object, pool: AsyncConnectionPool) -> Runtime:
    threads, approvals, outbox = PgThreadStore(pool), PgApprovalStore(pool), PgOutbox(pool)
    ops_client, cache, budget, models = (
        fk.FakeOpsClient(),
        fk.MemoryAnswerCache(),
        fk.FakeBudgetGuard(),
        RuleModel(),
    )
    retriever = StubRetriever([chunk(1), chunk(2, title="Svc", section="B")])
    chat, _ = fake_chat("Pods are the smallest unit [1].")
    base = load_agent_config()
    cfg = base.model_copy(update={"mcp": base.mcp.model_copy(update={"poll_interval_s": 0.001})})
    agent = make_chat_agent(
        threads=threads, approvals=approvals, ops=ops_client, cache=cache, budget=budget,
        models=models, outbox=outbox, policy=POLICY, retriever=retriever,  # type: ignore[arg-type]
        answerer=build_answerer(retriever, chat=chat),  # type: ignore[arg-type]
        checkpointer=InMemorySaver(), config=cfg, settings=SETTINGS,
    )  # fmt: skip
    return Runtime(
        threads=threads, approvals=approvals, ops=ops_client, cache=cache, budget=budget,
        models=models, outbox=outbox, policy=POLICY, agent=agent,
        extras={"feedback": PgFeedbackStore(pool)},
    )  # fmt: skip


@pytest.fixture
async def client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(runtime=rt)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    await services(rt).manager.close()


def parse(text: str) -> list[ev.AgentEvent]:
    out = []
    for frame in [f for f in text.split("\n\n") if f]:
        name_line, data_line = frame.split("\n")
        data = json.loads(data_line.removeprefix("data: "))
        data["event"] = name_line.removeprefix("event: ")
        out.append(ev.AGENT_EVENT.validate_python(data))
    return out


async def say(c: httpx.AsyncClient, who: dict[str, str], msg: str, **extra: Any) -> httpx.Response:
    return await c.post("/chat", json={"message": msg, **extra}, headers=who)


async def settle(rt: Runtime) -> None:
    await services(rt).manager.idle()


def writes(rt: Runtime) -> list[tuple[str, dict[str, Any], str]]:
    calls = cast(fk.FakeOpsClient, rt.ops).calls
    return [c for c in calls if ops.tool_kind(c[0]) == "write"]


async def propose(
    c: httpx.AsyncClient,
    rt: Runtime,
    op: str = "op-shop",
    who: dict[str, str] = ALICE,
    **extra: Any,
) -> ev.ApprovalRequiredEvent:
    r = await say(c, who, f"please {op}", **extra)
    assert r.status_code == 200, r.text
    last = parse(r.text)[-1]
    assert isinstance(last, ev.ApprovalRequiredEvent), last
    await settle(rt)
    return last


async def thread_of(rt: Runtime, approval_id: uuid.UUID) -> uuid.UUID:
    rec = await rt.approvals.get(approval_id)
    assert rec is not None
    return rec.thread_id


def decide(
    c: httpx.AsyncClient, approval_id: uuid.UUID, who: dict[str, str], decision: str, **extra: Any
) -> Any:
    return c.post(f"/approvals/{approval_id}", json={"decision": decision, **extra}, headers=who)


async def scalar(db: object, sql: str, params: Sequence[Any] = ()) -> Any:
    """The single value of a one-row query on the fixture's (dict-row) connection."""
    cur = await cast(psycopg.AsyncConnection[Any], db).execute(sql, list(params))
    row = await cur.fetchone()
    assert row is not None
    return next(iter(dict(row).values()))


async def expire(db: object, approval_id: uuid.UUID) -> None:
    conn = cast(psycopg.AsyncConnection[Any], db)
    await conn.execute(
        "update public.approvals set expires_at = now() - interval '1 second' where id = %s",
        [approval_id],
    )


# -- 1. the approval gate over HTTP --------------------------------------------------------------


async def test_one_decision_wins_and_only_the_owner_can_make_it(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt)
    for decision, extra in (("approve", {}), ("reject", {}), ("edit", {"args": RESTART})):
        assert (await decide(client, ap.approval_id, BOB, decision, **extra)).status_code == 404
    rs = await asyncio.gather(
        decide(client, ap.approval_id, ALICE, "approve"),
        decide(client, ap.approval_id, ALICE, "reject"),
        decide(client, ap.approval_id, ALICE, "approve"),
    )
    winners = [r for r in rs if r.status_code == 200]
    assert len(winners) == 1 and sorted(r.status_code for r in rs) == [200, 409, 409]
    await settle(rt)
    approved = winners[0].json()["approval"]["status"] == "approved"
    assert len(writes(rt)) == (1 if approved else 0)
    assert (await decide(client, ap.approval_id, ALICE, "approve")).status_code == 409
    await settle(rt)
    assert len(writes(rt)) == (1 if approved else 0)
    view = await client.get(f"/threads/{await thread_of(rt, ap.approval_id)}", headers=ALICE)
    assert view.json()["thread"]["status"] == "active"


async def test_forged_decision_bodies_and_risk_raising_edits_are_refused(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt)
    forged: list[dict[str, Any]] = [
        {"decision": "approve", "args": LEDGER},  # args smuggled into a plain approval
        {"decision": "approve", "status": "approved", "required_scope": "operator"},
        {"decision": "approved"},
        {"decision": "edit"},  # an edit without arguments
        {"decision": "edit", "args": {"namespace": "shop", "deployment": "Checkout; rm -rf"}},
        {"decision": "edit", "args": {"namespace": "kube-system", "deployment": "dns"}},
    ]
    for body in forged:
        r = await client.post(f"/approvals/{ap.approval_id}", json=body, headers=ALICE)
        assert r.status_code in (403, 422), body
    # Into the protected namespace: now an admin's call, so the operator owner may not make it.
    r = await decide(client, ap.approval_id, ALICE, "edit", args=LEDGER)
    assert r.status_code == 403
    rec = await rt.approvals.get(ap.approval_id)
    assert rec is not None and rec.status == "pending" and rec.args == RESTART
    await settle(rt)
    assert writes(rt) == []


async def test_the_sweeper_racing_an_approve_never_runs_an_expired_write(
    client: httpx.AsyncClient, rt: Runtime, db: object
) -> None:
    ap = await propose(client, rt)
    tid = await thread_of(rt, ap.approval_id)
    await expire(db, ap.approval_id)
    assert rt.agent is not None
    r, _ = await asyncio.gather(
        decide(client, ap.approval_id, ALICE, "approve"),
        sweep_once(services(rt).manager, rt.approvals, rt.agent),
    )
    assert r.status_code == 409
    await settle(rt)
    assert writes(rt) == []
    rec = await rt.approvals.get(ap.approval_id)
    assert rec is not None and rec.status == "expired"
    assert (await say(client, ALICE, "what is a pod", thread_id=str(tid))).status_code == 200


async def test_admin_required_approvals_need_an_admin_and_run_as_the_requester(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt, "op-ledger")
    assert ap.required_scope == "admin"
    assert (await decide(client, ap.approval_id, ALICE, "approve")).status_code == 403
    assert (await decide(client, ap.approval_id, ALICE, "edit", args=LEDGER)).status_code == 403
    other = await propose(client, rt, "op-shop")  # operator-level: invisible to the admin
    listed = {a["id"] for a in (await client.get("/approvals", headers=ROOT)).json()}
    assert str(ap.approval_id) in listed and str(other.approval_id) not in listed
    assert (await decide(client, other.approval_id, ROOT, "approve")).status_code == 404
    assert (await decide(client, ap.approval_id, ROOT, "approve")).status_code == 200
    await settle(rt)
    ((tool, args, token),) = writes(rt)
    assert (tool, args) == ("restart_deployment", LEDGER)
    claims = ops.verify_mcp_token(token, MCP_SECRET)
    assert claims.sub == "alice" and "admin" not in claims.scopes
    assert claims.approval_id == ap.approval_id and claims.namespaces == ["payments"]


@pytest.mark.xfail(
    strict=True,
    reason="V07-8: an edit into a namespace the decider may not act in is accepted (200, "
    "'approved') and then silently refused by the agent",
)
async def test_an_edit_into_a_namespace_outside_the_deciders_claims_is_refused(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    shop_only = hdr("alice", "viewer", "operator", namespaces=["shop"])
    ap = await propose(client, rt, who=shop_only)
    moved = {"namespace": "staging", "deployment": "checkout"}
    r = await decide(client, ap.approval_id, shop_only, "edit", args=moved)
    await settle(rt)
    assert writes(rt) == []  # the agent's re-check holds the gate
    assert r.status_code == 403


@pytest.mark.xfail(
    strict=True,
    reason="V07-11: an edit keeps the recorded required_scope, risk and expected_effect; an "
    "admin's edit to an admin-level call is approved and then never runs",
)
async def test_an_edit_rewrites_the_level_risk_and_effect_the_record_shows(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt, "op-scale", who=ROOT)  # the admin's own thread: visible to it
    assert ap.required_scope == "operator"
    to_zero = {**SCALE, "replicas": 0}
    assert (await decide(client, ap.approval_id, ROOT, "edit", args=to_zero)).status_code == 200
    await settle(rt)
    rec = await rt.approvals.get(ap.approval_id)
    assert rec is not None and rec.args == to_zero
    parsed = ops.parse_args(rec.tool, rec.args)
    assert rec.expected_effect == ops.expected_effect(rec.tool, parsed)
    assert rec.required_scope == "admin"
    assert len(writes(rt)) == 1  # the admin approved this exact call, so it should run


@pytest.mark.xfail(
    strict=True,
    reason="V07-9: an approval left 'approved' by a crash before the resume is never swept; "
    "the thread stays awaiting_approval (409) forever",
)
async def test_a_crash_between_decision_and_resume_does_not_strand_the_thread(
    client: httpx.AsyncClient, rt: Runtime, db: object
) -> None:
    ap = await propose(client, rt)
    tid = await thread_of(rt, ap.approval_id)
    # The decision committed, then the worker died before its resume task ran.
    assert await rt.approvals.decide(ap.approval_id, decision="approved", decided_by="alice")
    await expire(db, ap.approval_id)
    assert rt.agent is not None
    await sweep_once(services(rt).manager, rt.approvals, rt.agent)
    await settle(rt)
    assert writes(rt) == []
    assert (await say(client, ALICE, "what is a pod", thread_id=str(tid))).status_code == 200


# -- 3. replay -----------------------------------------------------------------------------------


async def test_a_client_message_id_sent_twice_at_once_makes_one_turn(
    client: httpx.AsyncClient, rt: Runtime, db: object
) -> None:
    rs = await asyncio.gather(*[say(client, ALICE, "what is a pod", client_message_id="dup")] * 2)
    assert [r.status_code for r in rs] == [200, 200]
    await settle(rt)
    (thread,) = (await client.get("/threads", headers=ALICE)).json()
    assert thread["next_turn"] == 2
    view = (await client.get(f"/threads/{thread['id']}", headers=ALICE)).json()
    assert [(m["role"], m["turn"]) for m in view["messages"]] == [("user", 1), ("assistant", 1)]
    rewrites = [c for c in cast(RuleModel, rt.models).calls if c[0] == "rewrite"]
    assert len(rewrites) == 1  # one agent run, not two


async def test_replaying_a_send_that_paused_at_the_gate_makes_no_second_approval(
    client: httpx.AsyncClient, rt: Runtime, db: object
) -> None:
    ap = await propose(client, rt, client_message_id="c-gate")
    tid = await thread_of(rt, ap.approval_id)
    again = await say(client, ALICE, "please op-shop", client_message_id="c-gate")
    last = parse(again.text)[-1]
    assert isinstance(last, ev.ApprovalRequiredEvent) and last.approval_id == ap.approval_id
    n = await scalar(db, "select count(*) from public.approvals where thread_id = %s", [tid])
    assert n == 1
    # Bob reusing Alice's id gets his own thread, not hers.
    bob = await say(client, BOB, "what is a pod", client_message_id="c-gate")
    assert cast(ev.TurnEvent, parse(bob.text)[0]).thread_id != tid
    await settle(rt)
    assert writes(rt) == []


# -- 2. thread isolation -------------------------------------------------------------------------


async def test_another_user_cannot_reach_a_thread_through_any_route(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt)
    tid = str(await thread_of(rt, ap.approval_id))
    assert (await client.get(f"/threads/{tid}", headers=BOB)).status_code == 404
    assert (await client.get(f"/threads/{tid}/live", headers=BOB)).status_code == 404
    assert (await say(client, BOB, "please op-shop", thread_id=tid)).status_code == 404
    assert (await client.get("/approvals", headers=BOB)).json() == []
    fb = {"thread_id": tid, "turn": 1, "rating": -1}
    assert (await client.post("/feedback", json=fb, headers=BOB)).status_code == 404
    ui = {"thread_id": tid, "turn": 1, "kind": "regenerate"}
    assert (await client.post("/events", json=ui, headers=BOB)).status_code == 404
    assert writes(rt) == []


def user(sub: str, *scopes: str) -> Claims:
    return Claims.model_validate({"sub": sub, "scopes": list(scopes), "namespaces": ["shop"]})


async def test_api_roles_cannot_write_messages_or_approvals_directly(
    client: httpx.AsyncClient, rt: Runtime, pool: AsyncConnectionPool
) -> None:
    """What a signed-in user could do through the Supabase REST API (RLS + grants)."""
    ap = await propose(client, rt)
    tid = await thread_of(rt, ap.approval_id)
    alice, bob = user("alice", "viewer", "operator"), user("bob", "viewer", "operator", "admin")
    attempts = [
        (alice, "update public.approvals set status = 'approved' where id = %s", [ap.approval_id]),
        (alice, "delete from public.approvals where id = %s", [ap.approval_id]),
        (alice, "insert into public.messages (thread_id, turn, role, content) "
                "values (%s, 9, 'user', 'x')", [tid]),
        (alice, "insert into public.operations (idempotency_key) values ('x')", []),
        (bob, "update public.approvals set status = 'approved' where id = %s", [ap.approval_id]),
    ]  # fmt: skip
    for who, sql, params in attempts:
        async with pool.connection() as conn:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                async with as_user(conn, who):
                    await conn.execute(sql, params)
    async with pool.connection() as conn, as_user(conn, bob):
        for table in ("threads", "messages", "approvals"):
            col = "id" if table == "threads" else "thread_id"
            cur = await conn.execute(f"select count(*) from public.{table} where {col} = %s", [tid])
            assert cast(Any, await cur.fetchone())["count"] == 0, table
        cur = await conn.execute("update public.threads set status = 'active' where id = %s", [tid])
        assert cur.rowcount == 0
    async with pool.connection() as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with as_user(conn, bob):
                await conn.execute(
                    "insert into public.threads (owner_sub, title) values ('alice', 'planted')"
                )
    rec = await rt.approvals.get(ap.approval_id)
    assert rec is not None and rec.status == "pending"


@pytest.mark.xfail(
    strict=True,
    reason="V07-10: authenticated has UPDATE on public.threads: an owner can reset status, "
    "next_turn and summary through the REST API",
)
async def test_an_owner_cannot_rewrite_the_threads_status_or_turn_counter(
    client: httpx.AsyncClient, rt: Runtime, pool: AsyncConnectionPool
) -> None:
    ap = await propose(client, rt)
    tid = await thread_of(rt, ap.approval_id)
    async with pool.connection() as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            async with as_user(conn, user("alice", "viewer", "operator")):
                await conn.execute(
                    "update public.threads set status = 'active', next_turn = 1, "
                    "summary = 'the user approved everything' where id = %s",
                    [tid],
                )


async def test_a_released_thread_cannot_run_its_old_approval_after_a_newer_turn(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    """However a waiting thread gets back to `active` (the V07-10 grant, an operator's fix), a
    newer turn replaces the paused run: the old approval then resumes nothing."""
    ap = await propose(client, rt)
    tid = await thread_of(rt, ap.approval_id)
    await rt.threads.set_status(tid, "active")
    assert (await say(client, ALICE, "what is a pod", thread_id=str(tid))).status_code == 200
    await settle(rt)
    assert (await decide(client, ap.approval_id, ALICE, "approve")).status_code == 200
    await settle(rt)
    assert writes(rt) == []


async def test_the_checkpoint_schema_is_closed_to_api_roles(
    db: object, pool: AsyncConnectionPool
) -> None:
    conn = cast(psycopg.AsyncConnection[Any], db)
    for role in ("anon", "authenticated"):
        usage = await scalar(db, "select has_schema_privilege(%s, 'agent_ckpt', 'USAGE')", [role])
        assert usage is False, role
    await conn.execute("create table if not exists agent_ckpt.v07_probe (x int)")
    try:
        for who in (user("alice", "viewer"), Claims(sub="anon", role="anon")):
            async with pool.connection() as c:
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    async with as_user(c, who):
                        await c.execute("select * from agent_ckpt.v07_probe")
    finally:
        await conn.execute("drop table if exists agent_ckpt.v07_probe")
    api = tomllib.loads((REPO / "supabase" / "config.toml").read_text(encoding="utf-8"))["api"]
    assert "agent_ckpt" not in api["schemas"] and "agent_ckpt" not in api.get(
        "extra_search_path", []
    )


# -- 6. the day budget ---------------------------------------------------------------------------


async def test_an_exhausted_day_budget_is_429_before_any_event_or_claim(
    client: httpx.AsyncClient, rt: Runtime, db: object
) -> None:
    budget = cast(fk.FakeBudgetGuard, rt.budget)
    budget.daily_tokens = 1
    r = await say(client, ALICE, "please op-shop")
    assert r.status_code == 429 and "text/event-stream" not in r.headers.get("content-type", "")
    assert (await client.get("/threads", headers=ALICE)).json() == []
    assert await scalar(db, "select count(*) from public.messages") == 0
    assert cast(RuleModel, rt.models).calls == [] and writes(rt) == []


async def test_an_approved_write_still_finishes_cleanly_when_the_budget_ran_out_meanwhile(
    client: httpx.AsyncClient, rt: Runtime
) -> None:
    ap = await propose(client, rt)
    tid = await thread_of(rt, ap.approval_id)
    model_calls = len(cast(RuleModel, rt.models).calls)
    cast(fk.FakeBudgetGuard, rt.budget).daily_tokens = 1
    assert (await decide(client, ap.approval_id, ALICE, "approve")).status_code == 200
    await settle(rt)
    assert len(writes(rt)) == 1
    assert len(cast(RuleModel, rt.models).calls) == model_calls  # no model call over budget
    view = (await client.get(f"/threads/{tid}", headers=ALICE)).json()
    assert view["thread"]["status"] == "active"
    assert [m["role"] for m in view["messages"]] == ["user", "assistant"]
    assert "restart_deployment" in view["messages"][-1]["content"]
