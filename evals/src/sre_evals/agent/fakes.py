"""Fakes for the agent runner: an ops client over `FakeCluster`, fault wrappers, an "oracle" agent
that plays each item's expected behaviour (with knobs to misbehave), and a rig factory.

The oracle is not the agent under test. It exists so the runner, the metrics and the PR workflow
run without a model: with default behaviour it passes every item, and each knob breaks one thing a
metric must catch (a wrong argument, a bad rewrite, a duplicated write, a followed injection).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from sre_core.agent.testing import MemoryApprovalStore, MemoryThreadStore, new_approval
from sre_evals.agent.cluster import FakeCluster
from sre_evals.agent.items import AgentItem
from sre_evals.agent.rig import AgentRig, FaultPlan, RigFactory
from sre_evals.runner.testing import FakeRetriever, chunk
from sre_shared.contracts import agent as ev
from sre_shared.contracts import ops
from sre_shared.models import Citation, Claims, GoldenItem

SECRET = "eval-rig-only-secret-not-a-credential-0000"  # noqa: S105  (signs throwaway tokens inside a test rig)


class ClusterOps:
    """An `OpsClient` over a `FakeCluster`: the idempotency key is built from the token's thread
    and turn, as mcp-ops does, so a repeated call with one token is one execution."""

    def __init__(self, cluster: FakeCluster, secret: str = SECRET) -> None:
        self.cluster = cluster
        self.secret = secret
        self.operations: dict[uuid.UUID, ops.OperationRecord] = {}
        self.last_write: tuple[str, dict[str, Any], str] | None = None

    async def read(self, tool: str, args: Mapping[str, Any], token: str) -> dict[str, Any]:
        ops.verify_mcp_token(token, self.secret)
        return self.cluster.read(tool, args)

    async def _run(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        claims = ops.verify_mcp_token(token, self.secret)
        key = ops.idempotency_key(claims.thread_id, claims.turn, tool, args)
        parsed = ops.parse_args(tool, args)
        ns, target = ops.target_of(tool, parsed)
        status: ops.OperationStatus = "succeeded"
        result: dict[str, Any] | None
        error = None
        try:
            result = self.cluster.execute(tool, args, key)
        except ValueError as e:
            status, result, error = "failed", None, str(e)
        rec = ops.OperationRecord(
            id=uuid.uuid4(), idempotency_key=key, kind=ops.tool_kind(tool), tool=tool,
            args=ops.normalize_args(tool, args), namespace=ns, target=target,
            subject=claims.sub, scopes=list(claims.scopes), thread_id=claims.thread_id,
            turn=claims.turn, status=status, result=result, error=error,
        )  # fmt: skip
        self.operations[rec.id] = rec
        return rec

    async def write(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        self.last_write = (tool, dict(args), token)
        return await self._run(tool, args, token)

    async def escalate(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        self.last_write = (tool, dict(args), token)
        return await self._run(tool, args, token)

    async def operation(self, operation_id: uuid.UUID, token: str) -> ops.OperationRecord:
        return self.operations[operation_id]

    async def redeliver(self) -> None:
        """The task body arrives a second time (same token, so the same idempotency key)."""
        if self.last_write is not None:
            tool, args, token = self.last_write
            await self._run(tool, args, token)


class DroppedReplyOps:
    """Fault: every write is sent twice and the first reply is lost on the way back."""

    def __init__(self, inner: ClusterOps) -> None:
        self.inner = inner

    async def read(self, tool: str, args: Mapping[str, Any], token: str) -> dict[str, Any]:
        return await self.inner.read(tool, args, token)

    async def write(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        await self.inner.write(tool, args, token)  # executed; the reply never arrives
        return await self.inner.write(tool, args, token)  # the client sends it again

    async def escalate(self, tool: str, args: Mapping[str, Any], token: str) -> ops.OperationRecord:
        await self.inner.escalate(tool, args, token)
        return await self.inner.escalate(tool, args, token)

    async def operation(self, operation_id: uuid.UUID, token: str) -> ops.OperationRecord:
        return await self.inner.operation(operation_id, token)


class LeakyApprovalStore(MemoryApprovalStore):
    """A broken store for tests: `decide` succeeds for every caller (no once-only rule)."""

    async def decide(
        self, approval_id: uuid.UUID, *, decision: Any, decided_by: str, args: Any = None
    ) -> ops.ApprovalRecord | None:
        a = self.items[approval_id]
        self.items[approval_id] = a.model_copy(
            update={"status": decision, "decided_by": decided_by}
        )
        return self.items[approval_id]


@dataclass(frozen=True)
class Behavior:
    """Ways the oracle agent can go wrong; the default is correct on every item."""

    wrong_args: bool = False  # propose the right tool on the wrong target
    bad_rewrite: bool = False  # leave a follow-up as typed instead of rewriting it
    no_cite: bool = False
    follow_injection: Literal["no", "propose", "execute"] = "no"
    duplicate_write: bool = False  # the agent itself writes twice
    write_without_approval: bool = False  # run a write the moment it is asked


def _wrong(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(args)
    if "deployment" in out:
        out["deployment"] = "web" if out["deployment"] != "web" else "cart"
    elif "namespace" in out:
        out["namespace"] = "staging"
    return out


@dataclass
class OracleAgent:
    item: AgentItem
    golden: Mapping[str, GoldenItem]
    threads: MemoryThreadStore
    approvals: MemoryApprovalStore
    ops_client: Any
    policy: ops.NamespacePolicy
    behavior: Behavior = field(default_factory=Behavior)
    planted: list[str] = field(default_factory=list)

    def _token(
        self, claims: Claims, thread_id: uuid.UUID, turn: int, approval: uuid.UUID | None
    ) -> str:
        return ops.mint_mcp_token(
            secret=SECRET, sub=claims.sub, scopes=list(claims.scopes),
            namespaces=list(claims.namespaces), thread_id=thread_id, turn=turn,
            approval_id=approval,
        )  # fmt: skip

    def _call(self) -> tuple[str, dict[str, Any]]:
        e = self.item.expected
        assert e.tool is not None
        args = dict(e.args)
        if e.tool == "get_logs":  # the item does not name the pod; the agent finds it
            args.setdefault("pod", "checkout-0")
        if e.tool == "create_incident":  # the item fixes only the severity and namespace
            args.setdefault("title", "Incident opened from chat")
            args.setdefault("summary", "Opened at the user's request from the chat agent.")
        return e.tool, (_wrong(e.tool, args) if self.behavior.wrong_args else args)

    async def stream_turn(
        self, claims: Claims, thread_id: uuid.UUID, turn: int, message: str
    ) -> AsyncIterator[ev.AgentEvent]:
        yield ev.TurnEvent(thread_id=thread_id, turn=turn)
        s = self.item.stratum
        if s == "multi_turn":
            for e in self._answer_turn(thread_id, turn, message):
                yield e
            return
        if s == "injection":
            async for e in self._injection(claims, thread_id, turn):
                yield e
            return
        tool, args = self._call()
        yield ev.RouteEvent(route="action", standalone_question=message)
        reason = None
        try:
            reason = ops.allowed(
                tool,
                ops.parse_args(tool, args),
                list(claims.scopes),
                list(claims.namespaces),
                self.policy,
            )
        except ValueError as exc:
            reason = str(exc)
        if reason and not self.behavior.write_without_approval:
            yield ev.DoneEvent(thread_id=thread_id, turn=turn, route="action",
                               answer=f"I can't do that: {reason}.")  # fmt: skip
            return
        kind = ops.tool_kind(tool)
        yield ev.ToolProposedEvent(tool=tool, args=args, kind=kind)
        if kind == "read":
            data = await self.ops_client.read(
                tool, args, self._token(claims, thread_id, turn, None)
            )
            yield ev.DoneEvent(
                thread_id=thread_id, turn=turn, route="action", answer=f"Result: {data}"
            )
            return
        if self.behavior.write_without_approval:
            op = await self.ops_client.write(tool, args, self._token(claims, thread_id, turn, None))
            yield ev.DoneEvent(
                thread_id=thread_id, turn=turn, route="action", answer=f"Done {op.status}"
            )
            return
        a = await self.approvals.create(
            new_approval(thread_id, turn, tool, args, policy=self.policy, requested_by=claims.sub,
                         now=self.approvals.clock())
        )  # fmt: skip
        await self.threads.set_status(thread_id, "awaiting_approval")
        yield ev.ApprovalRequiredEvent(
            approval_id=a.id, tool=a.tool, args=a.args, namespace=a.namespace, target=a.target,
            risk=a.risk, required_scope=a.required_scope, expected_effect=a.expected_effect,
            expires_at=a.expires_at,
        )  # fmt: skip

    def _answer_turn(self, thread_id: uuid.UUID, turn: int, message: str) -> list[ev.AgentEvent]:
        expect = self.item.expected.turns[turn - 1]
        gold = [self.golden[g] for g in expect.golden_ids]
        standalone = message
        if gold and not self.behavior.bad_rewrite:
            terms = ", ".join(expect.rewrite_terms)  # what a good rewrite restores
            standalone = f"{gold[0].question} ({terms})" if terms else gold[0].question
        events: list[ev.AgentEvent] = [
            ev.RouteEvent(route="simple", standalone_question=standalone)
        ]
        cites = [
            Citation(chunk_id=100 + i, source_path=p.source_path, k8s_version=p.k8s_version,
                     url="https://example.test/", title="t")
            for i, g in enumerate(gold) for p in g.gold_passages
        ]  # fmt: skip
        if self.behavior.no_cite:
            cites = []
        events.append(
            ev.DoneEvent(thread_id=thread_id, turn=turn, route="simple",
                         answer="Answer from the documentation [1].", citations=cites)
        )  # fmt: skip
        return events

    async def _injection(
        self, claims: Claims, thread_id: uuid.UUID, turn: int
    ) -> AsyncIterator[ev.AgentEvent]:
        e = self.item.expected
        yield ev.RouteEvent(route="action", standalone_question=self.item.turns[0])
        if e.tool is not None:
            tool, args = e.tool, dict(e.args)
            if tool == "get_logs" and self.item.expected.injection:
                inj = self.item.expected.injection
                args.setdefault("pod", f"{inj.deployment}-0")
            yield ev.ToolProposedEvent(tool=tool, args=args, kind=ops.tool_kind(tool))
            await self.ops_client.read(tool, args, self._token(claims, thread_id, turn, None))
        if self.behavior.follow_injection != "no" and e.forbidden:
            f = e.forbidden[0]
            args = dict(f.args or {})
            yield ev.ToolProposedEvent(tool=f.tool, args=args, kind=ops.tool_kind(f.tool))
            if self.behavior.follow_injection == "execute":
                await self.ops_client.write(
                    f.tool, args, self._token(claims, thread_id, turn, None)
                )
        yield ev.DoneEvent(
            thread_id=thread_id, turn=turn, route="action", answer="Here is what I found."
        )

    async def resume(
        self, claims: Claims, thread_id: uuid.UUID, approval: ops.ApprovalRecord
    ) -> AsyncIterator[ev.AgentEvent]:
        if approval.status != "approved":
            await self.threads.set_status(thread_id, "active")
            yield ev.DoneEvent(thread_id=thread_id, turn=approval.turn, route="action",
                               answer=f"Not run: the approval was {approval.status}.")  # fmt: skip
            return
        token = self._token(claims, thread_id, approval.turn, approval.id)
        send = (
            self.ops_client.escalate
            if ops.tool_kind(approval.tool) == "escalation"
            else self.ops_client.write
        )
        op = await send(approval.tool, approval.args, token)
        if self.behavior.duplicate_write:
            await send(approval.tool, approval.args, token)
        yield ev.OperationEvent(operation_id=op.id, tool=op.tool, status="running")
        yield ev.OperationEvent(
            operation_id=op.id, tool=op.tool, status=op.status, result=op.result
        )
        await self.approvals.mark_executed(approval.id)
        await self.threads.set_status(thread_id, "active")
        yield ev.DoneEvent(thread_id=thread_id, turn=approval.turn, route="action",
                           answer=f"Done: {approval.expected_effect}")  # fmt: skip


def make_fake_rig_factory(
    scenarios: Mapping[str, Any],
    policy: ops.NamespacePolicy,
    golden: Mapping[str, GoldenItem],
    *,
    behavior: Behavior | None = None,
    dedupe: bool = True,
    leaky_approvals: bool = False,
) -> RigFactory:
    """A rig of fakes per item: fresh stores, a `FakeCluster`, the oracle agent."""
    by_question = {" ".join(g.question.lower().split()): g for g in golden.values()}

    async def factory(item: AgentItem, faults: FaultPlan) -> AgentRig:
        cluster = FakeCluster(scenarios, policy, dedupe=dedupe)
        cluster_ops = ClusterOps(cluster)
        client: Any = DroppedReplyOps(cluster_ops) if faults.dup_write else cluster_ops
        threads = MemoryThreadStore()
        approvals = LeakyApprovalStore() if leaky_approvals else MemoryApprovalStore()
        agent = OracleAgent(
            item, golden, threads, approvals, client, policy, behavior or Behavior()
        )

        def script(req: Any, exact: bool) -> tuple[list[Any], float]:
            q = " ".join(req.query.lower().split())
            g = next((x for k, x in by_question.items() if q.startswith(k)), None)
            if g is None:
                return [chunk(900, "unrelated text about nothing in particular here")], 1.0
            return [
                chunk(i + 1, p.quote, p.source_path, p.k8s_version)
                for i, p in enumerate(g.gold_passages)
            ], 1.0

        planter: Callable[[str], None] = agent.planted.append
        return AgentRig(
            agent=agent, threads=threads, approvals=approvals, cluster=cluster,
            retriever=FakeRetriever(script), plant_log=cluster.plant_log, plant_chunk=planter,
            redeliver=cluster_ops.redeliver if faults.redeliver else None,
        )  # fmt: skip

    return factory
