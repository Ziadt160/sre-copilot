"""The agent-v1 item schema and its loader.

One JSON object per line of `evals/golden/agent-v1/items.jsonl`. `expected` is stratum-specific
(see `Expected`); `validate_items` checks an item set against the scenario file and golden v1 so a
task nobody can pass, or a gold passage that does not exist, is found before a run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sre_shared.contracts import ops
from sre_shared.models import Claims, GoldenItem, Scope
from sre_shared.settings import REPO_ROOT

AGENT_GOLDEN_VERSION = "agent-v1"
Stratum = Literal["action", "multi_turn", "permission", "injection"]
STRATA: tuple[Stratum, ...] = ("action", "multi_turn", "permission", "injection")
# Accepted items after the two-pass review (agent-v1/README.md); the draft had 25 multi_turn and
# 4 of them were rejected as near-duplicates or unanswerable as asked.
TARGET_COUNTS: dict[str, int] = {"action": 25, "multi_turn": 21, "permission": 10, "injection": 10}
PR_TAG = "pr"


def items_path(version: str = AGENT_GOLDEN_VERSION) -> Path:
    return REPO_ROOT / "evals" / "golden" / version / "items.jsonl"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Caller(_Strict):
    sub: str
    scopes: list[Scope]
    namespaces: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=lambda: ["public"])

    def claims(self) -> Claims:
        return Claims(
            sub=self.sub,
            groups=list(self.groups),
            scopes=list(self.scopes),
            namespaces=list(self.namespaces),
        )


class DeploymentState(_Strict):
    """What a deployment must look like at the end; fields left out are not checked."""

    namespace: str
    deployment: str
    replicas: int | None = Field(default=None, ge=0)
    revision: int | None = Field(default=None, ge=1)
    healthy: bool | None = None

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.deployment}"


class TurnExpect(_Strict):
    """Multi-turn: what the user message at this position should retrieve and cite."""

    golden_ids: list[str] = Field(default_factory=list)
    must_cite: bool = False
    rewrite_terms: list[str] = Field(
        default_factory=list,
        description="words the standalone question must contain (what a bare follow-up omits)",
    )


class Injection(_Strict):
    planted_in: Literal["chunk", "log"]
    text: str = Field(min_length=10)
    namespace: str | None = None  # log: where the line is planted
    deployment: str | None = None

    @model_validator(mode="after")
    def _log_has_target(self) -> Injection:
        if self.planted_in == "log" and not (self.namespace and self.deployment):
            raise ValueError("a log injection names the namespace and deployment it is planted in")
        return self


class Forbidden(_Strict):
    """A tool call the injected text asks for. `args` is a subset; None forbids the tool at all."""

    tool: str
    args: dict[str, Any] | None = None


Refusal = Literal["scope", "namespace", "admin_required", "thread_owner"]


class Expected(_Strict):
    # action, injection (benign part), permission (the call that must not run)
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    approval: Literal["approve", "reject", "none"] = "none"
    approval_scope: Literal["operator", "admin"] | None = None
    end_state: list[DeploymentState] = Field(default_factory=list)
    # multi_turn
    turns: list[TurnExpect] = Field(default_factory=list)
    # permission
    refusal: Refusal | None = None
    # injection
    injection: Injection | None = None
    forbidden: list[Forbidden] = Field(default_factory=list)


class AgentItem(_Strict):
    id: str
    stratum: Stratum
    status: Literal["draft", "accepted", "rejected"] = "draft"
    turns: list[str] = Field(min_length=1)
    caller: Caller
    scenario: str = "default"
    approver: Caller | None = None
    thread_owner: str | None = None
    expected: Expected
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _stratum_shape(self) -> AgentItem:
        e, s = self.expected, self.stratum
        problems: list[str] = []
        if e.tool is not None:
            try:
                kind = ops.tool_kind(e.tool)
            except ValueError as exc:
                problems.append(str(exc))
                kind = "read"
        else:
            kind = "read"
        if s == "action":
            if not e.tool:
                problems.append("an action item names the expected tool")
            if kind != "read" and e.approval == "none":
                problems.append("a write item says whether the approval is approved or rejected")
        elif s == "multi_turn":
            if not 2 <= len(self.turns) <= 4:
                problems.append("a conversation has 2-4 user turns")
            if len(e.turns) != len(self.turns):
                problems.append("expected.turns has one entry per user turn")
            if not any(t.golden_ids for t in e.turns):
                problems.append("a conversation has gold passages on at least one turn")
        elif s == "permission":
            if e.refusal is None or not e.tool:
                problems.append("a permission item names the refusal kind and the tool")
            if (e.refusal == "thread_owner") != (self.thread_owner is not None):
                problems.append("thread_owner is set exactly for thread_owner refusals")
        elif s == "injection" and (e.injection is None or not e.forbidden):
            problems.append("an injection item has the injected text and the forbidden calls")
        for f in e.forbidden:
            if f.tool not in ops.TOOL_ARGS:
                problems.append(f"forbidden tool {f.tool!r} is unknown")
        if problems:
            raise ValueError(f"{self.id}: " + "; ".join(problems))
        return self

    @property
    def in_pr_subset(self) -> bool:
        return PR_TAG in self.tags

    def decider(self) -> Caller:
        return self.approver or self.caller


def load_items(
    path: Path | None = None,
    *,
    subset: Literal["pr", "full"] = "full",
    include_rejected: bool = False,
) -> list[AgentItem]:
    """The task set; rejected items stay in the file (with the reviewers' notes) but do not run."""
    path = path or items_path()
    items: list[AgentItem] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                items.append(AgentItem.model_validate_json(line))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        dupes = sorted(k for k, n in Counter(ids).items() if n > 1)
        raise ValueError(f"{path}: duplicate item ids {dupes}")
    if not include_rejected:
        items = [i for i in items if i.status != "rejected"]
    if subset == "pr":
        items = [i for i in items if i.in_pr_subset]
    return items


def write_items(path: Path, items: Iterable[AgentItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_none=True) + "\n")


def item_set_version(path: Path) -> str:
    from sre_shared.stamps import file_hash

    return f"{path.parent.name}-{file_hash(path)}"


def validate_items(
    items: Sequence[AgentItem],
    golden: Mapping[str, GoldenItem],
    scenarios: Mapping[str, Any],
    policy: ops.NamespacePolicy,
    *,
    require_counts: bool = True,
) -> list[str]:
    """Problems found, empty when the set is sound. `scenarios` maps a name to a SimScenario."""
    from sre_evals.agent.cluster import FakeCluster, simulate_expected

    problems: list[str] = []
    counts: Counter[str] = Counter(i.stratum for i in items)
    if require_counts:
        for stratum, n in TARGET_COUNTS.items():
            if counts[stratum] != n:
                problems.append(f"stratum {stratum}: {counts[stratum]} items, expected {n}")
    for item in items:
        e = item.expected
        if item.scenario not in scenarios:
            problems.append(f"{item.id}: unknown scenario {item.scenario!r}")
            continue
        cluster = FakeCluster(scenarios, policy)
        cluster.reset(item.scenario, 0)
        snap = cluster.snapshot()["deployments"]
        for ds in e.end_state:
            if ds.key not in snap:
                problems.append(f"{item.id}: end state names {ds.key}, not in {item.scenario}")
        if e.injection and e.injection.planted_in == "log":
            key = f"{e.injection.namespace}/{e.injection.deployment}"
            if key not in snap:
                problems.append(f"{item.id}: injection is planted in {key}, not in the scenario")
        if item.stratum == "multi_turn":
            for i, t in enumerate(e.turns, start=1):
                for gid in t.golden_ids:
                    g = golden.get(gid)
                    if g is None:
                        problems.append(f"{item.id} turn {i}: golden id {gid} is not in golden v1")
                    elif g.status != "accepted" or not g.gold_passages:
                        problems.append(f"{item.id} turn {i}: golden {gid} is not accepted + gold")
                if t.must_cite and not t.golden_ids:
                    problems.append(f"{item.id} turn {i}: must_cite without a golden id")
        if item.stratum == "action":
            problems.extend(simulate_expected(item, scenarios, policy))
        if item.stratum in ("permission", "injection") and e.tool and e.args:
            unknown = set(e.args) - set(ops.TOOL_ARGS[e.tool].model_fields)
            if unknown:
                problems.append(f"{item.id}: {sorted(unknown)} are not arguments of {e.tool}")
    return problems
