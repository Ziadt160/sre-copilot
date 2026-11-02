"""Drive the agent over the agent-v1 items, score each, and store `agent` / `conversation` runs.

One item: build a rig, reset the cluster to the item's scenario, plant any injection, then play the
user's turns. Each turn claims its number from the thread store (as the API does), streams the
agent's events, and, when the agent asks for approval, plays the item's decision through the
`ApprovalStore` (the decider must hold the approval's required scope, which `decide` itself does
not check). A crash of the agent is recorded as an `error` event: a failed task, not a failed run.

Stored runs (`eval_runs`): kind `agent` for action / permission / injection items, kind
`conversation` for multi-turn. `summary` keys:

    experiment, subset, faults, seed, n_items, n_error_events
    task_success, duplicate_side_effects (sum), steps (mean)
    by_stratum   {stratum: {"n", "task_success", "tool_correct", "args_correct", "steps",
                            "duplicate_side_effects", "refusal_correct", "injection_resisted",
                            "rewrite_recall@5", "rewrite_terms", "approval_once"}} (those that apply)
    items        {item id: {metric: value}}
    item_strata  {item id: stratum} (the report's confidence intervals need it)

`eval_scores` rows: one per item and metric (k = 0, except rewrite_recall@5 with k = 5), plus
`agent_detail` (value 1.0) whose `details` hold the transcript, end state and mutations.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from sre_core.agent.ports import ThreadBusy, ThreadNotFound
from sre_evals.agent.items import AGENT_GOLDEN_VERSION, AgentItem
from sre_evals.agent.rig import AgentRig, FaultPlan, RigFactory
from sre_evals.agent.score import (
    ApprovalOutcome,
    TaskRecord,
    TurnRecord,
    of_type,
    score_record,
    standalone_questions,
)
from sre_evals.metrics import score_item
from sre_shared.contracts import agent as ev
from sre_shared.models import Claims, GoldenItem, GoldPassage, RetrievalFilters, RetrievalRequest
from sre_shared.stamps import VersionStamps, config_hash

RECALL_K = 5
NO_FAULTS = FaultPlan()
SCORE_KEYS = (
    "task_success",
    "tool_correct",
    "args_correct",
    "steps",
    "duplicate_side_effects",
    "refusal_correct",
    "injection_resisted",
    "rewrite_recall@5",
    "rewrite_terms",
    "approval_once",
)


class AgentRunConfig(BaseModel):
    name: str = "agent-v1"
    subset: str = "full"
    seed: int = 7
    faults: dict[str, bool] = Field(default_factory=dict)
    rig: str = "fake"


async def _collect(stream: Any, sink: list[dict[str, Any]]) -> None:
    """Drain an agent stream into `sink`; an exception becomes an `error` event."""
    try:
        async for e in stream:
            sink.append(e.model_dump(mode="json"))
    except Exception as exc:  # an agent crash is a failed task, not a failed run
        sink.append(
            ev.ErrorEvent(type=type(exc).__name__, detail=str(exc)[:500]).model_dump(mode="json")
        )


async def _play_approval(
    rig: AgentRig,
    item: AgentItem,
    claims: Claims,
    thread_id: Any,
    required: Mapping[str, Any],
    faults: FaultPlan,
    sink: list[dict[str, Any]],
) -> ApprovalOutcome:
    approval_id = uuid.UUID(str(required["approval_id"]))  # events are JSON: ids are strings
    approval = await rig.approvals.get(approval_id)
    base = ApprovalOutcome(
        approval_id=str(approval_id),
        tool=str(required["tool"]),
        required_scope=str(required["required_scope"]),
        decision="not_decided",
    )
    want = item.expected.approval
    if approval is None or want == "none":
        return base
    decider = item.decider()
    if want == "approve" and base.required_scope not in decider.scopes:
        base.decision = "blocked"  # a person without the scope cannot approve it
        return base
    decision = "approved" if want == "approve" else "rejected"
    copies = 2 if (faults.double_approval and decision == "approved") else 1
    base.concurrent = copies > 1
    decided = await asyncio.gather(
        *(
            rig.approvals.decide(approval_id, decision=decision, decided_by=decider.sub)  # type: ignore[arg-type]
            for _ in range(copies)
        )
    )
    winners = [d for d in decided if d is not None]
    base.winners = len(winners)
    if not winners:
        base.decision = "refused"
        return base
    base.decision = decision
    # The API resumes the run once per successful decision; a store that lets two decisions
    # through resumes it twice, at the same time.
    sinks: list[list[dict[str, Any]]] = [[] for _ in winners]
    await asyncio.gather(
        *(
            _collect(rig.agent.resume(claims, thread_id, w), s)
            for w, s in zip(winners, sinks, strict=True)
        )
    )
    for s in sinks:
        sink.extend(s)
    if faults.redeliver and rig.redeliver is not None and decision == "approved":
        await rig.redeliver()
    return base


async def _rewrite_recall(
    rig: AgentRig,
    rec: TaskRecord,
    golden: Mapping[str, GoldenItem],
    claims: Claims,
) -> None:
    item = rec.item
    questions = standalone_questions(rec)
    for turn, expect in enumerate(item.expected.turns, start=1):
        gold: list[GoldPassage] = []
        filters: RetrievalFilters | None = None
        for gid in expect.golden_ids:
            g = golden[gid]
            gold.extend(g.gold_passages)
            filters = filters or g.filters
        if not gold:
            continue
        rec.gold[turn] = gold
        q = questions.get(turn)
        if not q or rig.retriever is None:
            rec.rewrite_recall[turn] = 0.0
            continue
        res = await rig.retriever.retrieve(
            RetrievalRequest(query=q, filters=filters or RetrievalFilters(), k=RECALL_K), claims
        )
        scores = score_item(res.chunks, gold, [RECALL_K])
        rec.rewrite_recall[turn] = scores.recall[RECALL_K] if scores else 0.0


async def run_item(
    item: AgentItem,
    factory: RigFactory,
    faults: FaultPlan,
    seed: int,
    golden: Mapping[str, GoldenItem],
) -> TaskRecord:
    rig = await factory(item, faults)
    rig.cluster.reset(item.scenario, seed)
    inj = item.expected.injection
    if inj is not None:
        if inj.planted_in == "log" and rig.plant_log is not None:
            rig.plant_log(inj.namespace or "", inj.deployment or "", inj.text)
        elif inj.planted_in == "chunk" and rig.plant_chunk is not None:
            rig.plant_chunk(inj.text)
    claims = item.caller.claims()
    owner = Claims(sub=item.thread_owner, scopes=["viewer"]) if item.thread_owner else claims
    thread = await rig.threads.create(owner)
    rec = TaskRecord(item=item, before=rig.cluster.snapshot())
    for text in item.turns:
        try:
            turn = await rig.threads.claim_turn(claims, thread.id)
        except ThreadNotFound:
            rec.turns.append(
                TurnRecord(turn=len(rec.turns) + 1, user=text, refusal="thread_not_found")
            )
            break
        except ThreadBusy:
            rec.turns.append(TurnRecord(turn=len(rec.turns) + 1, user=text, refusal="thread_busy"))
            break
        tr = TurnRecord(turn=turn, user=text)
        await _collect(rig.agent.stream_turn(claims, thread.id, turn, text), tr.events)
        for required in of_type(tr.events, "approval_required"):
            rec.approvals.append(
                await _play_approval(rig, item, claims, thread.id, required, faults, tr.events)
            )
        rec.turns.append(tr)
    rec.after = rig.cluster.snapshot()
    rec.mutations = rig.cluster.mutations()
    if item.stratum == "multi_turn":
        await _rewrite_recall(rig, rec, golden, claims)
    return rec


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(
    records: Sequence[TaskRecord],
    scores: Mapping[str, Mapping[str, float]],
    cfg: AgentRunConfig,
) -> dict[str, Any]:
    by: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by[r.item.stratum].append(r.item.id)

    def means(ids: Sequence[str]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(ids)}
        for key in SCORE_KEYS:
            vals = [scores[i][key] for i in ids if key in scores[i]]
            if not vals:
                continue
            out[key] = float(sum(vals)) if key == "duplicate_side_effects" else _mean(vals)
        return out

    all_ids = [r.item.id for r in records]
    return {
        "experiment": cfg.name,
        "subset": cfg.subset,
        "faults": cfg.faults,
        "seed": cfg.seed,
        "rig": cfg.rig,
        "n_items": len(records),
        "n_error_events": sum(len(of_type(r.events(), "error")) for r in records),
        **{k: v for k, v in means(all_ids).items() if k != "n"},
        "by_stratum": {s: means(ids) for s, ids in sorted(by.items())},
        "items": {i: dict(scores[i]) for i in all_ids},
        "item_strata": {r.item.id: r.item.stratum for r in records},
    }


def _detail(rec: TaskRecord) -> dict[str, Any]:
    return {
        "stratum": rec.item.stratum,
        "turns": [
            {"turn": t.turn, "user": t.user, "refusal": t.refusal, "events": t.events}
            for t in rec.turns
        ],
        "approvals": [a.__dict__ for a in rec.approvals],
        "before": rec.before,
        "after": rec.after,
        "mutations": rec.mutations,
        "rewrite_recall": {str(k): v for k, v in rec.rewrite_recall.items()},
    }


def score_rows(run_id: int, rec: TaskRecord, scores: Mapping[str, float]) -> list[list[Any]]:
    iid = rec.item.id
    rows: list[list[Any]] = []
    for metric, value in scores.items():
        if metric == "rewrite_recall@5":
            rows.append([run_id, iid, "rewrite_recall", RECALL_K, value, None])
        else:
            rows.append([run_id, iid, metric, 0, value, None])
    rows.append([run_id, iid, "agent_detail", 0, 1.0, Jsonb(_detail(rec))])
    return rows


async def _insert_run(
    conn: AsyncConnection[Any],
    kind: str,
    loop: str,
    golden_version: str,
    cfg: AgentRunConfig,
    summary: dict[str, Any],
) -> int:
    stamps = VersionStamps(config_hash=config_hash(cfg.model_dump()), golden_version=golden_version)
    row = await (
        await conn.execute(
            "insert into public.eval_runs (kind, loop, experiment, config, stamps, golden_version,"
            " status, summary, finished_at) values (%s, %s, %s, %s, %s, %s, 'done', %s, now())"
            " returning id",
            [
                kind, loop, cfg.name, Jsonb(cfg.model_dump(mode="json")),
                Jsonb(stamps.model_dump(mode="json")), golden_version, Jsonb(summary),
            ],
        )
    ).fetchone()  # fmt: skip
    assert row is not None
    return int(row["id"] if isinstance(row, dict) else row[0])


@dataclass
class AgentRunResult:
    summary: dict[str, Any]  # all items together
    summaries: dict[str, dict[str, Any]]  # per stored kind: "agent", "conversation"
    run_ids: dict[str, int]
    records: list[TaskRecord]


async def run_agent_evals(
    items: Sequence[AgentItem],
    factory: RigFactory,
    golden: Mapping[str, GoldenItem],
    *,
    cfg: AgentRunConfig,
    faults: FaultPlan = NO_FAULTS,
    conn: AsyncConnection[Any] | None = None,
    loop: str = "local",
    golden_version: str = AGENT_GOLDEN_VERSION,
) -> AgentRunResult:
    """Run every item (in order, one at a time: determinism over speed) and score it. With a
    connection, store one `agent` run and one `conversation` run (when there are such items)."""
    # Evals must not trace (tracing would send item transcripts to a hosted service).
    os.environ["SRE_TRACING"] = "off"
    cfg = cfg.model_copy(update={"faults": faults.__dict__.copy()})
    records = [await run_item(i, factory, faults, cfg.seed, golden) for i in items]
    scores = {r.item.id: score_record(r) for r in records}
    groups = {
        "agent": [r for r in records if r.item.stratum != "multi_turn"],
        "conversation": [r for r in records if r.item.stratum == "multi_turn"],
    }
    summaries = {k: summarize(g, scores, cfg) for k, g in groups.items() if g}
    run_ids: dict[str, int] = {}
    if conn is not None:
        async with conn.transaction():
            for kind, group in groups.items():
                if not group:
                    continue
                run_id = await _insert_run(conn, kind, loop, golden_version, cfg, summaries[kind])
                rows = [row for r in group for row in score_rows(run_id, r, scores[r.item.id])]
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "insert into public.eval_scores (run_id, item_id, metric, k, value,"
                        " details) values (%s, %s, %s, %s, %s, %s)",
                        rows,
                    )
                run_ids[kind] = run_id
    overall = summarize(records, scores, cfg)
    return AgentRunResult(overall, summaries, run_ids, records)
