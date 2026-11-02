"""Agent evals on fakes: the task set is sound, the metrics catch what they must, runs are
deterministic, injected faults are survived by a correct backend and bite a broken one."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from sre_evals.agent.cluster import FakeCluster
from sre_evals.agent.fakes import Behavior, make_fake_rig_factory
from sre_evals.agent.items import (
    TARGET_COUNTS,
    AgentItem,
    items_path,
    load_items,
    validate_items,
)
from sre_evals.agent.report import render_agent_report
from sre_evals.agent.rig import FaultPlan
from sre_evals.agent.runner import AgentRunConfig, run_agent_evals
from sre_evals.agent.score import duplicate_side_effects
from sre_shared.config import load_mcp_ops_config, load_simulator_scenarios
from sre_shared.golden_io import golden_path, load_golden

SCENARIOS = load_simulator_scenarios().scenarios
POLICY = load_mcp_ops_config().policy()
GOLDEN = {g.id: g for g in load_golden(golden_path("v1"))}
ITEMS = load_items()
BY_ID = {i.id: i for i in ITEMS}


async def run(
    ids: list[str] | None = None,
    *,
    faults: FaultPlan = FaultPlan(),
    dedupe: bool = True,
    leaky: bool = False,
    **behavior: Any,
) -> dict[str, Any]:
    factory = make_fake_rig_factory(
        SCENARIOS, POLICY, GOLDEN, behavior=Behavior(**behavior), dedupe=dedupe,
        leaky_approvals=leaky,
    )  # fmt: skip
    items = [BY_ID[i] for i in ids] if ids else ITEMS
    out = await run_agent_evals(items, factory, GOLDEN, cfg=AgentRunConfig(), faults=faults)
    return out.summary


def item_scores(summary: dict[str, Any], iid: str) -> dict[str, float]:
    return summary["items"][iid]  # type: ignore[no-any-return]


# --- the task set -------------------------------------------------------------------------------


def test_items_validate_and_have_the_designed_shape() -> None:
    assert len(ITEMS) == 70
    assert Counter(i.stratum for i in ITEMS) == TARGET_COUNTS
    assert all(i.status == "draft" for i in ITEMS)
    assert validate_items(ITEMS, GOLDEN, SCENARIOS, POLICY) == []
    pr = load_items(items_path(), subset="pr")
    assert {i.stratum for i in pr} == set(TARGET_COUNTS)


def test_validation_finds_an_unreachable_end_state_and_a_missing_gold_id() -> None:
    bad = BY_ID["act-009"].model_copy(deep=True)
    bad.expected.end_state[0].healthy = False  # a restart fixes the OOM: unhealthy is unreachable
    ghost = BY_ID["mt-001"].model_copy(deep=True)
    ghost.expected.turns[0].golden_ids = ["fac-9999"]
    problems = validate_items([bad, ghost], GOLDEN, SCENARIOS, POLICY, require_counts=False)
    assert any("act-009" in p and "unreachable" in p for p in problems)
    assert any("fac-9999" in p for p in problems)


def test_schema_refuses_a_write_item_without_an_approval_decision() -> None:
    data = BY_ID["act-009"].model_dump(mode="json")
    data["expected"]["approval"] = "none"
    with pytest.raises(ValueError, match="approved or rejected"):
        AgentItem.model_validate(data)


def test_fake_cluster_applies_scenario_semantics() -> None:
    c = FakeCluster(SCENARIOS, POLICY)
    c.reset("default", 0)
    c.execute("scale_deployment", {"namespace": "shop", "deployment": "checkout", "replicas": 6})
    assert c.snapshot()["deployments"]["shop/checkout"]["healthy"] is False  # scaling fixes nothing
    c.execute(
        "rollback_deployment", {"namespace": "shop", "deployment": "checkout", "to_revision": 1}
    )
    assert c.snapshot()["deployments"]["shop/checkout"]["healthy"] is False  # 1 is not a good one
    c.execute(
        "rollback_deployment", {"namespace": "shop", "deployment": "checkout", "to_revision": 2}
    )
    assert c.snapshot()["deployments"]["shop/checkout"]["healthy"] is True


# --- the metrics --------------------------------------------------------------------------------


async def test_oracle_passes_everything_with_no_duplicates() -> None:
    s = await run()
    assert s["task_success"] == 1.0
    assert s["duplicate_side_effects"] == 0
    assert s["n_error_events"] == 0


async def test_a_deliberately_duplicated_write_is_counted() -> None:
    # The agent writes twice and the backend does not deduplicate: two identical mutations.
    s = await run(["act-009"], dedupe=False, duplicate_write=True)
    assert item_scores(s, "act-009")["duplicate_side_effects"] == 1.0
    # The second apply of a restart on an already-fixed deployment changes nothing; it still counts.
    c = FakeCluster(SCENARIOS, POLICY, dedupe=False)
    c.reset("default", 0)
    args = {"namespace": "shop", "deployment": "cart"}
    c.execute("restart_deployment", args, "k")
    c.execute("restart_deployment", args, "k")
    assert [m["changed"] for m in c.mutations()] == [True, False]
    assert duplicate_side_effects(c.mutations()) == 1
    # A correct backend turns the same double send into one execution.
    s = await run(["act-009"], duplicate_write=True)
    assert item_scores(s, "act-009")["duplicate_side_effects"] == 0.0


def test_duplicate_count_by_hand() -> None:
    m = {"tool": "restart_deployment", "args": {"namespace": "shop", "deployment": "cart"}}
    other = {
        "tool": "scale_deployment",
        "args": {"namespace": "shop", "deployment": "web", "replicas": 3},
    }
    assert duplicate_side_effects([]) == 0
    assert duplicate_side_effects([m, other]) == 0
    assert duplicate_side_effects([m, other, dict(m, idempotency_key="k2"), dict(m)]) == 1
    # Per key (O-2): a second apply under one key counts, even when it changed nothing.
    keyed = dict(m, idempotency_key="k1", changed=True)
    assert duplicate_side_effects([keyed, dict(keyed, changed=False)]) == 1
    assert duplicate_side_effects([keyed, dict(keyed, idempotency_key="k2")]) == 0


async def test_a_wrong_argument_fails_args_correct() -> None:
    s = await run(["act-009", "act-011"], wrong_args=True)
    for iid in ("act-009", "act-011"):
        sc = item_scores(s, iid)
        assert sc["tool_correct"] == 1.0
        assert sc["args_correct"] == 0.0
        assert sc["task_success"] == 0.0  # the wrong deployment got restarted


async def test_a_correct_refusal_passes_and_a_write_that_runs_anyway_fails() -> None:
    ids = ["perm-001", "perm-004", "perm-007", "perm-009"]
    s = await run(ids)
    assert all(item_scores(s, i)["refusal_correct"] == 1.0 for i in ids)
    s = await run(ids[:2], write_without_approval=True)
    assert all(item_scores(s, i)["refusal_correct"] == 0.0 for i in ids[:2])


async def test_an_operator_cannot_approve_an_admin_request() -> None:
    summary = await run(["perm-007"])
    assert item_scores(summary, "perm-007")["refusal_correct"] == 1.0


async def test_an_injected_instruction_that_triggers_a_tool_call_fails_injection_resisted() -> None:
    ids = ["inj-001", "inj-003", "inj-005"]
    clean = await run(ids)
    assert all(item_scores(clean, i)["injection_resisted"] == 1.0 for i in ids)
    for mode in ("propose", "execute"):
        s = await run(ids, follow_injection=mode)
        for i in ids:
            assert item_scores(s, i)["injection_resisted"] == 0.0, (mode, i)
            assert item_scores(s, i)["task_success"] == 0.0


async def test_rewrite_recall_and_terms_drop_when_the_follow_up_is_not_rewritten() -> None:
    good = await run(["mt-001", "mt-004"])
    bad = await run(["mt-001", "mt-004"], bad_rewrite=True)
    assert item_scores(good, "mt-001")["rewrite_recall@5"] == 1.0
    assert item_scores(good, "mt-001")["rewrite_terms"] == 1.0
    assert item_scores(bad, "mt-001")["rewrite_recall@5"] == 0.5  # turn 1 is already standalone
    assert item_scores(bad, "mt-001")["rewrite_terms"] == 0.0
    assert item_scores(bad, "mt-004")["rewrite_recall@5"] == pytest.approx(1 / 3)


async def test_must_cite_fails_without_a_citation() -> None:
    s = await run(["mt-001"], no_cite=True)
    assert item_scores(s, "mt-001")["task_success"] == 0.0


async def test_a_rejected_write_runs_nothing() -> None:
    s = await run(["act-022", "act-023", "act-024"])
    assert all(item_scores(s, i)["task_success"] == 1.0 for i in ("act-022", "act-023", "act-024"))


# --- determinism and faults ---------------------------------------------------------------------


async def test_two_runs_have_identical_scores() -> None:
    a, b = await run(faults=FaultPlan.all()), await run(faults=FaultPlan.all())
    assert a["items"] == b["items"]
    assert a["by_stratum"] == b["by_stratum"]


async def test_a_correct_backend_survives_all_three_faults() -> None:
    s = await run(faults=FaultPlan.all())
    assert s["task_success"] == 1.0
    assert s["duplicate_side_effects"] == 0


async def test_the_faults_bite_a_backend_that_does_not_deduplicate() -> None:
    ids = ["act-009", "act-010", "act-011"]
    for plan in (FaultPlan(dup_write=True), FaultPlan(redeliver=True)):
        s = await run(ids, faults=plan, dedupe=False)
        assert s["duplicate_side_effects"] == len(ids), plan


async def test_two_concurrent_approvals_with_a_store_that_lets_both_through() -> None:
    plan = FaultPlan(double_approval=True)
    ok = await run(["act-009"], faults=plan)
    assert item_scores(ok, "act-009")["approval_once"] == 1.0
    leaky = await run(["act-009"], faults=plan, leaky=True)
    assert item_scores(leaky, "act-009")["approval_once"] == 0.0
    both = await run(["act-009"], faults=plan, leaky=True, dedupe=False)
    assert item_scores(both, "act-009")["duplicate_side_effects"] == 1.0


async def test_report_sections_render_with_intervals() -> None:
    s = await run(faults=FaultPlan.all())
    text = render_agent_report([s])
    assert "| action | 25 |" in text
    assert "Duplicate side effects: **0** (gate: 0). Gate met." in text
    assert "[" in text  # confidence intervals


# --- storage (needs the test database) ----------------------------------------------------------


@pytest.mark.integration
async def test_runs_are_stored_as_agent_and_conversation_kinds(db: Any) -> None:
    factory = make_fake_rig_factory(SCENARIOS, POLICY, GOLDEN)
    items = [BY_ID[i] for i in ("act-009", "perm-001", "inj-001", "mt-001")]
    out = await run_agent_evals(
        items, factory, GOLDEN, cfg=AgentRunConfig(), conn=db, loop="ci",
        golden_version="agent-v1-test",
    )  # fmt: skip
    assert set(out.run_ids) == {"agent", "conversation"}
    rows = await (
        await db.execute("select kind, status, summary from public.eval_runs order by id")
    ).fetchall()
    assert [r["kind"] for r in rows] == ["agent", "conversation"]
    assert rows[0]["summary"]["n_items"] == 3
    scores = await (
        await db.execute(
            "select item_id, metric, k, value from public.eval_scores where run_id = %s",
            [out.run_ids["conversation"]],
        )
    ).fetchall()
    metrics = {(r["metric"], r["k"]) for r in scores}
    assert ("task_success", 0) in metrics and ("rewrite_recall", 5) in metrics
    assert ("agent_detail", 0) in metrics
