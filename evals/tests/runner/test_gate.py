from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sre_evals.runner import check_gate, load_baseline, make_baseline, write_baseline
from sre_evals.runner.gate import DEFAULT_BASELINE, SCHEMA_VERSION, gate_tolerance

BASE: dict[str, Any] = {
    "schema_version": 2,
    "experiment": "dense-gte-small",
    "golden_version": "v1-abc",
    "search": "exact",
    "metrics": {"recall@5": 0.80, "mrr@10": 0.60},
    "ann_recall_floor": {"ef_search": 100, "min": 0.95},
}


def run(
    recall: float = 0.80, mrr: float = 0.60, n: int = 100, exact: bool = True, ann: float = 0.97
) -> dict[str, Any]:
    return {
        "recall@5": recall,
        "mrr@10": mrr,
        "n_scored": n,
        "exact": exact,
        "ann_recall": {"100": ann},
    }


def gate(summary: dict[str, Any], golden: str = "v1-abc", exp: str = "dense-gte-small") -> Any:
    return check_gate(exp, golden, summary, BASE)


def test_tolerance_is_one_question_and_never_below_0_01() -> None:
    assert gate_tolerance({"n_scored": 69}) == 1 / 69  # 0.0145
    assert gate_tolerance({"n_scored": 500}) == 0.01
    assert gate_tolerance({}) == 0.01


def test_losing_one_question_passes_and_two_fail() -> None:
    # n = 69: one question is 0.0145 of recall@5.
    assert gate(run(recall=0.80 - 1 / 69, n=69)).status == "pass"
    assert gate(run(recall=0.80 - 2 / 69, n=69)).status == "fail"


def test_drop_just_over_tolerance_fails_and_under_passes() -> None:
    # n = 100: tolerance 0.01.
    assert gate(run(recall=0.789)).status == "fail"
    assert gate(run(recall=0.791)).status == "pass"
    assert gate(run(mrr=0.589)).status == "fail"
    assert gate(run(mrr=0.591)).status == "pass"


def test_improvement_passes() -> None:
    assert gate(run(recall=0.95, mrr=0.9)).status == "pass"


def test_golden_experiment_or_search_mismatch_fails() -> None:
    r = gate(run(), golden="v1-other")
    assert r.status == "fail" and any("golden version" in m for m in r.messages)
    assert gate(run(), exp="hybrid-gte-small").status == "fail"
    r = gate(run(exact=False))
    assert r.status == "fail" and any("search mode" in m for m in r.messages)


def test_ann_recall_floor() -> None:
    assert gate(run(ann=0.95)).status == "pass"
    r = gate(run(ann=0.949))
    assert r.status == "fail" and any("below the floor" in m for m in r.messages)
    unmeasured = run()
    del unmeasured["ann_recall"]
    r = gate(unmeasured)
    assert r.status == "fail" and any("--ann-recall" in m for m in r.messages)


def test_empty_baseline_skips() -> None:
    r = check_gate("dense-gte-small", "v1-abc", run(), {})
    assert r.status == "skipped" and r.ok


def test_old_schema_fails() -> None:
    assert (
        check_gate("dense-gte-small", "v1-abc", run(), {**BASE, "schema_version": 1}).status
        == "fail"
    )


def test_missing_metric_fails() -> None:
    assert gate({"recall@5": 0.8, "exact": True, "ann_recall": {"100": 0.99}}).status == "fail"


def test_baseline_roundtrip_and_missing_file(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "none.json") == {}
    b = make_baseline(
        "dense-gte-small", "v1-abc", {"recall@5": 0.81234567, "mrr@10": 0.5, "exact": True},
        run_id=7, git_sha="abc", corpus_version_id=3, config_hash="h", ann_floor=(100, 0.95),
    )  # fmt: skip
    assert b["search"] == "exact" and b["ann_recall_floor"] == {"ef_search": 100, "min": 0.95}
    path = tmp_path / "b.json"
    write_baseline(b, path)
    assert json.loads(path.read_text()) == b
    assert load_baseline(path)["metrics"] == {"recall@5": 0.812346, "mrr@10": 0.5}
    same = {**run(recall=0.812346, mrr=0.5), "ann_recall": {"100": 0.99}}
    assert check_gate("dense-gte-small", "v1-abc", same, b).status == "pass"


def test_committed_baseline_is_an_exact_dense_gate() -> None:
    data = json.loads(DEFAULT_BASELINE.read_text(encoding="utf-8"))
    if not data:  # before the first baseline is written
        return
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["experiment"] == "dense-gte-small" and data["search"] == "exact"
    assert data["ann_recall_floor"]["min"] >= 0.9
