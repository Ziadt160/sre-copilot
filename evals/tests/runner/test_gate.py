from __future__ import annotations

import json
from pathlib import Path

from sre_evals.runner import check_gate, load_baseline, make_baseline, write_baseline

BASE = {
    "schema_version": 1,
    "experiment": "hybrid-gte-small",
    "golden_version": "v1-abc",
    "metrics": {"recall@5": 0.80, "mrr@10": 0.60},
}


def run(recall: float = 0.80, mrr: float = 0.60) -> dict[str, float]:
    return {"recall@5": recall, "mrr@10": mrr}


def gate(summary: dict[str, float], golden: str = "v1-abc", exp: str = "hybrid-gte-small"):
    return check_gate(exp, golden, summary, BASE)


def test_recall_drop_of_0_011_fails() -> None:
    # 0.80 - 0.789 = 0.011 > 0.01 -> fail.
    assert gate(run(recall=0.789)).status == "fail"


def test_recall_drop_of_0_009_passes() -> None:
    # 0.80 - 0.791 = 0.009 <= 0.01 -> pass.
    assert gate(run(recall=0.791)).status == "pass"


def test_drop_of_exactly_tolerance_passes() -> None:
    assert gate(run(recall=0.79)).status == "pass"


def test_mrr_drop_of_0_011_fails_and_0_009_passes() -> None:
    assert gate(run(mrr=0.589)).status == "fail"
    assert gate(run(mrr=0.591)).status == "pass"


def test_improvement_passes() -> None:
    assert gate(run(recall=0.95, mrr=0.9)).status == "pass"


def test_golden_or_experiment_mismatch_fails() -> None:
    r = gate(run(), golden="v1-other")
    assert r.status == "fail" and any("golden version" in m for m in r.messages)
    assert gate(run(), exp="dense-gte-small").status == "fail"


def test_empty_baseline_skips() -> None:
    r = check_gate("hybrid-gte-small", "v1-abc", run(), {})
    assert r.status == "skipped" and r.ok


def test_missing_metric_fails() -> None:
    assert gate({"recall@5": 0.8}).status == "fail"


def test_baseline_roundtrip_and_missing_file(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "none.json") == {}
    b = make_baseline(
        "hybrid-gte-small", "v1-abc", {"recall@5": 0.81234567, "mrr@10": 0.5},
        run_id=7, git_sha="abc", corpus_version_id=3, config_hash="h",
    )  # fmt: skip
    path = tmp_path / "b.json"
    write_baseline(b, path)
    assert json.loads(path.read_text()) == b
    assert load_baseline(path)["metrics"] == {"recall@5": 0.812346, "mrr@10": 0.5}
    assert (
        check_gate("hybrid-gte-small", "v1-abc", {"recall@5": 0.812346, "mrr@10": 0.5}, b).status
        == "pass"
    )


def test_committed_placeholder_is_empty_json_object() -> None:
    from sre_evals.runner.gate import DEFAULT_BASELINE

    assert load_baseline(DEFAULT_BASELINE) == {}
