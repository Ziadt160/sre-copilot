"""CI gate: compare a run's headline metrics with the committed baseline.

Baseline file `evals/baselines/retrieval.json`:

    {
      "schema_version": 2,
      "experiment": "dense-gte-small",
      "golden_version": "v1-<hash>",       # golden set the numbers were measured on
      "search": "exact",                   # dense leg as an exact scan; "index" = HNSW
      "metrics": {"recall@5": 0.0, "mrr@10": 0.0},
      "ann_recall_floor": {"ef_search": 100, "min": 0.95},   # optional
      "source": {"run_id": 1, "git_sha": "...", "corpus_version_id": 1, "config_hash": "..."}
    }

`{}` means "no baseline yet": the gate skips.

Why exact search: HNSW builds are randomised. Rebuilding the index with the same parameters
moved recall@5 by one question (sweep `hnsw-gte-small-repeat`), and CI builds its own index.
With the dense leg as an exact scan the metrics are deterministic for a given corpus; the
index is checked separately, through the floor on ANN recall@10 at the config's ef_search.

Tolerance: one question's worth, 1 / n_scored (0.0145 at 69 items), never below 0.01. CI
embeds the corpus on another CPU, and float differences can flip a near-tie; losing two
questions fails. Improvements never fail. A different experiment, golden version or search
mode than the baseline's also fails: the numbers would not be comparable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sre_shared.settings import REPO_ROOT

MIN_TOLERANCE = 0.01
TOLERANCE_ITEMS = 1
GATED_METRICS = ("recall@5", "mrr@10")
SCHEMA_VERSION = 2
DEFAULT_BASELINE = REPO_ROOT / "evals" / "baselines" / "retrieval.json"
Search = Literal["exact", "index"]


@dataclass
class GateResult:
    status: Literal["pass", "fail", "skipped"]
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status != "fail"


def load_baseline(path: Path = DEFAULT_BASELINE) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: baseline must be a JSON object")
    return data


def run_search(summary: dict[str, Any]) -> Search:
    return "exact" if summary.get("exact") else "index"


def make_baseline(
    experiment: str,
    golden_version: str,
    summary: dict[str, Any],
    *,
    run_id: int,
    git_sha: str,
    corpus_version_id: int | None,
    config_hash: str,
    ann_floor: tuple[int, float] | None = None,
) -> dict[str, Any]:
    """`ann_floor` = (ef_search, minimum ANN recall@10) to require of every gated run."""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "golden_version": golden_version,
        "search": run_search(summary),
        "metrics": {m: round(float(summary[m]), 6) for m in GATED_METRICS},
        "source": {
            "run_id": run_id,
            "git_sha": git_sha,
            "corpus_version_id": corpus_version_id,
            "config_hash": config_hash,
        },
    }
    if ann_floor is not None:
        out["ann_recall_floor"] = {"ef_search": ann_floor[0], "min": ann_floor[1]}
    return out


def write_baseline(baseline: dict[str, Any], path: Path = DEFAULT_BASELINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(baseline, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")  # same bytes on every OS


def gate_tolerance(summary: dict[str, Any]) -> float:
    n = int(summary.get("n_scored") or 0)
    return max(MIN_TOLERANCE, TOLERANCE_ITEMS / n) if n else MIN_TOLERANCE


def check_gate(
    experiment: str,
    golden_version: str,
    summary: dict[str, Any],
    baseline: dict[str, Any],
    tolerance: float | None = None,
) -> GateResult:
    if not baseline:
        return GateResult("skipped", ["baseline is empty ({}): gate skipped"])
    tol = round(gate_tolerance(summary) if tolerance is None else tolerance, 9)  # as drops are
    problems: list[str] = []
    if baseline.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"baseline schema_version must be {SCHEMA_VERSION}")
    if baseline.get("experiment") != experiment:
        problems.append(
            f"baseline is for experiment {baseline.get('experiment')!r}, run is {experiment!r}"
        )
    if baseline.get("golden_version") != golden_version:
        problems.append(
            f"golden version changed: baseline {baseline.get('golden_version')!r}, "
            f"run {golden_version!r}; rewrite the baseline after review"
        )
    if baseline.get("search") != run_search(summary):
        problems.append(
            f"search mode differs: baseline {baseline.get('search')!r}, run "
            f"{run_search(summary)!r} (run the gate with --exact for an exact baseline)"
        )
    base_metrics = baseline.get("metrics", {})
    messages: list[str] = []
    for metric in GATED_METRICS:
        if metric not in base_metrics or metric not in summary:
            problems.append(f"{metric} missing from baseline or run")
            continue
        base, got = float(base_metrics[metric]), float(summary[metric])
        drop = round(base - got, 9)
        verdict = "FAIL" if drop > tol else "ok"
        messages.append(
            f"{metric}: baseline {base:.4f}, run {got:.4f}, drop {drop:+.4f} "
            f"(tolerance {tol:.4f}) {verdict}"
        )
        if drop > tol:
            problems.append(f"{metric} dropped by {drop:.4f} (> {tol:.4f})")
    floor = baseline.get("ann_recall_floor")
    if floor:
        ef, minimum = int(floor["ef_search"]), float(floor["min"])
        got_ann = (summary.get("ann_recall") or {}).get(str(ef))
        if got_ann is None:
            problems.append(f"ANN recall at ef_search {ef} not measured (run with --ann-recall)")
        else:
            verdict = "FAIL" if got_ann < minimum else "ok"
            messages.append(
                f"ANN recall@10 at ef_search {ef}: {got_ann:.4f} (min {minimum}) {verdict}"
            )
            if got_ann < minimum:
                problems.append(f"ANN recall {got_ann:.4f} below the floor {minimum}")
    if problems:
        return GateResult("fail", [*messages, *problems])
    return GateResult("pass", messages)
