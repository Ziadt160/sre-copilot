"""CI gate: compare a run's headline metrics with the committed baseline.

Baseline file `evals/baselines/retrieval.json`:

    {
      "schema_version": 1,
      "experiment": "hybrid-gte-small",
      "golden_version": "v1-<hash>",       # golden set the numbers were measured on
      "metrics": {"recall@5": 0.0, "mrr@10": 0.0},
      "source": {"run_id": 1, "git_sha": "...", "corpus_version_id": 1, "config_hash": "..."}
    }

`{}` means "no baseline yet": the gate skips. Retrieval is deterministic, so the tolerance is
0.01 absolute: the gate fails when `baseline - run > 0.01` for any gated metric (improvements
never fail). A different experiment or golden version than the baseline's also fails: the
numbers would not be comparable, and the fix is to rewrite the baseline after review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sre_shared.settings import REPO_ROOT

TOLERANCE = 0.01
GATED_METRICS = ("recall@5", "mrr@10")
SCHEMA_VERSION = 1
DEFAULT_BASELINE = REPO_ROOT / "evals" / "baselines" / "retrieval.json"


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


def make_baseline(
    experiment: str,
    golden_version: str,
    summary: dict[str, Any],
    *,
    run_id: int,
    git_sha: str,
    corpus_version_id: int | None,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "golden_version": golden_version,
        "metrics": {m: round(float(summary[m]), 6) for m in GATED_METRICS},
        "source": {
            "run_id": run_id,
            "git_sha": git_sha,
            "corpus_version_id": corpus_version_id,
            "config_hash": config_hash,
        },
    }


def write_baseline(baseline: dict[str, Any], path: Path = DEFAULT_BASELINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_gate(
    experiment: str,
    golden_version: str,
    summary: dict[str, Any],
    baseline: dict[str, Any],
    tolerance: float = TOLERANCE,
) -> GateResult:
    if not baseline:
        return GateResult("skipped", ["baseline is empty ({}): gate skipped"])
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
    base_metrics = baseline.get("metrics", {})
    messages: list[str] = []
    for metric in GATED_METRICS:
        if metric not in base_metrics or metric not in summary:
            problems.append(f"{metric} missing from baseline or run")
            continue
        base, got = float(base_metrics[metric]), float(summary[metric])
        drop = round(base - got, 9)
        verdict = "FAIL" if drop > tolerance else "ok"
        messages.append(
            f"{metric}: baseline {base:.4f}, run {got:.4f}, drop {drop:+.4f} "
            f"(tolerance {tolerance}) {verdict}"
        )
        if drop > tolerance:
            problems.append(f"{metric} dropped by {drop:.4f} (> {tolerance})")
    if problems:
        return GateResult("fail", [*messages, *problems])
    return GateResult("pass", messages)
