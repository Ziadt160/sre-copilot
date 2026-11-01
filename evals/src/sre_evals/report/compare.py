"""Paired comparison of experiments against a baseline: the evidence behind every adoption.

The rule (design doc, Phase 2): a technique stays only if it beats the baseline, on both fixed
halves of the golden set (`details.fold_of`), with a paired bootstrap interval reported. Ties
lose. Per metric this computes, over the items both runs scored:

  * the mean per-item difference with a 95% paired bootstrap interval (10,000 resamples,
    seeded per comparison, so a result does not depend on what else was compared)
  * the mean difference on each half (A, B)

Verdict on the primary metric (recall@5): `win` when the interval is above zero and both
halves improve, `loss` when the interval is below zero, otherwise `tie`. Decisions also look
at the other metrics; the verdict is the minimum bar, not the whole argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection

from sre_evals.answers.stats import paired_bootstrap_ci
from sre_evals.report.details import fold_of

METRICS: tuple[tuple[str, int], ...] = (("recall", 5), ("mrr", 10), ("ndcg", 10))
PRIMARY = ("recall", 5)
RESAMPLES = 10_000
SEED = 7


@dataclass(frozen=True)
class Delta:
    metric: str
    n: int
    mean: float
    low: float
    high: float
    fold_a: float
    fold_b: float

    @property
    def both_halves(self) -> bool:
        return self.fold_a > 0 and self.fold_b > 0


def paired_delta(metric: str, exp: Mapping[str, float], base: Mapping[str, float]) -> Delta | None:
    ids = sorted(set(exp) & set(base))
    if not ids:
        return None
    diffs = [exp[i] - base[i] for i in ids]
    ci = paired_bootstrap_ci(diffs, resamples=RESAMPLES, seed=SEED)
    assert ci is not None
    folds = fold_of(ids)
    half = {
        f: sum(d for i, d in zip(ids, diffs, strict=True) if folds[i] == f)
        / max(1, sum(1 for i in ids if folds[i] == f))
        for f in "AB"
    }
    return Delta(metric, len(ids), ci[0], ci[1], ci[2], half["A"], half["B"])


def verdict(primary: Delta | None) -> str:
    if primary is None:
        return "no data"
    if primary.low > 0 and primary.both_halves:
        return "win"
    if primary.high < 0:
        return "loss"
    return "tie"


async def latest_item_scores(
    conn: AsyncConnection[Any], experiment: str, metric: str, k: int, kind: str = "retrieval"
) -> dict[str, float]:
    cur = await conn.execute(
        "select s.item_id, s.value from public.eval_scores s where s.run_id = ("
        " select max(id) from public.eval_runs where experiment = %s and kind = %s"
        " and status = 'done') and s.metric = %s and s.k = %s",
        [experiment, kind, metric, k],
    )
    return {r["item_id"]: float(r["value"]) for r in await cur.fetchall()}


async def compare(
    conn: AsyncConnection[Any], base: str, experiments: Sequence[str], kind: str = "retrieval"
) -> dict[str, list[Delta | None]]:
    out: dict[str, list[Delta | None]] = {}
    for exp in experiments:
        row: list[Delta | None] = []
        for metric, k in METRICS:
            a = await latest_item_scores(conn, exp, metric, k, kind)
            b = await latest_item_scores(conn, base, metric, k, kind)
            row.append(paired_delta(f"{metric}@{k}", a, b))
        out[exp] = row
    return out


def render(base: str, results: Mapping[str, Sequence[Delta | None]]) -> list[str]:
    head = " | ".join(f"{m}@{k}" for m, k in METRICS)
    lines = [
        f"vs `{base}` (95% paired bootstrap; halves A / B)",
        "",
        f"| experiment | n | {head} | verdict |",
        "|---|---|" + "---|" * len(METRICS) + "---|",
    ]
    for exp, deltas in results.items():
        cells = [
            "-"
            if d is None
            else f"{d.mean:+.3f} [{d.low:+.3f}, {d.high:+.3f}] ({d.fold_a:+.3f} / {d.fold_b:+.3f})"
            for d in deltas
        ]
        n = next((d.n for d in deltas if d is not None), 0)
        lines.append(f"| `{exp}` | {n} | " + " | ".join(cells) + f" | {verdict(deltas[0])} |")
    return lines
