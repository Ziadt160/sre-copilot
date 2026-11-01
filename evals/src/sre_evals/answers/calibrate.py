"""Judge calibration: re-judge human-labelled answers with each candidate judge and measure
agreement with the human labels (and with a reference judge).

A judge's faithfulness score in [0, 1] becomes a label with `stats.score_to_label` (>= 0.9
supported, >= 0.5 partly, else not supported); the binary view is supported vs not.
Per judge: accuracy and Cohen's kappa against the human labels (3-class and binary), and, for
a reference judge, kappa between the two judges on the items both scored. Items a judge fails
to score (bad JSON) are left out of its agreement numbers and counted in `n_failed`.
Below `KAPPA_TARGET` (0.7) the design says to revise the judge prompt and recalibrate.

Hosted judges are checked against their quota before the first call: the planned tokens for
the whole labelled set must fit what is left of the model's daily budget, else QuotaExceeded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from sre_evals.answers.judge import (
    JUDGE_PROMPT_VERSION,
    Judge,
    check_judge_budget,
    estimate_judge_tokens,
    metric_version,
)
from sre_evals.answers.label import Label, LabelItem
from sre_evals.answers.stats import (
    NOT_SUPPORTED,
    PARTLY,
    PARTLY_MIN,
    SUPPORTED,
    SUPPORTED_MIN,
    accuracy,
    cohens_kappa,
    score_to_label,
    to_binary,
)
from sre_shared.llm import ChatClient
from sre_shared.stamps import VersionStamps, config_hash

KAPPA_TARGET = 0.7


@dataclass
class JudgeHandle:
    """A judge by name; `client` is set for model-backed judges (quota guard, token use)."""

    name: str
    judge: Judge
    client: ChatClient | None = None


@dataclass
class JudgeAgreement:
    name: str
    n: int = 0
    n_failed: int = 0
    accuracy: float | None = None
    kappa: float | None = None
    accuracy_binary: float | None = None
    kappa_binary: float | None = None
    vs_reference: dict[str, float | None] = field(default_factory=dict)
    labels: dict[tuple[int, str], str] = field(default_factory=dict, repr=False)
    human: dict[tuple[int, str], str] = field(default_factory=dict, repr=False)  # joined labels

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_failed": self.n_failed,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "accuracy_binary": self.accuracy_binary,
            "kappa_binary": self.kappa_binary,
            "vs_reference": self.vs_reference,
        }


def plan_tokens(items: Sequence[LabelItem]) -> int:
    return sum(
        estimate_judge_tokens(i.answer, [p.text for p in i.passages], relevancy=False)
        for i in items
    )


def check_budgets(handles: Sequence[JudgeHandle], items: Sequence[LabelItem]) -> None:
    """Raise `QuotaExceeded` before any call if a hosted judge cannot cover the whole set."""
    planned = plan_tokens(items)
    for h in handles:
        if h.client is not None:
            check_judge_budget(h.client, planned)


def agreement(
    name: str, human: dict[tuple[int, str], str], judged: dict[tuple[int, str], str | None]
) -> JudgeAgreement:
    keys = [k for k in human if judged.get(k) is not None]
    h = [human[k] for k in keys]
    j = [str(judged[k]) for k in keys]
    out = JudgeAgreement(name, n=len(keys), n_failed=len(human) - len(keys))
    out.labels = {k: str(judged[k]) for k in keys}
    out.human = dict(human)
    out.accuracy, out.kappa = accuracy(h, j), cohens_kappa(h, j)
    hb, jb = [to_binary(x) for x in h], [to_binary(x) for x in j]
    out.accuracy_binary, out.kappa_binary = accuracy(hb, jb), cohens_kappa(hb, jb)
    return out


async def calibrate(
    labels: Sequence[Label],
    items: Sequence[LabelItem],
    handles: Sequence[JudgeHandle],
    *,
    reference: str | None = None,
    concurrency: int = 1,
) -> list[JudgeAgreement]:
    """Judge every labelled answer with every handle and compare with the human labels."""
    by_key = {i.key: i for i in items}
    labelled = [lab for lab in labels if (lab.run_id, lab.item_id) in by_key]
    human = {(lab.run_id, lab.item_id): lab.label for lab in labelled}
    chosen = [by_key[k] for k in human]
    check_budgets(handles, chosen)
    if reference is not None and reference not in {h.name for h in handles}:
        raise ValueError(f"reference judge {reference!r} is not among the judges")

    results: list[JudgeAgreement] = []
    sem = asyncio.Semaphore(max(1, concurrency))
    for h in handles:

        async def one(
            item: LabelItem, judge: Judge = h.judge
        ) -> tuple[tuple[int, str], str | None]:
            async with sem:
                score = await judge.faithfulness(
                    item.question, item.answer, [p.text for p in item.passages]
                )
            return item.key, None if score is None else score_to_label(score)

        judged = dict(await asyncio.gather(*(one(i) for i in chosen)))
        results.append(agreement(h.name, human, judged))

    if reference is not None:
        ref = next(r for r in results if r.name == reference)
        for r in results:
            if r is ref:
                continue
            keys = [k for k in r.labels if k in ref.labels]
            a, b = [r.labels[k] for k in keys], [ref.labels[k] for k in keys]
            r.vs_reference = {
                "n": float(len(keys)),
                "kappa": cohens_kappa(a, b),
                "kappa_binary": cohens_kappa([to_binary(x) for x in a], [to_binary(x) for x in b]),
            }
    return results


def calibration_summary(
    results: Sequence[JudgeAgreement],
    labels: Sequence[Label],
    labels_file: str,
    reference: str | None,
) -> dict[str, Any]:
    """`n_labels` and the distribution cover the labels that joined a stored answer;
    `n_unjoined` counts the ones that did not (their run or item is gone)."""
    joined = results[0].human if results else {(lab.run_id, lab.item_id): "" for lab in labels}
    used = [lab for lab in labels if (lab.run_id, lab.item_id) in joined]
    dist: dict[str, int] = {SUPPORTED: 0, PARTLY: 0, NOT_SUPPORTED: 0}
    for lab in used:
        dist[lab.label] += 1
    return {
        "labels_file": labels_file,
        "n_labels": len(used),
        "n_unjoined": len(labels) - len(used),
        "labellers": sorted({lab.labeller for lab in used}),
        "human_distribution": dist,
        "reference": reference,
        "judges": {r.name: r.as_dict() for r in results},
        "below_target": [r.name for r in results if r.kappa is None or r.kappa < KAPPA_TARGET],
        "kappa_target": KAPPA_TARGET,
        "thresholds": {"supported_min": SUPPORTED_MIN, "partly_min": PARTLY_MIN},
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "metric_version": metric_version(),
    }


async def store_calibration(
    conn: AsyncConnection[Any], name: str, summary: dict[str, Any], golden_version: str = "-"
) -> int:
    """Persist a calibration as an `eval_runs` row of kind `calibration` (read by the report)."""
    config = {"labels_file": summary["labels_file"], "judges": list(summary["judges"])}
    stamps = VersionStamps(
        config_hash=config_hash(config),
        golden_version=golden_version,
        prompt_versions={"judge": JUDGE_PROMPT_VERSION},
    )
    row = await (
        await conn.execute(
            "insert into public.eval_runs (kind, loop, experiment, config, stamps, golden_version,"
            " status, summary, finished_at) values ('calibration', 'local', %s, %s, %s, %s,"
            " 'done', %s, now()) returning id",
            [
                name,
                Jsonb(config),
                Jsonb(stamps.model_dump(mode="json")),
                golden_version,
                Jsonb(summary),
            ],
        )
    ).fetchone()
    assert row is not None
    return int(row["id"] if isinstance(row, dict) else row[0])
