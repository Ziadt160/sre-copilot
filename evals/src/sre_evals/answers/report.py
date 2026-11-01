"""Answer sections of `docs/results.md`: answers per experiment, noise spread, paired
differences with bootstrap CIs, the two-halves check, and the judge calibration table.

`fetch_answer_data` reads the latest batch (= one `answer run` invocation, all its repeats) of
each experiment and the latest calibration; `render_answer_sections` is pure and deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection

from sre_evals.answers.calibrate import KAPPA_TARGET
from sre_evals.answers.stats import noise_stats, paired_bootstrap_ci
from sre_evals.report.details import fold_of

BASELINE = "answer-k5"
PAIRED_METRICS = ("faithfulness", "answer_relevancy", "citation_hit")
SUMMARY_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "citation_hit_rate",
    "cited_rate",
    "abstain_rate",
)


@dataclass
class AnswerExperiment:
    experiment: str
    runs: list[dict[str, Any]]  # summaries of the batch's repeats, repeat order
    per_item: dict[str, dict[str, float]] = field(default_factory=dict)  # metric -> item -> mean
    folds: dict[str, str] = field(default_factory=dict)  # item -> 'A' | 'B' (all run items)


@dataclass
class AnswerData:
    experiments: list[AnswerExperiment] = field(default_factory=list)
    calibration: dict[str, Any] | None = None


async def fetch_answer_data(conn: AsyncConnection[Any]) -> AnswerData:
    cur = await conn.execute(
        "select id, experiment, summary from public.eval_runs where kind = 'answer'"
        " and status = 'done' order by id"
    )
    rows = [dict(r) for r in await cur.fetchall()]
    latest_batch: dict[str, str] = {}
    for r in rows:  # ascending id: the last batch seen per experiment is the latest
        latest_batch[r["experiment"]] = str(r["summary"].get("batch", r["id"]))
    data = AnswerData()
    for name in sorted(latest_batch):
        runs = [
            r
            for r in rows
            if r["experiment"] == name
            and str(r["summary"].get("batch", r["id"])) == latest_batch[name]
        ]
        runs.sort(key=lambda r: int(r["summary"].get("repeat", 1)))
        exp = AnswerExperiment(name, [r["summary"] for r in runs])
        # The halves come from every item the run covered, not from the ones that got a score.
        exp.folds = fold_of(sorted({i for r in runs for i in r["summary"].get("items", {})}))
        cur = await conn.execute(
            "select item_id, metric, avg(value) as value from public.eval_scores"
            " where run_id = any(%s) and metric = any(%s) group by item_id, metric",
            [[r["id"] for r in runs], list(PAIRED_METRICS)],
        )
        for s in await cur.fetchall():
            exp.per_item.setdefault(s["metric"], {})[s["item_id"]] = float(s["value"])
        data.experiments.append(exp)
    cur = await conn.execute(
        "select summary from public.eval_runs where kind = 'calibration' and status = 'done'"
        " order by id desc limit 1"
    )
    row = await cur.fetchone()
    data.calibration = dict(row["summary"]) if row else None
    return data


def _f(x: Any, digits: int = 3) -> str:
    return "-" if x is None else f"{float(x):.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def _mean_over_runs(runs: Sequence[Mapping[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in runs if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _signed(x: float) -> str:
    return f"{x:+.3f}"


def _fold(exp: AnswerExperiment, item_id: str) -> str:
    """Fixed half of an item. Stored runs know all their items (`exp.folds`); without that,
    fall back to the parity of the id's number, which also never depends on what was scored."""
    if item_id in exp.folds:
        return exp.folds[item_id]
    digits = "".join(ch for ch in item_id.rsplit("-", 1)[-1] if ch.isdigit())
    return "A" if digits and int(digits) % 2 == 1 else "B"


def _halves(exp: AnswerExperiment, metric: str) -> str:
    values = exp.per_item.get(metric, {})
    if not values:
        return "-"
    parts = []
    for half in "AB":
        v = [x for i, x in values.items() if _fold(exp, i) == half]
        parts.append(_f(sum(v) / len(v)) if v else "-")
    return " / ".join(parts)


JUDGE_KEYS = ("judge", "judge_prompt_version", "metric_version", "golden_version")
NOISE_ITEM_METRICS = ("faithfulness", "answer_relevancy", "citation_hit", "cited", "abstained")


def _not_comparable(a: AnswerExperiment, b: AnswerExperiment) -> str | None:
    """Why two experiments' scores must not be paired (None: they can be)."""
    diff = [k for k in JUDGE_KEYS if a.runs[0].get(k) != b.runs[0].get(k)]
    return ", ".join(diff) if diff else None


def _noise_rows(e: AnswerExperiment) -> list[list[str]]:
    """Per metric, the per-repeat mean over the items scored in EVERY repeat, then the spread of
    those means. Without per-item data (old runs) the stored summary means are used."""
    rows: list[list[str]] = []
    items: list[dict[str, dict[str, float | None]]] = [r.get("items") or {} for r in e.runs]
    if all(items):
        for m in NOISE_ITEM_METRICS:
            common = set.intersection(
                *({i for i, v in it.items() if v.get(m) is not None} for it in items)
            )
            if not common:
                continue
            means = [{m: sum(float(it[i][m] or 0) for i in common) / len(common)} for it in items]
            s = noise_stats(means)[m]
            rows.append([
                f"`{e.experiment}`", m, str(s.n), str(len(common)), _f(s.mean), _f(s.min),
                _f(s.max), _f(s.stdev),
            ])  # fmt: skip
        return rows
    stats = noise_stats(
        [{m: float(r[m]) for m in SUMMARY_METRICS if r.get(m) is not None} for r in e.runs]
    )
    return [
        [f"`{e.experiment}`", m, str(s.n), "-", _f(s.mean), _f(s.min), _f(s.max), _f(s.stdev)]
        for m, s in stats.items()
    ]


def _answered(runs: Sequence[Mapping[str, Any]]) -> str:
    vals = [int(r.get("n_answered", 0)) for r in runs]
    lo, hi = min(vals), max(vals)
    n = runs[0].get("n_items", 0)
    return f"{hi}/{n}" if lo == hi else f"{lo}-{hi}/{n}"


def render_answer_sections(data: AnswerData) -> list[str]:
    """Markdown lines to append to docs/results.md (empty when there is nothing to show)."""
    if not data.experiments and data.calibration is None:
        return []
    lines: list[str] = ["", "## Answers", ""]
    exps = data.experiments
    if exps:
        lines += [
            "Latest invocation of each answer experiment (repeats averaged). Faithfulness and "
            "answer relevancy are DeepEval scores from the judge shown; `citation hit` is the "
            "share of items with gold passages whose answer cites a chunk matching one "
            "(an abstention is a miss); `cited` is the share of answered items with a valid "
            "citation. Faithfulness A / B are two fixed halves of the golden set.",
            "",
        ]
        rows = []
        for e in exps:
            r0 = e.runs[0]
            rows.append([
                f"`{e.experiment}`", str(r0.get("model", "-")), str(r0.get("judge", "-")),
                str(len(e.runs)), _answered(e.runs),
                _f(_mean_over_runs(e.runs, "faithfulness")),
                _f(_mean_over_runs(e.runs, "answer_relevancy")),
                _f(_mean_over_runs(e.runs, "citation_hit_rate")),
                _f(_mean_over_runs(e.runs, "cited_rate")),
                _f(_mean_over_runs(e.runs, "abstain_rate")),
                _f(_mean_over_runs(e.runs, "invalid_citations"), 1),
                _halves(e, "faithfulness"),
                _f(_mean_over_runs([r["latency_ms"] for r in e.runs], "p50"), 0),
                _f(_mean_over_runs(e.runs, "prompt_tokens"), 0),
            ])  # fmt: skip
        lines += _table(
            [
                "Experiment", "Model", "Judge", "Repeats", "Answered", "Faithfulness",
                "Relevancy", "Citation hit", "Cited", "Abstain", "Invalid cites",
                "Faithfulness A / B", "p50 ms", "Prompt tokens",
            ],
            rows,
        )  # fmt: skip

        base = next((e for e in exps if e.experiment == BASELINE), exps[0])
        others = [e for e in exps if e is not base]
        if others:
            lines += [
                "",
                f"### Paired differences vs `{base.experiment}`",
                "",
                "Mean per-item difference (experiment minus baseline) over the items both "
                "scored, with a 95% paired bootstrap interval; `*` marks an interval that "
                "excludes 0. Per-item values are means over the repeats.",
                "",
            ]
            rows = []
            for e in others:
                cells = [f"`{e.experiment}`"]
                why = _not_comparable(e, base)
                for m in PAIRED_METRICS:
                    if why:
                        cells.append(f"not comparable ({why} differs)")
                        continue
                    a, b = e.per_item.get(m, {}), base.per_item.get(m, {})
                    common = sorted(a.keys() & b.keys())
                    ci = paired_bootstrap_ci([a[i] - b[i] for i in common])
                    if ci is None:
                        cells.append("-")
                    else:
                        mean, lo, hi = ci
                        star = "*" if lo > 0 or hi < 0 else ""
                        cells.append(
                            f"{_signed(mean)} [{_signed(lo)}, {_signed(hi)}]{star} (n={len(common)})"
                        )
                rows.append(cells)
            lines += _table(["Experiment", *PAIRED_METRICS], rows)
            if base.experiment != BASELINE:
                lines += ["", f"_`{BASELINE}` has no run: the baseline is `{base.experiment}`._"]
            half_rows = []
            for e in others:
                if _not_comparable(e, base):
                    continue
                a, b = e.per_item.get("faithfulness", {}), base.per_item.get("faithfulness", {})
                cells = [f"`{e.experiment}`"]
                for half in "AB":
                    d = [a[i] - b[i] for i in sorted(a.keys() & b.keys()) if _fold(base, i) == half]
                    cells.append(f"{_signed(sum(d) / len(d))} (n={len(d)})" if d else "-")
                half_rows.append(cells)
            if half_rows:
                lines += ["", "Faithfulness difference per fixed half of the golden set:", ""]
                lines += _table(["Experiment", "Half A", "Half B"], half_rows)

        noisy = [e for e in exps if len(e.runs) > 1]
        lines += ["", "### Noise (same commit, repeated runs)", ""]
        if noisy:
            rows = [row for e in noisy for row in _noise_rows(e)]
            lines += _table(
                [
                    "Experiment",
                    "Metric",
                    "Runs",
                    "Items",
                    "Mean",
                    "Min",
                    "Max",
                    "Stdev (sample, n-1)",
                ],
                rows,
            )
            lines += [
                "",
                "Each repeat's mean covers only the items scored in every repeat (`Items`), so"
                " a judge failure or an abstention in one repeat does not show up as spread.",
            ]
            lines += [
                "",
                "A difference between two experiments smaller than this spread is not a result.",
            ]
        else:
            lines.append("_No repeated runs yet (`sre-eval answers run --repeat 3`)._")

    lines += ["", "### Judge calibration (faithfulness vs the human labels)", ""]
    cal = data.calibration
    if cal is None:
        lines.append("_Not calibrated yet (`sre-eval answers calibrate`)._")
    else:
        dist = cal["human_distribution"]
        t = cal["thresholds"]
        lines += [
            f"{cal['n_labels']} labels by {', '.join(cal['labellers']) or '-'} "
            f"(supported {dist['supported']}, partly {dist['partly']}, "
            f"not supported {dist['not_supported']}). Judge score to label: >= "
            f"{t['supported_min']:g} supported, >= {t['partly_min']:g} partly, else not "
            f"supported. Judge prompt `{cal['judge_prompt_version']}`, "
            f"metrics `{cal['metric_version']}`. Target kappa {KAPPA_TARGET:g}."
            + (
                f" {cal['n_unjoined']} labels could not be matched to a stored answer and are"
                " left out."
                if cal.get("n_unjoined")
                else ""
            ),
            "",
        ]
        rows = []
        for name, j in cal["judges"].items():
            ref = j.get("vs_reference") or {}
            rows.append([
                f"`{name}`" + (" (reference)" if name == cal.get("reference") else ""),
                f"{j['n']}/{j['n'] + j['n_failed']}", _f(j["accuracy"]), _f(j["kappa"]),
                _f(j["accuracy_binary"]), _f(j["kappa_binary"]),
                _f(ref.get("kappa")) if ref else "-",
                "below target" if name in cal["below_target"] else "ok",
            ])  # fmt: skip
        lines += _table(
            [
                "Judge", "Scored", "Accuracy", "Kappa", "Accuracy (binary)", "Kappa (binary)",
                "Kappa vs reference", "Verdict",
            ],
            rows,
        )  # fmt: skip
    return lines
