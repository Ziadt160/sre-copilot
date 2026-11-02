"""Markdown sections for docs/results.md: agent evals, overall and by stratum, with 95% CIs.

A CI is the percentile bootstrap of the item mean (`sre_evals.answers.stats.paired_bootstrap_ci`,
fixed seed, 2000 resamples). With few items it is wide; that is the point of printing it.
`duplicate_side_effects` is a count, not a rate: the gate is its sum being 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sre_evals.answers.stats import paired_bootstrap_ci

COLUMNS = (
    ("task_success", "task success"),
    ("tool_correct", "tool"),
    ("args_correct", "args"),
    ("refusal_correct", "refusal"),
    ("injection_resisted", "injection resisted"),
    ("rewrite_recall@5", "rewrite recall@5"),
)


def _ci(values: Sequence[float]) -> str:
    r = paired_bootstrap_ci(values)
    if r is None:
        return "-"
    mean, lo, hi = r
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def _values(summary: Mapping[str, Any], ids: Sequence[str], key: str) -> list[float]:
    items = summary["items"]
    return [items[i][key] for i in ids if key in items[i]]


def _table(summary: Mapping[str, Any]) -> list[str]:
    strata: dict[str, list[str]] = {}
    for iid, s in summary["item_strata"].items():
        strata.setdefault(s, []).append(iid)
    groups = {**{s: ids for s, ids in sorted(strata.items())}, "all": list(summary["items"])}
    used = [
        (k, label)
        for k, label in COLUMNS
        if any(_values(summary, ids, k) for ids in groups.values())
    ]
    head = ["stratum", "n", *[label for _, label in used], "steps", "duplicates"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, ids in groups.items():
        cells = [name, str(len(ids))]
        for k, _ in used:
            vals = _values(summary, ids, k)
            cells.append(_ci(vals) if vals else "-")
        steps = _values(summary, ids, "steps")
        cells.append(f"{sum(steps) / len(steps):.1f}" if steps else "-")
        cells.append(str(int(sum(_values(summary, ids, "duplicate_side_effects")))))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def render_agent_section(summary: Mapping[str, Any], *, title: str = "Agent") -> str:
    faults = [k for k, v in (summary.get("faults") or {}).items() if v]
    dupes = int(sum(_values(summary, list(summary["items"]), "duplicate_side_effects")))
    lines = [
        f"### {title} ({summary['experiment']}, {summary['subset']} set, rig {summary.get('rig')})",
        "",
        f"Injected faults: {', '.join(faults) if faults else 'none'}. Items: {summary['n_items']}."
        f" Agent crashes (error events): {summary['n_error_events']}.",
        "",
        *_table(summary),
        "",
        f"Duplicate side effects: **{dupes}** (gate: 0)."
        + (" Gate met." if dupes == 0 else " **Gate failed.**"),
        "",
        "Cells are the item mean with a 95% bootstrap interval; `-` means the metric does not"
        " apply to that stratum. `steps` is the mean of proposed tool calls plus final answers.",
    ]
    return "\n".join(lines) + "\n"


def render_agent_report(summaries: Sequence[Mapping[str, Any]]) -> str:
    """The `## Agent` block of docs/results.md for the given stored summaries."""
    return "## Agent\n\n" + "\n".join(render_agent_section(s) for s in summaries)
