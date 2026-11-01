"""Rewrite the README results block between the `results:start` / `results:end` markers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

START = "<!-- results:start -->"
END = "<!-- results:end -->"

EMPTY_NOTE = (
    "_No eval runs yet. Phase 1 fills this table with retrieval baselines: dense-only vs "
    "keyword-only vs hybrid, per embedding model._"
)

HEADER = (
    "| Experiment | Model | recall@5 | MRR@10 | nDCG@10 | p95 ms | Items | Golden | Git SHA |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def _short_golden(golden_version: str) -> str:
    name, _, digest = golden_version.partition("-")
    return f"{name}-{digest[:8]}" if digest else name


def run_model(config: dict[str, Any]) -> str:
    """Embedding model of a run; keyword-only experiments do not use one."""
    legs = config.get("legs", ["dense", "keyword"])
    return str(config.get("embedding_model", "-")) if "dense" in legs else "-"


def table_row(run: dict[str, Any]) -> str:
    s = run["summary"]
    cells = [
        f"`{run['experiment']}`",
        run_model(run["config"]),
        f"{s['recall@5']:.3f}",
        f"{s['mrr@10']:.3f}",
        f"{s['ndcg@10']:.3f}",
        f"{s['latency_ms']['p95']:.0f}",
        str(s["n_scored"]),
        _short_golden(run["golden_version"]),
        f"`{run['stamps'].get('git_sha', 'unknown')}`",
    ]
    return "| " + " | ".join(cells) + " |"


def render_block(runs: Sequence[dict[str, Any]]) -> str:
    """Block body (no markers), with `\\n` newlines. Deterministic: sorted by experiment name."""
    if not runs:
        return EMPTY_NOTE
    rows = [table_row(r) for r in sorted(runs, key=lambda r: r["experiment"])]
    return "\n".join([HEADER, *rows])


def replace_block(text: str, block: str) -> str:
    """Return `text` with only the content between the markers replaced by `block`.

    Newline style follows the file (CRLF stays CRLF). Everything outside the markers is
    preserved byte for byte.
    """
    s = text.find(START)
    e = text.find(END)
    if s < 0 or e < 0 or e < s or text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"README needs exactly one {START} ... {END} pair")
    nl = "\r\n" if "\r\n" in text else "\n"
    body = block.replace("\r\n", "\n").replace("\n", nl)
    return text[: s + len(START)] + nl + body + nl + text[e:]
