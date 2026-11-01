from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sre_evals.report import END, START, build_reports, render_block, replace_block
from sre_evals.runner import run_experiment
from sre_evals.runner.config import ExperimentConfig
from sre_evals.runner.testing import SRC, VER, FakeRetriever, W, chunk
from sre_shared.models import GoldenItem, GoldPassage, RetrievalRequest
from sre_shared.testing.seed import create_corpus_version

BEFORE = "# Title\n\nintro text  \n\n## Results\n\n"
AFTER = "\n## Architecture\n\ntrailing text\n"
README = f"{BEFORE}{START}\nold table\n{END}{AFTER}"


def test_replace_block_touches_only_inside_markers() -> None:
    out = replace_block(README, "| new |")
    assert out == f"{BEFORE}{START}\n| new |\n{END}{AFTER}"
    assert out.startswith(BEFORE) and out.endswith(AFTER)


def test_replace_block_is_idempotent() -> None:
    once = replace_block(README, "| a |\n| b |")
    assert replace_block(once, "| a |\n| b |") == once


def test_replace_block_preserves_crlf() -> None:
    crlf = README.replace("\n", "\r\n")
    out = replace_block(crlf, "l1\nl2")
    assert out == f"{BEFORE}{START}\nl1\nl2\n{END}{AFTER}".replace("\n", "\r\n")


def test_replace_block_needs_exactly_one_marker_pair() -> None:
    with pytest.raises(ValueError):
        replace_block("no markers here", "x")
    with pytest.raises(ValueError):
        replace_block(README + README, "x")
    with pytest.raises(ValueError):
        replace_block(f"{END}\n{START}", "x")


def test_render_block_empty_and_sorted() -> None:
    assert "No eval runs yet" in render_block([])

    def r(name: str, model: str, legs: list[str]) -> dict[str, Any]:
        return {
            "experiment": name,
            "config": {"embedding_model": model, "legs": legs},
            "golden_version": "v1-0123456789abcdef",
            "stamps": {"git_sha": "abc123"},
            "summary": {
                "recall@5": 0.8, "mrr@10": 0.5, "ndcg@10": 0.6,
                "latency_ms": {"p95": 41.6}, "n_scored": 70,
            },
        }  # fmt: skip

    block = render_block(
        [r("kw-english", "gte-small", ["keyword"]), r("dense-x", "bge-m3", ["dense"])]
    )
    lines = block.splitlines()
    assert lines[0].startswith("| Experiment | Model | recall@5 | MRR@10 | nDCG@10 | p95 ms |")
    assert (
        lines[2]
        == "| `dense-x` | bge-m3 | 0.800 | 0.500 | 0.600 | 42 | 70 | v1-01234567 | `abc123` |"
    )
    assert lines[3].startswith("| `kw-english` | - |")  # keyword-only runs show no model


@pytest.mark.integration
async def test_build_reports_from_runs_is_idempotent(db: Any, tmp_path: Path) -> None:
    await create_corpus_version(db)

    def script(req: RetrievalRequest, exact: bool) -> tuple[list[Any], float]:
        return [chunk(1, W(1, 10))], 12.0

    item = GoldenItem(
        id="fac-0001", stratum="factual", question="q", status="accepted",
        gold_passages=[GoldPassage(source_path=SRC, k8s_version=VER, quote=W(1, 10))],
    )  # fmt: skip
    for name in ("hybrid-gte-small", "kw-english"):
        legs: Any = ["keyword"] if name.startswith("kw") else ["dense", "keyword"]
        cfg = ExperimentConfig(name=name, legs=legs)
        await run_experiment(
            cfg, [item], FakeRetriever(script), db, loop="local", golden_version="v1-abcdef0123"
        )
    # A later run of the same experiment replaces the earlier one in the report; failed runs do not count.
    await db.execute(
        "update public.eval_runs set status = 'failed' where experiment = 'kw-english'"
    )

    readme = tmp_path / "README.md"
    readme.write_bytes(README.encode())
    details = tmp_path / "docs" / "results.md"
    runs = await build_reports(db, readme, details)
    assert [r["experiment"] for r in runs] == ["hybrid-gte-small"]

    first_readme, first_details = readme.read_bytes(), details.read_bytes()
    text = first_readme.decode()
    assert text.startswith(BEFORE) and text.endswith(AFTER)
    assert (
        "| `hybrid-gte-small` | gte-small | 1.000 | 1.000 | 1.000 | 12 | 1 | v1-abcdef01 |" in text
    )
    assert "kw-english" not in text

    await build_reports(db, readme, details)
    assert readme.read_bytes() == first_readme
    assert details.read_bytes() == first_details
    md = first_details.decode()
    assert "## Per stratum" in md and "## Weight sweep" in md and "## ANN recall" in md
    assert "## Embedding models" in md and "factual" in md
    assert "Git SHA:" in md and "Golden version: `v1-abcdef0123`" in md
