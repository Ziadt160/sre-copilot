"""Non-interactive review: a decisions file goes through the same ReviewSession rules."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from sre_evals.golden import cli
from sre_evals.golden.review import ReviewDecision, ReviewSession, apply_decisions
from sre_shared.golden_io import load_golden, write_golden
from sre_shared.models import GoldenItem, GoldPassage

FIXED = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
QUOTE = "Use --max-surge to bound extra Pods during a rolling update. Extra words follow here."


def _item(
    id_: str, question: str, stratum: str = "factual", tags: list[str] | None = None
) -> GoldenItem:
    return GoldenItem(
        id=id_,
        stratum=stratum,  # type: ignore[arg-type]
        question=question,
        expected_answer=f"answer to {id_}",
        gold_passages=[
            GoldPassage(source_path="content/en/docs/d.md", k8s_version="1.36", quote=QUOTE)
        ],
        status="candidate",
        tags=tags or [],
    )


def _session(tmp_path: Path) -> ReviewSession:
    src = tmp_path / "candidates.jsonl"
    if not src.exists():
        write_golden(
            src,
            [
                _item("fac-0001", "What bounds extra pods?"),
                _item("fac-0002", "Vague question?"),
                _item(
                    "exs-0001", "What does --max-surge do?", "exact_string", ["exact:--max-surge"]
                ),
            ],
        )
    return ReviewSession.open(src, tmp_path / "items.jsonl", "claude-opus-5-5", now=lambda: FIXED)


def test_accept_reject_rewrite_and_trim(tmp_path: Path) -> None:
    s = _session(tmp_path)
    report = apply_decisions(
        s,
        [
            ReviewDecision(id="fac-0002", action="reject", reason="too vague"),
            ReviewDecision(
                id="fac-0001",
                action="accept",
                question="How do I cap extra pods during a rollout?",
                quote="Use --max-surge to bound extra Pods during a rolling update.",
                reason="rewritten in an operator's words",
            ),
            ReviewDecision(id="exs-0001", action="accept", add_tags=["check:accept"]),
        ],
    )
    assert (report.accepted, report.rejected, report.errors) == (2, 1, [])
    items = {i.id: i for i in load_golden(tmp_path / "items.jsonl")}
    assert items["fac-0002"].status == "rejected" and items["fac-0002"].notes == "too vague"
    fac = items["fac-0001"]
    assert fac.status == "accepted" and fac.origin == "rewritten"
    assert fac.question == "How do I cap extra pods during a rollout?"
    assert (
        fac.gold_passages[0].quote == "Use --max-surge to bound extra Pods during a rolling update."
    )
    assert fac.reviewer == "claude-opus-5-5" and fac.reviewed_at == FIXED
    assert items["exs-0001"].origin == "synthetic"
    assert items["exs-0001"].tags == ["exact:--max-surge", "check:accept"]


def test_refused_edits_leave_the_item_pending(tmp_path: Path) -> None:
    s = _session(tmp_path)
    report = apply_decisions(
        s,
        [
            ReviewDecision(id="exs-0001", action="accept", question="What caps surge pods?"),
            ReviewDecision(id="fac-0001", action="accept", quote="text that is not in the quote"),
            ReviewDecision(id="nope-0001", action="accept"),
        ],
    )
    assert report.accepted == 0 and len(report.errors) == 3
    assert "exact string" in report.errors[0] and "sub-span" in report.errors[1]
    assert not (tmp_path / "items.jsonl").exists() or load_golden(tmp_path / "items.jsonl") == []
    assert set(s.queue) == {"fac-0001", "fac-0002", "exs-0001"}


def test_cli_apply_resumes_and_reports_errors(tmp_path: Path) -> None:
    _session(tmp_path)  # writes the candidates file
    dec = tmp_path / "decisions.jsonl"
    dec.write_text(
        ReviewDecision(id="fac-0001", action="accept").model_dump_json()
        + "\n"
        + ReviewDecision(id="fac-0002", action="reject").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    args = [
        "apply",
        "--in",
        str(tmp_path / "candidates.jsonl"),
        "--decisions",
        str(dec),
        "--out",
        str(tmp_path / "items.jsonl"),
        "--reviewer",
        "claude-opus-5-5",
    ]
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert "accepted 1, rejected 1" in r.output
    r = CliRunner().invoke(cli.app, args)  # already decided: reported, non-zero exit
    assert r.exit_code == 1 and "not pending" in r.output
    assert len(load_golden(tmp_path / "items.jsonl")) == 2
