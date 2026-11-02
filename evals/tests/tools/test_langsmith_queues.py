"""tools/langsmith_queues.py against a fake LangSmith client: no network, no real key."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tools" / "langsmith_queues.py"
_spec = importlib.util.spec_from_file_location("langsmith_queues", _PATH)
assert _spec and _spec.loader
lq = importlib.util.module_from_spec(_spec)
sys.modules["langsmith_queues"] = lq
_spec.loader.exec_module(lq)


class FakeLangSmith:
    """Stores what the script creates; `writes` lists every mutating call."""

    def __init__(self, *, project: bool = True, honours_retention: bool = True) -> None:
        self.queues: list[SimpleNamespace] = []
        self.rules: list[dict[str, Any]] = []
        self.writes: list[str] = []
        self.reads: list[str] = []
        self.project = project
        self.honours_retention = honours_retention

    def list_annotation_queues(self, *, name: str | None = None, **_: Any) -> Any:
        self.reads.append("list_annotation_queues")
        return iter([q for q in self.queues if q.name == name])

    def create_annotation_queue(self, *, name: str, **kw: Any) -> SimpleNamespace:
        self.writes.append("create_annotation_queue")
        q = SimpleNamespace(id=f"q{len(self.queues) + 1}", name=name, **kw)
        self.queues.append(q)
        return q

    def read_project(self, *, project_name: str) -> SimpleNamespace:
        self.reads.append("read_project")
        if not self.project:
            raise LookupError(project_name)
        return SimpleNamespace(id="proj-1")

    def request_with_retries(self, method: str, path: str, **kw: Any) -> Any:
        assert path == "/runs/rules"
        if method == "GET":
            self.reads.append("GET /runs/rules")
            return SimpleNamespace(json=lambda: list(self.rules))
        assert method == "POST"
        self.writes.append("POST /runs/rules")
        body = dict(kw["json"])
        if not self.honours_retention:
            body.pop("extend_annotation_queue_trace_retention")
        self.rules.append(body)
        return SimpleNamespace(json=lambda: body)


def test_dry_run_with_a_client_reads_but_calls_no_write_method(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLangSmith()
    lq.plan(fake, "sre-copilot")
    assert fake.writes == []
    assert fake.reads  # it did look at the workspace

    # And through the CLI entry point, without --apply.
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-not-real")
    assert lq.main([], client_factory=lambda _k: fake) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "sre-copilot-review" in out and "retention extension OFF" in out
    assert fake.writes == []


def test_dry_run_without_a_key_needs_no_client(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert lq.main([]) == 0
    assert "would create queue" in capsys.readouterr().out


def test_apply_creates_each_object_once_and_a_second_apply_creates_nothing() -> None:
    fake = FakeLangSmith()
    first = lq.apply(fake, "sre-copilot")
    assert fake.writes == ["create_annotation_queue", "POST /runs/rules", "POST /runs/rules"]
    assert len(first.created) == 3 and first.manual == []

    before = list(fake.writes)
    second = lq.apply(fake, "sre-copilot")
    assert fake.writes == before
    assert second.created == [] and len(second.existing) == 3


def test_rules_send_thumbs_down_runs_and_a_ten_percent_sample_with_retention_off() -> None:
    fake = FakeLangSmith()
    lq.apply(fake, "sre-copilot")
    by_name = {r["display_name"]: r for r in fake.rules}
    down, sample = by_name["sre-copilot-thumbs-down"], by_name["sre-copilot-review-sample"]
    assert down["sampling_rate"] == 1.0 and "feedback_score" in down["filter"]
    assert sample["sampling_rate"] == 0.1
    for rule in fake.rules:
        assert rule["extend_annotation_queue_trace_retention"] is False
        assert rule["add_to_annotation_queue_id"] == fake.queues[0].id
    assert "faithful" in fake.queues[0].rubric_instructions
    assert "right tool" in fake.queues[0].rubric_instructions


def test_when_the_api_does_not_confirm_retention_the_user_is_told_to_switch_it_off() -> None:
    fake = FakeLangSmith(honours_retention=False)
    report = lq.apply(fake, "sre-copilot")
    assert len(report.manual) == 2 and "by hand" in report.manual[0]


def test_missing_project_creates_the_queue_and_says_to_come_back_for_the_rules() -> None:
    fake = FakeLangSmith(project=False)
    report = lq.apply(fake, "sre-copilot")
    assert fake.writes == ["create_annotation_queue"]
    assert "does not exist yet" in report.manual[0]


def test_apply_without_a_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert lq.main(["--apply"]) == 2


def test_rule_body_is_json_serialisable() -> None:
    json.dumps(lq.rule_body(lq.RULES[0], project_id="p", queue_id="q"))
