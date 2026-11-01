"""`sre-eval retrieval -e A -e B` runs both experiments (a single-value option kept only the
last one, silently)."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from sre_evals import cli


def test_repeated_experiment_options_run_every_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    async def fake_run_all(dsn: str, selected: list[Any], *a: Any) -> bool:
        seen.append([c.name for c in selected])
        return False

    monkeypatch.setattr(cli, "_run_all", fake_run_all)
    args = ["retrieval", "-e", "dense-gte-small-rerank20", "-e", "dense-gte-small",
            "-e", "dense-gte-small-rerank20", "--database-url", "postgresql://unused"]  # fmt: skip
    res = CliRunner().invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    assert seen == [["dense-gte-small-rerank20", "dense-gte-small"]]  # order kept, once each


def test_an_unknown_experiment_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*a: Any) -> bool:
        raise AssertionError("must not run")

    monkeypatch.setattr(cli, "_run_all", fail)
    res = CliRunner().invoke(cli.app, ["retrieval", "-e", "dense-gte-small", "-e", "nope"])
    assert res.exit_code != 0 and "nope" in res.output
