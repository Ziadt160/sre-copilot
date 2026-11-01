"""Regression: `synthesize` with several strata used one asyncio.run() per stratum, so the real
async HTTP client of the second stratum talked to a closed loop and every call failed."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sre_evals.golden import cli
from sre_evals.golden.contexts import ContextRow, write_contexts
from sre_shared.golden_io import load_golden


def _ctx(tag: str, stratum: str, matched: str | None = None) -> ContextRow:
    body = f"[T:{tag}] " + ("Body text for the synthesizer. " * 10) + (matched or "")
    return ContextRow(
        chunk_key=f"key-{tag}",
        source_path=f"content/en/docs/{tag}.md",
        k8s_version="1.36",
        url=f"https://kubernetes.io/{tag}/",
        title=tag,
        body=body,
        stratum=stratum,  # type: ignore[arg-type]
        matched=matched,
        match_kind="flag" if matched else None,
    )


def test_all_strata_share_one_event_loop(
    tmp_path: Path, make_llm: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm, fake = make_llm(questions={"ex": "When should I set --max-surge?"})
    loops: set[int] = set()
    plain = fake.achat.completions.create

    async def recording(**kw: Any) -> Any:
        loops.add(id(asyncio.get_running_loop()))
        return await plain(**kw)

    fake.achat.completions.create = recording
    monkeypatch.setattr(cli, "make_llm", lambda model: llm)
    ctx = tmp_path / "ctx.jsonl"
    write_contexts(ctx, [_ctx("fa", "factual"), _ctx("ex", "exact_string", "--max-surge")])
    out = tmp_path / "cand.jsonl"
    r = CliRunner().invoke(
        cli.app,
        ["synthesize", "--contexts", str(ctx), "--model", "ollama:qwen2.5:7b", "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert sorted(i.stratum for i in load_golden(out)) == ["exact_string", "factual"]
    assert len(loops) == 1
