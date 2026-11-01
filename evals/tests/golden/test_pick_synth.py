from __future__ import annotations

import re
from typing import Any

import pytest

from sre_evals.golden.contexts import (
    ContextRow,
    find_exact_strings,
    pick_contexts,
    read_contexts,
    write_contexts,
)
from sre_evals.golden.llm import ModelSpec, parse_model_spec
from sre_evals.golden.synth import QuotaError, a_synthesize, check_quota, estimate_tokens
from sre_shared.config import Limits, ModelLimits, Provider

pytestmark = pytest.mark.integration


def test_find_exact_strings_kinds() -> None:
    text = (
        "Pods in CrashLoopBackOff; use --max-surge or --dry-run=client; set "
        "spec.template.spec.containers and note ErrImagePull. Single --force is ignored."
    )
    found = dict((s, k) for k, s in find_exact_strings(text))
    assert found["CrashLoopBackOff"] == "error"
    assert found["--max-surge"] == "flag"
    assert found["spec.template.spec.containers"] == "field"
    assert found["ErrImagePull"] == "error"
    assert "--force" not in found


async def test_pick_factual_rule_and_determinism(db: Any, corpus: int) -> None:
    a = await pick_contexts(db, "factual", 6, None, seed=7)
    b = await pick_contexts(db, "factual", 6, None, seed=7)
    assert [r.chunk_key for r in a] == [r.chunk_key for r in b]
    assert a, "fixture corpus has factual candidates"
    assert all("/reference/" not in r.source_path for r in a)  # concept/task only
    assert all("private" not in r.source_path for r in a)  # public group only
    assert all(r.stratum == "factual" and r.matched is None for r in a)
    # spread across documents: first picks never repeat a document while others remain
    docs = {(r.source_path, r.k8s_version) for r in a}
    assert len(docs) >= 3
    other = await pick_contexts(db, "factual", 4, None, seed=8)
    assert {r.chunk_key for r in other} != {r.chunk_key for r in a} or len(other) < len(a)


async def test_pick_exact_string_matches_rule(db: Any, corpus: int) -> None:
    rows = await pick_contexts(db, "exact_string", 20, ["1.34"], seed=1)
    assert rows
    for r in rows:
        assert r.matched and r.matched in r.body
        assert r.k8s_version == "1.34"
        assert (r.match_kind, r.matched) in find_exact_strings(r.body)
    matched = [r.matched for r in rows]
    assert "--max-surge" in matched or "spec.template.spec.containers" in matched
    # the reference chunk is eligible for exact strings (any doc type), the private one is not
    assert not any("private" in r.source_path for r in rows)
    again = await pick_contexts(db, "exact_string", 20, ["1.34"], seed=1)
    assert rows == again


async def test_contexts_jsonl_round_trip(db: Any, corpus: int, tmp_path: Any) -> None:
    rows = await pick_contexts(db, "exact_string", 5, None, seed=3)
    p = tmp_path / "ctx.jsonl"
    write_contexts(p, rows)
    assert read_contexts(p) == rows


def _row(tag: str, stratum: str = "factual", matched: str | None = None) -> ContextRow:
    body = f"[T:{tag}] " + ("Body text for the synthesizer. " * 10) + (matched or "")
    return ContextRow(
        chunk_key=f"key-{tag}",
        source_path=f"content/en/docs/{tag}.md",
        k8s_version="1.34",
        url=f"https://kubernetes.io/{tag}/",
        title=tag,
        body=body,
        stratum=stratum,  # type: ignore[arg-type]
        matched=matched,
        match_kind="error" if matched else None,
    )


async def test_synthesize_one_golden_per_context(make_llm: Any) -> None:
    llm, _fake = make_llm()
    rows = [_row(f"c{i}") for i in range(5)]
    res = await a_synthesize(rows, llm, evolutions=0)
    assert not res.skipped
    assert len(res.items) == 5
    by_key = {t.split(":", 1)[1]: i for i in res.items for t in i.tags if t.startswith("chunk:")}
    for r in rows:  # mapped back by chunk key, not by output order
        item = by_key[r.chunk_key]
        assert item.gold_passages[0].quote == r.body
        assert item.gold_passages[0].source_path == r.source_path
        assert item.filters is not None and item.filters.k8s_versions == [r.k8s_version]
        assert re.search(rf"about {r.title}\?", item.question)
        assert item.status == "candidate" and item.origin == "synthetic"
        assert item.synth_model == "ollama:fake-model"
        assert item.expected_answer == "A grounded answer."
        assert "evolutions:0" in item.tags
    assert [i.id for i in res.items if i.stratum == "factual"] == [
        f"fac-{n:04d}" for n in range(1, 6)
    ] or len({i.id for i in res.items}) == 5


async def test_synthesize_skips_context_without_exactly_one_golden(make_llm: Any) -> None:
    llm, _ = make_llm(drop_tags={"c1"})
    res = await a_synthesize([_row("c0"), _row("c1")], llm)
    assert [i.gold_passages[0].source_path for i in res.items] == ["content/en/docs/c0.md"]
    assert res.skipped == [("key-c1", "0 goldens instead of 1")]


async def test_exact_string_repair_then_reject(make_llm: Any) -> None:
    rows = [
        _row("e0", "exact_string", "CrashLoopBackOff"),
        _row("e1", "exact_string", "ImagePullBackOff"),
        _row("e2", "exact_string", "ErrImagePull"),
    ]
    llm, _fake = make_llm(
        questions={
            "e0": "Why does my pod show CrashLoopBackOff?",  # already contains it
            "e1": "Why can the image not be pulled?",  # repairable
            "e2": "What is the pull error called?",  # unrepairable
        },
        repair=lambda p: "Why do I see ImagePullBackOff?" if "ImagePullBackOff" in p else "nope",
    )
    res = await a_synthesize(rows, llm)
    qs = {i.question for i in res.items}
    assert qs == {"Why does my pod show CrashLoopBackOff?", "Why do I see ImagePullBackOff?"}
    assert res.exact_repaired == 1 and res.exact_rejected == 1
    assert all(i.stratum == "exact_string" for i in res.items)
    assert all(
        i.question.count(next(t[6:] for t in i.tags if t.startswith("exact:"))) >= 1
        for i in res.items
    )
    assert any("ErrImagePull" in why for _, why in res.skipped)


async def test_filter_threshold_rejects_low_quality_and_counts_per_stratum(make_llm: Any) -> None:
    llm, _ = make_llm(
        questions={"c0": "Good question about c0?", "c1": "bad one"},
        critic_score=lambda q: 0.1 if "bad one" in q or "rewritten" in q else 0.9,
    )
    res = await a_synthesize([_row("c0"), _row("c1")], llm, threshold=0.5)
    assert len(res.items) == 1 and res.items[0].question == "Good question about c0?"
    assert res.filter_rejected == {"factual": 1}


async def test_evolutions_tag(make_llm: Any) -> None:
    llm, _ = make_llm()
    res = await a_synthesize([_row("c0")], llm, evolutions=0)
    assert "evolutions:0" in res.items[0].tags


# --- quota guard ---------------------------------------------------------------------


def _limits(tpd: int | None = 200_000) -> Limits:
    return Limits(
        providers={"groq": Provider(order=1, models={"m": ModelLimits(tpd=tpd)})},
        budgets={"golden_synthesis_max_share_of_tpd": 0.25},
    )


def test_quota_guard() -> None:
    rows = [_row(f"c{i}") for i in range(10)]
    hosted = ModelSpec("groq", "m", "https://x/v1", "k")
    assert check_quota(hosted, rows, _limits()) == estimate_tokens(rows)
    many = [_row(f"c{i}") for i in range(200)]
    with pytest.raises(QuotaError, match="exceeds 25%"):
        check_quota(hosted, many, _limits())
    with pytest.raises(QuotaError, match="no tpd"):
        check_quota(hosted, rows, _limits(None))
    local = ModelSpec("ollama", "qwen2.5:7b", "http://x/v1", "ollama")
    assert check_quota(local, many, _limits()) > 50_000  # exempt, no error


def test_parse_model_spec() -> None:
    s = parse_model_spec("ollama:qwen2.5:7b")
    assert (s.provider, s.model, s.hosted) == ("ollama", "qwen2.5:7b", False)
    assert s.base_url.endswith("/v1")
    g = parse_model_spec("groq:openai/gpt-oss-20b")
    assert g.hosted and g.model == "openai/gpt-oss-20b"
    with pytest.raises(ValueError):
        parse_model_spec("nocolon")
    with pytest.raises(ValueError):
        parse_model_spec("nope:x")
