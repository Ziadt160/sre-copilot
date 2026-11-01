"""Candidate generation with DeepEval's Synthesizer, one context (= one chunk) per golden."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, FiltrationConfig

from sre_evals.golden.contexts import ContextRow
from sre_evals.golden.llm import ModelSpec, OpenAICompatLLM
from sre_shared.config import Limits
from sre_shared.models import GoldenItem, GoldPassage

log = logging.getLogger("sre_evals.golden")

ID_PREFIX = {"factual": "fac", "exact_string": "exs"}
CALL_OVERHEAD_TOKENS = 500  # prompt templates and the reply, on top of the chunk text
DEFAULT_QUALITY_THRESHOLD = 0.5


class QuotaError(RuntimeError):
    """The planned synthesis would use too much of a hosted model's daily token allowance."""


def estimate_tokens(contexts: Sequence[ContextRow], evolutions: int = 0) -> int:
    """~3 model calls per context (inputs, quality critic, expected output) plus one per
    evolution; the chunk text counts as chars/4 tokens in each call."""
    calls = 3 + evolutions
    return sum(calls * (len(c.body) // 4 + CALL_OVERHEAD_TOKENS) for c in contexts)


def check_quota(
    spec: ModelSpec,
    contexts: Sequence[ContextRow],
    limits: Limits,
    *,
    evolutions: int = 0,
) -> int:
    """Refuse a hosted model when the plan exceeds the share of its tpd. Ollama is exempt."""
    est = estimate_tokens(contexts, evolutions)
    if not spec.hosted:
        return est
    share = limits.budgets.get("golden_synthesis_max_share_of_tpd", 0.25)
    tpd = limits.model_limits(spec.provider, spec.model).tpd
    if tpd is None:
        raise QuotaError(f"{spec.label}: no tpd in config/limits.yaml, cannot verify the budget")
    cap = int(tpd * share)
    if est > cap:
        raise QuotaError(
            f"{spec.label}: planned ~{est} tokens for {len(contexts)} contexts exceeds "
            f"{share:.0%} of its {tpd} tokens/day ({cap}); pick fewer contexts or use ollama"
        )
    return est


@dataclass
class SynthResult:
    items: list[GoldenItem] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (chunk_key, reason)
    filter_rejected: Counter[str] = field(default_factory=Counter)  # per stratum
    exact_repaired: int = 0
    exact_rejected: int = 0
    llm_calls: int = 0


def _repair_prompt(question: str, matched: str, body: str) -> str:
    return (
        f"Rewrite the question so that it contains the exact text `{matched}` verbatim, while "
        "staying a natural question that the passage answers. Keep its meaning.\n\n"
        f"Passage:\n{body}\n\nQuestion: {question}\n\nReply with only the rewritten question."
    )


def build_synthesizer(
    llm: OpenAICompatLLM,
    *,
    evolutions: int = 0,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    max_concurrent: int = 4,
) -> Synthesizer:
    return Synthesizer(
        model=llm,
        async_mode=True,
        max_concurrent=max_concurrent,
        filtration_config=FiltrationConfig(
            synthetic_input_quality_threshold=threshold, critic_model=llm
        ),
        evolution_config=EvolutionConfig(num_evolutions=evolutions),
    )


async def a_synthesize(
    contexts: Sequence[ContextRow],
    llm: OpenAICompatLLM,
    *,
    evolutions: int = 0,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
    id_start: int = 1,
    max_concurrent: int = 4,
    synthesizer: Synthesizer | None = None,
) -> SynthResult:
    """One golden per context. Goldens are mapped back by `source_file == chunk_key` because the
    async path returns them in completion order, and `golden.context` must equal `[body]`."""
    result = SynthResult()
    if not contexts:
        return result
    by_key = {c.chunk_key: c for c in contexts}
    if len(by_key) != len(contexts):
        raise ValueError("duplicate chunk_key in contexts")
    synth = synthesizer or build_synthesizer(
        llm, evolutions=evolutions, threshold=threshold, max_concurrent=max_concurrent
    )
    calls_before = llm.stats.calls
    goldens = await synth.a_generate_goldens_from_contexts(
        contexts=[[c.body] for c in contexts],
        include_expected_output=True,
        max_goldens_per_context=1,
        source_files=[c.chunk_key for c in contexts],
    )
    grouped: dict[str, list[Any]] = {}
    for g in goldens:
        grouped.setdefault(str(g.source_file), []).append(g)

    counters: Counter[str] = Counter()
    for c in contexts:
        got = grouped.get(c.chunk_key, [])
        if len(got) != 1:
            result.skipped.append((c.chunk_key, f"{len(got)} goldens instead of 1"))
            continue
        g = got[0]
        if g.context != [c.body]:
            result.skipped.append((c.chunk_key, "golden context differs from the given chunk"))
            continue
        question = str(g.input).strip()
        quality = (g.additional_metadata or {}).get("synthetic_input_quality")
        if isinstance(quality, int | float) and quality < threshold:
            result.filter_rejected[c.stratum] += 1
            result.skipped.append((c.chunk_key, f"critic score {quality:.2f} < {threshold}"))
            continue
        if not question:
            result.skipped.append((c.chunk_key, "empty question"))
            continue
        if c.stratum == "exact_string":
            assert c.matched is not None
            if c.matched not in question:
                repaired = llm.complete(_repair_prompt(question, c.matched, c.body)).strip()
                if c.matched in repaired:
                    question = repaired
                    result.exact_repaired += 1
                else:
                    result.exact_rejected += 1
                    result.skipped.append((c.chunk_key, f"question lacks {c.matched!r}"))
                    continue
        counters[c.stratum] += 1
        tags = [f"evolutions:{evolutions}", f"chunk:{c.chunk_key}", f"url:{c.url}"]
        if c.match_kind:
            tags.append(f"match:{c.match_kind}")
        if c.matched:
            tags.append(f"exact:{c.matched}")
        result.items.append(
            GoldenItem(
                id=f"{ID_PREFIX[c.stratum]}-{id_start + counters[c.stratum] - 1:04d}",
                stratum=c.stratum,
                question=question,
                expected_answer=(g.expected_output or None),
                gold_passages=[
                    GoldPassage(source_path=c.source_path, k8s_version=c.k8s_version, quote=c.body)
                ],
                origin="synthetic",
                status="candidate",
                synth_model=llm.spec.label,
                tags=tags,
            )
        )
    result.llm_calls = llm.stats.calls - calls_before
    return result
