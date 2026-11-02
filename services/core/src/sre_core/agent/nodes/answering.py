"""The two documentation routes and the grounding check that follows them."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import openai
from langchain_core.runnables import RunnableConfig

from sre_core.agent.fallback import AllProvidersFailed, extractive_answer
from sre_core.agent.nodes.common import Ctx, Meter, Node, ParseFailure, Update
from sre_core.agent.ports import CachedAnswer
from sre_core.agent.prompts import AGENT_PROMPT_VERSION, decompose_messages, synthesize_messages
from sre_core.agent.schemas import GroundedOut, PlanOut
from sre_core.agent.state import AgentState
from sre_core.answer.citations import resolve_citations
from sre_core.answer.prompts import ABSTAIN_TEXT, group_sources
from sre_shared.contracts.agent import SourcesEvent
from sre_shared.llm import QuotaExceeded
from sre_shared.models import AnswerRequest, Citation, Claims, RetrievedChunk

log = logging.getLogger("sre_core.agent")

# Failures of the model side that the extractive answer covers (not bugs: those propagate).
_MODEL_DOWN = (AllProvidersFailed, QuotaExceeded, openai.OpenAIError)
MAX_COMPLEX_CHUNKS = 8


def citation_of(c: RetrievedChunk) -> Citation:
    return Citation(
        chunk_id=c.chunk_id,
        source_path=c.source_path,
        k8s_version=c.k8s_version,
        url=c.url,
        title=c.title,
    )


def _dump(cites: Sequence[Citation]) -> list[dict[str, object]]:
    return [c.model_dump() for c in cites]


def merge_chunks(
    per_question: Sequence[Sequence[RetrievedChunk]], cap: int
) -> list[RetrievedChunk]:
    """Round-robin over the sub-questions' results so each keeps its best chunks; no duplicates."""
    out: list[RetrievedChunk] = []
    seen: set[int] = set()
    depth = max((len(r) for r in per_question), default=0)
    for i in range(depth):
        for chunks in per_question:
            if i < len(chunks) and chunks[i].chunk_id not in seen:
                seen.add(chunks[i].chunk_id)
                out.append(chunks[i])
    return out[:cap]


def make_answer_nodes(ctx: Ctx) -> dict[str, Node]:
    cfg = ctx.cfg
    floor = ctx.answer_cfg.relevance_floor

    def no_support(chunks: Sequence[RetrievedChunk]) -> bool:
        sims = [c.dense_similarity for c in chunks if c.dense_similarity is not None]
        return not chunks or (bool(sims) and max(sims) < floor)

    def extractive(question: str, chunks: Sequence[RetrievedChunk], confidence: float) -> Update:
        """The last fallback: quoted passages, no model. Not cached (it is a degraded answer)."""
        text, _ = extractive_answer(question, chunks)
        if not text:
            return {"abstained": True, "answer": ABSTAIN_TEXT}
        clean, cites, invalid = resolve_citations(text, chunks, group_sources(chunks))
        if cites:
            ctx.emit(SourcesEvent(sources=cites))
        return {
            "answer": clean,
            "citations": _dump(cites),
            "invalid_citations": invalid,
            "confidence": confidence,
            "degraded": True,
        }

    async def simple(state: AgentState, config: RunnableConfig) -> Update:
        """Retrieve, compress and answer with citations: the existing answer path, unchanged."""

        async def body(meter: Meter, claims: Claims) -> Update:
            question = state["standalone"]
            await ctx.charge(meter, claims, ctx.answer_cfg.max_tokens)
            try:
                res = await ctx.answerer.answer(AnswerRequest(question=question), claims)
            except _MODEL_DOWN:
                log.warning("answer model unavailable: extractive fallback")
                found = await ctx.retrieve(question, claims)
                if no_support(found.chunks):
                    return {
                        "abstained": True,
                        "answer": ABSTAIN_TEXT,
                        "context_citations": _dump([citation_of(c) for c in found.chunks]),
                        "corpus_version_id": found.corpus_version_id,
                        "answer_route": "simple",
                    }
                out = extractive(question, found.chunks, state["route_confidence"])
                return {
                    **out,
                    "context_citations": _dump([citation_of(c) for c in found.chunks]),
                    "corpus_version_id": found.corpus_version_id,
                    "answer_route": "simple",
                }
            ctx.corpus_version_id = res.corpus_version_id
            if not res.abstained or res.prompt_tokens or res.completion_tokens:
                await ctx.record(meter, claims, res.prompt_tokens, res.completion_tokens)
            if res.citations:
                ctx.emit(SourcesEvent(sources=res.citations))
            return {
                "answer": res.answer,
                "citations": _dump(res.citations),
                "invalid_citations": res.invalid_citations,
                "confidence": state["route_confidence"],
                "abstained": res.abstained,
                "answer_route": "simple",
                "context_citations": _dump([citation_of(c) for c in res.context]),
                "corpus_version_id": res.corpus_version_id,
            }

        return await ctx.run(state, config, body)

    async def decompose(state: AgentState, config: RunnableConfig) -> Update:
        async def body(meter: Meter, claims: Claims) -> Update:
            limit = cfg.complex.max_subquestions
            try:
                plan = await ctx.structured(
                    meter,
                    claims,
                    "router",
                    decompose_messages(state["standalone"], limit),
                    PlanOut,
                    max_tokens=300,
                )
                subs = [q.strip() for q in plan.subquestions if q.strip()]
            except (ParseFailure, AllProvidersFailed):
                subs = []
            subs = list(dict.fromkeys(subs))[:limit] or [state["standalone"]]
            return {"subquestions": subs}

        return await ctx.run(state, config, body)

    async def complex_answer(state: AgentState, config: RunnableConfig) -> Update:
        """Parallel retrieval per sub-question, then one synthesis over the merged sources."""

        async def body(meter: Meter, claims: Claims) -> Update:
            question = state["standalone"]
            results = await asyncio.gather(
                *(ctx.retrieve(q, claims) for q in state["subquestions"])
            )
            chunks = merge_chunks([r.chunks for r in results], MAX_COMPLEX_CHUNKS)
            base: Update = {
                "answer_route": "complex",
                "context_citations": _dump([citation_of(c) for c in chunks]),
                "corpus_version_id": results[0].corpus_version_id if results else None,
            }
            if no_support(chunks):
                return {**base, "abstained": True, "answer": ABSTAIN_TEXT}
            try:
                out = await ctx.structured(
                    meter,
                    claims,
                    "synthesize",
                    synthesize_messages(question, chunks),
                    GroundedOut,
                    max_tokens=ctx.answer_cfg.max_tokens,
                )
            except (ParseFailure, *_MODEL_DOWN):
                log.warning("synthesis unavailable: extractive fallback")
                return {**base, **extractive(question, chunks, state["route_confidence"])}
            clean, cites, invalid = resolve_citations(out.answer, chunks)
            if cites:
                ctx.emit(SourcesEvent(sources=cites))
            return {
                **base,
                "answer": clean.strip(),
                "citations": _dump(cites),
                "invalid_citations": invalid,
                "confidence": min(state["route_confidence"], out.confidence),
            }

        return await ctx.run(state, config, body)

    async def grounding(state: AgentState, config: RunnableConfig) -> Update:
        """Pass: respond (and cache). Fail: abstain and show what was found."""
        claims = ctx.claims(config)
        reason: str | None = None
        if state["abstained"]:
            reason = ABSTAIN_TEXT
        elif state["invalid_citations"]:
            reason = "The draft cited sources it was not given, so I did not trust it."
        elif not state["citations"]:
            reason = "I could not tie an answer to the documentation."
        elif state["confidence"] < cfg.confidence_floor:
            reason = "I am not confident enough in an answer from the sources I found."
        if reason is not None:
            return {
                "final_kind": "abstain",
                "final_route": state["answer_route"],
                "abstain_reason": reason,
            }
        version = state.get("corpus_version_id")
        if version is not None and not state["degraded"]:
            try:
                await ctx.cache.store(
                    CachedAnswer(
                        question=state["standalone"],
                        answer=state["answer"],
                        citations=[Citation.model_validate(c) for c in state["citations"]],
                        route="complex" if state["answer_route"] == "complex" else "simple",
                    ),
                    claims=claims,
                    corpus_version_id=version,
                    prompt_version=AGENT_PROMPT_VERSION,
                )
            except Exception:  # a cache outage must not lose a good answer
                log.warning("answer cache store failed", exc_info=True)
        return {"final_kind": "done", "final_route": state["answer_route"]}

    return {
        "simple": simple,
        "decompose": decompose,
        "complex": complex_answer,
        "grounding": grounding,
    }
