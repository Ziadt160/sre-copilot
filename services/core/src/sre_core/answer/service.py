"""The single-turn answer path: retrieve, compress, generate with citations, validate them.

Grounding rules (design doc): the answer cites chunk IDs, and citations outside the retrieved
set are rejected; when no chunk clears the relevance floor the bot says it does not know and
shows what it found, without calling the model. Retrieved text is delimited data
(`prompts.py`). Retrieval runs under the caller's claims, so RLS decides what can be cited.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from sre_core.answer.base import Compressor
from sre_core.answer.citations import resolve_citations
from sre_core.answer.compress import NoCompressor, SentenceCompressor, count_tokens
from sre_core.answer.prompts import (
    ABSTAIN_TEXT,
    PROMPT_VERSION,
    build_messages,
)
from sre_core.retrieval import Retriever
from sre_shared.config import load_answer_config, load_retrieval_defaults
from sre_shared.embed import Embedder, get_embedder
from sre_shared.llm import ChatClient, Message, estimate_tokens, parse_model_spec
from sre_shared.models import (
    ANONYMOUS,
    AnswerConfig,
    AnswerRequest,
    AnswerResult,
    Claims,
    EmbeddingModelName,
    RetrievalFilters,
    RetrievalRequest,
    RetrievedChunk,
)

Event = dict[str, Any]


class _Prepared:
    def __init__(
        self,
        cfg: AnswerConfig,
        version: int,
        retrieved: list[RetrievedChunk],
        context: list[RetrievedChunk],
        abstain: bool,
        timings: dict[str, float],
        t0: float,
    ) -> None:
        self.cfg = cfg
        self.version = version
        self.retrieved = retrieved
        self.context = context
        self.abstain = abstain
        self.timings = timings
        self.t0 = t0


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        *,
        config: AnswerConfig | None = None,
        chat: ChatClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config
        self._chat = chat
        self._chats: dict[str, ChatClient] = {}
        self._embedder = embedder

    # -- wiring -------------------------------------------------------------------------

    def _cfg(self, req: AnswerRequest) -> AnswerConfig:
        cfg = req.config or self._config or load_answer_config()
        if cfg.prompt_version != PROMPT_VERSION:
            raise ValueError(f"unknown prompt_version {cfg.prompt_version!r}")
        return cfg

    def _chat_for(self, cfg: AnswerConfig) -> ChatClient:
        if self._chat is not None:
            return self._chat
        if cfg.model not in self._chats:
            self._chats[cfg.model] = ChatClient(parse_model_spec(cfg.model))
        return self._chats[cfg.model]

    def _compressor(self, cfg: AnswerConfig) -> Compressor:
        if cfg.compression == "none":
            return NoCompressor()
        if self._embedder is None:
            self._embedder = get_embedder("gte-small")
        return SentenceCompressor(self._embedder)

    @staticmethod
    def _retrieval_request(
        question: str,
        cfg: AnswerConfig,
        filters: RetrievalFilters,
        corpus_version_id: int | None,
    ) -> RetrievalRequest:
        d = load_retrieval_defaults()
        model: EmbeddingModelName = d.embedding_model  # type: ignore[assignment]
        return RetrievalRequest(
            query=question,
            filters=filters,
            embedding_model=model,
            use_dense=d.mode in ("hybrid", "dense"),
            use_keyword=d.mode in ("hybrid", "keyword"),
            ts_config=d.ts_config,
            keyword_match=d.keyword_match,
            candidates_per_leg=d.candidates_per_leg,
            k=cfg.k,
            w_dense=d.weights.get("dense", 1.0),
            w_keyword=d.weights.get("keyword", 1.0),
            rrf_k=d.rrf_k,
            ef_search=d.hnsw.ef_search,
            iterative_scan=d.hnsw.iterative_scan,
            corpus_version_id=corpus_version_id,
        )

    # -- pipeline -----------------------------------------------------------------------

    async def _prepare(self, req: AnswerRequest, claims: Claims) -> _Prepared:
        t0 = time.perf_counter()
        cfg = self._cfg(req)
        rr = self._retrieval_request(req.question, cfg, req.filters, req.corpus_version_id)
        result = await self._retriever.retrieve(rr, claims)
        t1 = time.perf_counter()
        retrieved = result.chunks
        sims = [c.dense_similarity for c in retrieved if c.dense_similarity is not None]
        # Without a dense leg there is no similarity to compare, so the floor cannot apply.
        abstain = not retrieved or (bool(sims) and max(sims) < cfg.relevance_floor)
        context: list[RetrievedChunk] = retrieved
        if not abstain:
            context = await self._compressor(cfg).compress(
                req.question, retrieved, cfg.context_token_budget
            )
        t2 = time.perf_counter()
        timings = {"retrieve": (t1 - t0) * 1000, "compress": (t2 - t1) * 1000}
        return _Prepared(cfg, result.corpus_version_id, retrieved, context, abstain, timings, t0)

    def _finish(
        self,
        req: AnswerRequest,
        p: _Prepared,
        text: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        t_gen_start: float,
    ) -> AnswerResult:
        clean, citations, invalid = resolve_citations(text, p.context)
        clean = clean.strip()
        t_end = time.perf_counter()
        timings = {
            **p.timings,
            "generate": (t_end - t_gen_start) * 1000,
            "total": (t_end - p.t0) * 1000,
        }
        return AnswerResult(
            question=req.question,
            answer=clean,
            citations=citations,
            invalid_citations=invalid,
            abstained=clean == ABSTAIN_TEXT and not citations,
            context=p.context,
            config=p.cfg,
            model=model,
            corpus_version_id=p.version,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            context_tokens=sum(count_tokens(c.header) + count_tokens(c.body) for c in p.context),
            timings_ms=timings,
        )

    def _abstained(self, req: AnswerRequest, p: _Prepared) -> AnswerResult:
        t_end = time.perf_counter()
        return AnswerResult(
            question=req.question,
            answer=ABSTAIN_TEXT,
            citations=[],
            abstained=True,
            context=p.retrieved,
            config=p.cfg,
            model=p.cfg.model,
            corpus_version_id=p.version,
            timings_ms={**p.timings, "generate": 0.0, "total": (t_end - p.t0) * 1000},
        )

    async def answer(self, req: AnswerRequest, claims: Claims = ANONYMOUS) -> AnswerResult:
        p = await self._prepare(req, claims)
        if p.abstain:
            return self._abstained(req, p)
        messages = build_messages(req.question, p.context)
        t_gen = time.perf_counter()
        res = await self._chat_for(p.cfg).acomplete(
            messages, max_tokens=p.cfg.max_tokens, temperature=p.cfg.temperature
        )
        return self._finish(
            req, p, res.text, res.model, res.prompt_tokens, res.completion_tokens, t_gen
        )

    async def stream(self, req: AnswerRequest, claims: Claims = ANONYMOUS) -> AsyncIterator[Event]:
        """Events: `sources` {context}, `token` {text}, `done` {AnswerResult}. Errors (no active
        corpus, quota) are raised before the first event is yielded, so a caller that reads the
        first event can still turn them into an HTTP status."""
        p = await self._prepare(req, claims)
        if p.abstain:
            result = self._abstained(req, p)
            yield {"event": "sources", "data": {"context": _dump(p.retrieved)}}
            yield {"event": "token", "data": {"text": result.answer}}
            yield {"event": "done", "data": result.model_dump(mode="json")}
            return
        chat = self._chat_for(p.cfg)
        messages: list[Message] = build_messages(req.question, p.context)
        chat.check_quota(estimate_tokens(messages, p.cfg.max_tokens))
        yield {"event": "sources", "data": {"context": _dump(p.context)}}
        t_gen = time.perf_counter()
        parts: list[str] = []
        async for delta in chat.astream(
            messages, max_tokens=p.cfg.max_tokens, temperature=p.cfg.temperature
        ):
            parts.append(delta)
            yield {"event": "token", "data": {"text": delta}}
        text = "".join(parts)
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        result = self._finish(
            req, p, text, chat.spec.model, prompt_tokens, count_tokens(text), t_gen
        )
        yield {"event": "done", "data": result.model_dump(mode="json")}


def _dump(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in chunks]


def build_answerer(
    retriever: Retriever,
    *,
    config: AnswerConfig | None = None,
    chat: ChatClient | None = None,
    embedder: Embedder | None = None,
) -> Answerer:
    """The answer path on `retriever`. `config` defaults to config/answer.yaml (a request's own
    `config` overrides it); `chat` defaults to a client for `config.model`; `embedder` serves the
    sentence compressor (gte-small by default)."""
    return Answerer(retriever, config=config, chat=chat, embedder=embedder)
