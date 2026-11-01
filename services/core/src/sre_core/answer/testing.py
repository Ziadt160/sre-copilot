"""Fakes for the answer tests (not used in production): a chat client without a network, a
retriever stub, and chunk factories."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from sre_shared.config import Limits, ModelLimits, Provider
from sre_shared.llm import ChatClient, MemoryLedger, ModelSpec
from sre_shared.models import (
    ANONYMOUS,
    Claims,
    DocType,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)


class FakeAsyncCompletions:
    def __init__(self, reply: str = "Pods are the smallest unit [1].") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if kw.get("stream"):
            return self._stream()
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(
            model=kw["model"],
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=7),
        )

    async def _stream(self) -> Any:
        words = self.reply.split(" ")
        for i, w in enumerate(words):
            piece = w if i == 0 else " " + w
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])


def fake_chat(
    reply: str = "Pods are the smallest unit [1].", *, tpd: int | None = None
) -> tuple[ChatClient, FakeAsyncCompletions]:
    comps = FakeAsyncCompletions(reply)
    aclient = SimpleNamespace(chat=SimpleNamespace(completions=comps))
    if tpd is None:
        spec = ModelSpec("ollama", "fake", "http://127.0.0.1:1/v1", "x")
        return ChatClient(spec, aclient=aclient, client=object()), comps
    limits = Limits(
        providers={
            "groq": Provider(
                order=1,
                base_url="https://x/v1",
                key_env="GROQ_API_KEY",
                models={"fake": ModelLimits(tpd=tpd)},
            )
        },
        budgets={"stop_routing_at_share_of_tpd": 0.9},
    )
    spec = ModelSpec("groq", "fake", "https://x/v1", "k")
    chat = ChatClient(spec, aclient=aclient, client=object(), ledger=MemoryLedger(), limits=limits)
    return chat, comps


def chunk(
    i: int,
    body: str = "Body text.",
    *,
    sim: float | None = 0.9,
    title: str = "Pods",
    section: str = "Overview",
    path: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i,
        chunk_key=f"key{i}",
        source_path=path or f"content/en/docs/p{i}.md",
        k8s_version="1.31",
        doc_type=DocType.CONCEPT,
        title=title,
        section_path=[section],
        url=f"https://example.test/p{i}",
        header=f"{title} > {section}",
        body=body,
        score=1.0 / i,
        dense_similarity=sim,
    )


class StubRetriever:
    """Returns fixed chunks and records the request and claims it was called with."""

    def __init__(self, chunks: list[RetrievedChunk], version: int = 7) -> None:
        self.chunks = chunks
        self.version = version
        self.calls: list[tuple[RetrievalRequest, Claims]] = []

    async def retrieve(self, req: RetrievalRequest, claims: Claims = ANONYMOUS) -> RetrievalResult:
        self.calls.append((req, claims))
        return RetrievalResult(
            request=req,
            corpus_version_id=self.version,
            chunks=self.chunks[: req.k],
            timings_ms={"embed": 0.0, "sql": 0.0, "fuse": 0.0, "total": 0.0},
        )
