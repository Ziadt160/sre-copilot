"""A fake Retriever for runner and report tests (no database, no models)."""

from __future__ import annotations

from collections.abc import Callable

from sre_shared.models import (
    ANONYMOUS,
    Claims,
    DocType,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)

SRC = "content/en/docs/a.md"
VER = "1.33"


def W(a: int, b: int) -> str:
    return " ".join(f"w{i}" for i in range(a, b + 1))


def chunk(chunk_id: int, body: str, src: str = SRC, ver: str = VER) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_key=f"key{chunk_id}",
        source_path=src,
        k8s_version=ver,
        doc_type=DocType.CONCEPT,
        title="t",
        section_path=[],
        url="https://example.test/",
        header="h",
        body=body,
        score=1.0 / chunk_id,
    )


Script = Callable[[RetrievalRequest, bool], tuple[list[RetrievedChunk], float]]


class FakeRetriever:
    """`script(req, exact)` returns (chunks, total_ms). Records every request."""

    def __init__(self, script: Script, corpus_version_id: int = 1) -> None:
        self.script = script
        self.corpus_version_id = corpus_version_id
        self.calls: list[tuple[RetrievalRequest, bool]] = []

    async def retrieve(
        self, req: RetrievalRequest, claims: Claims = ANONYMOUS, *, exact: bool = False
    ) -> RetrievalResult:
        self.calls.append((req, exact))
        chunks, total = self.script(req, exact)
        return RetrievalResult(
            request=req,
            corpus_version_id=self.corpus_version_id,
            chunks=chunks[: req.k],
            timings_ms={"total": total},
        )
