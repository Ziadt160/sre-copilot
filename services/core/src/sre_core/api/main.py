"""FastAPI application. Phase 1 exposes health and search; chat and SSE arrive in Phase 3."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from sre_core.api.auth import claims_from_authorization
from sre_core.retrieval import NoActiveCorpusError, Retriever
from sre_shared.config import load_retrieval_defaults
from sre_shared.db import create_pool
from sre_shared.models import (
    Claims,
    DocType,
    EmbeddingModelName,
    RetrievalFilters,
    RetrievalRequest,
    RetrievalResult,
    TsConfig,
)


def get_claims(authorization: Annotated[str | None, Header()] = None) -> Claims:
    return claims_from_authorization(authorization)


def get_retriever(request: Request) -> Retriever:
    r: Retriever = request.app.state.retriever
    return r


def create_app(retriever: Retriever | None = None) -> FastAPI:
    """Build the app. Tests pass a `Retriever` on their own pool; production opens one at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if retriever is not None:
            app.state.retriever = retriever
            yield
            return
        pool = create_pool()
        await pool.open()
        app.state.retriever = Retriever(pool)
        try:
            yield
        finally:
            await pool.close()

    app = FastAPI(title="SRE Copilot core", version="0.1.0", lifespan=lifespan)
    if retriever is not None:
        app.state.retriever = retriever

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search", response_model=RetrievalResult)
    async def search(
        retr: Annotated[Retriever, Depends(get_retriever)],
        claims: Annotated[Claims, Depends(get_claims)],
        q: Annotated[str, Query(min_length=1)],
        k: Annotated[int, Query(ge=1, le=100)] = 5,
        model: EmbeddingModelName = "gte-small",
        version: Annotated[list[str] | None, Query(description="Kubernetes version(s)")] = None,
        doc_type: Annotated[list[DocType] | None, Query()] = None,
        mode: Annotated[
            Literal["hybrid", "dense", "keyword"] | None,
            Query(description="Defaults to `mode` in config/retrieval.yaml (dense)"),
        ] = None,
        ts_config: TsConfig = "english",
        corpus_version: Annotated[
            int | None, Query(description="Defaults to the active one")
        ] = None,
    ) -> RetrievalResult:
        mode = mode or load_retrieval_defaults().mode
        req = RetrievalRequest(
            query=q,
            k=k,
            embedding_model=model,
            filters=RetrievalFilters(k8s_versions=version or None, doc_types=doc_type or None),
            use_dense=mode in ("hybrid", "dense"),
            use_keyword=mode in ("hybrid", "keyword"),
            ts_config=ts_config,
            corpus_version_id=corpus_version,
        )
        try:
            return await retr.retrieve(req, claims)
        except NoActiveCorpusError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """psycopg async cannot use the Proactor loop that uvicorn picks on Windows."""
    return asyncio.SelectorEventLoop()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "sre_core.api.main:app",
        host="127.0.0.1",
        port=8000,
        loop="sre_core.api.main:selector_loop_factory" if sys.platform == "win32" else "auto",
    )
