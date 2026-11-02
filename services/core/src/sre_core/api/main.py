"""FastAPI application. Phase 1 exposes health and search; chat and SSE arrive in Phase 3."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from sre_core.answer import Answerer, build_answerer
from sre_core.answer.service import ConfigError, error_event
from sre_core.api.auth import claims_from_authorization
from sre_core.api.chat import router as chat_router
from sre_core.api.chat import start_background, stop_background
from sre_core.api.runtime import Runtime, build_runtime, close_runtime
from sre_core.retrieval import NoActiveCorpusError, Retriever
from sre_core.retrieval.rerank import default_reranker
from sre_shared.config import load_retrieval_defaults
from sre_shared.db import create_pool
from sre_shared.llm import QuotaExceeded
from sre_shared.models import (
    AnswerRequest,
    AnswerResult,
    Claims,
    DocType,
    EmbeddingModelName,
    RetrievalFilters,
    RetrievalRequest,
    RetrievalResult,
    TsConfig,
)
from sre_shared.settings import get_settings


def get_claims(authorization: Annotated[str | None, Header()] = None) -> Claims:
    return claims_from_authorization(authorization)


def get_retriever(request: Request) -> Retriever:
    r: Retriever = request.app.state.retriever
    return r


def get_answerer(request: Request) -> Answerer:
    a: Answerer = request.app.state.answerer
    return a


def sse_frame(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def create_app(
    retriever: Retriever | None = None,
    answerer: Answerer | None = None,
    allow_request_config: bool = False,
    runtime: Runtime | None = None,
) -> FastAPI:
    """Build the app. Tests pass a `Retriever` (and optionally an `Answerer`) on their own pool;
    production opens a pool at startup. `allow_request_config` lets `/answer` callers send their
    own `AnswerConfig` (model, floor, tokens): off by default, because it would let anyone pick
    the model and disable abstention. The chat routes are mounted when a `runtime` is given, or
    in production, where the lifespan builds one (chat answers 503 if that failed)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if runtime is not None:
            app.state.runtime = runtime
            start_background(runtime)
        try:
            if retriever is not None:
                app.state.retriever = retriever
                app.state.answerer = answerer or build_answerer(retriever)
                yield
                return
            async with _production(app):
                yield
        finally:
            if runtime is not None:
                await stop_background(runtime)

    @asynccontextmanager
    async def _production(app: FastAPI) -> AsyncIterator[None]:
        pool = create_pool()
        await pool.open()
        replica_url = get_settings().database_replica_url
        # A dead replica must not stall requests for the pool's default 30 s: after 2 s the
        # read falls back to the primary (Retriever catches PoolTimeout).
        read_pool = create_pool(replica_url, timeout=2.0) if replica_url else None
        if read_pool is not None:
            await read_pool.open()  # failures fall back to the primary per request
        app.state.retriever = Retriever(pool, read_pool=read_pool, reranker=default_reranker())
        app.state.answerer = build_answerer(app.state.retriever)
        built: Runtime | None = None
        if runtime is None:
            try:
                built = await build_runtime(
                    get_settings(), retriever=app.state.retriever, answerer=app.state.answerer
                )
                app.state.runtime = built
                start_background(built)
            except Exception:
                logging.getLogger(__name__).exception("chat runtime unavailable; /chat answers 503")
        try:
            yield
        finally:
            if built is not None:
                await stop_background(built)
                await close_runtime(built)
            if read_pool is not None:
                await read_pool.close()
            await pool.close()

    app = FastAPI(title="SRE Copilot core", version="0.1.0", lifespan=lifespan)
    if retriever is not None:
        app.state.retriever = retriever
        app.state.answerer = answerer or build_answerer(retriever)
    if runtime is not None:
        app.state.runtime = runtime
    if runtime is not None or retriever is None:
        app.include_router(chat_router)

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

    @app.post("/answer", response_model=AnswerResult)
    async def answer(
        body: AnswerRequest,
        request: Request,
        ans: Annotated[Answerer, Depends(get_answerer)],
        claims: Annotated[Claims, Depends(get_claims)],
    ) -> AnswerResult | StreamingResponse:
        """JSON `AnswerResult`; with `Accept: text/event-stream`, SSE events `sources`, `token`
        (repeated) and `done` (the full `AnswerResult`)."""
        if body.config is not None and not allow_request_config:
            raise HTTPException(status_code=422, detail="config is not accepted on this endpoint")
        try:
            if "text/event-stream" in request.headers.get("accept", ""):
                events = ans.stream(body, claims)
                first = await anext(events)  # errors surface here, before the 200 is sent

                async def frames() -> AsyncIterator[str]:
                    yield sse_frame(first["event"], first["data"])
                    try:
                        async for ev in events:
                            yield sse_frame(ev["event"], ev["data"])
                    except Exception as exc:  # the 200 is sent: end with an error event
                        ev = error_event(exc)
                        yield sse_frame(ev["event"], ev["data"])

                return StreamingResponse(
                    frames(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return await ans.answer(body, claims)
        except NoActiveCorpusError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except QuotaExceeded as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ConfigError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
