"""FastAPI application. Phase 1 exposes health and search; chat and SSE arrive in Phase 3."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="SRE Copilot core", version="0.1.0")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    import uvicorn

    uvicorn.run("sre_core.api.main:app", host="127.0.0.1", port=8000)
