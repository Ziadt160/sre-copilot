# sre-core service image. Build from the repository root:
#   docker build -f infra/docker/core.Dockerfile -t sre-core .
# Installs sre-core and its dependency sre-shared only (no ingest, no evals, no torch: the BGE
# models run in the lab GPU server, and gte-small runs on onnxruntime).

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# The workspace needs every member's pyproject to resolve; only core and shared are installed.
COPY pyproject.toml uv.lock ./
COPY services/ingest/pyproject.toml services/ingest/pyproject.toml
COPY evals/pyproject.toml evals/pyproject.toml
COPY packages/shared packages/shared
COPY services/core services/core
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package sre-core

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY config /app/config
ENV PATH="/app/.venv/bin:$PATH" \
    SRE_REPO_ROOT=/app \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "sre_core.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
