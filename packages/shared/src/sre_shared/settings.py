"""Runtime settings, read from the environment and `.env`.

Secrets are only ever read here and referenced by name elsewhere; nothing logs their values.
Free-tier limits and model lists are not settings: they live in `config/*.yaml` (see `config.py`)
because they change often and are reviewed as data.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root() -> Path:
    """The directory holding `config/limits.yaml`: SRE_REPO_ROOT, else the nearest parent of
    this file or of the working directory (installed images set SRE_REPO_ROOT)."""
    env = os.environ.get("SRE_REPO_ROOT")
    if env:
        return Path(env).resolve()
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in (start, *start.parents):
            if (parent / "config" / "limits.yaml").is_file():
                return parent
    return Path.cwd().resolve()


REPO_ROOT = _find_repo_root()


class Environment(StrEnum):
    LAB = "lab"
    CI = "ci"
    DEMO = "demo"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sre_env: Environment = Environment.LAB
    config_dir: Path = REPO_ROOT / "config"
    data_dir: Path = REPO_ROOT / "data"

    # Postgres. Lab default is the local Supabase stack started with `npx supabase start`.
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    # Lab only: retrieval reads go here; writes and everything else stay on database_url.
    database_replica_url: str | None = None
    db_pool_min: int = 1
    db_pool_max: int = 5

    # Supabase HTTP API (Edge Functions, Auth).
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_jwt_secret: SecretStr | None = None  # verifies Supabase Auth JWTs (HS256)
    # Browser origins allowed to call the core API (the web UI); a JSON list in the environment.
    cors_origins: list[str] = ["http://127.0.0.1:3000", "http://localhost:3000"]

    # Model providers. Keys are optional: without them dev and CI use local Ollama.
    groq_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    # build.nvidia.com (NIM); lab judges (decision 39). Either variable name is read.
    nvidia_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("NVIDIA_API_KEY", "NIM_API_KEY")
    )
    ollama_base_url: str = "http://127.0.0.1:11434"

    # Phase 3: agent and mcp-ops.
    redis_url: str | None = None  # lab: redis://127.0.0.1:6379/0 (compose profile agent)
    mcp_ops_url: str = "http://127.0.0.1:8100/mcp"
    mcp_token_secret: SecretStr | None = (
        None  # HS256 key core signs mcp-ops tokens with (32+ bytes)
    )
    fastmcp_tasks_encryption_key: SecretStr | None = None  # encrypts credentials in task snapshots
    cloudflare_account_id: str | None = None  # Workers AI (fallback chain)
    cloudflare_api_token: SecretStr | None = None

    # Observability.
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "sre-copilot"

    # Embeddings. Query-time model for core; ingest and evals choose explicitly.
    embedding_model: str = "gte-small"
    gpu_embed_url: str = "http://127.0.0.1:8091"  # tools/gpu_embed_server.py (lab only)
    hf_offline: bool = Field(default=False, description="Load HF models from the local cache only")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
