"""Version stamps: what produced a number, so a drop arrives with its cause."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sre_shared.settings import REPO_ROOT


def git_sha(repo: Path = REPO_ROOT) -> str:
    """HEAD's short SHA, with "-dirty" when tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def config_hash(obj: Any) -> str:
    """Stable 16-hex-digit hash of a JSON-serializable config (pydantic models allowed)."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class VersionStamps(BaseModel):
    git_sha: str = Field(default_factory=git_sha)
    config_hash: str
    embedding_model: str | None = None
    embedding_revision: str | None = None
    corpus_version_id: int | None = None
    chunker_version: str | None = None
    golden_version: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    models_per_node: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
