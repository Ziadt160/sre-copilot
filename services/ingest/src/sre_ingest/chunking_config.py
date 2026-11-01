"""Chunking experiment parameters, loaded from `config/chunking/<name>.yaml`."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sre_shared.settings import get_settings

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ChunkingConfig(BaseModel):
    """One chunking variant. `version` is the chunker version recorded on the corpus row and
    appended to the corpus label, so each variant ingests into its own corpus version."""

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    version: str = "1"
    max_tokens: int = Field(510, ge=16, le=510)
    """gte-small tokens, header included."""
    overlap_tokens: int = Field(0, ge=0)
    header_in_embedding: bool = True
    """False: the embedding text is the body only (the header stays stored and keyword-indexed)."""


def load_chunking(name: str = "default", config_dir: Path | None = None) -> ChunkingConfig:
    if not NAME_RE.match(name):
        raise ValueError(f"invalid chunking name {name!r}")
    path = (config_dir or get_settings().config_dir) / "chunking" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no chunking config {name!r} at {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = ChunkingConfig.model_validate(data)
    if cfg.name != name:
        raise ValueError(f"{path}: name {cfg.name!r} does not match the file name")
    return cfg
