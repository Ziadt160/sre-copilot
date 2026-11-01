"""Reading and writing golden-set files (JSONL, one GoldenItem per line)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sre_shared.models import GoldenItem
from sre_shared.settings import REPO_ROOT
from sre_shared.stamps import file_hash

GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


def golden_path(version: str = "v1") -> Path:
    return GOLDEN_DIR / version / "items.jsonl"


def load_golden(path: Path, *, accepted_only: bool = False) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = GoldenItem.model_validate_json(line)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
            if not accepted_only or item.status == "accepted":
                items.append(item)
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items


def write_golden(path: Path, items: Iterable[GoldenItem]) -> None:
    """Write items atomically, one JSON object per line, in the given order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(item.model_dump_json(exclude_none=True) + "\n")
    tmp.replace(path)


def golden_version(path: Path) -> str:
    """`<dir name>-<content hash>`, e.g. "v1-3f2a9c...": changes whenever any item changes."""
    return f"{path.parent.name}-{file_hash(path)}"
