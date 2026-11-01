"""Generated answers saved as they are produced, so a failed run never re-spends tokens.

A Groq pass over 69 questions costs ~165k of a ~180k daily allowance; one connection error
after 40 answers used to discard the whole run (it is stored `failed`, with no scores). Each
answer is now appended to `data/cache/answers/<key>.jsonl` the moment it exists, keyed by the
run's config hash, the repeat number and the golden version; a rerun of the same
configuration reuses those answers and only generates the missing ones. Noise repeats have
their own keys, so they still generate afresh.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sre_shared.models import AnswerResult
from sre_shared.settings import get_settings


def cache_dir() -> Path:
    return get_settings().data_dir / "cache" / "answers"


def cache_key(config_hash: str, repeat: int, golden_version: str) -> str:
    raw = f"{config_hash}|r{repeat}|{golden_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class AnswerCache:
    """Append-only JSONL of {item_id, question, result}; the last line per item wins."""

    def __init__(self, key: str, directory: Path | None = None) -> None:
        self.path = (directory or cache_dir()) / f"{key}.jsonl"
        self._items: dict[str, AnswerResult] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self._items[row["item_id"]] = AnswerResult.model_validate(row["result"])
                except (ValueError, KeyError):
                    continue  # a line cut off by a crash: that item is generated again

    def get(self, item_id: str, question: str) -> AnswerResult | None:
        hit = self._items.get(item_id)
        return hit if hit is not None and hit.question == question else None

    def put(self, item_id: str, result: AnswerResult) -> None:
        self._items[item_id] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"item_id": item_id, "result": result.model_dump(mode="json")})
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    def __len__(self) -> int:
        return len(self._items)
