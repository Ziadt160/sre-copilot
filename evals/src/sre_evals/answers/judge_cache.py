"""Judge scores kept across calibration runs, so a hosted judge's quota is spent once.

Calibration re-judges every labelled answer with every judge. With about 60 labels a hosted
reference judge needs more than one day of the free quota; without a cache a run stopped by
the quota guard loses everything it judged. Scores are appended to
`data/cache/judge/<judge>-<prompt version>.jsonl` as they arrive, keyed by a hash of the
question, the answer and the passages; a later run reuses them. Failures (None) are not kept,
so they are retried. Noise runs (`answers judge --repeat`) must not use this: they measure
how much the same judge varies on the same answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sre_evals.answers.judge import JUDGE_PROMPT_VERSION, Judge
from sre_shared.settings import get_settings


def cache_dir() -> Path:
    return get_settings().data_dir / "cache" / "judge"


def item_key(metric: str, question: str, answer: str, contexts: Sequence[str] = ()) -> str:
    raw = json.dumps([metric, question, answer, list(contexts)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class CachedJudge:
    """Wraps a `Judge`; counters and planning attributes are the wrapped judge's."""

    def __init__(self, judge: Judge, directory: Path | None = None) -> None:
        self.judge = judge
        self.label = judge.label
        self.prompt_version = str(getattr(judge, "prompt_version", JUDGE_PROMPT_VERSION))
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{self.label}-{self.prompt_version}")
        self.path = (directory or cache_dir()) / f"{name}.jsonl"
        self._scores: dict[str, float] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._scores[row["key"]] = float(row["score"])
                except (ValueError, KeyError, TypeError):
                    continue  # a line cut off by a crash: that answer is judged again
        self.hits = 0

    def __getattr__(self, name: str) -> Any:  # counters, faithfulness_calls, reply_tokens
        return getattr(self.judge, name)

    def cached(self, metric: str, question: str, answer: str, contexts: Sequence[str] = ()) -> bool:
        return item_key(metric, question, answer, contexts) in self._scores

    def _put(self, key: str, score: float) -> None:
        self._scores[key] = score
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"key": key, "score": score}) + "\n")

    async def faithfulness(
        self, question: str, answer: str, contexts: Sequence[str]
    ) -> float | None:
        key = item_key("faithfulness", question, answer, contexts)
        if key in self._scores:
            self.hits += 1
            return self._scores[key]
        score = await self.judge.faithfulness(question, answer, contexts)
        if score is not None:
            self._put(key, score)
        return score

    async def relevancy(self, question: str, answer: str) -> float | None:
        key = item_key("relevancy", question, answer)
        if key in self._scores:
            self.hits += 1
            return self._scores[key]
        score = await self.judge.relevancy(question, answer)
        if score is not None:
            self._put(key, score)
        return score
